import logging
import math

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BertTokenizer

from ignite.engine import Engine, Events
from ignite.contrib.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.metrics import Loss, MetricsLambda

from text_classifier.deeplearning.transformers.pretraining_model import TransformerWithLMHead
from text_classifier.deeplearning.transformers.datautils import get_and_tokenize_dataset, average_distributed_scalar
from text_classifier.deeplearning.transformers.datautils import add_logging_and_checkpoint_saving


logger = logging.getLogger(__file__)


def get_data_loaders(args, tokenizer):
    """ Prepare the dataloaders for training and evaluation """
    datasets = get_and_tokenize_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Convert to Tensor and reshape in blocks of the transformer's input length")
    for split_name in ['train', 'valid']:
        tensor = torch.tensor(datasets[split_name], dtype=torch.long)
        num_sequences = (tensor.size(0) // args.num_max_positions) * args.num_max_positions
        datasets[split_name] = tensor.narrow(0, 0, num_sequences).view(-1, args.num_max_positions)

    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        datasets['train']) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        datasets['valid']) if args.distributed else None
    train_loader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(datasets['valid'], sampler=valid_sampler, batch_size=args.valid_batch_size,
                              shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(datasets['train'].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(datasets['valid'].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler, \
           datasets['train_num_words'], datasets['valid_num_words']


def train(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
    args.num_embeddings = len(tokenizer.vocab)  # We need this to create the model at next line (number of embeddings to use)
    model = TransformerWithLMHead(args)
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info("Model has %s parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler, train_num_words, valid_num_words = get_data_loaders(args,
                                                                                                                tokenizer)

    # Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original
    def mask_tokens(inputs):
        labels = inputs.clone()
        masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).byte()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).byte() & masked_indices
        inputs[indices_replaced] = tokenizer.vocab["[MASK]"]  # 80% of the time, replace masked input tokens with [MASK]
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).byte() & masked_indices & ~indices_replaced
        random_words = torch.randint(args.num_embeddings, labels.shape, dtype=torch.long, device=args.device)
        inputs[indices_random] = random_words[
            indices_random]  # 10% of the time, replace masked input tokens with random word
        return inputs, labels

    def update(engine, batch):
        model.train()
        inputs = batch.transpose(0, 1).contiguous().to(args.device)
        inputs, labels = mask_tokens(inputs) if args.mlm else (inputs, inputs)
        logits, loss = model(inputs, labels=labels)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs = batch.transpose(0, 1).contiguous().to(args.device)
            inputs, labels = mask_tokens(inputs) if args.mlm else (
            inputs, inputs)  # Prepare masked input/labels if we use masked LM
            logits = model(inputs)
            shift_logits = logits[:-1] if not args.mlm else logits
            shift_labels = labels[1:] if not args.mlm else labels
            return shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1)
    evaluator = Engine(inference)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_every > 0:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED,
            lambda engine: evaluator.run(val_loader) if engine.state.iteration % args.eval_every == 0 else None)
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

    # Learning rate schedule: linearly warm-up to lr and then decrease the learning rate to zero with cosine schedule
    cos_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, len(train_loader) * args.n_epochs)
    scheduler = create_lr_scheduler_with_warmup(cos_scheduler, 0.0, args.lr, args.n_warmup)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we average distributed metrics using average_distributed_scalar
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    # Let's convert sub-word perplexities in word perplexities. If you need details: http://sjmielke.com/comparing-perplexities.htm
    metrics["average_word_ppl"] = MetricsLambda(
        lambda x: math.exp(x * val_loader.dataset.numel() / valid_num_words), metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model and configuration before we start to train
    if args.local_rank in [-1, 0]:
        checkpoint_handler, tb_logger = add_logging_and_checkpoint_saving(trainer, evaluator, metrics, model,
                                                                          optimizer, args)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

