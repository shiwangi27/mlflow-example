import logging
import importlib
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, MetricsLambda

from transformers import BertTokenizer, cached_path

from text_classifier.deeplearning.transformers.datautils import (get_and_tokenize_dataset, average_distributed_scalar,
                                                                 pad_dataset, add_logging_and_checkpoint_saving,
                                                                 WEIGHTS_NAME, CONFIG_NAME)

logger = logging.getLogger(__file__)


def get_data_loaders(args, tokenizer, max_length, clf_token):
    """ Prepare the dataloaders for training and evaluation.
        Add a classification token at the end of each sample if needed. """
    datasets = get_and_tokenize_dataset(tokenizer, args.dataset_path, args.dataset_cache, with_labels=True)

    logger.info("Convert to Tensor, pad and trim to trim_length")
    for split_name in ['train', 'valid']:
        datasets[split_name] = [x[:max_length-1] + [clf_token] for x in datasets[split_name]]  # trim dataset
        datasets[split_name] = pad_dataset(datasets[split_name])  # pad dataset
        tensor = torch.tensor(datasets[split_name], dtype=torch.long)
        labels = torch.tensor(datasets[split_name + '_labels'], dtype=torch.long)
        datasets[split_name] = TensorDataset(tensor, labels)

    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(datasets['train']) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(datasets['valid']) if args.distributed else None
    train_loader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(datasets['valid'], sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(datasets['train'].tensors[0].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(datasets['valid'].tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train(args):
    # Loading tokenizer, pretrained model and optimizer
    logger.info("Prepare tokenizer, model and optimizer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)  # Let's use a pre-defined tokenizer

    logger.info("Create model from class %s and configuration %s", args.finetuning_model_class,
                os.path.join(args.model_checkpoint, CONFIG_NAME))
    ModelClass = getattr(importlib.import_module("finetuning_model"), args.finetuning_model_class)
    pretraining_args = torch.load(cached_path(os.path.join(args.model_checkpoint, CONFIG_NAME)))
    model = ModelClass(config=pretraining_args, fine_tuning_config=args).to(args.device)

    logger.info("Load pretrained weigths from %s", os.path.join(args.model_checkpoint, WEIGHTS_NAME))
    state_dict = torch.load(cached_path(os.path.join(args.model_checkpoint, WEIGHTS_NAME)), map_location='cpu')
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    logger.info("Parameters discarded from the pretrained model: %s", incompatible_keys.unexpected_keys)
    logger.info("Parameters added in the adaptation model: %s", incompatible_keys.missing_keys)

    model.tie_weights()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info("Model has %s parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    logger.info("Prepare datasets")
    loaders = get_data_loaders(args, tokenizer, pretraining_args.num_max_positions, clf_token=tokenizer.vocab['[CLS]'])
    train_loader, val_loader, train_sampler, valid_sampler = loaders

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch, labels = (t.to(args.device) for t in batch)
        inputs = batch.transpose(0, 1).contiguous()  # to shape [seq length, batch]
        _, (clf_loss, lm_loss) = model(inputs, clf_tokens_mask=(inputs == tokenizer.vocab['[CLS]']),
                                       clf_labels=labels,
                                       lm_labels=inputs,
                                       padding_mask=(batch == tokenizer.vocab['[PAD]']))
        loss = (max(0, args.clf_loss_coef) * clf_loss
                + max(0, args.lm_loss_coef) * lm_loss) / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch, labels = (t.to(args.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()  # to shape [seq length, batch]
            _, clf_logits = model(inputs, clf_tokens_mask=(inputs == tokenizer.vocab['[CLS]']),
                                  padding_mask=(batch == tokenizer.vocab['[PAD]']))
        return clf_logits, labels

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate at the end of each epoch and every 'eval_every' iterations if needed
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_every > 0:
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  lambda engine: evaluator.run(
                                      val_loader) if engine.state.iteration % args.eval_every == 0 else None)
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

    # Learning rate schedule: linearly warm-up to lr and then to zero
    scheduler = PiecewiseLinear(optimizer, 'lr',
                                [(0, 0.0), (args.n_warmup, args.lr), (len(train_loader) * args.n_epochs, 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we average distributed metrics using average_distributed_scalar
    metrics = {"accuracy": Accuracy()}
    metrics.update({"average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model and configuration before we start to train
    if args.local_rank in [-1, 0]:
        checkpoint_handler, tb_logger = add_logging_and_checkpoint_saving(trainer, evaluator, metrics, model,
                                                                          optimizer, args, prefix="finetune_")

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint for easy re-loading
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir,
                                                                     WEIGHTS_NAME))
        tb_logger.close()

