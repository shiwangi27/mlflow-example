import logging
import math
import os

import torch
from torch.optim import Adam

from transformers import BertTokenizer
from ignite.engine import Engine, Events

from text_classifier.deeplearning.transformer import TransformerWithLMHead
from text_classifier.deeplearning.datautils import get_data_loaders


logger = logging.getLogger(__file__)


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

