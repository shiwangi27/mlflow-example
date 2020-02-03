import math
import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import cached_path

logger = logging.getLogger(__file__)


DATASETS_URL = {
    'simplebooks-2-raw': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-2-raw/train.txt",
                          'valid': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-2-raw/valid.txt"},
    'simplebooks-92-raw': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-92-raw/train.txt",
                           'valid': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-92-raw/valid.txt"},
    'imdb': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/train.txt",
             'test': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/test.txt"}
    }

DATASETS_LABELS_URL = {
    'imdb': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/train.labels.txt",
             'test': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/test.labels.txt"}
    }


DATASETS_LABELS_CONVERSION = {
    'imdb': {'pos': 0, 'neg': 1}
    }

WEIGHTS_NAME = 'model_checkpoint.pth'
CONFIG_NAME = 'model_training_args.bin'


def get_dataset(dataset_dir, with_labels=False):
    """ Retrieve and cache a dataset with optional labels """

    # If the dataset is in our list of DATASETS_URL, use this url, otherwise, look for 'train.txt' and 'valid.txt' files
    if dataset_dir in DATASETS_URL:
        dataset_map = DATASETS_URL[dataset_dir]
    else:
        dataset_map = {'train': os.path.join(dataset_dir, 'train.txt'),
                       'valid': os.path.join(dataset_dir, 'valid.txt')}

    logger.info("Get dataset from %s", dataset_dir)

    # Download and read dataset and replace a few token for compatibility with the Bert tokenizer we are using
    dataset = {}
    for split_name in dataset_map.keys():
        dataset_file = cached_path(dataset_map[split_name])
        with open(dataset_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            dataset[split_name] = [
                line.strip(' ').replace('<unk>', '[UNK]').replace('\n', '[SEP]' if not with_labels else '')
                for line in tqdm(all_lines)]

    return dataset


def tokenize_dataset(dataset, tokenizer, dataset_cache=None):
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load encoded dataset from cache at %s", dataset_cache)
        encoded_dataset = torch.load(dataset_cache)
    else:
        # Tokenize and encode the dataset
        logger.info("Tokenize and encode the dataset")

        def encode(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, encode(o)) for n, o in obj.items())
            return list(encode(o) for o in tqdm(obj))

        encoded_dataset = encode(dataset)

        # Save to cache
        if dataset_cache:
            logger.info("Save encoded dataset to cache at %s", dataset_cache)
            torch.save(encoded_dataset, dataset_cache)

    return encoded_dataset


def get_data_loaders(args, tokenizer):
    """ Prepare the dataloaders for training and evaluation """
    datasets = get_dataset(args.dataset_path, args.dataset_cache)
    encoded_datasets = tokenize_dataset(datasets, tokenizer)

    logger.info("Convert to Tensor and reshape in blocks of the transformer's input length")
    for split_name in ['train', 'valid']:
        tensor = torch.tensor(encoded_datasets[split_name], dtype=torch.long)
        num_sequences = (tensor.size(0) // args.num_max_positions) * args.num_max_positions
        encoded_datasets[split_name] = tensor.narrow(0, 0, num_sequences).view(-1, args.num_max_positions)

    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(encoded_datasets['train']) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(encoded_datasets['valid']) if args.distributed else None
    train_loader = DataLoader(encoded_datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(encoded_datasets['valid'], sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(encoded_datasets['train'].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(encoded_datasets['valid'].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler, encoded_datasets['train_num_words'], encoded_datasets['valid_num_words']


