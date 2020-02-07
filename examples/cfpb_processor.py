import os

import torch
from transformers import DataProcessor, InputExample


class CfpbProcessor(DataProcessor):
    """Processor for the CFPB data set (Custom data)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['Debt collection',
                'Credit reporting, credit repair services, or other personal consumer reports',
                'Credit card or prepaid card',
                'Checking or savings account',
                'Vehicle loan or lease',
                'Payday loan, title loan, or personal loan',
                'Mortgage',
                'Student loan',
                'Money transfer, virtual currency, or money service']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

