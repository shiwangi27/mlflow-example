import logging
from argparse import ArgumentParser
import torch

logger = logging.getLogger(__file__)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='wikitext-2', help="One of ('wikitext-103', 'wikitext-2') or a dict of splits paths.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")

    parser.add_argument("--embed_dim", type=int, default=410, help="Embeddings dim")
    parser.add_argument("--hidden_dim", type=int, default=2100, help="Hidden dimension")
    parser.add_argument("--num_max_positions", type=int, default=256, help="Max input length")
    parser.add_argument("--num_heads", type=int, default=10, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=16, help="NUmber of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Normal initialization standard deviation")
    parser.add_argument("--sinusoidal_embeddings", action="store_true", help="Use sinusoidal embeddings")

    parser.add_argument("--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n_warmup", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--eval_every", type=int, default=-1, help="Evaluate every X steps (-1 => end of epoch)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradient")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log on main process only, logger.warning => log on all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", format(args))  # This is a logger.info: only printed on the first process

