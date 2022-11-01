import argparse

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from utils import prepare_dataset_for_train, get_collate_fn

# Parse Arguments
parser=argparse.ArgumentParser(description="Training Arguments")
# Required
parser.add_argument("--corpus", type=str, required=True, help="Training Corpus Type: general|domain")
parser.add_argument("--use-maxmatch", type=str, required=True, help="Whether Use MaxMatch-Dropout or Not: True|False")
#parser.add_argument("", type=, required=True, help="")
# NOT Required
parser.add_argument("--model-size", type=str, default="base", help="Size of PLM(BERT): base|large")
#parser.add_argument("", type=, default=, help="")
args=parser.parse_args()

# NOT Logging Lower than ERROR Level
transformers.logging.set_verbosity_error()

def train():
    """
    Train with a Single Device (GPU or CPU)
    """
    # Path of Pre-Trained LM
    if args.model_size=="base":
        # 110M Params
        model_path="bert-base-uncased"
    elif args.model_size=="large":
        # 340M Params
        model_path="bert-large-uncased"
    else:
        print("Wrong Model Size!")
        return

    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    pretrained=AutoModel.from_pretrained(model_path)

    # Load Dataset
    dataset_train=prepare_dataset_for_train(
        corpus=args.corpus,
        use_maxmatch=args.use_maxmatch,
        tokenizer=tokenizer
    )
    # Load Collate Function
    collate_fn=get_collate_fn(pad_token_id=tokenizer.pad_token_id)
    # Load DataLoader
    dataloader_train=DataLoader(dataset_train, batch_size=3, collate_fn=collate_fn)

    for step, (sent, pos) in enumerate(dataloader_train):
        print(sent)
        print(pos)
        break

def train_ddp():
    print("Train with Multi-GPU!")

def main():
    # Arguments Validation
    if args.corpus not in ["general", "domain"]:
        print("Wrong Corpus Type!")
        return
    if args.use_maxmatch not in ["True", "False"]:
        print("Use MaxMatch-Dropout? Answer: True or False")
        return

    # CUDA Available
    if torch.cuda.is_available():
        # Number of GPUs
        world_size=torch.cuda.device_count()

        # Multi-GPU
        if world_size>=2:
            train_ddp()
        # Single GPU
        else:
            train()
    # CPU
    else:
        train()

if __name__=="__main__":
    main()
