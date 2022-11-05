import argparse

import torch
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from utils import prepare_dataset_for_train, get_collate_fn
from models import SimCSE

# Parse Arguments
parser=argparse.ArgumentParser(description="Training Arguments")
# Required
parser.add_argument("--corpus", type=str, required=True, help="Training Corpus Type: general|domain")
parser.add_argument("--use-maxmatch", type=str, required=True, help="Whether Use MaxMatch-Dropout or Not: True|False")
#parser.add_argument("", type=, required=True, help="")
# NOT Required
parser.add_argument("--model-size", type=str, default="base", help="Size of PLM(BERT): base|large")
parser.add_argument("--max-seq-len", type=int, default=128, help="Max Input Sequence Length")
parser.add_argument("--batch-size", type=int, default=64, help="Batch Size")
parser.add_argument("--accum-steps", type=int, default=1, help="Accumulation Steps")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning Rate")
parser.add_argument("--epochs", type=int, default=1, help="Epochs")
parser.add_argument("--p-maxmatch", type=float, default=0.3, help="MaxMatch-Dropout Rate")
#parser.add_argument("", type=, default=, help="")
args=parser.parse_args()

# NOT Logging Lower than ERROR Level
transformers.logging.set_verbosity_error()

def train(device):
    """
    Train with a Single Device (GPU or CPU)
    """
    # Path of Pre-Trained LM
    if args.model_size=="base":
        # 110M Params
        pretrained_path="bert-base-uncased"
    elif args.model_size=="large":
        # 340M Params
        pretrained_path="bert-large-uncased"
    else:
        print("Wrong Model Size!")
        return

    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(pretrained_path)
    pretrained=AutoModel.from_pretrained(pretrained_path).to(device)

    # Load Dataset
    dataset_train=prepare_dataset_for_train(
        corpus=args.corpus,
        use_maxmatch=args.use_maxmatch,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        p_maxmatch=args.p_maxmatch
    )
    # Load Collate Function
    collate_fn=get_collate_fn(pad_token_id=tokenizer.pad_token_id)
    # Load DataLoader
    dataloader_train=DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    model=SimCSE(pretrained=pretrained).to(device)
    model.train()
    # Optimizer, Scheduler
    optimizer=AdamW(model.parameters(), lr=args.lr, no_deprecation_warning=True)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(args.epochs*len(dataset_train)/(args.batch_size*args.accum_steps))
    )
    # Mixed Precision: GradScaler
    scaler=amp.GradScaler()

    # Tensorboard
    writer=SummaryWriter()
    # Training
    step_global=0
    for epoch in range(args.epochs):
        _loss=0
        optimizer.zero_grad()
        
        for step, (sent, pos) in enumerate(dataloader_train):
            # Load Data on Device
            sent=sent.to(device)
            pos=pos.to(device)
            
            # Forward
            with amp.autocast():
                loss=model(sent, pos)
                loss=loss/args.accum_steps
            # Backward
            scaler.scale(loss).backward()
            _loss+=loss.item()

            # Step
            if (step+1)%args.accum_steps==0:
                step_global+=1
                
                # Model Path
                model_path="_".join([
                    "simcse-"+args.model_size,
                    args.corpus,
                    "batch"+str(args.batch_size*args.accum_steps),
                    "lr"+str(args.lr)
                ])
                if args.use_maxmatch=="True":
                    model_path="maxmatch-"+model_path

                # Tensorboard
                writer.add_scalar(
                    "loss_train/"+model_path+f'_epochs{args.epochs}',
                    _loss,
                    step_global
                )
                _loss=0
                
                # Optimizer, Scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Eval Phase, Save Model
                if (step_global)%250==0:
                    # Save Model
                    torch.save(
                        model.state_dict(),
                        "./model/"+model_path+f'_step{step_global}.pth'
                    )

def train_ddp(rank, world_size):
    """
    Train with Multi-GPUs
    """
    # Create Default Process Group
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:8973", rank=rank, world_size=world_size)

    # Path of Pre-Trained LM
    if args.model_size=="base":
        # 110M Params
        pretrained_path="bert-base-uncased"
    elif args.model_size=="large":
        # 340M Params
        pretrained_path="bert-large-uncased"
    else:
        print("Wrong Model Size!")
        return

    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(pretrained_path)
    pretrained=AutoModel.from_pretrained(pretrained_path).to(rank)

    # Load Dataset
    dataset_train=prepare_dataset_for_train(
        corpus=args.corpus,
        use_maxmatch=args.use_maxmatch,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        p_maxmatch=args.p_maxmatch
    )
    # Load Collate Function
    collate_fn=get_collate_fn(pad_token_id=tokenizer.pad_token_id)
    # Load DataLoader
    sampler=DistributedSampler(dataset_train)
    dataloader_train=DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler)

    # Model
    model=SimCSE(pretrained=pretrained).to(rank)
    model_ddp=DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_ddp.train()
    # Optimizer, Scheduler
    optimizer=AdamW(model_ddp.parameters(), lr=args.lr, no_deprecation_warning=True)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(args.epochs*len(dataset_train)/(args.batch_size*args.accum_steps*world_size))
    )
    # Mixed Precision: GradScaler
    scaler=amp.GradScaler()

    # Tensorboard
    writer=SummaryWriter()
    # Training
    step_global=0
    for epoch in range(args.epochs):
        _loss=0
        optimizer.zero_grad()
        
        for step, (sent, pos) in enumerate(dataloader_train):
            # Load Data on Device
            sent=sent.to(rank)
            pos=pos.to(rank)
            
            # Forward
            with amp.autocast():
                loss=model_ddp(sent, pos)
                loss=loss/args.accum_steps
            # Backward
            scaler.scale(loss).backward()
            _loss+=loss.item()

            # Step
            if (step+1)%args.accum_steps==0:
                step_global+=1
                
                # Model Path
                model_path="_".join([
                    "simcse-"+args.model_size,
                    args.corpus,
                    "batch"+str(args.batch_size*args.accum_steps*world_size),
                    "lr"+str(args.lr)
                ])
                if args.use_maxmatch=="True":
                    model_path="maxmatch-"+model_path

                # Tensorboard
                if rank==0:
                    writer.add_scalar(
                        "loss_train/"+model_path+f'_epochs{args.epochs}',
                        _loss,
                        step_global
                    )
                _loss=0
                
                # Optimizer, Scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Eval Phase, Save Model
                if (step_global)%250==0:
                    # Save Model
                    if rank==0:
                        torch.save(
                            model_ddp.module.state_dict(),
                            "./model/"+model_path+f'_step{step_global}.pth'
                        )
                    # Block Process
                    dist.barrier()
                    # Load Model
                    model_ddp.module.load_state_dict(torch.load(
                        "./model/"+model_path+f'_step{step_global}.pth',
                        map_location={'cuda:%d' % 0: 'cuda:%d' % rank}
                    ))

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
            mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
        # Single GPU
        else:
            train(device=torch.device("cuda:0"))
    # CPU
    else:
        train()

if __name__=="__main__":
    main()
