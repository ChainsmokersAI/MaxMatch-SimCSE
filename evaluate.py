import argparse

import torch

import transformers
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset

from models import SimCSE

import numpy as np
from scipy import spatial, stats
from sklearn.metrics import classification_report

# Parse Arguments
parser=argparse.ArgumentParser(description="Evaluation Arguments")
# Required
parser.add_argument("--model-path", type=str, required=True, help="Path of Trained Model")
parser.add_argument("--testset", type=str, required=True, help="Test Set: sts|casehold")
parser.add_argument("--split", type=str, required=True, help="Test Set Split: dev|test")
#parser.add_argument("", type=, required=True, help="")
# NOT Required
#parser.add_argument("", type=, default=, help="")
args=parser.parse_args()

# NOT Logging Lower than ERROR Level
transformers.logging.set_verbosity_error()

def load_trained_model(device, model_path):
    """
    Return Trained Tokenizer, Model
    """
    # Model Size
    model_size=model_path.split("/")[-1].split("_")[0].split("-")[-1]

    # Load Pre-Trained
    if model_size=="base":
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
        pretrained=AutoModel.from_pretrained("bert-base-uncased").to(device)
    elif model_size=="large":
        tokenizer=AutoTokenizer.from_pretrained("bert-large-uncased")
        pretrained=AutoModel.from_pretrained("bert-large-uncased").to(device)

    # Load Trained
    model=SimCSE(pretrained=pretrained)
    model.load_state_dict(torch.load(model_path))
    model=model.to(device)

    return tokenizer, model

def evaluate_on_sts(device, model_path, split):
    """
    Evaluate Trained Model on STS Benchmark
    """
    # Load Trained Tokenizer, Model
    tokenizer, model=load_trained_model(device=device, model_path=model_path)

    # Load Dataset
    if split=="dev":
        dataset=open("./dataset/stsbenchmark/sts-dev.csv", "r").read()
    elif split=="test":
        dataset=open("./dataset/stsbenchmark/sts-test.csv", "r").read()

    # Evaluate
    preds=[]
    labels=[]
    
    model.eval()
    with torch.no_grad():
        for data in dataset.split("\n")[:-1]:
            # Parse
            label, sent1, sent2=data.split('\t')[4:7]

            # Encode
            enc1=tokenizer.encode(sent1)
            enc2=tokenizer.encode(sent2)

            # Prediction
            pred=1-spatial.distance.cosine(
                np.array(model.get_embedding(torch.tensor([enc1]).to(device)).detach().cpu()),
                np.array(model.get_embedding(torch.tensor([enc2]).to(device)).detach().cpu())
            )
            preds.append(pred)
            # Labels
            labels.append(float(label))

    # Results
    print(np.corrcoef(preds, labels))
    print(stats.spearmanr(preds, labels))

def evaluate_on_casehold(device, model_path, split):
    """
    Evaluate Trained Model on CaseHOLD
    """
    # Load Trained Tokenizer, Model
    tokenizer, model=load_trained_model(device=device, model_path=model_path)

    # Load Dataset
    if split=="dev":
        dataset=load_dataset("lex_glue", "case_hold")["dev"]
    elif split=="test":
        dataset=load_dataset("lex_glue", "case_hold")["test"]

    # Evaluate
    preds=[]
    labels=[]

    model.eval()
    with torch.no_grad():
        for data in dataset:
            # Context
            enc_context=tokenizer.encode(data["context"])
            embd_context=model.get_embedding(torch.tensor([enc_context]).to(device))

            # Prediction
            pred=-1
            max_sim=-1
            for idx, ending in enumerate(data["endings"]):
                # Ending
                enc_ending=tokenizer.encode(ending)
                #
                sim=1-spatial.distance.cosine(
                    np.array(embd_context.detach().cpu()),
                    np.array(model.get_embedding(torch.tensor([enc_ending]).to(device)).detach().cpu())
                )
                #
                if sim>max_sim:
                    pred=idx
                    max_sim=sim
            preds.append(pred)
            # Labels
            labels.append(data["label"])

    # Results
    print(classification_report(labels, preds))

def main():
    # Arguments Validation
    if args.split not in ["dev", "test"]:
        print("Wrong Split: dev or test")
        return

    # Device
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Evaluate
    if args.testset=="sts":
        # STS Benchmark (General Dataset)
        evaluate_on_sts(
            device=device,
            model_path=args.model_path,
            split=args.split
        )
    elif args.testset=="casehold":
        # CaseHOLD (Domain-Specific Dataset)
        evaluate_on_casehold(
            device=device,
            model_path=args.model_path,
            split=args.split
        )
    else:
        print("Wrong Test Set: sts or casehold")
        return

if __name__=="__main__":
    main()
