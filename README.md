# SimCSE with Subword Regularization on Domain-Specific Dataset
**Subword Regularization**([Kudo, 2018](https://arxiv.org/abs/1804.10959)) employs multiple subword segmentations, which means that tokenization results of same corpus can change for every epoch (like following example).
```python
for i in range(3):
    tokenizer.tokenize("chocolate ice cream")
>> ["chocolate", "ice", "cream"]
>> ["cho", "##col", "##ate", "ice", "cream"]
>> ["chocolate", "ice", "cr", "##ea", "##m"]
```
Subword Regularization is largely adopted in NLP systems because of its effect of data augmentation and improving model robustness.<br/><br/>
I hypothesized that *Subword Regularization is more effective on domain-specific task where words distribution is far different from general corpus which tokenizer is trained on*.<br/>
For proving my hypothesis, I applied **MaxMatch-Dropout**([Hiraoka, 2022](https://arxiv.org/abs/2209.04126)), Subword Regularization method for WordPiece Tokenizer, on unsupervised **SimCSE**([Gao et al., 2021](https://arxiv.org/abs/2104.08821)).
### Datasets
I used [Wikipedia corpus](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse)(from [official SimCSE Repo](https://github.com/princeton-nlp/SimCSE)) and [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) for **general** train/test set.<br/><br/>
I adopted [CaseHOLD dataset](https://huggingface.co/datasets/lex_glue) of [LexGLUE](https://github.com/coastalcph/lex-glue), legal NLU benchmark, as **domain-specific** train/test set.<br/>
I used **accuracy** (single prediction-single ground truth of 5 candidates) metric for evaluating model's performance on CaseHOLD, though it is multiple choice (of relevant holdings to present case) task.<br/>
Also, note that labels in CaseHOLD train set **are not used** during training. Only sentences in train set are used.
### Special Thanks :pray:
Thanks for good research and [codes](https://github.com/tatHi/maxmatch_dropout) of MaxMatch-Dropout!
## Usage (OS: Ubuntu)
### Dependencies
* datasets (2.6.1)
* pytorch (1.11.0)
* transformers (4.18.0)
* tensorboard
* numpy
* scipy
* scikit-learn
### Initialization
Clone this repo and download datasets.
```bash
git clone https://github.com/ChainsmokersAI/MaxMatch-SimCSE.git
cd MaxMatch-SimCSE/
sh download_datasets.sh
```
### Training
Training uses *all* available GPUs (*real* batch size is `n(GPUs)*batch_size*accum_steps`). If not, uses CPU instead.<br/>
If `--corpus=general`, model(unsupervised SimCSE) is trained on Wikipedia corpus and when `--corpus=domain`, trained on CaseHOLD train set without labels.<br/>
You can apply MaxMatch-Dropout or not with `--use-maxmatch` option and set its dropout rate via `--p-maxmatch`.<br/>
Model is based on [BERT](https://huggingface.co/bert-base-uncased) of which the size(base or large) is determined by `--model-size`.
```bash
python train.py --corpus=domain \
--use-maxmatch=True \
--model-size=base \
--batch-size=16 \
--accum-steps=1 \
--lr=3e-5 \
--epochs=1 \
--p-maxmatch=0.07
```
### Evaluation
Trained model is evaluated on `sts` or `casehold` dataset (`dev` or `test` split).
```bash
python evaluate.py \
--model-path=./model/maxmatch-simcse-base_domain_batch64_lr3e-05_p0.07_step4000.pth \
--testset=casehold \
--split=dev
```
### Results
All experiments were conducted on *4 x GeForce RTX 3090 GPUs*.<br/>
Model checkpoints were saved per every *250 training steps* and the one which showed best performance (*spearmanr*/*accuracy*) on dev set was picked for final evaluation (on test set).<br/><br/>
On **general** corpus, models were trained with *batch size 32*, *learning rate 3e-5*.<br/>
I picked the best checkpoints on *STS Benchmark dev set* for final one.
|dropout rate|0.0|0.01|0.03|0.05|0.07|0.1|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|||||||||
|**STS Benchmark**|***76.28***|73.79|72.55|73.15|72.48|71.95|
|**CaseHOLD**|**43.99**|40.61|37.98|42.58|40.15|42.28|

On **domain-specific** corpus, models were trained with *batch size 64*.<br/>
And the best checkpoints on *CaseHOLD dev set* were picked finally.
|dropout rate|0.0|0.01|0.03|0.05|0.07|0.1|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
||||||||
|**learning rate**|7e-5|3e-5|3e-5|3e-5|3e-5|3e-5|
|**STS Benchmark**|**49.21**|42.62|43.54|42.34|44.23|48.41|
|**CaseHOLD**|***49.01***|48.25|48.99|***49.56***|***50.66***|***50.51***|

