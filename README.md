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
```bash
git clone https://github.com/ChainsmokersAI/MaxMatch-SimCSE.git
cd MaxMatch-SimCSE/
sh download_datasets.sh
```
### Training
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
```bash
python evaluate.py \
--model-path=./model/maxmatch-simcse-base_domain_batch64_lr3e-05_p0.07_step4000.pth \
--testset=casehold \
--split=dev
```

