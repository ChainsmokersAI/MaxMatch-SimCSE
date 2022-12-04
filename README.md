# SimCSE with Subword Regularization on Domain-Specific Dataset
***Subword Regularization***([Kudo, 2018](https://arxiv.org/abs/1804.10959)) employs multiple subword segmentations, which means that tokenization results of same corpus can change for every epoch (like following example).
```
for i in range(3):
    tokenizer.tokenize("chocolate ice cream")
>> [chocolate, ice, cream]
>> [cho, ##col, ##ate, ice, cream]
>> [chocolate, ice, cr, ##ea, ##m]
```
*Subword Regularization* is largely adopted in NLP systems because of its effect of data augmentation and improving model robustness.<br/><br/>

I hypothesized that *Subword Regularization* is **more effective on domain-specific task where words distribution is far different from general corpus** which tokenizer is trained on.<br/>
For proving my hypothesis, I applied ***MaxMatch-Dropout***([Hiraoka, 2022](https://arxiv.org/abs/2209.04126)), *Subword Regularization* method for WordPiece Tokenizer, on unsupervised ***SimCSE***([Gao et al., 2021](https://arxiv.org/abs/2104.08821)).
### Datasets

