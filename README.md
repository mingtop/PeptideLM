# Baselinses for Peptile
`This repo contains two baseline models for peptile prediction.  MLM-BERT and CLM-GPT2.`

## Requirement
Before pretrain BERT/GPT2 model, please install Hugging Face Transformer 
        ``` pip install -q transformers```


**Training bash**
```
python3 run_mlm.py \
   --model_name_or_path Rostlab/prot_bert \
   --train_file ./uniprot_sprot_trembl_60_withSpace.txt \
   --do_train \
   --num_train_epochs 200 \
   --logging_steps 5000 \
   --max_seq_length=64 \
   --logging_dir runs/bert_base_AMP_ALL \
   --per_device_train_batch_size 64 \
   --line_by_line \
   --overwrite_output_dir \
   --learning_rate 1e-5 \
   --validation_split_percentage=5 \
   --output_dir ./test-mlm-BERT-base-AMP-ALL-Epoch200-maxLen64
```

## Code Structure: 

> + src : pretrain codes
> + model   trained model
> + runs    tensorboard-version experimental logs
> + fig     experimental Results
> > + demo.ipynb: demostrate the usage of pretrained models
> > + run_clm/run_mlm.py: the pretrained codes 
> > + run.bash     bash to run




## Experiments Results

**Data distribution of AMP**
<img src=figure/fig_dataDistribution.png>

**Demo to BERT**
<img src=figure/fig_demo_Bert.png>

**Demo to GPT2**
<img src=figure/fig_demo_Gpt2.png>