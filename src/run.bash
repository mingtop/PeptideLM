# Peptide  uniprot_sprot_60.txt   AMP
python -m torch.distributed.launch --nproc_per_node=2 run_mlm.py \
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



# GPT-2 --> Peptide  uniprot_sprot_60.txt   AMP
python -m torch.distributed.launch --nproc_per_node=2  run_clm.py \
    --model_name_or_path gpt2 \
    --train_file ./uniprot_sprot_60_withSpace.txt \
    --do_train \
    --num_train_epochs 200 \
    --logging_steps 1000 \
    --per_device_train_batch_size 2 \
    --overwrite_output_dir \
    --logging_dir runs/gpt2_AMP_test \
    --validation_split_percentage=5 \
    --output_dir ./test-clm-GPT2-test
