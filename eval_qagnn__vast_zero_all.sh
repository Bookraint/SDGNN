#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`


dataset="vast_zero_sent"
model='roberta-large'
shift
shift
args=$@


echo "******************************"
echo "dataset: $dataset"
echo "******************************"



###### Eval ######
python3 -u qagnn.py --dataset $dataset \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements data/${dataset}/statement/train.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_attn \
      --save_dir saved_models \
      --mode eval_detail \
      --load_model_path /home/yjx/ZSSD/qagnn-main/saved_models/vast_zero_sent/enc-roberta-large__k5__gnndim200__bs64__seed0__20230114_154924/model.pt.14 \
      $args
