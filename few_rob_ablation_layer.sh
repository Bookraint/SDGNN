#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
dt=`date '+%Y%m%d_%H%M%S'`


dataset="vast_few"
# model='SentiX_Base_Model'
model='bert-base-uncased'
model_fix='SentiX_Base_Model'
shift
shift
args=$@


elr="3e-5"
dlr="1e-4"
bs=128
mbs=2
n_epochs=40
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges
output_mode=3
lr_schedule=fixed

# k=5 #num of gnn layers
# k=4 #num of gnn layers
# k=3 #num of gnn layers
# k=6 #num of gnn layers
k=7 #num of gnn layers
gnndim=200
max_seq_len=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs

###### Training ######
for seed in 0; do
  python3 -u qagnn.py --dataset $dataset \
      --encoder $model --encoder_fix $model_fix -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed --max_seq_len $max_seq_len --lr_schedule $lr_schedule \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --output_mode ${output_mode} \
      --save_model \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}_outputmode${output_mode} $args \
  > logs/train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done
