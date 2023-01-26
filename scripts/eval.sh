# model=sage
model=gcn
dataset=cora


CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
    --config ./config/${model}.yaml \
    --dataset ${dataset} \
    --saved_model_path $2