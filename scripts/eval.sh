# model=sage
model=gcn
dataset=cora
output_dir=./finetune

CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
    --config ./config/${model}.yaml \
    --dataset ${dataset} \
    --output_dir ${output_dir} \
    --saved_model_path $2