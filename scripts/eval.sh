# model=sage
model=gcn
dataset=cora
output_dir=./finetune

if ! [ -d "./${output_dir}/${dataset}" ]; then
    mkdir "./${output_dir}/${dataset}"
fi

CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
    --config ./config/${model}.yaml \
    --dataset ${dataset} \
    --output_dir ${output_dir} \
    --saved_model_path $2 2>&1 | tee ${output_dir}/${dataset}/${model}_eval.log