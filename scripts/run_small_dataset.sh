
output_dir=./ckpts
# for model in gcn sage; do
# for dataset in cora citeseer pubmed; do
#     if ! [ -d "./${output_dir}/${dataset}" ]; then
#         mkdir "./${output_dir}/${dataset}"
#     fi
#     CUDA_VISIBLE_DEVICES=$1 python ./train.py \
#         --config ./config/${model}.yaml \
#         --dataset ${dataset} \
#         --output_dir ${output_dir} 2>&1 | tee ${output_dir}/${dataset}/${model}.log
# done
# done


for model in gcn sage; do
for dataset in arxiv; do
    if ! [ -d "./${output_dir}/${dataset}" ]; then
        mkdir "./${output_dir}/${dataset}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./train.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} 2>&1 | tee ${output_dir}/${dataset}/${model}.log
done
done