# model=sage
model=gcn
dataset=cora

CUDA_VISIBLE_DEVICES=$1 python ./main.py --config ./config/${model}.yaml --dataset ${dataset} 