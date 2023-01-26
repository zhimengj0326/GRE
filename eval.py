import argparse
import torch
import shutil
import pdb
import os
import torch.nn.functional as F
import yaml
import editable_gnn.models as models
from data import get_data
from utils import prepare_dataset, compute_micro_f1, get_optimizer, set_seeds_all, sorted_checkpoints
from logger import Logger


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--saved_model_path', type=str, required=True,
                    help='the path to the traiend model')

if __name__ == '__main__':
    args = parser.parse_args()
    set_seeds_all(args.seed)
    with open(args.config, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        if args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        else:
            model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize
    print(args)
    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    GNN = getattr(models, model_config['arch_name'])
    data, num_features, num_classes = get_data(args.root, args.dataset)
    # model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    model = GNN.from_pretrained(in_channels=num_features, 
                                out_channels=num_classes, 
                                saved_ckpt_path=args.saved_model_path,
                                **model_config['architecture'])

    loss_op = F.binary_cross_entropy_with_logits if multi_label else F.cross_entropy
    optimizer = get_optimizer(model_config, model)
    print(model)
    model.cuda()
