import argparse
import torch
import shutil
import numpy as np
import random
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


@torch.no_grad()
def test(model, data):
    out = get_prediction(model, data)
    y_true = data.y
    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)
    test_acc = compute_micro_f1(out, y_true, data.test_mask)
    return train_acc, valid_acc, test_acc


@torch.no_grad()
def get_prediction(model, data):
    model.eval()
    return model(data.x, data.adj_t)


def edit(model, data, optimizer, loss_op, node_idx_2flip, flipped_label, max_num_step):
    model.train()
    for i in range(max_num_step):
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = loss_op(out[node_idx_2flip].unsqueeze(0), flipped_label)
        loss.backward()
        optimizer.step()
        y_pred = out.argmax(dim=-1)[node_idx_2flip]
        # pdb.set_trace()
        print(f'{i}-th edit step, loss: {loss.item()}, model pred: {y_pred.item()}, label: {flipped_label.item()}')
        if y_pred == flipped_label:
            print(f'successfully flip the model with {i} grad decent steps, break')
            break


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
    model = GNN.from_pretrained(in_channels=num_features, 
                                out_channels=num_classes, 
                                saved_ckpt_path=args.saved_model_path,
                                **model_config['architecture'])

    print(model)
    model.cuda()
    train_data, whole_data = prepare_dataset(model_config, data)
    del data
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    N = 1000
    result = test(model, whole_data)
    train_acc, valid_acc, test_acc = result
    print(f'before edit, train acc {train_acc}, valid acc {valid_acc}, test acc {test_acc}')
    bef_edit_logits = get_prediction(model, whole_data)
    bef_edit_pred = bef_edit_logits.argmax(dim=-1)

    val_nodes = whole_data.val_mask.nonzero().squeeze()
    # pdb.set_trace()
    node_idx_2flip = bef_edit_pred.ne(whole_data.y).nonzero()[123].item()
    # node_idx_2flip = val_nodes[random.randint(0, len(val_nodes))].item()
    bef_edit_y_pred = bef_edit_pred[node_idx_2flip].item()
    y_true = whole_data.y[node_idx_2flip].item()

    loss_op = F.binary_cross_entropy_with_logits if multi_label else F.cross_entropy
    optimizer = get_optimizer(model_config, model)
    # flipped_label = torch.randint(high=num_classes, size=(1,), device='cuda')
    flipped_label = whole_data.y[node_idx_2flip].unsqueeze(dim=0)
    edit(model, whole_data, optimizer, loss_op, node_idx_2flip, flipped_label, 10)

    result = test(model, whole_data)
    train_acc, valid_acc, test_acc = result
    print(f'after edit, train acc {train_acc}, valid acc {valid_acc}, test acc {test_acc}')
    after_edit_logits = get_prediction(model, whole_data)
    after_edit_pred = after_edit_logits.argmax(dim=-1)
    after_edit_y_pred = after_edit_pred[node_idx_2flip].item()
    print(f'edit node_idx: {node_idx_2flip}, true label: {y_true}, before edit model predict: {bef_edit_y_pred},'
          f'corrected label: {flipped_label.item()}, after edit model predict: {after_edit_y_pred}')
