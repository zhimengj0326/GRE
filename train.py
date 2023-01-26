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
parser.add_argument('--runs', default=1, type=int,
                    help='number of runs')
parser.add_argument('--output_dir', default='./ckpts', type=str)



def train(model, optimizer, data, loss_op):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = loss_op(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.adj_t)
    y_true = data.y
    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)
    test_acc = compute_micro_f1(out, y_true, data.test_mask)
    return train_acc, valid_acc, test_acc



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
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    loss_op = F.binary_cross_entropy_with_logits if multi_label else F.cross_entropy
    optimizer = get_optimizer(model_config, model)
    model.cuda()

    print(model)
    train_data, whole_data = prepare_dataset(model_config, data)
    del data
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')

    logger = Logger(args.runs, args)

    save_path = os.path.join(args.output_dir, args.dataset)
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = model_config['arch_name']

    for run in range(args.runs):
        set_seeds_all(args.seed + run)
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        best_val = -1
        checkpoint_prefix = f'{model_name}_run{run}'
        for epoch in range(1, model_config['epochs'] + 1):
            train(model, optimizer, train_data, loss_op)
            result = test(model, whole_data)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            # save the model with the best valid acc
            if valid_acc > best_val:
                best_model_checkpoint = os.path.join(save_path, f'{checkpoint_prefix}_{epoch}.pt')
                best_val = valid_acc
                torch.save(model.state_dict(), best_model_checkpoint)
                checkpoints_sorted = sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, save_path)
                number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
                checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
                for checkpoint in checkpoints_to_be_deleted:
                    os.remove(f'./{checkpoint}')

            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train f1: {100 * train_acc:.2f}%, '
                    f'Valid f1: {100 * valid_acc:.2f}% '
                    f'Test f1: {100 * test_acc:.2f}%')
        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()