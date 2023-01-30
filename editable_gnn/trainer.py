import os
import pdb
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import numpy as np
import re
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data.data import Data
from editable_gnn.models.base import BaseModel
from editable_gnn.logger import Logger
from editable_gnn.utils import set_seeds_all


class WholeGraphTrainer(object):
    def __init__(self, 
                 model: BaseModel, 
                 train_data: Data, 
                 whole_data: Data,
                 model_config: Dict,
                 output_dir: str,
                 dataset_name: str,
                 is_multi_label_task: bool,
                 amp_mode: bool = False,
                 runs: int = 10,
                 seed: int = 0) -> None:
        self.model = model
        self.train_data = train_data
        self.whole_data = whole_data
        self.model_config = model_config
        self.model_name = model_config['arch_name']
        if amp_mode is True:
            raise NotImplementedError
        self.logger = Logger(runs)
        self.runs = runs
        self.optimizer = None
        self.save_path = os.path.join(output_dir, dataset_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.loss_op = F.binary_cross_entropy_with_logits if is_multi_label_task else F.cross_entropy
        self.seed = seed


    @staticmethod
    def train_loop(model: BaseModel, 
                   optimizer: torch.optim.Optimizer, 
                   train_data: Data, 
                   loss_op):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.adj_t)
        loss = loss_op(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()


    def single_run(self, run: int):
        self.model.reset_parameters()
        optimizer = self.get_optimizer(self.model_config, self.model)
        best_val = -1.
        checkpoint_prefix = f'{self.model_name}_run{run}'
        for epoch in range(1, self.model_config['epochs'] + 1):
            train_loss = self.train_loop(self.model, optimizer, self.train_data, self.loss_op)
            result = self.test(self.model, self.whole_data)
            self.logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            # save the model with the best valid acc
            if valid_acc > best_val:
                self.save_model(checkpoint_prefix, epoch)
                best_val = valid_acc

            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train f1: {100 * train_acc:.2f}%, '
                    f'Valid f1: {100 * valid_acc:.2f}% '
                    f'Test f1: {100 * test_acc:.2f}%')
        self.logger.print_statistics(run)

    def save_model(self, checkpoint_prefix: str, epoch: int):
        best_model_checkpoint = os.path.join(self.save_path, f'{checkpoint_prefix}_{epoch}.pt')
        torch.save(self.model.state_dict(), best_model_checkpoint)
        checkpoints_sorted = self.sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, self.save_path)
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            os.remove(f'./{checkpoint}')


    def train(self):
        for run in range(self.runs):
            set_seeds_all(self.seed + run)
            self.single_run(run)
        self.logger.print_statistics()


    @staticmethod
    def compute_micro_f1(logits, y, mask=None) -> float:
        if mask is not None:
            logits, y = logits[mask], y[mask]

        if y.dim() == 1:
            return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
            
        else:
            y_pred = logits > 0
            y_true = y > 0.5

            tp = int((y_true & y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())

            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                return 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                return 0.


    @torch.no_grad()
    def test(self, model, data):
        out = self.prediction(model, data)
        y_true = data.y
        train_acc = self.compute_micro_f1(out, y_true, data.train_mask)
        valid_acc = self.compute_micro_f1(out, y_true, data.val_mask)
        test_acc = self.compute_micro_f1(out, y_true, data.test_mask)
        return train_acc, valid_acc, test_acc


    @staticmethod
    @torch.no_grad()
    def prediction(model, data):
        model.eval()
        return model(data.x, data.adj_t)


    @staticmethod
    def sorted_checkpoints(
        checkpoint_prefix, best_model_checkpoint, output_dir=None, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
            checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted


    @staticmethod
    def get_optimizer(model_config, model):
        if model_config['optim'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
        elif model_config['optim'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
        else:
            raise NotImplementedError
        if model_config['optim'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
        elif model_config['optim'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
        else:
            raise NotImplementedError
        return optimizer


    def eval_edit_quality(self, node_idx_2flip, flipped_label, whole_data, max_num_step, bef_edit_results): 
        bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = bef_edit_results
        results_temporary = self.edit(node_idx_2flip, flipped_label, whole_data, max_num_step)
        train_acc, val_acc, test_acc, succeses, steps = zip(*results_temporary)
        # pdb.set_trace()
        tra_drawdown = np.mean(train_acc) - bef_edit_tra_acc
        val_drawdown = np.mean(val_acc) - bef_edit_val_acc
        test_drawdown = np.mean(test_acc) - bef_edit_tst_acc
        return dict(bef_edit_tra_acc=bef_edit_tra_acc, 
                    bef_edit_val_acc=bef_edit_val_acc, 
                    bef_edit_tst_acc=bef_edit_tst_acc, 
                    tra_drawdown=-tra_drawdown * 100, 
                    val_drawdown=-val_drawdown * 100, 
                    test_drawdown=-test_drawdown * 100, 
                    success_rate=np.mean(succeses),
                    mean_complexity=np.mean(steps)
                    )

    def single_edit(self, idx, label, optimizer, max_num_step):
        success = False
        for step in range(1, max_num_step + 1):
            optimizer.zero_grad()
            out = self.model(self.whole_data.x, self.whole_data.adj_t)
            loss = self.loss_op(out[idx], label)
            loss.backward()
            optimizer.step()
            y_pred = out.argmax(dim=-1)[idx]
            if y_pred == label:
                # print(f'successfully flip the model with {i} grad decent steps, break')
                success = True
                break
        return success, loss, step
        

    def edit(self, node_idx_2flip, flipped_label, whole_data, max_num_step):
        self.model.train()
        optimizer = self.get_optimizer(self.model_config, self.model)
        results_temporary = []
        for idx, f_label in tqdm(zip(node_idx_2flip, flipped_label)):
            success, loss, steps = self.single_edit(idx, f_label, optimizer, max_num_step)
            results_temporary.append((*self.test(self.model, whole_data), success, steps))
        return results_temporary


    def select_node(self, whole_data: Data, 
                    num_classes: int, 
                    num_samples: int, 
                    criterion: str, 
                    from_valid_set: bool = True):
        self.model.eval()
        bef_edit_logits = self.prediction(self.model, whole_data)
        bef_edit_pred = bef_edit_logits.argmax(dim=-1)
        val_y_true = whole_data.y[whole_data.val_mask]
        val_y_pred = bef_edit_pred[whole_data.val_mask]
        if from_valid_set:
            nodes_set = whole_data.val_mask.nonzero().squeeze()
        else:
            raise NotImplementedError
        assert criterion in ['wrong2correct', 'random']
        if criterion == 'wrong2correct':
            wrong_pred_set = val_y_pred.ne(val_y_true).nonzero()
            val_node_idx_2flip = wrong_pred_set[torch.randperm(len(wrong_pred_set))[:num_samples]]
            node_idx_2flip = nodes_set[val_node_idx_2flip]
            flipped_label = whole_data.y[node_idx_2flip]
        elif criterion == 'random':
            node_idx_2flip = nodes_set[torch.randint(high=len(nodes_set), size=(num_samples, 1))]
            flipped_label = torch.randint(high=num_classes, size=(num_samples, 1))
        else:
            raise NotImplementedError
        return node_idx_2flip, flipped_label