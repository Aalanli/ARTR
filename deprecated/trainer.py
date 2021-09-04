from itertools import cycle

import torch
from model.trainer import Trainer

from torch.utils.tensorboard import SummaryWriter


class TrainerV1(Trainer):
    """Specific implementation for baseline"""
    def __init__(
        self, 
        model: torch.nn.Module = None, 
        criterion: torch.nn.Module = None, 
        optimizer: torch.optim.Optimizer = None, 
        directory: str = None, 
        metric_step: int = None, 
        checkpoint_step: int = None, 
        mixed_precision: bool = None, 
        lr_scheduler: bool = None, 
        max_checkpoints: int = None,
        loss_weight_dict: dict = {}) -> None:
        super().__init__(model, 
                         criterion=criterion, 
                         optimizer=optimizer, 
                         directory=directory, 
                         metric_step=metric_step, 
                         checkpoint_step=checkpoint_step, 
                         mixed_precision=mixed_precision, 
                         lr_scheduler=lr_scheduler, 
                         max_checkpoints=max_checkpoints)
        self.loss_weight_dict = loss_weight_dict

    def train_step(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.mixed_precison):
            self.model.train()
            pred = self.model(*x)  # List[im], List[List[query_im]]
            loss_dict = self.criterion(pred, y)
            weight_dict = self.loss_weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        self.steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.steps)
        self.optimizer.zero_grad()
        self.scaler.scale(losses).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # disconnect from graph, for logging purposes
        loss = {k: float(v) for k, v in loss_dict.items()}
        loss.update({'total_loss': float(losses)})
        return loss
    
    def eval_step(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.mixed_precison):
            with torch.no_grad():
                self.model.eval()
                self.criterion.eval()
                pred = self.model(*x)  # logits = [batch, seq_len, classes]
                loss_dict = self.criterion(pred, y)
                weight_dict = self.loss_weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_dict.update({'total_loss': float(losses)})
                return loss_dict
    
    @staticmethod
    def call_functions(tensor, attr_args):
        """Helper function for recursive_cast"""
        for k in attr_args:    
            tensor = tensor.__getattribute__(k)(*attr_args[k])
        return tensor

    @staticmethod
    def recursive_cast(nested_tensor_list: list, attr_args):
        """Recursively casts nested tensor lists by attr_args, inplace if list, else returns the casted tensor"""
        if isinstance(nested_tensor_list, list):
            for i in range(len(nested_tensor_list)):
                if isinstance(nested_tensor_list[i], torch.Tensor):
                    nested_tensor_list[i] = TrainerV1.call_functions(nested_tensor_list[i], attr_args)
                elif isinstance(nested_tensor_list[i], list):
                    TrainerV1.recursive_cast(nested_tensor_list[i], attr_args)
        else:
            return TrainerV1.call_functions(nested_tensor_list, attr_args)

    def train(self, train_data, eval_data=None):
        """train one epoch"""
        summary_writer = SummaryWriter(self.dir)

        if eval_data is not None:
            eval_data = cycle(eval_data)
            eval_losses: dict = None  # placeholder

        train_losses: dict = None  # placeholder

        for ims, qrs, target in train_data:
            self.recursive_cast(ims, {'cuda': []})
            self.recursive_cast(qrs, {'cuda': []})
            for t in target:
                for k in t:
                    t[k] = t[k].cuda()
 
            new_train_log = self.train_step((ims, qrs), target)
            if train_losses is None:
                train_losses = new_train_log
            else:
                # average logging across self.metric_step(s)
                self.sum_scalars(train_losses, new_train_log)
            
            if eval_data is not None:
                ims, qrs, target = next(eval_data)
                self.recursive_cast(ims, {'cuda': []})
                self.recursive_cast(qrs, {'cuda': []})
                for t in target:
                    for k in t:
                        t[k] = t[k].cuda()
                new_eval_log = self.eval_step((ims, qrs), target)
                if eval_losses is None:
                    eval_losses = new_eval_log
                else:
                    self.sum_scalars(eval_losses, new_eval_log)

            if (self.steps + 1) % self.metric_step == 0:
                self.log_scalars(summary_writer, train_losses, 'train')
                self.zero_scalars(train_losses)
                if eval_data is not None:
                    self.log_scalars(summary_writer, eval_losses, 'eval')
                    self.zero_scalars(eval_losses)
            
            if (self.steps + 1) % self.checkpoint_step == 0:
                self.regulate_checkpoints()
        self.regulate_checkpoints()  # save final checkpoint