# %%
import os
import dill
import json
import pathlib
from itertools import cycle

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    """The general trainer base class"""
    def __init__(
        self, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module = None, 
        optimizer: torch.optim.Optimizer = None, 
        directory: str = None, 
        metric_step: int = None, 
        checkpoint_step: int = None, 
        mixed_precision: bool = False, 
        lr_scheduler: bool = None,
        max_checkpoints: int = 5
    ) -> None:
    
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mixed_precison = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.dir = directory
        self.metric_step = metric_step
        self.checkpoint_step = checkpoint_step

        self.max_checkpoints_saved = max_checkpoints
        self.steps = 0

        self.checkpointable = ['model', 'criterion', 'optimizer', 'scaler', 'steps']
        
        if isinstance(self.model, torch.nn.Module):
            if os.path.exists(self.dir):
                self.restore_latest_checkpoint()
            else:
                # first time loading model
                os.makedirs(self.dir)
                with open(self.dir + '/model_params.json', 'w') as f:
                    json.dump(self.model.config, f)
                # save a pickled referance of the objects
                # in case original class and arguments were forgotten
                with open(os.path.join(self.dir, 'obj_ref.pkl'), 'wb') as f:
                    dill.dump(self.obj_ref, f)
        elif isinstance(self.model, str):
            # if model is a string to the model directory
            self.restore_objects(os.path.join(self.model, 'obj_ref.pkl'))
            self.restore_latest_checkpoint()
    
    @property
    def obj_ref(self):
        return self.__dict__

    def restore_objects(self, file):
        self.__dict__ = torch.load(file)

    def restore_checkpoint(self, file):
        checkpoint = torch.load(file)
        for k in checkpoint:
            if getattr(self.obj_ref[k], 'load_state_dict', None) is not None:
                self.obj_ref[k].load_state_dict(checkpoint[k])
            else:
                self.obj_ref[k] = checkpoint[k]
    
    def restore_latest_checkpoint(self):
        ckpt = self.get_checkpoints()
        if ckpt is not None and len(ckpt) != 0:
            self.restore_checkpoint(ckpt[-1])
            print('Restored checkpoint', ckpt[-1])
    
    def save_checkpoint(self, file):
        checkpoints = {}
        for k in self.checkpointable:
            if getattr(self.obj_ref[k], 'state_dict', None) is not None:
                checkpoints[k] = self.obj_ref[k].state_dict()
            else:
                checkpoints[k] = self.obj_ref[k]
        torch.save(checkpoints, file + '.tar')
    
    def get_checkpoints(self):
        x = sorted(pathlib.Path(self.dir).glob('**/*.tar'))
        if x is None:
            return None
        x = [str(i) for i in x]
        x = [int(os.path.basename(i)[:-4]) for i in x]  # sort by int
        x.sort()
        x = [self.dir + f'/{str(i)}.tar' for i in x]
        return x

    def regulate_checkpoints(self):
        ckpt = self.get_checkpoints()
        if ckpt is not None and len(ckpt) > self.max_checkpoints_saved:
            os.remove(ckpt[0])
        self.save_checkpoint(f'{self.dir}/{self.steps}')
    
    def save_model(self, path=None):
        if path is None:
            torch.save(self.model, f'{self.dir}/model')
        else:
            torch.save(self.model, path)
    
    def load_saved_model(self, path=None):
        """loads a model from the saved model file"""
        if path is None:
            self.model = torch.load(f'{self.dir}/model')
        else:
            self.model = torch.load(path)

    def sum_scalars(self, log: dict, new_log: dict):
        for k in log:
            log[k] += new_log[k]
    
    def zero_scalars(self, log: dict):
        for k in log:
            log[k] = 0
    
    def log_scalars(self, writer: SummaryWriter, log: dict, prefix=''):
        for k in log:
            writer.add_scalar(f'{prefix}/{k}', log[k] / self.metric_step, self.steps)


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
                pred = self.model(x)  # logits = [batch, seq_len, classes]
                loss_dict = self.criterion(pred, y)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                return loss_dict + {'total_loss': float(losses)}
    
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
            
            if eval_data is not None and eval_losses is not None:
                ims, qrs, target = next(eval_data)
                new_eval_log = self.eval_step((ims, qrs), target)
                if eval_losses is None:
                    eval_losses = new_eval_log
                else:
                    self.sum_scalars(eval_losses, new_eval_log)

            if (self.steps + 1) % self.metric_step:
                self.log_scalars(summary_writer, train_losses, 'train')
                self.zero_scalars(train_losses)
                if eval_data is not None:
                    self.log_scalars(summary_writer, eval_losses, 'eval')
                    self.zero_scalars(eval_losses)
            
            if (self.steps + 1) % self.checkpoint_step == 0:
                self.regulate_checkpoints()
        self.regulate_checkpoints()  # save final checkpoint
