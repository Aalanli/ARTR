# %%
# 
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import FeatureQueries
from model.trainer import Trainer
from model.feature_extractor import VAE
from model.misc import calculate_param_size
from utils.ops import unnormalize

class VAETrainer(Trainer):
    @torch.no_grad()
    def log_dict(self, writer: SummaryWriter, log: dict, prefix: str):
        for k, tensor in log.items():
            name = prefix + '_' + k
            if list(tensor.shape) == []:
                writer.add_scalar(name, tensor, self.steps)
                tensor.zero_()
            elif tensor.dim() == 4:
                tensor = unnormalize(tensor)
                writer.add_images(name, tensor, self.steps)
            else:
                print(f'{k}: {type(tensor)}')

    @torch.no_grad()    
    def average_log(self, log_mean, log_dict):
        for k in log_dict:
            if k not in log_mean:
                log_mean[k] = log_dict[k]
            elif list(log_dict[k].shape) == []:
                log_mean[k].add_(log_dict[k].div_(self.metric_step))
            else:
                log_mean[k] = log_dict[k]

    def train(self, epochs, dataset):
        self.model.train()
        summary_writer = SummaryWriter(self.dir)
        log_mean = {}
        for i in range(epochs):
            for im, im_t in dataset:
                im, im_t = im.cuda(), im_t
                with torch.cuda.amp.autocast(enabled=self.mixed_precison):
                    loss, log_dict = self.model.mse_reconstruction_loss(im)
                self.steps += 1
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.steps)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.average_log(log_mean, log_dict)
                if (self.steps + 1) % self.metric_step == 0:
                    self.log_dict(summary_writer, log_mean, 'train')
                if (self.steps + 1) % self.checkpoint_step == 0:
                    self.regulate_checkpoints()
            self.regulate_checkpoints()  # save final checkpoint



batch_size = 4
layers = [2, 1, 1, 1]
dataset = FeatureQueries('datasets/coco/train2017_query_pool')
dataset = DataLoader(dataset, 
                     batch_size=batch_size, 
                     shuffle=True, 
                     collate_fn=FeatureQueries.collate_fn, 
                     num_workers=4)

vae = VAE(layers=layers, latent_dim=512).cuda()
print('model parameters:', calculate_param_size(vae))
optimizer = torch.optim.Adam(vae.parameters(), 1e-4)

# %%
trainer = VAETrainer(vae,
                     criterion=None,
                     optimizer=optimizer,
                     directory='experiments/feature_extractor/v5',
                     metric_step=200,
                     checkpoint_step=500,
                     mixed_precision=True)
# %%
trainer.train(1, dataset)

