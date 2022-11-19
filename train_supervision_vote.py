import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from tools.metricv2 import Evaluatorv2
from pytorch_lightning.loggers import CSVLogger
import random
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=False)
    return parser.parse_args()


def get_mma_loss(weight):
    '''
    MMA regularization in PyTorch
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    # for convolutional layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())

    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))

    # maxmize the minimum angle
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss

def l2_penalty(w,item_count = 6):
    tmp = torch.norm(w,dim=1)
    return torch.abs(torch.sum(tmp)/item_count)

def norm_balance(w,item_count = 6):
    p_norms = torch.norm(w,dim=1).squeeze(-1).squeeze(-1)
    p = p_norms.repeat(item_count).view(item_count,-1).t()
    p_t = p.t()
    final_p = torch.abs(p_t - p)
    norm_balance= torch.sum(final_p)
    return norm_balance

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False

        self.loss = config.loss
        self.loss.mode = self.net.currentTrainStage

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # supervision stage
        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad()

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log_dict = {"train_loss": loss, 'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def validation_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        log_dict = {"val_loss": loss, 'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        if self.net.currentTrainStage == self.net.TrainMainClassifier:
            optimizer = self.config.optimizer
            lr_scheduler = self.config.lr_scheduler
            return [optimizer], [lr_scheduler]
        elif self.net.currentTrainStage == self.net.TrainFullUncertaintyClassifier:
            optimizer = self.config.full_optimizer
            lr_scheduler = self.config.full_scheduler
            return [optimizer], [lr_scheduler]
        elif self.net.currentTrainStage == self.net.TrainShadowUncertaintyClassifier:
            optimizer = self.config.sub_optimizer
            lr_scheduler = self.config.sub_scheduler
            return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader




# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    #stage0 train epoch  
    config.net.setCurrentTrainStage(config.net.TrainMainClassifier)
    config.net.setCurrentTestStage(config.net.TestMain)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger)
    trainer.fit(model=model)

    #stage1 train epoch 
    config.net.setCurrentTrainStage(config.net.TrainFullUncertaintyClassifier)
    config.net.setCurrentTestStage(config.net.TestFull)

    fullcls_model = Supervision_Train(config)
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.full_uncertainty_max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger)
    trainer.fit(model=fullcls_model)

    #stage2 train epoch 
    config.net.setCurrentTrainStage(config.net.TrainShadowUncertaintyClassifier)
    config.net.setCurrentTestStage(config.net.TestShadow)

    subcls_model = Supervision_Train(config)
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.sub_uncertainty_max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger)
    trainer.fit(model=subcls_model)


if __name__ == "__main__":
   main()
    # trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
    #                      check_val_every_n_epoch=config.check_val_every_n_epoch,
    #                      callbacks=[checkpoint_callback], strategy=config.strategy,
    #                      resume_from_checkpoint=config.resume_ckpt_path, logger=logger)