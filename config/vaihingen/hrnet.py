from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from geoseg.models.HRNet import HRNet_W48_Proto
from geoseg.loss.loss_helper import FSAuxCELoss

from catalyst.contrib.nn import Lookahead
from catalyst import utils
from geoseg.utils.tools.configer import Configer

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 1
val_batch_size = 1
lr = 6e-4
momentum = 0.9
weight_decay = 0.0005
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "hr18-r18-1024crop512-crop-ms-e100"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "hr18-r18-1024crop512-crop-ms-e100"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [1]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
configer = Configer(configs='./config/vaihingen/H_48_D_4_proto.json')
net = HRNet_W48_Proto(configer)

# define the loss
loss = FSAuxCELoss(configer)
use_aux_loss = False

# define the dataloader

train_dataset = VaihingenDataset(data_root='./data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='./data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
optimizer = torch.optim.SGD(net_params, lr=lr, momentum = momentum,weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100 ,gamma=0.5)

