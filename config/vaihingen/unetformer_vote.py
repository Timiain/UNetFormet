from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from geoseg.models.UNetFormerVote import UNetFormerVote
from catalyst.contrib.nn import Lookahead
from catalyst import utils

ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

# training hparam
#  define the network # define the loss
net = UNetFormerVote(num_classes=num_classes,output_uncertainty=False)
#net.setCurrentTestStage(net.TestMain)
loss = VoteStageTrainingLoss(ignore_index=ignore_index)
use_aux_loss = True

# normal train main classifer only

#max_epoch = 105
max_epoch = 105
# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

#train main classifer Uncertainty only
full_uncertainty_max_epoch = 45
full_params=[]
for param in net.decoder.segmentation_head_0.parameters():
    full_params.append(param)
for param in net.decoder.uncertainty_0.parameters():
    full_params.append(param)
full_optimizer = torch.optim.Adam(full_params, lr=0.001, weight_decay=0)
full_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(full_optimizer, T_0=15, T_mult=2)

#train sub classifer and Uncertainty
sub_uncertainty_max_epoch = 45
sub_params=[]
for param in net.decoder.segmentation_head_1.parameters():
    sub_params.append(param)
for param in net.decoder.uncertainty_1.parameters():
    sub_params.append(param)
sub_optimizer = torch.optim.Adam(sub_params, lr=0.001, weight_decay=0)
sub_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(sub_optimizer, T_0=15, T_mult=2)

# other definition
weights_name = "unetformer-cascadvote-r18-512-crop-ms-e100"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "unetformer-cascadvote-r18-512-crop-ms-e100"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None


# define the dataloader
train_dataset = VaihingenDataset(data_root='./data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='./data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=0,
                          pin_memory=False,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)



