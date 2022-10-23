import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from data.data_RGB import get_training_data
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from model.MSSNet import DeblurNet
from train.trainer_dp import Trainer

parser = argparse.ArgumentParser(description='deblur arguments')
parser.add_argument("--batchsize",type = int, default = 16)
parser.add_argument("--cropsize",type = int, default = 256)
parser.add_argument("--numworker",type = int, default = 2)
parser.add_argument("--lr_initial", type = float, default = 2e-4)
parser.add_argument("--lr_min", type = float, default = 1e-6)
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=3000)

parser.add_argument("--train_datalist",type=str, default='./datalist/datalist_gopro.txt')
parser.add_argument("--val_datalist",type=str, default='./datalist/datalist_gopro_test.txt')
parser.add_argument("--checkdir",type=str,default='./checkpoint')
parser.add_argument("--loadchdir",type=str,default='./checkpoint/model_00600E.pt')
parser.add_argument("--data_root_dir",type=str,default='./dataset/GOPRO_Large/train')
parser.add_argument("--val_root_dir",type=str,default='./dataset/validation_data')

parser.add_argument("--isloadch", action="store_true")
parser.add_argument("--isval", action="store_true")
parser.add_argument("--mgpu", action="store_true")

parser.add_argument("--wf",type=int,default=54)
parser.add_argument("--scale",type=int,default=42)
parser.add_argument("--vscale",type=int,default=42)

parser.set_defaults(isloadch=False)
parser.set_defaults(isval=False)
parser.set_defaults(mgpu=False)
args = parser.parse_args()

#Hyper Parameters
lr_initial = args.lr_initial
lr_min = args.lr_min
gpu = args.gpu
NUM_WORKER = args.numworker
BATCH_SIZE = args.batchsize
CROP_SIZE = args.cropsize

#initial
train_log_dir = os.path.join(args.checkdir, 'tlog')

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def main():
    print("start")

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)

    if os.path.exists(args.checkdir) == False:
        os.makedirs(args.checkdir)

    train_writer = SummaryWriter(logdir=train_log_dir)
    deblur_model = DeblurNet(wf=args.wf, scale=args.scale, vscale=args.vscale)

    optimizer = torch.optim.Adam(deblur_model.parameters(), lr=lr_initial, betas=(0.9, 0.999),eps=1e-8)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch-warmup_epochs, lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    ############################

    if args.isloadch: 
        load_path = os.path.join(args.loadchdir)
        if os.path.exists(load_path):
            checkpoint = torch.load(str(load_path))
            deblur_model.load_state_dict(checkpoint['model_state_dict'])
            deblur_model.cuda(gpu)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            all_step = checkpoint['all_step']

            for i in range(1, start_epoch):
                scheduler.step()
            new_lr = scheduler.get_lr()[0]

            print('------------------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            print('==> start epoch:',start_epoch)
            print("==> load DeblurNet success!")
            print('------------------------------------------------------------------------------')
    else:
        print("initializing....")
        deblur_model.cuda(gpu)
        start_epoch = 1
        all_step = 0

    if args.mgpu:
        deblur_model = nn.DataParallel(deblur_model)
        print('use data parallel')

    ######### DataLoaders ###########
    train_dataset = get_training_data(args.train_datalist, args.data_root_dir, {'patch_size':CROP_SIZE})
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER, drop_last=False, pin_memory=True)
    print('train_batch_num:' ,len(train_loader))
    Trainer(args).train(deblur_model,train_loader,optimizer,scheduler,train_writer,start_epoch,all_step)

    train_writer.close()

if __name__ == '__main__':
    main()
