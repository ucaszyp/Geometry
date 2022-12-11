#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from os.path import exists, join
from datetime import datetime

import numpy as np
import shutil

from tqdm import tqdm
import sys
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.distributed as dist
import torch.utils.data.distributed

import torch.nn.functional as F

from model.models import DPTDepthModel, DPTNormalUncertainModel, DPTDepthUncertainModel
from utils import *
from dataset import *
from losses import compute_loss, silog_loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

task_list = None
middle_task_list = None

def validate_normal(args, val_loader, model, total_iter):
    print("=============validation begin==============")
    val_loader = tqdm(val_loader, desc="Loop: Validation")
    model.eval()
    sub_epoch = total_iter / (args.val_freq * args.batch_size)
    with torch.no_grad():
        total_normal_errors = None
        
        for i, data_dict in enumerate(val_loader):

            # data to device
            img = data_dict['img'].cuda()
            gt_norm = data_dict['norm'].cuda()
            gt_norm_mask = data_dict['norm_valid_mask'].cuda()
            img_path = data_dict['img_path'][0]

            # forward pass

            norm_out_list, _, _ = model(img, gt_norm_mask=gt_norm_mask, mode='val')
            norm_out = norm_out_list[-1]

            # upsample if necessary
            if norm_out.size(2) != gt_norm.size(2):
                norm_out = F.interpolate(norm_out, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)

            pred_norm = norm_out[:, :3, :, :]  # (B, 3, H, W)
            pred_kappa = norm_out[:, 3:, :, :]  # (B, 1, H, W)
            if dist.get_rank()== 0 and args.vis and sub_epoch % 5 == 0:
                vis(args, img_path, norm_out_list, gt_norm, i, sub_epoch)

            prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            E = torch.acos(prediction_error) * 180.0 / np.pi

            mask = gt_norm_mask[:, 0, :, :]
            if total_normal_errors is None:
                total_normal_errors = E[mask]
            else:
                total_normal_errors = torch.cat((total_normal_errors, E[mask]), dim=0)

        total_normal_errors = total_normal_errors.data.cpu().numpy()
        metrics = compute_normal_errors(total_normal_errors)
        log_path = join(args.log_dir, "eval.txt")
        if args.loacl_rank == 0:
            log_normal_errors(metrics, log_path, first_line='epoch: {}'.format(sub_epoch + 1))    

        return metrics

def validate_depth(args, val_loader, model, total_iter):
    print("=============validation begin==============")
    val_loader = tqdm(val_loader, desc="Loop: Validation")
    model.eval()
    depth_metrics = {'silog': [], 'abs_rel': [], 'log10': [], 'rms': [], 'sq_rel': [], 
                    'log_rms': [], 'd1': [], 'd2': [], 'd3': []}
    sub_epoch = total_iter / (args.val_freq * args.batch_size)

    with torch.no_grad():    
        for i, data_dict in enumerate(val_loader):

            # data to device
            img = data_dict['img'].cuda()
            gt_depth = data_dict['depth'].cuda()
            gt_depth_mask = data_dict['depth_valid_mask'].cuda()
            img_path = data_dict['img_path'][0]
            
            if args.use_uncertain == 0:
                pred_depth = model(img)
                pred_depth = nn.functional.interpolate(pred_depth, size=[gt_depth.size(2), gt_depth.size(3)],
                                                    mode='bilinear', align_corners=True)
            elif args.use_uncertain:   
                out_depth_list, pred_list, _ = model(img, gt_depth_mask, mode='val')
                out_depth = out_depth_list[-1]

                out_depth = nn.functional.interpolate(out_depth, size=[gt_depth.size(2), gt_depth.size(3)],
                                                    mode='bilinear', align_corners=True)
                pred_depth = out_depth[:, 0:1, :, :]
                if dist.get_rank()== 0 and args.vis and sub_epoch % 5 == 0:
                    vis(args, img_path, out_depth_list, gt_depth, i, sub_epoch)
            
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            
            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        
            metrics = compute_depth_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            for k in depth_metrics:
                depth_metrics[k].append(metrics[k])

        depth_metrics = {k: np.mean(v).item() for k, v in depth_metrics.items()}
        log_path = join(args.log_dir, "eval.txt")
        if args.loacl_rank == 0:
            log_depth_errors(depth_metrics, log_path, first_line='epoch: {}'.format(sub_epoch + 1)) 

        return metrics

def train_normal(args, train_loader, val_loader, model, criterion, optimizer, epoch, best_prec, train_writer):
    prec1 = 0
    local_rank = args.local_rank
    train_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train Normal")
    model.train()
    for i, data_dict in enumerate(train_loader):
        
        img = data_dict['img'].clone().cuda(local_rank, non_blocking=True)
        gt_norm = data_dict['norm'].clone().cuda(local_rank, non_blocking=True)
        gt_norm_mask = data_dict['norm_valid_mask'].clone().cuda(local_rank, non_blocking=True)

        _, pred_list, coord_list = model(img, gt_norm_mask=gt_norm_mask, mode='train')
        loss = criterion(pred_list, coord_list, gt_norm, gt_norm_mask)
        loss_ = float(loss.data.cpu().numpy())
        train_loader.set_description(f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train. Loss: {'%.5f' % loss_}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_iter = epoch * len(train_loader) + i
        if local_rank == 0:
            train_writer.add_scalar('train_normal_loss', loss.item(), total_iter)

            if total_iter % (args.val_freq * args.batch_size) == 0 and total_iter > 0:
                is_best = 0
                metrics = validate_normal(args, val_loader, model, total_iter)
                prec1 = metrics["a1"]
                if prec1 > best_prec:
                    best_prec = prec1
                    is_best = 1
            
                checkpoint_path = join(args.log_dir, 'model_best.pth.tar')

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec,
                }, is_best, filename=checkpoint_path)
        
    return best_prec 

def train_depth(args, train_loader, val_loader, model, criterion, optimizer, epoch, best_prec, train_writer):
    prec1 = 0

    local_rank = args.local_rank
    train_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train Depth")
    model.train()
    loss = 0
    for i, data_dict in enumerate(train_loader):

        img = data_dict['img'].clone().cuda(local_rank, non_blocking=True)
        gt_depth = data_dict['depth'].clone().cuda(local_rank, non_blocking=True)
        gt_depth_mask = data_dict['depth_valid_mask'].clone().cuda(local_rank, non_blocking=True)

        if args.use_uncertain:
            depth_list, pred_list, coord_list = model(img, gt_depth_mask, mode='train')
            loss = criterion(pred_list, coord_list, gt_depth, gt_depth_mask)
        
        else:
            depth_est = model(img)
            loss = criterion(depth_est, gt_depth, gt_depth_mask)
        loss_ = float(loss.data.cpu().numpy())
        if local_rank == 0:
            train_loader.set_description(f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train. Loss: {'%.5f' % loss_}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_iter = epoch * len(train_loader) + i
        if dist.get_rank() == 0:
            train_writer.add_scalar('train_depth_loss', loss.item(), total_iter)

        if total_iter % (args.val_freq * args.batch_size) == 0 and total_iter > 0:
            is_best = 0
            metrics = validate_depth(args, val_loader, model, total_iter)
            prec1 = metrics["d1"]
            if prec1 > best_prec:
                best_prec = prec1
                is_best = 1

            if local_rank == 0:
                checkpoint_path = join(args.log_dir, "model_best.pth.tar")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec,
                }, is_best, filename=checkpoint_path)
 
    return best_prec 

def train_single(args):
    
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    assert local_rank == dist.get_rank()
    batch_size = args.batch_size
    num_workers = args.workers

    if local_rank == 0:
        if not exists(args.log_dir):
            os.makedirs(args.log_dir)
        print(' '.join(sys.argv))
        print(args)
        train_writer = SummaryWriter(args.log_dir)
        print(' '.join(sys.argv))
        for k, v in args.__dict__.items():
            print(k, ':', v)
        shutil.copyfile("./run.sh", join(args.log_dir, "run.sh"))
    else:
        train_writer = None
    
    single_model = None
    criterion = None

    if args.task == "normal":
        single_model = DPTNormalUncertainModel(backbone="vitb_rn50_384", sampling_ratio=0.4, importance_ratio=0.7)

    if args.task == "depth":
        if args.use_uncertain:
            single_model = DPTDepthUncertainModel(backbone="vitb_rn50_384", sampling_ratio=0.3, importance_ratio=0.6)   
        else:    
            single_model = DPTDepthModel(backbone="vitb_rn50_384")
    
    single_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model).cuda(local_rank)
    single_model = torch.nn.parallel.DistributedDataParallel(single_model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=True)
    model = single_model.cuda(local_rank)


    if args.task == "normal":
        criterion = compute_loss(args)

    if args.task == 'depth':
        if args.use_uncertain:
            criterion = compute_loss(args)
        else:
            criterion = silog_loss(args)
    criterion.cuda(local_rank)

    # Data loading code
    data_dir = args.data_dir

    if args.task == "normal": 
        train_set = Normal_Single(args, data_dir, 'train')    
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, drop_last=True, sampler=train_sampler
        )

        val_loader = torch.utils.data.DataLoader(
            Normal_Single(args, data_dir, 'val'),
            batch_size=1, shuffle=False, num_workers=num_workers,
            pin_memory=True
        )

    if args.task == "depth":
        train_set = Depth_Single(args, data_dir, 'train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, drop_last=True, sampler=train_sampler
        )

        val_loader = torch.utils.data.DataLoader(
            Depth_Single(args, data_dir, 'val'),
            batch_size=1, shuffle=False, num_workers=num_workers,
            pin_memory=True
        )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    best_prec = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        if local_rank == 0:
            logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch

        if args.task == "normal": 
            best_prec = train_normal(args, train_loader, val_loader, model, criterion, optimizer, epoch, best_prec, train_writer) 
        
        if args.task == "depth":
            best_prec = train_depth(args, train_loader, val_loader, model, criterion, optimizer, epoch, best_prec, train_writer)

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['single', 'multi'])
    parser.add_argument('-d', '--data-dir', default='../dataset/nyud2')
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--phase', default='train')

    parser.add_argument("--use_uncertain", type=int, default=0)
    parser.add_argument("--task", type=str, default="depth")
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument("--data_augmentation_rotation", default=False, action="store_true")
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_hflip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_random_crop", default=False, action="store_true")
    parser.add_argument('--loss_fn', default='UG_NLL_ours', type=str, help='{L1, L2, AL, NLL_vMF, NLL_ours, UG_NLL_vMF, UG_NLL_ours}')
    parser.add_argument('--variance_focus', type=float, default=0.85)
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10.0)
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default="./exp_logs")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--val_freq", type=int, default=200)

    args = parser.parse_args()
    args.log_dir = join(args.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    return args

def main():
    args = parse_args()
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    
    if args.cmd == 'single':
        train_single(args)

if __name__ == '__main__':
    main()