import numpy as np
import os
from PIL import Image
import random
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import torch


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)

def random_crop(img, norm, norm_mask, height, width):
    """randomly crop the input image & surface normal
    """
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, :]
    norm = norm[y:y + height, x:x + width, :]
    norm_mask = norm_mask[y:y + height, x:x + width, :]
    return img, norm, norm_mask


def color_augmentation(image, indoors=True):
    """color augmentation
    """
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    if indoors:
        brightness = random.uniform(0.75, 1.25)
    else:
        brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug

def compute_normal_errors(total_normal_errors):
    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / total_normal_errors.shape),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / total_normal_errors.shape[0]),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / total_normal_errors.shape[0]),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / total_normal_errors.shape[0]),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / total_normal_errors.shape[0]),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / total_normal_errors.shape[0])
    }
    return metrics


# log normal errors
def log_normal_errors(metrics, where_to_write, first_line):
    print(first_line)
    print("mean median rmse 5 7.5 11.25 22.5 30")
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
        metrics['mean'], metrics['median'], metrics['rmse'],
        metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

    with open(where_to_write, 'a') as f:
        f.write('%s\n' % first_line)
        f.write("mean median rmse 5 7.5 11.25 22.5 30\n")
        f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n\n" % (
            metrics['mean'], metrics['median'], metrics['rmse'],
            metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return {'silog': silog, 'abs_rel': abs_rel, 'log10': log10, 'rms': rms,  
            'sq_rel': sq_rel, "log_rms": log_rms, "d1": d1, 'd2': d2, 'd3': d3}

# log normal errors
def log_depth_errors(metrics, where_to_write, first_line):
    print(first_line)
    print("silog abs_rel log10 rms sq_rel log_rms d1 d2 d3")
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
        metrics['silog'], metrics['abs_rel'], metrics['log10'], metrics['rms'], 
        metrics['sq_rel'], metrics['log_rms'], metrics['d1'], metrics['d2'], metrics['d3'],))

    with open(where_to_write, 'a') as f:
        f.write('%s\n' % first_line)
        f.write("silog abs_rel log10 rms sq_rel log_rms d1 d2 d3\n")
        f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" % (
        metrics['silog'], metrics['abs_rel'], metrics['log10'], metrics['rms'], 
        metrics['sq_rel'], metrics['log_rms'], metrics['d1'], metrics['d2'], metrics['d3'],))

def vis(args, img_path, pred_list, gt, index, sub_epoch):
    save_dir = "./vis_{}/{}".format(args.task, "%04d" %sub_epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, "%04d" %index + ".jpg")
    pred_depth = pred_sigma = pred_norm = pred_kappa = None
    pred_output = pred_uncertain = None
    pred = pred_list[-1]
    if args.task == "depth":
        pred_depth, pred_sigma = pred[:, 0:1, :, :], pred[:, 1:, :, :]
        pred_depth.clamp(max=10.0, min=0.001)
        pred_depth = pred_depth[0][0].detach().cpu().numpy()
        pred_output = pred_depth
        pred_sigma = torch.exp(pred_sigma)
        pred_uncertain = pred_sigma
        pred_sigma = pred_sigma[0][0].detach().cpu().numpy()
        gt = gt[0][0].detach().cpu().numpy()
    
    elif args.task == "normal":
        pred_norm, pred_kappa = pred[:, 0:3, :, :], pred[:, 3:, :, :]
        pred_norm = pred_norm[0].detach().cpu().numpy()
        pred_norm = ((pred_norm + 1) / 2) * 255
        pred_output = pred_norm
        pred_kappa = pred_kappa[0][0].detach().cpu().numpy()
        pred_uncertain = pred_kappa
        gt = gt[0].detach().cpu().numpy()
        gt = ((gt + 1) / 2) * 255       

    img = plt.imread(img_path)
     
    plt.subplot(221)
    plt.axis("equal")
    plt.axis("off")
    plt.imshow(img)

    plt.subplot(222)
    plt.axis("equal")
    plt.axis("off")
    plt.imshow(pred_output)

    plt.subplot(223)
    plt.axis("equal")
    plt.axis("off")
    plt.imshow(pred_uncertain)

    plt.subplot(224)
    plt.axis("equal")
    plt.axis("off")
    plt.imshow(gt)

    plt.savefig(save_file, dpi=400, bbox_inches='tight')
    plt.close()

            






