import torch
import random
from PIL import Image
import numpy as np
from os.path import exists, join, split
from torchvision import transforms
import torchvision.transforms.functional as TF
import utils as data_utils

TASK =None
if TASK == 'ATTRIBUTE':
    task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
    FILE_DESCRIPTION = '_attribute'
elif TASK == 'AFFORDANCE':
    task_list = ['L','M','R','S','W']
    FILE_DESCRIPTION = '_affordance'

elif TASK =='SEGMENTATION':
    task_list = ['Segmentation']
    FILE_DESCRIPTION = ''

else:
    task_list = None
    FILE_DESCRIPTION = ''
    PALETTE = None
    EVAL_METHOD = None

class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        if self.label_list is not None:
       	    data.append(Image.open(join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class ConcatSegList(torch.utils.data.Dataset):
    def __init__(self, at, af, seg):
        self.at = at
        self.af = af
        self.seg = seg

    def __getitem__(self, index):
        at_input = self.at[index]
        af_input = self.af[index]
        seg_input = self.seg[index]
        return (at_input, af_input, seg_input)
    
    def __len__(self):
        return len(self.at)

class Depth_Single(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, phase):
        self.args = args
        self.task = args.task
        self.data_dir = data_dir
        self.phase = phase
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_list = None
        self.label_list = None
        self.read_lists()

    def __getitem__(self, index):
        img_path = join(self.data_dir, self.image_list[index])
        depth_path = join(self.data_dir, self.label_list[index])
        # TODO: random resize
        img = Image.open(img_path).convert("RGB")
        depth_gt = Image.open(depth_path)

        if self.phase == 'train':

            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    img = TF.hflip(img)
                    depth_gt = TF.hflip(depth_gt)

            # # depth_gt = depth_gt.crop((43, 45, 608, 472))
            # # img = img.crop((43, 45, 608, 472))

            if self.args.data_augmentation_rotation is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                img = self.rotate_image(img, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # to array
            img = np.array(img).astype(np.float32) / 255.0
            depth_gt = np.array(depth_gt).astype(np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)    
            depth_gt = depth_gt / 1000.0      
            depth_valid_mask = depth_gt > 0.1
            # depth_valid_mask = depth_valid_mask[:, :, np.newaxis]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, depth_gt, depth_valid_mask = data_utils.random_crop(img, depth_gt, depth_valid_mask, 
                                                                        height=416, width=544)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = data_utils.color_augmentation(img, indoors=True)
        else:
            img = np.array(img).astype(np.float32) / 255.0

            depth_gt = np.array(depth_gt).astype(np.float32)
            depth_gt = depth_gt / 1000.0
            depth_gt = np.expand_dims(depth_gt, axis=2) 
            depth_valid_mask = depth_gt > 0.1

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1)                    # (3, H, W)
        depth_valid_mask = torch.from_numpy(depth_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                    'depth': depth_gt,
                    'depth_valid_mask': depth_valid_mask, 
                    'img_path': img_path}


        return sample

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.data_dir, self.task, self.phase + '_{}'.format(self.task) + '_images.txt')
        label_path = join(self.data_dir, self.task, self.phase + '_{}'.format(self.task) + '_gt.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

    def rotate_image(self, img, angle, flag=Image.BILINEAR):
        result = img.rotate(angle, resample=flag)
        return result

class Normal_Single(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, phase):
        self.args = args
        self.task = args.task
        self.data_dir = data_dir
        self.phase = phase
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_list = None
        self.label_list = None
        self.read_lists()

    def __getitem__(self, index):
        img_path = join(self.data_dir, self.image_list[index])
        norm_path = join(self.data_dir, self.label_list[index])
        # TODO: random resize
        img = Image.open(img_path).convert("RGB").resize(size=(self.args.input_width, self.args.input_height), 
                                                            resample=Image.BILINEAR)
        norm_gt = Image.open(norm_path).convert("RGB").resize(size=(self.args.input_width, self.args.input_height), 
                                                            resample=Image.NEAREST)

        if self.phase == 'train':
            # horizontal flip (default: True)
            DA_hflip = False
            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    img = TF.hflip(img)
                    norm_gt = TF.hflip(norm_gt)

            # to array
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)

            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                    norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

            if DA_hflip:
                norm_gt[:, :, 0] = - norm_gt[:, :, 0]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, norm_gt, norm_valid_mask = data_utils.random_crop(img, norm_gt, norm_valid_mask, 
                                                                        height=416, width=544)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = data_utils.color_augmentation(img, indoors=True)
        else:
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)

            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                    norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                    'norm': norm_gt,
                    'norm_valid_mask': norm_valid_mask,
                    'img_path': img_path}

        return sample


    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.data_dir, self.task, self.phase + '_{}'.format(self.task) + '_images.txt')
        label_path = join(self.data_dir, self.task, self.phase + '_{}'.format(self.task) + '_gt.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class Seg_Single(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
       	        label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class SegListMSMultiHead(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
       	        label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
