import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=400, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=320, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-5, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')

args = parser.parse_args()

gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self,degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img,brightness_factor)
        img_ = TF.adjust_contrast(img_,contrast_factor)
        img_ = TF.adjust_saturation(img_,saturation_factor)
        img_ = TF.adjust_hue(img_,hue_factor)
        
        return img_


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:,0]
        self.file_labels_tool = file_labels[:,range(1,8)]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_tool = self.file_labels_tool[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase, labels_tool

    def __len__(self):
        return len(self.file_paths)


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.fc_h = nn.Linear(512, 512)
        self.fc2 = nn.Linear(2048, 7)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc_h.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = F.relu(self.fc_h(y))
        y = self.fc(y)
        return y, z


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])(crop) for crop in crops]))
        ])

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
#    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each


# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


sig_f = nn.Sigmoid()


def valMinibatch(testloader,model):
    model.eval()
    criterion_tool = nn.BCEWithLogitsLoss(size_average=False)
    criterion_phase = nn.CrossEntropyLoss(size_average=False)
    with torch.no_grad():
        val_loss_tool = 0.0
        val_corrects_tool = 0.0
        val_loss_phase = 0.0
        val_corrects_phase = 0.0
        for data in testloader:
            if use_gpu:
                inputs, labels_phase, labels_tool = data[0].to(device), data[1].to(device), data[2].to(device)
            else:
                inputs, labels_phase, labels_tool = data[0], data[1], data[2]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase, outputs_tool = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            sig_out = sig_f(outputs_tool.data)
            preds_tool = (sig_out.cpu() > 0.5).mul_(1)
            preds_tool = preds_tool.float()

            labels_tool = labels_tool.data.float()
            loss_tool = criterion_tool(outputs_tool, labels_tool)

            val_loss_tool += loss_tool.data.item()
            val_loss_phase += loss_phase.data.item()

            val_corrects_tool += torch.sum(preds_tool == labels_tool.data.cpu())
            val_corrects_phase += torch.sum(preds_phase == labels_phase.data)

    model.train()
    return (val_loss_phase, val_loss_tool), (val_corrects_phase, val_corrects_tool)


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    # TensorBoard
    writer = SummaryWriter('runs/log_tool+phase')

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx)
    num_val_we_use = len(val_useful_start_idx)
    # num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    # num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # num_train_we_use = 8000
    # num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    #    np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num of train dataset: {:6d}'.format(num_train))
    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of valid dataset: {:6d}'.format(num_val))
    print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    print('num of valid we use : {:6d}'.format(num_val_we_use))
    print('num of all valid use: {:6d}'.format(num_val_all))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False
    )

    model = resnet_lstm()
    model.load_state_dict(torch.load("./best_model_phase/lstm_epoch_10_length_10_opt_0_mulopt_1_flip_1_crop_1_batch_400_train_9940_val_7786.pth"),strict=False)
    model.load_state_dict(torch.load("./temp/lr5e-5/latest_model_tool_3.pth"),strict=False)
    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    criterion_tool = nn.BCEWithLogitsLoss(size_average=False)
    criterion_phase = nn.CrossEntropyLoss(size_average=False)

    optimizer = None
    exp_lr_scheduler = None

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_h.parameters(), 'lr': learning_rate},
                {'params': model.module.fc2.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_h.parameters(), 'lr': learning_rate},
                {'params': model.module.fc2.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_tool = 0.0
    correspond_train_acc_tool = 0.0
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        # np.random.seed(epoch)
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=workers,
            pin_memory=False
        )

        # Sets the module in training mode.
        model.train()
        train_loss_tool = 0.0
        train_corrects_tool = 0
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_tool = 0.0
        minibatch_correct_tool = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase, labels_tool = data[0].to(device), data[1].to(device),data[2].to(device)
            else:
                inputs, labels_phase, labels_tool = data[0], data[1], data[2]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase, outputs_tool = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            sig_out = sig_f(outputs_tool.data)
            preds_tool = (sig_out.cpu() > 0.5).mul_(1)
            preds_tool = preds_tool.float()

            labels_tool = labels_tool.data.float()
            loss_tool = criterion_tool(outputs_tool, labels_tool)

            loss = loss_tool + loss_phase
            loss.backward()
            optimizer.step()

            running_loss_tool += loss_tool.data.item()
            train_loss_tool += loss_tool.data.item()
            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_tool = torch.sum(preds_tool == labels_tool.data.cpu())
            train_corrects_tool += batch_corrects_tool
            minibatch_correct_tool += batch_corrects_tool

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase


            if i % 500 == 499:
                # ...log the running loss
                batch_iters = epoch * num_train_all/sequence_length + i*train_batch_size/sequence_length
                writer.add_scalar('training loss tool',
                                  running_loss_tool / (train_batch_size*500) / 7,
                                  batch_iters)
                # ...log the training acc
                writer.add_scalar('training acc tool',
                                  float(minibatch_correct_tool) / (float(train_batch_size)*500) / 7,
                                  batch_iters)
                writer.add_scalar('training loss phase',
                                  running_loss_phase / (train_batch_size*500/sequence_length) ,
                                  batch_iters)
                # ...log the training acc
                writer.add_scalar('training acc phase',
                                  float(minibatch_correct_phase) / (float(train_batch_size)*500/sequence_length),
                                  batch_iters)
                # ...log the val acc loss

                (val_loss_phase, val_loss_tool), (val_corrects_phase, val_corrects_tool) = valMinibatch(val_loader, model)
                writer.add_scalar('validation acc miniBatch tool',
                                  float(val_corrects_tool) / float(num_val_all) / 7,
                                  batch_iters)
                writer.add_scalar('validation loss miniBatch tool',
                                  float(val_loss_tool) / float(num_val_all) / 7,
                                  batch_iters)
                writer.add_scalar('validation acc miniBatch phase',
                                  float(val_corrects_phase) / float(num_val_we_use),
                                  batch_iters)
                writer.add_scalar('validation loss miniBatch phase',
                                  float(val_loss_phase) / float(num_val_we_use),
                                  batch_iters)

                running_loss_tool = 0.0
                minibatch_correct_tool = 0.0
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            if (i+1)*train_batch_size >= num_train_all:               
                running_loss_tool = 0.0
                minibatch_correct_tool = 0.0
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            batch_progress += 1
            if batch_progress*train_batch_size >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            else:
                percent = round(batch_progress*train_batch_size / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress*train_batch_size, num_train_all), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_tool = float(train_corrects_tool) / float(num_train_all) / 7
        train_average_loss_tool = train_loss_tool / num_train_all / 7
        train_accuracy_phase = float(train_corrects_phase) / float(num_train_all) * sequence_length
        train_average_loss_phase = train_loss_phase / num_train_all * sequence_length

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_tool = 0.0
        val_corrects_tool = 0
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_tool = []
        val_all_labels_tool = []
        val_all_preds_phase = []
        val_all_labels_phase = []

        with torch.no_grad():
            for data in val_loader:
                if use_gpu:
                    inputs, labels_phase, labels_tool = data[0].to(device), data[1].to(device),data[2].to(device)
                else:
                    inputs, labels_phase, labels_tool = data[0], data[1], data[2]

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase, outputs_tool = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                sig_out = sig_f(outputs_tool.data)
                preds_tool = (sig_out.cpu() > 0.5).mul_(1)
                preds_tool = preds_tool.float()

                labels_tool = labels_tool.data.float()
                loss_tool = criterion_tool(outputs_tool, labels_tool)

                val_loss_tool += loss_tool.data.item()
                val_loss_phase += loss_phase.data.item()

                val_corrects_tool += torch.sum(preds_tool == labels_tool.data.cpu())
                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                for i in range(len(preds_tool)):
                    val_all_preds_tool.append(list(preds_tool.data.cpu()[i]))
                for i in range(len(labels_tool)):
                    val_all_labels_tool.append(list(labels_tool.data.cpu()[i]))
                for i in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
                for i in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))


                val_progress += 1
                if val_progress*val_batch_size >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_tool = float(val_corrects_tool) / num_val_all / 7
        val_average_loss_tool = val_loss_tool / num_val_all / 7
        val_accuracy_phase = float(val_corrects_phase) / float(num_val_we_use)
        val_average_loss_phase = val_loss_phase / num_val_we_use

        val_all_preds_tool = np.array(val_all_preds_tool)
        val_all_labels_tool = np.array(val_all_labels_tool)
        val_precision_each_tool = metrics.precision_score(val_all_labels_tool,val_all_preds_tool, average=None)
        val_recall_each_tool = metrics.recall_score(val_all_labels_tool,val_all_preds_tool, average=None)
        val_precision_tool = metrics.precision_score(val_all_labels_tool,val_all_preds_tool, average="macro")
        val_recall_tool = metrics.recall_score(val_all_labels_tool,val_all_preds_tool, average="macro")

        val_recall_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        val_jaccard_phase = metrics.jaccard_similarity_score(val_all_labels_phase,val_all_preds_phase)
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average=None)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average=None)

        writer.add_scalar('validation acc epoch tool',
                          float(val_accuracy_tool),epoch)
        writer.add_scalar('validation loss epoch tool',
                          float(val_average_loss_tool),epoch)
        writer.add_scalar('validation acc epoch phase',
                          float(val_accuracy_phase),epoch)
        writer.add_scalar('validation loss epoch phase',
                          float(val_average_loss_phase),epoch)

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss(phase/tool): {:4.4f}/{:4.4f}'
              ' train accu(phase/tool): {:.4f}/{:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid loss(phase/tool): {:4.4f}/{:4.4f}'
              ' valid accu(phase/tool): {:.4f}/{:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss_phase,
                      train_average_loss_tool,
                      train_accuracy_phase,
                      train_accuracy_tool,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss_phase,
                      val_average_loss_tool,
                      val_accuracy_phase,
                      val_accuracy_tool))

        print("val_precision_each_tool:", val_precision_each_tool)
        print("val_recall_each_tool:", val_recall_each_tool)
        print("val_precision_tool", val_precision_tool)
        print("val_recall_tool", val_recall_tool)

        print("val_precision_each_phase:", val_precision_each_phase)
        print("val_recall_each_phase:", val_recall_each_phase)
        print("val_precision_phase", val_precision_phase)
        print("val_recall_phase", val_recall_phase)
        print("val_jaccard_phase", val_jaccard_phase)

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss_tool+val_average_loss_phase)

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            best_val_accuracy_tool = val_accuracy_tool
            correspond_train_acc_tool = train_accuracy_tool
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch
        elif val_accuracy_phase == best_val_accuracy_phase:
            if val_accuracy_tool > best_val_accuracy_tool:
                correspond_train_acc_tool = train_accuracy_tool
                correspond_train_acc_phase = train_accuracy_phase
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_epoch = epoch
            elif val_accuracy_tool == best_val_accuracy_tool:
                if train_accuracy_phase > correspond_train_acc_phase:
                    correspond_train_acc_phase = train_accuracy_phase
                    correspond_train_acc_tool = train_accuracy_tool
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                    best_epoch = epoch
                elif train_accuracy_phase == correspond_train_acc_phase:
                    if train_accuracy_tool > best_val_accuracy_tool:
                        correspond_train_acc_tool = train_accuracy_tool
                        best_model_wts = copy.deepcopy(model.module.state_dict())
                        best_epoch = epoch

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_val_tool = int("{:4.0f}".format(best_val_accuracy_tool * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        save_train_tool = int("{:4.0f}".format(correspond_train_acc_tool * 10000))
        public_name = "cnn_lstm_phase+tool" \
                      + "_epoch_" + str(best_epoch) \
                      + "_length_" + str(sequence_length) \
                      + "_opt_" + str(optimizer_choice) \
                      + "_mulopt_" + str(multi_optim) \
                      + "_flip_" + str(use_flip) \
                      + "_crop_" + str(crop_type) \
                      + "_batch_" + str(train_batch_size) \
                      + "_trainPhase_" + str(save_train_phase) \
                      + "_trainTool_" + str(save_train_tool) \
                      + "_valPhase_" + str(save_val_phase) \
                      + "_valTool_" + str(save_val_tool)

        torch.save(best_model_wts, "./best_model/"+public_name+".pth")
        print("best_epoch",str(best_epoch))

        torch.save(model.module.state_dict(), "./temp_tool+phase/latest_model_"+str(epoch)+".pth")


    print('best accuracy: {:.4f} cor train accu: {:.4f}'
          .format(best_val_accuracy_tool, correspond_train_acc_tool))




def main():
    train_dataset, train_num_each, val_dataset, val_num_each = get_data('./train_val_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
