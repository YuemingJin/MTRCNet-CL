import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import os
from PIL import Image, ImageOps
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers

parser = argparse.ArgumentParser(description='cnn_lstm_kl training')
parser.add_argument('-g', '--gpu', default=[2], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train', default=100, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val', default=8, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-3, type=float, help='learning rate for optimizer, default 1e-3')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=0, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('-a', '--alpha', default=1.0, type=float, help='kl loss ratio, default 1.0')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
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
alpha = args.alpha

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

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
print('alpha for kl    : {:.4f}'.format(alpha))


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
        self.count += 1
        random.seed(seed)
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        # self.target_transform=target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)


class multi_lstm_p2t(torch.nn.Module):
    def __init__(self):
        super(multi_lstm_p2t, self).__init__()
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
        self.lstm = nn.LSTM(2048, 512, batch_first=True, dropout=1)
        self.fc = nn.Linear(512, 7)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 7)
        self.relu = nn.ReLU()
        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        init.xavier_uniform(self.fc.weight)
        init.xavier_uniform(self.fc2.weight)
        init.xavier_uniform(self.fc3.weight)
        self.fc_p2t = nn.Linear(512, 7)
        init.xavier_uniform(self.fc_p2t.weight)

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        z = self.fc3(self.relu(z))
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y1 = self.fc(self.relu(y))
        p2t = self.fc_p2t(self.relu(y))
        return z, y1, p2t


class multi_lstm(torch.nn.Module):
    def __init__(self):
        super(multi_lstm, self).__init__()
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
        self.lstm = nn.LSTM(2048, 512, batch_first=True, dropout=1)
        self.fc = nn.Linear(512, 7)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 7)
        self.relu = nn.ReLU()
        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        init.xavier_uniform(self.fc.weight)
        init.xavier_uniform(self.fc2.weight)
        init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        z = self.fc3(self.relu(z))
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fc(self.relu(y))
        return z, y


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
    test_paths = train_test_paths_labels[2]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    if use_flip == 0:
        train_transforms = transforms.Compose([
            RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop) for crop in crops]))
        ])

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)

    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # num_train_we_use = 800
    # num_val_we_use = 80

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

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

    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train dataset: {:6d}'.format(num_train))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    print('num of valid dataset: {:6d}'.format(num_val))
    print('num of valid we use : {:6d}'.format(num_val_we_use))
    print('num of all valid use: {:6d}'.format(num_val_all))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_idx,
        num_workers=workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_idx,
        num_workers=workers,
        pin_memory=False
    )

    model_old = multi_lstm()
    model_old = DataParallel(model_old)
    model_old.load_state_dict(torch.load(
        "cnn_lstm_1_epoch_25_length_10_opt_1_mulopt_1_flip_0_crop_1_batch_300_train1_9991_train2_9958_val1_9725_val2_8864.pth"))

    model = multi_lstm_p2t()
    model.share = model_old.module.share
    model.lstm = model_old.module.lstm
    model.fc = model_old.module.fc
    model.fc2 = model_old.module.fc2
    model.fc3 = model_old.module.fc3

    model = DataParallel(model)
    for param in model.module.fc_p2t.parameters():
        param.requires_grad = False
    model.module.fc_p2t.load_state_dict(torch.load(
        "fc_epoch_25_length_4_opt_1_mulopt_1_flip_0_crop_1_batch_800_train1_9951_train2_9713_val1_9686_val2_7867_p2t.pth"))

    if use_gpu:
        model = model.cuda()
        model.module.fc_p2t = model.module.fc_p2t.cuda()

    criterion_1 = nn.BCEWithLogitsLoss(size_average=False)
    criterion_2 = nn.CrossEntropyLoss(size_average=False)
    criterion_3 = nn.KLDivLoss(size_average=False)
    sigmoid_cuda = nn.Sigmoid().cuda()

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), },
                {'params': model.module.fc.parameters()},
                {'params': model.module.fc2.parameters()},
                {'params': model.module.fc3.parameters()}],
                lr=learning_rate, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_step, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), },
                {'params': model.module.fc.parameters()},
                {'params': model.module.fc2.parameters()},
                {'params': model.module.fc3.parameters()}], lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
                {'params': model.module.fc2.parameters(), 'lr': learning_rate},
                {'params': model.module.fc3.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_step, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
                {'params': model.module.fc2.parameters(), 'lr': learning_rate},
                {'params': model.module.fc3.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy_1 = 0.0
    best_val_accuracy_2 = 0.0
    correspond_train_acc_1 = 0.0
    correspond_train_acc_2 = 0.0

    # 要存储2个train的准确率 2个valid的准确率 3个train 3个loss的loss, 一共10个数据要记录
    record_np = np.zeros([epochs, 12])

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
            sampler=train_idx,
            num_workers=workers,
            pin_memory=False
        )

        model.train()
        train_loss_1 = 0.0
        train_loss_2 = 0.0
        train_loss_3 = 0.0
        train_corrects_1 = 0
        train_corrects_2 = 0
        train_corrects_3 = 0

        train_start_time = time.time()
        for data in train_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels_1 = Variable(labels_1.cuda())
                labels_2 = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels_1 = Variable(labels_1)
                labels_2 = Variable(labels_2)

            optimizer.zero_grad()

            outputs_1, outputs_2, outputs_3 = model.forward(inputs)

            _, preds_2 = torch.max(outputs_2.data, 1)
            train_corrects_2 += torch.sum(preds_2 == labels_2.data)

            sig_output_1 = sigmoid_cuda(outputs_1)
            sig_output_3 = sigmoid_cuda(outputs_3)

            sig_average = (sig_output_1.data + sig_output_3.data) / 2

            preds_1 = torch.cuda.ByteTensor(sig_output_1.data > 0.5)
            preds_1 = preds_1.long()
            train_corrects_1 += torch.sum(preds_1 == labels_1.data)

            preds_3 = torch.cuda.ByteTensor(sig_average > 0.5)
            preds_3 = preds_3.long()
            train_corrects_3 += torch.sum(preds_3 == labels_1.data)

            labels_1 = Variable(labels_1.data.float())
            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)

            sig_output_3 = Variable(sig_output_3.data, requires_grad=False)
            loss_3 = torch.abs(criterion_3(sig_output_1, sig_output_3))
            loss = loss_1 + loss_2 + loss_3 * alpha
            loss.backward()
            optimizer.step()

            train_loss_1 += loss_1.data[0]
            train_loss_2 += loss_2.data[0]
            train_loss_3 += loss_3.data[0]

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_1 = train_corrects_1 / num_train_all / 7
        train_accuracy_2 = train_corrects_2 / num_train_all
        train_accuracy_3 = train_corrects_3 / num_train_all / 7
        train_average_loss_1 = train_loss_1 / num_train_all / 7
        train_average_loss_2 = train_loss_2 / num_train_all
        train_average_loss_3 = train_loss_3 / num_train_all

        # begin eval

        model.eval()
        val_loss_1 = 0.0
        val_loss_2 = 0.0
        val_loss_3 = 0.0
        val_corrects_1 = 0
        val_corrects_2 = 0
        val_corrects_3 = 0

        val_start_time = time.time()
        for data in val_loader:
            inputs, labels_1, labels_2 = data
            labels_2 = labels_2[(sequence_length - 1):: sequence_length]
            if use_gpu:
                inputs = Variable(inputs.cuda(), volatile=True)
                labels_1 = Variable(labels_1.cuda(), volatile=True)
                labels_2 = Variable(labels_2.cuda(), volatile=True)
            else:
                inputs = Variable(inputs, volatile=True)
                labels_1 = Variable(labels_1, volatile=True)
                labels_2 = Variable(labels_2, volatile=True)

            outputs_1, outputs_2, outputs_3 = model.forward(inputs)
            outputs_2 = outputs_2[(sequence_length - 1):: sequence_length]
            _, preds_2 = torch.max(outputs_2.data, 1)
            val_corrects_2 += torch.sum(preds_2 == labels_2.data)

            sig_output_1 = sigmoid_cuda(outputs_1)
            sig_output_3 = sigmoid_cuda(outputs_3)

            sig_average = (sig_output_1.data + sig_output_3.data) / 2

            preds_1 = torch.cuda.ByteTensor(sig_output_1.data > 0.5)
            preds_1 = preds_1.long()
            val_corrects_1 += torch.sum(preds_1 == labels_1.data)

            preds_3 = torch.cuda.ByteTensor(sig_average > 0.5)
            preds_3 = preds_3.long()
            val_corrects_3 += torch.sum(preds_3 == labels_1.data)

            labels_1 = Variable(labels_1.data.float())
            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)

            sig_output_3 = Variable(sig_output_3.data, requires_grad=False)
            loss_3 = torch.abs(criterion_3(sig_output_1, sig_output_3))

            val_loss_1 += loss_1.data[0]
            val_loss_2 += loss_2.data[0]
            val_loss_3 += loss_3.data[0]

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_1 = val_corrects_1 / (num_val_all * 7)
        val_accuracy_2 = val_corrects_2 / num_val_we_use
        val_accuracy_3 = val_corrects_3 / (num_val_all * 7)
        val_average_loss_1 = val_loss_1 / (num_val_all * 7)
        val_average_loss_2 = val_loss_2 / num_val_we_use
        val_average_loss_3 = val_loss_3 / num_val_all

        print('epoch: {:3d}'
              ' train time: {:2.0f}m{:2.0f}s'
              ' train accu_1: {:.4f}'
              ' train accu_3: {:.4f}'
              ' train accu_2: {:.4f}'
              ' train loss_1: {:4.4f}'
              ' train loss_2: {:4.4f}'
              ' train loss_3: {:4.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_accuracy_1,
                      train_accuracy_3,
                      train_accuracy_2,
                      train_average_loss_1,
                      train_average_loss_2,
                      train_average_loss_3))
        print('epoch: {:3d}'
              ' valid time: {:2.0f}m{:2.0f}s'
              ' valid accu_1: {:.4f}'
              ' valid accu_3: {:.4f}'
              ' valid accu_2: {:.4f}'
              ' valid loss_1: {:4.4f}'
              ' valid loss_2: {:4.4f}'
              ' valid loss_3: {:4.4f}'
              .format(epoch,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_accuracy_1,
                      val_accuracy_3,
                      val_accuracy_2,
                      val_average_loss_1,
                      val_average_loss_2,
                      val_average_loss_3))

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss_1 + val_average_loss_2 + alpha * val_average_loss_3)

        if val_accuracy_2 > best_val_accuracy_2 and val_accuracy_1 > 0.95:
            best_val_accuracy_2 = val_accuracy_2
            best_val_accuracy_1 = val_accuracy_1
            correspond_train_acc_1 = train_accuracy_1
            correspond_train_acc_2 = train_accuracy_2
            best_model_wts = copy.deepcopy(model.state_dict())
        elif val_accuracy_2 == best_val_accuracy_2 and val_accuracy_1 > 0.95:
            if val_accuracy_1 > best_val_accuracy_1:
                correspond_train_acc_1 = train_accuracy_1
                correspond_train_acc_2 = train_accuracy_2
                best_model_wts = copy.deepcopy(model.state_dict())
            elif val_accuracy_1 == best_val_accuracy_1:
                if train_accuracy_2 > correspond_train_acc_2:
                    correspond_train_acc_2 = train_accuracy_2
                    correspond_train_acc_1 = train_accuracy_1
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif train_accuracy_2 == correspond_train_acc_2:
                    if train_accuracy_1 > best_val_accuracy_1:
                        correspond_train_acc_1 = train_accuracy_1
                        best_model_wts = copy.deepcopy(model.state_dict())

        record_np[epoch, 0] = train_accuracy_1
        record_np[epoch, 1] = train_accuracy_3
        record_np[epoch, 2] = train_accuracy_2
        record_np[epoch, 3] = train_average_loss_1
        record_np[epoch, 4] = train_average_loss_2
        record_np[epoch, 5] = train_average_loss_3

        record_np[epoch, 6] = val_accuracy_1
        record_np[epoch, 7] = val_accuracy_3
        record_np[epoch, 7] = val_accuracy_2
        record_np[epoch, 9] = val_average_loss_1
        record_np[epoch, 10] = val_average_loss_2
        record_np[epoch, 11] = val_average_loss_3

    print('best accuracy_1: {:.4f} cor train accu_1: {:.4f}'.format(best_val_accuracy_1, correspond_train_acc_1))
    print('best accuracy_2: {:.4f} cor train accu_2: {:.4f}'.format(best_val_accuracy_2, correspond_train_acc_2))

    save_val_1 = int("{:4.0f}".format(best_val_accuracy_1 * 10000))
    save_val_2 = int("{:4.0f}".format(best_val_accuracy_2 * 10000))
    save_train_1 = int("{:4.0f}".format(correspond_train_acc_1 * 10000))
    save_train_2 = int("{:4.0f}".format(correspond_train_acc_2 * 10000))
    public_name = "cnn_lstm1_p2t" \
                  + "_epoch_" + str(epochs) \
                  + "_length_" + str(sequence_length) \
                  + "_opt_" + str(optimizer_choice) \
                  + "_mulopt_" + str(multi_optim) \
                  + "_flip_" + str(use_flip) \
                  + "_crop_" + str(crop_type) \
                  + "_batch_" + str(train_batch_size) \
                  + "_train1_" + str(save_train_1) \
                  + "_train2_" + str(save_train_2) \
                  + "_val1_" + str(save_val_1) \
                  + "_val2_" + str(save_val_2)
    model_name = public_name + ".pth"
    torch.save(best_model_wts, model_name)

    record_name = public_name + ".npy"
    np.save(record_name, record_np)


def main():
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('train_val_test_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
