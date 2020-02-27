import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=400, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=10, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('-fz', '--freeze', default=False, type=bool, help='freeze net, default True')

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

freeze_net = args.freeze

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
print("freeze net      :",freeze_net)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels = self.file_labels[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels

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
        # self.fcDropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 7)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        # y = self.fcDropout(y)
        y = self.fc(y)
        return y


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

    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.RandomCrop(224),
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
            transforms.Resize((250, 250)),
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


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
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
    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    criterion = nn.CrossEntropyLoss(size_average=False)

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
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0
    best_epoch = 0

    record_np = np.zeros([epochs, 4])

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
        train_loss = 0.0
        train_corrects = 0
        batch_progress = 0.0
        train_start_time = time.time()
        for data in train_loader:           
            optimizer.zero_grad()
             # 释放显存
            torch.cuda.empty_cache()

            if use_gpu:
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels[(sequence_length - 1)::sequence_length]
            else:
                inputs, labels = data[0], data[1]
                labels = labels[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)

            outputs = model.forward(inputs)
            outputs = outputs[sequence_length - 1::sequence_length]

            _, preds = torch.max(outputs.data, 1)            

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            batch_corrects = torch.sum(preds == labels.data)
            train_corrects += batch_corrects

            batch_acc=float(batch_corrects)/train_batch_size*sequence_length

            batch_progress += 1
            if batch_progress*train_batch_size >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d] Batch acc:%.2f' % (str(percent) + '%', num_train_all, num_train_all, batch_acc), end='\n')
            else:
                percent = round(batch_progress*train_batch_size / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d] Batch acc:%.2f' % (str(percent) + '%', batch_progress*train_batch_size, num_train_all, batch_acc), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = float(train_corrects) / float(num_train_all)*sequence_length
        train_average_loss = train_loss / num_train_all*sequence_length

        # Sets the module in evaluation mode.
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_start_time = time.time()
        val_progress = 0

        with torch.no_grad():
            for data in val_loader:
                # 释放显存
                torch.cuda.empty_cache()
                if use_gpu:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels = labels[(sequence_length - 1)::sequence_length]
                else:
                    inputs, labels = data[0], data[1]
                    labels = labels[(sequence_length - 1)::sequence_length]

                if crop_type == 0 or crop_type == 1:
                    inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                    outputs = model.forward(inputs)
                elif crop_type == 5:
                    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                    inputs = inputs.view(-1, 3, 224, 224)
                    outputs = model.forward(inputs)
                    outputs = outputs.view(5, -1, 7)
                    outputs = torch.mean(outputs, 0)
                elif crop_type == 10:
                    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                    inputs = inputs.view(-1, 3, 224, 224)
                    outputs = model.forward(inputs)
                    outputs = outputs.view(10, -1, 7)
                    outputs = torch.mean(outputs, 0)

                outputs = outputs[sequence_length - 1::sequence_length]

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data)

                val_progress += 1
                if val_progress*val_batch_size >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy = float(val_corrects) / float(num_val_we_use)
        val_average_loss = val_loss / num_val_we_use
        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss: {:4.4f}'
              ' train accu: {:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid loss: {:4.4f}'
              ' valid accu: {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss,
                      train_accuracy,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss,
                      val_accuracy))

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch
        if val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_epoch = epoch

        record_np[epoch, 0] = train_accuracy
        record_np[epoch, 1] = train_average_loss
        record_np[epoch, 2] = val_accuracy
        record_np[epoch, 3] = val_average_loss

        save_val = int("{:4.0f}".format(best_val_accuracy * 10000))
        save_train = int("{:4.0f}".format(correspond_train_acc * 10000))
        model_name = "lstm" \
                     + "_epoch_" + str(best_epoch) \
                     + "_length_" + str(sequence_length) \
                     + "_opt_" + str(optimizer_choice) \
                     + "_mulopt_" + str(multi_optim) \
                     + "_flip_" + str(use_flip) \
                     + "_crop_" + str(crop_type) \
                     + "_batch_" + str(train_batch_size) \
                     + "_train_" + str(save_train) \
                     + "_val_" + str(save_val) \
                     + ".pth"

        torch.save(best_model_wts, model_name)
        print("best_epoch",str(best_epoch))

        record_name = "lstm" \
                      + "_epoch_" + str(best_epoch) \
                      + "_length_" + str(sequence_length) \
                      + "_opt_" + str(optimizer_choice) \
                      + "_mulopt_" + str(multi_optim) \
                      + "_flip_" + str(use_flip) \
                      + "_crop_" + str(crop_type) \
                      + "_batch_" + str(train_batch_size) \
                      + "_train_" + str(save_train) \
                      + "_val_" + str(save_val) \
                      + ".npy"
        np.save(record_name, record_np)

    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))




def main():
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('./train40_val8_test32_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
