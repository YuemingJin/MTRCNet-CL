import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import DataParallel
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
from torchvision.transforms import Lambda

parser = argparse.ArgumentParser(description='cnn_lstm testing')
parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int, help='index of gpu to use, default 1')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--test', default=800, type=int, help='test batch size, default 800')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 2')
parser.add_argument('-n', '--name', type=str, help='name of model')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')

args = parser.parse_args()
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
sequence_length = args.seq
test_batch_size = args.test
model_name = args.name
workers = args.work
crop_type = args.crop

model_pure_name, _ = os.path.splitext(model_name)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('test batch size : {:6d}'.format(test_batch_size))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('name of this model: {:s}'.format(model_name))  # so we can store all result in the same file


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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

    train_transforms = transforms.Compose([
        transforms.RandomCrop(224),
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


def test_model(test_dataset, test_num_each):
    num_test = len(test_dataset)
    test_count = 0
    for i in range(len(test_num_each)):
        test_count += test_num_each[i]

    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    num_test_we_use = len(test_useful_start_idx)
    # 其实需要除以gpu个数再乘以gpu个数，但是为了保证所有都测试到，尽量保证test个数完整
    # num_test_we_use = 804

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    print('num of test dataset: {:6d}'.format(num_test))
    print('num of test we use : {:6d}'.format(num_test_we_use))
    print('num of all test use: {:6d}'.format(num_test_all))

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_idx,
        num_workers=1,
        pin_memory=False
    )
    model = multi_lstm_p2t()
    model = DataParallel(model)
    model.load_state_dict(torch.load(model_name))
    # model = model.module
    # model = DataParallel(model)

    if use_gpu:
        model = model.cuda()
    # model = DataParallel(model)
    # model = model.module
    criterion_1 = nn.BCEWithLogitsLoss(size_average=False)
    criterion_2 = nn.CrossEntropyLoss(size_average=False)
    sig_f = nn.Sigmoid()

    model.eval()
    test_loss_1 = 0.0
    test_loss_2 = 0.0
    test_corrects_2 = 0

    test_start_time = time.time()
    all_preds_1 = []
    all_labels_1 = []
    all_preds_2 = []

    for data in test_loader:
        inputs, labels_1, labels_2 = data

        # labels_1 = labels_1[(sequence_length - 1)::sequence_length]
        labels_2 = labels_2[(sequence_length - 1)::sequence_length]
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels_1 = Variable(labels_1.cuda(), volatile=True)
            labels_2 = Variable(labels_2.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels_1 = Variable(labels_1, volatile=True)
            labels_2 = Variable(labels_2, voatile=True)

        if crop_type == 0 or crop_type == 1:
            outputs_1, outputs_2, _ = model.forward(inputs)
        elif crop_type == 5:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            inputs = inputs.view(-1, 3, 224, 224)
            outputs_1, outputs_2, _ = model.forward(inputs)
            outputs_1 = outputs_1.view(5, -1, 7)
            outputs_1 = torch.mean(outputs_1, 0)
            outputs_2 = outputs_2.view(5, -1, 7)
            outputs_2 = torch.mean(outputs_2, 0)
        elif crop_type == 10:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            inputs = inputs.view(-1, 3, 224, 224)
            outputs_1, outputs_2, _ = model.forward(inputs)
            outputs_1 = outputs_1.view(10, -1, 7)
            outputs_1 = torch.mean(outputs_1, 0)
            outputs_2 = outputs_2.view(10, -1, 7)
            outputs_2 = torch.mean(outputs_2, 0)

        # outputs_1 = outputs_1[sequence_length-1::sequence_length]
        outputs_2 = outputs_2[sequence_length - 1::sequence_length]

        _, preds_2 = torch.max(outputs_2.data, 1)

        for i in range(len(outputs_1)):
            all_preds_1.append(outputs_1[i].data.cpu().numpy().tolist())
            all_labels_1.append(labels_1[i].data.cpu().numpy().tolist())
        for i in range(len(preds_2)):
            all_preds_2.append(preds_2[i])
        print('preds_1: {:6d} preds_2: {:6d}'.format(len(all_preds_1), len(all_preds_2)))

        # labels_1 = Variable(labels_1.data.float())
        # loss_1 = criterion_1(outputs_1, labels_1)

        # test_loss_1 += loss_1.data[0]
        loss_2 = criterion_2(outputs_2, labels_2)
        test_loss_2 += loss_2.data[0]
        test_corrects_2 += torch.sum(preds_2 == labels_2.data)

    all_preds_1_cor = []
    all_labels_1_cor = []
    cor_count = 0
    for i in range(len(test_num_each)):
        for j in range(cor_count, cor_count + test_num_each[i] - (sequence_length - 1)):
            if j == cor_count:
                for k in range(sequence_length - 1):
                    all_preds_1_cor.append(all_preds_1[sequence_length * j + k])
                    all_labels_1_cor.append(all_labels_1[sequence_length * j + k])
            all_preds_1_cor.append(all_preds_1[sequence_length * j + sequence_length - 1])
            all_labels_1_cor.append(all_labels_1[sequence_length * j + sequence_length - 1])
        cor_count += test_num_each[i] + 1 - sequence_length

    print('all_preds_1 : {:6d}'.format(len(all_preds_1)))
    print('all_labels_1: {:6d}'.format(len(all_labels_1)))
    print('cor_labels_1: {:6d}'.format(len(all_preds_1_cor)))
    print('cor_labels_1: {:6d}'.format(len(all_labels_1_cor)))

    pt_preds_1 = torch.from_numpy(np.asarray(all_preds_1_cor, dtype=np.float32))
    pt_labels_1 = torch.from_numpy(np.asarray(all_labels_1_cor, dtype=np.float32))
    pt_labels_1 = Variable(pt_labels_1, requires_grad=False)
    pt_preds_1 = Variable(pt_preds_1, requires_grad=False)
    loss_1 = criterion_1(pt_preds_1, pt_labels_1)
    test_loss_1 += loss_1.data[0]

    pt_labels_1 = pt_labels_1.data
    pt_preds_1 = pt_preds_1.data
    sig_out = sig_f(pt_preds_1)
    preds_cor = torch.ByteTensor(sig_out > 0.5)
    preds_cor = preds_cor.long()
    pt_labels_1 = pt_labels_1.long()
    test_corrects_1 = torch.sum(preds_cor == pt_labels_1)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy_1 = test_corrects_1 / num_test / 7
    test_accuracy_2 = test_corrects_2 / num_test_we_use
    test_average_loss_1 = test_loss_1 / num_test / 7
    test_average_loss_2 = test_loss_2 / num_test_we_use

    print('preds_1 num: {:6d} preds_2 num: {:6d}'.format(len(all_preds_1_cor), len(all_preds_2)))

    save_test1 = int("{:4.0f}".format(test_accuracy_1 * 10000))
    save_test2 = int("{:4.0f}".format(test_accuracy_2 * 10000))

    pred_1_name = model_pure_name + '_test1_' + str(save_test1) + '_crop_' + str(crop_type) + '.pkl'
    pred_2_name = model_pure_name + '_test2_' + str(save_test2) + '_crop_' + str(crop_type) + '.pkl'

    with open(pred_1_name, 'wb') as f:
        pickle.dump(all_preds_1_cor, f)
    with open(pred_2_name, 'wb') as f:
        pickle.dump(all_preds_2, f)

    print('test completed in:'
          ' {:2.0f}m{:2.0f}s'
          ' test loss_1: {:4.4f}'
          ' test loss_2: {:4.4f}'
          ' test accu_1: {:.4f}'
          ' test accu_2: {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_average_loss_1,
                  test_average_loss_2,
                  test_accuracy_1,
                  test_accuracy_2))


print()


def main():
    _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')

    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
