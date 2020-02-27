import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import DataParallel
import torch.nn.functional as F
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
from torchvision.transforms import Lambda
from sklearn import metrics

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='use gpu, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--test', default=400, type=int, help='test batch size, default 10')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 4')
parser.add_argument('-n', '--name', type=str, help='name of model')
parser.add_argument(
    '-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 2 resize, 5 five_crop, 10 ten_crop, default 2')

args = parser.parse_args()
'''
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
'''
sequence_length = args.seq
test_batch_size = args.test
workers = args.work
model_name = args.name
crop_type = args.crop
use_gpu = args.gpu

model_pure_name, _ = os.path.splitext(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('gpu             : ', device)
print('sequence length : {:6d}'.format(sequence_length))
print('test batch size : {:6d}'.format(test_batch_size))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('name of this model: {:s}'.format(model_name))  # so we can store all result in the same file
print('Result store path: {:s}'.format(model_pure_name))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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


def get_test_data(data_path):
    with open(data_path, 'rb') as f:
        test_paths_labels = pickle.load(f)

    test_paths = test_paths_labels[2]
    test_labels = test_paths_labels[5]
    test_num_each = test_paths_labels[8]

    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    test_labels = np.asarray(test_labels, dtype=np.int64)

    test_transforms = None
    if crop_type == 0:
        test_transforms = transforms.Compose([
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
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])(crop) for crop in crops]))
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 3:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])

    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return test_dataset, test_num_each


# TODO
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


def test_model(test_dataset, test_num_each):
    num_test = len(test_dataset)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    num_test_we_use = len(test_useful_start_idx)

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
# TODO sampler

    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             sampler=SeqSampler(test_dataset, test_idx),
                             num_workers=workers)

    model = resnet_lstm()
    model.load_state_dict(torch.load(model_name),strict = False)
    print(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")  
    model = DataParallel(model)

    if use_gpu:
        model.to(device)
    # 应该可以直接多gpu计算
    # model = model.module            #要测试一下
    criterion_tool = nn.BCEWithLogitsLoss(size_average=False)
    criterion_phase = nn.CrossEntropyLoss(size_average=False)
    sig_f = nn.Sigmoid()

    model.eval()
    test_start_time = time.time()

    all_preds_tool = []
    all_labels_tool = []
    val_corrects_tool = 0
    val_corrects_phase = 0
    val_loss_tool = 0
    val_loss_phase = 0

    with torch.no_grad():

        for data in test_loader:
            
            # 释放显存
            #  torch.cuda.empty_cache()            

            if use_gpu:
                inputs, labels_phase, labels_tool = data[0].to(device), data[1].to(device),data[2].to(device)
            else:
                inputs, labels_phase, labels_tool = data[0], data[1], data[2]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)

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

            for i in range(len(preds_tool)):
                all_preds_tool.append(preds_tool[i].data.cpu().numpy().tolist())
                all_labels_tool.append(labels_tool[i].data.cpu().numpy().tolist())

            print("all_preds length:",len(all_preds_tool))

    all_preds_tool_cor = []
    all_labels_tool_cor = []
    cor_count = 0
    for i in range(len(test_num_each)):
        for j in range(cor_count, cor_count + test_num_each[i] - (sequence_length - 1)):
            if j==cor_count:
                for k in range(sequence_length-1):
                    all_preds_tool_cor.append(all_preds_tool[sequence_length * j + k])
                    all_labels_tool_cor.append(all_labels_tool[sequence_length * j + k])
            all_preds_tool_cor.append(all_preds_tool[sequence_length * j + sequence_length - 1])
            all_labels_tool_cor.append(all_labels_tool[sequence_length * j + sequence_length - 1])
        cor_count += test_num_each[i] + 1 - sequence_length

    print('all_preds_tool : {:6d}'.format(len(all_preds_tool)))
    print('all_labels_tool: {:6d}'.format(len(all_labels_tool)))
    print('all_preds_tool_cor: {:6d}'.format(len(all_preds_tool_cor)))
    print('all_labels_tool_cor: {:6d}'.format(len(all_labels_tool_cor)))

    num_test = len(test_dataset)
    test_elapsed_time = time.time() - test_start_time
    test_accuracy = float(val_corrects_tool) / float(num_test_all) /7
    test_average_loss = val_loss_tool / num_test_all /7

    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    pred_name = model_pure_name + '_test_' + str(save_test) + '_crop_' + str(crop_type) + '.pkl'

    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds_tool_cor, f)
    print('test elapsed: {:2.0f}m{:2.0f}s'
          ' test loss: {:4.4f}'
          ' test accu: {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_average_loss, test_accuracy))

    val_all_preds_tool = np.array(all_preds_tool_cor)
    val_all_labels_tool = np.array(all_labels_tool_cor)
    val_precision_each_tool = metrics.precision_score(val_all_labels_tool,val_all_preds_tool, average=None)
    print("val_precision_each_tool:", val_precision_each_tool)


print()


def main():
    test_dataset, test_num_each = get_test_data(
        './test_paths_labels.pkl')

    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
