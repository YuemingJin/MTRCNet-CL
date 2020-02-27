import pickle
import shutil
import os
import argparse
import numpy as np

with open('test_paths_labels.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

test_num_each = train_test_paths_labels[7]
test_paths =train_test_paths_labels[1]
test_labels =train_test_paths_labels[4]

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-n', '--name', type=str, help='name of model')

args = parser.parse_args()
pred_name = args.name

with open(pred_name, 'rb') as f:
    ori_preds = pickle.load(f)

num_labels = len(test_labels)
num_preds = len(ori_preds)
print('num of labels : {:6d}'.format(num_labels))
print("num of preds  : {:6d}".format(num_preds))
print("preds example: ", ori_preds[0])

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

preds_all = []
count = 0
labels_count = 0
for i in range(4):
    filename = '../result/tool/video' + str(1 + i) + '-tool.txt'
    gt_filename = '../result/gt-tool/video' + str(1 + i) + '-tool.txt'
    f = open(filename, 'w+')
    f_gt = open(gt_filename, 'w+')
    f.write('Frame-1fps\tTool1\tTool2\tTool3\tTool4\tTool5\tTool6\tTool7')
    f.write('\n')
    f_gt.write('Frame-1fps\tTool1\tTool2\tTool3\tTool4\tTool5\tTool6\tTool7')
    f_gt.write('\n')
    preds_each = []
    labels_count += test_num_each[i]
    for j in range(count, count + test_num_each[i]):
        f.write(str(j - count))
        f.write('\t')
        f_gt.write(str(j - count))
        f_gt.write('\t')
        for k in range(7):
            temp_pred = sigmoid(ori_preds[j][k])
            f.write(str(temp_pred))
            f.write('\t')
            f_gt.write(str(test_labels[j][k+1]))
            f_gt.write('\t')
        f.write('\n')
        f_gt.write('\n')
    f.close()
    f_gt.close()
    count += test_num_each[i]

print('num of labels  : {:6d}'.format(labels_count))
print('ori_preds count: {:6d}'.format(count))
print('rsult of last  : {:6d}'.format(len(preds_each)))
print('rsult of all   : {:6d}'.format(len(preds_all)))
