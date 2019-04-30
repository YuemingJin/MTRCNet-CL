import pickle
import shutil
import os
import argparse

with open('train_val_test_paths_labels.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

test_num_each = train_test_paths_labels[8]
test_paths =train_test_paths_labels[2]
test_labels =train_test_paths_labels[5]

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

root_dir = './tool'
shutil.rmtree(root_dir)
os.mkdir(root_dir)

preds_all = []
count = 0
labels_count = 0
for i in range(40):
    filename = './tool/video' + str(41 + i) + '-tool.txt'
    f = open(filename, 'w+')
    f.write('Frame Tool1 Tool2 Tool3 Tool4 Tool5 Tool6 Tool7')
    f.write('\n')
    preds_each = []
    labels_count += test_num_each[i]
    for j in range(count, count + test_num_each[i]):
        f.write(str(25* (j - count)))
        f.write('\t')
        for k in range(7):
            f.write(str(ori_preds[j][k]))
            f.write('\t')
        f.write('\n')
    f.close()
    count += test_num_each[i]

print('num of labels  : {:6d}'.format(labels_count))
print('ori_preds count: {:6d}'.format(count))
print('rsult of last  : {:6d}'.format(len(preds_each)))
print('rsult of all   : {:6d}'.format(len(preds_all)))
