import pickle
import shutil
import os
import argparse

with open('train_val_test_paths_labels.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

test_num_each = train_test_paths_labels[8]
test_paths = train_test_paths_labels[2]
test_labels = train_test_paths_labels[5]

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-n', '--name', type=str, help='name of model')

args = parser.parse_args()
sequence_length = args.seq
pred_name = args.name

with open(pred_name, 'rb') as f:
    ori_preds = pickle.load(f)

num_labels = len(test_labels)
num_preds = len(ori_preds)

print('num of labels  : {:6d}'.format(num_labels))
print("num ori preds  : {:6d}".format(num_preds))
print("labels example : ", test_labels[0][7])
print("preds example  : ", ori_preds[0])

if num_labels == (num_preds + (sequence_length - 1) * 40):

    root_dir = './phase'
    shutil.rmtree(root_dir)
    os.mkdir(root_dir)
    phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
    preds_all = []
    count = 0
    for i in range(40):
        filename = './phase/video' + str(41 + i) + '-phase.txt'
        f = open(filename, 'a')
        f.write('Frame Phase')
        f.write('\n')
        preds_each = []
        for j in range(count, count + test_num_each[i] - (sequence_length - 1)):
            if j == count:
                for k in range(sequence_length - 1):
                    preds_each.append(ori_preds[j])
                    preds_all.append(ori_preds[j])
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        for k in range(len(preds_each)):
            f.write(str(25 * k))
            f.write('\t')
            f.write(phase_dict_key[preds_each[k]])
            f.write('\n')
        f.close()
        count += test_num_each[i] - (sequence_length - 1)
    test_corrects = 0

    for i in range(len(test_labels)):
        if test_labels[i][7] == preds_all[i]:
            test_corrects += 1

    print('last video num label: {:6d}'.format(test_num_each[-1]))
    print('last video num preds: {:6d}'.format(len(preds_each)))
    print('num of labels       : {:6d}'.format(num_labels))
    print('rsult of all preds  : {:6d}'.format(len(preds_all)))
    print('right number preds  : {:6d}'.format(test_corrects))
    print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')