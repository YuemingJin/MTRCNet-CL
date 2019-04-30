import os
import numpy as np
import pickle

root_dir = '/data/lhx/cholec'
img_dir = os.path.join(root_dir, 'data_resize')
tool_dir = os.path.join(root_dir, 'tool_annotations')
phase_dir = os.path.join(root_dir, 'phase_annotations')

print(root_dir)
print(img_dir)
print(tool_dir)
print(phase_dir)


def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths

def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


img_dir_names, img_dir_paths = get_dirs(img_dir)
tool_file_names, tool_file_paths = get_files(tool_dir)
phase_file_names, phase_file_paths = get_files(phase_dir)

phase_dict = {}
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)

# for i in range(1):
#     img_file_names, img_file_paths = get_files(img_dir_paths[i])
#     print(len(img_file_names))
#     print(len(img_file_paths))
#     # print(img_file_names[:10])
#     # print(img_file_paths[:10])

all_info_all = []

for j in range(len(tool_file_names)):
    last_tool_index = ''
    last_phase_index = ''
    tool_file = open(tool_file_paths[j])
    phase_file = open(phase_file_paths[j])
    tool_count = 0
    phase_count = 0
    info_all = []
    for tool_line in tool_file:
        tool_count += 1
        if tool_count > 1:
            tool_split = tool_line.split()
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths[j], img_dir_names[j] + '-' + str(tool_count - 1) + '.jpg')
            info_each.append(img_file_each_path)
            for l in range(1, len(tool_split)):
                info_each.append(int(tool_split[l]))
                last_tool_index = tool_split[0]
            info_all.append(info_each)
    for phase_line in phase_file:
        phase_count += 1
        if phase_count % 25 == 2 and (phase_count // 25) < len(info_all):
            phase_split = phase_line.split()
            # for m in range(len(phase_split)):
            info_all[phase_count // 25].append(phase_dict[phase_split[1]])
            last_phase_index = phase_split[0]
    print('the{:4d}th tool: {:6d} index_error{:2d}'.format(j, tool_count - 1,
                                                           int(last_tool_index) - int(last_phase_index)))

    # print(len(info_all))
    all_info_all.append(info_all)

# for k in range(10):
# print(all_info_all[0][k])
with open('cholec80.pkl', 'wb') as f:
    pickle.dump(all_info_all, f)

import pickle

with open('cholec80.pkl', 'rb') as f:
    all_info = pickle.load(f)

train_file_paths = []
test_file_paths = []
val_file_paths = []
val_labels = []
train_labels = []
test_labels = []

train_num_each = []
val_num_each = []
test_num_each = []

for i in range(40):
    train_num_each.append(len(all_info[i]))
    for j in range(len(all_info[i])):
        train_file_paths.append(all_info[i][j][0])
        train_labels.append(all_info[i][j][1:])

print(len(train_file_paths))
print(len(train_labels))

for i in range(40, 48):
    val_num_each.append(len(all_info[i]))
    for j in range(len(all_info[i])):
        val_file_paths.append(all_info[i][j][0])
        val_labels.append(all_info[i][j][1:])

print(len(val_file_paths))
print(len(val_labels))

for i in range(48, 80):
    test_num_each.append(len(all_info[i]))
    for j in range(len(all_info[i])):
        test_file_paths.append(all_info[i][j][0])
        test_labels.append(all_info[i][j][1:])

print(len(test_file_paths))
print(len(test_labels))

# for i in range(10):
#     print(train_file_paths[i], train_labels[i])
#     print(test_file_paths[i], test_labels[i])

train_val_test_paths_labels = []
train_val_test_paths_labels.append(train_file_paths)
train_val_test_paths_labels.append(val_file_paths)
train_val_test_paths_labels.append(test_file_paths)

train_val_test_paths_labels.append(train_labels)
train_val_test_paths_labels.append(val_labels)
train_val_test_paths_labels.append(test_labels)

train_val_test_paths_labels.append(train_num_each)
train_val_test_paths_labels.append(val_num_each)
train_val_test_paths_labels.append(test_num_each)

with open('train40_val8_test32_paths_labels.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)


print('Done')
print()
