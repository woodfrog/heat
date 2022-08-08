import os
from shutil import copyfile


data_base = './montefloor_data/'
dir_names = list(sorted(os.listdir(data_base)))
out_path = './s3d_floorplan'

wrong_s3d_annotations_list = [3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]

train_list = []
val_list = []
test_list = []

for dir_name in dir_names:
    data_dir = os.path.join(data_base, dir_name)
    annot_path = os.path.join(data_dir, 'annot.npy')
    if not os.path.exists(annot_path):
        continue
    data_id = int(dir_name[-5:])
    if data_id in wrong_s3d_annotations_list:
        continue
    annot_dst = os.path.join(out_path, 'annot', dir_name[-5:] + '.npy')
    density_dst = os.path.join(out_path, 'density', dir_name[-5:] + '.png')
    normal_dst = os.path.join(out_path, 'normals', dir_name[-5:] + '.png')
    density_src = os.path.join(data_dir, 'density.png')
    normal_src = os.path.join(data_dir, 'normals.png')
    copyfile(normal_src, normal_dst)
    copyfile(density_src, density_dst)
    copyfile(annot_path, annot_dst)
    if 0 <= data_id < 3000:
        train_list.append(dir_name[-5:])
    elif data_id < 3250:
        val_list.append(dir_name[-5:])
    else:
        test_list.append(dir_name[-5:])

with open(os.path.join(out_path, 'train_list.txt'), 'w') as f:
    for item in train_list:
        f.write(item + '\n')
with open(os.path.join(out_path, 'valid_list.txt'), 'w') as f:
    for item in val_list:
        f.write(item + '\n')
with open(os.path.join(out_path, 'test_list.txt'), 'w') as f:
    for item in test_list:
        f.write(item + '\n')
