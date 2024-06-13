import os
import shutil
from tqdm import tqdm
import random

def setup_directories(base_dir, subdirs):
    for key, dirs in subdirs.items():
        for dir in dirs:
            os.makedirs(os.path.join(base_dir, key, dir), exist_ok=True)




def copy_files_with_labels(file_list, source_dir, image_dir, label_dir, label_suffix='_0000.nii.gz'):
    filename = source_dir.split('/')[-1]
    # if filename == "image":
    #     filename = 'LIDC/image'
    for file in file_list:
        # 图像文件路径
        image_src_path = os.path.join(source_dir, file)
        image_dst_path = os.path.join(image_dir, file)
        shutil.copy(image_src_path, image_dst_path)
        
        

        # 标签文件路径
        label_file = file.replace(label_suffix, '.nii.gz')  # 根据规则修改
        label_src_path = os.path.join(source_dir.replace(filename,'labelsTr'), label_file)
        label_dst_path = os.path.join(label_dir, label_file)
        shutil.copy(label_src_path, label_dst_path)


def split_and_copy_files(folder_path, base_output_dir, train_ratio=0.8, val_ratio=0.1, label_suffix='_0000.nii.gz'):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz') and label_suffix in f]
    random.shuffle(all_files)

    # 分割数据集
    num_total = len(all_files)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]

    # 复制文件到图像和标签目录
    copy_files_with_labels(train_files, folder_path, os.path.join(base_output_dir, 'images', 'train'), os.path.join(base_output_dir, 'labels', 'train_labels'), label_suffix)
    copy_files_with_labels(val_files, folder_path, os.path.join(base_output_dir, 'images', 'val'), os.path.join(base_output_dir, 'labels', 'val_labels'), label_suffix)
    copy_files_with_labels(test_files, folder_path, os.path.join(base_output_dir, 'images', 'test'), os.path.join(base_output_dir, 'labels', 'test_labels'), label_suffix)

    
    
    
    return len(train_files), len(val_files), len(test_files)

def main(base_dir, output_dir):
    # 遍历基础目录中的所有子目录
    for folder_name in tqdm(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if folder_name == 'labelsTr' or folder_name == 'labelsTr.zip':
            continue

        # if folder_name == 'LIDC':
        #     folder_path = os.path.join(folder_path,'image')
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            train_count, val_count, test_count = split_and_copy_files(folder_path, output_dir)
            print(f"Finished {folder_name}: Train {train_count}, Val {val_count}, Test {test_count}")

if __name__ == '__main__':
    base_dir = 'xxx/yjx/MICCAI_challenge_data/Release-FLARE24-T1/Train-Labeled'  # 原始3D数据的根目录
    output_dir = 'xxx/yjx/MICCAI_challenge_data/labeled_volumes_0514'  # 输出目录
    
    # 设置目录结构
    directories = {
        "images": ["train", "val", "test"],
        "labels": ["train_labels", "val_labels", "test_labels"],  # 单独的标签目录
    }

    # 调用 setup_directories 来创建目录
    setup_directories(output_dir, directories)
    
    main(base_dir, output_dir)
