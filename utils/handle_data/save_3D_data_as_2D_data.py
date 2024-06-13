import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json

try:
    with open('/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/config/Dataset130_MICCAI_challenge/dataset_fingerprint.json','r') as f:
        dataset_fingerprint = json.load(f)
except Exception as e:
        raise Exception(f"Error loading dataset_fingerprint from : {e}\n") from e
    
foreground_intensity_properties_per_channel = dataset_fingerprint['foreground_intensity_properties_per_channel']

lock = Lock()

def preprocess_and_slice_3d_to_2d(input_dir, output_dir, preprocess_fn=None, retain_no_tumor_ratio=0.2, max_workers=12):
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels_images')
    
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels_images')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    splits = ['train', 'val', 'test']
    tasks = []

    for split in splits:
        image_files = sorted(os.listdir(os.path.join(image_dir, split)))
        for image_file in image_files:
            tasks.append((image_file, split, image_dir, label_dir, output_image_dir, output_label_dir, preprocess_fn, retain_no_tumor_ratio))

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(tasks)) as pbar:
        futures = {executor.submit(process_file, task): task for task in tasks}
        for future in as_completed(futures):
            future.result()
            with lock:
                pbar.update(1)

def process_file(task):
    image_file, split, image_dir, label_dir, output_image_dir, output_label_dir, preprocess_fn, retain_no_tumor_ratio = task

    label_file = image_file.replace('_0000.nii.gz', '.nii.gz')
    image_file_path = os.path.join(image_dir, split, image_file)
    label_file_path = os.path.join(label_dir, f"{split}_labels", label_file)
    
    # Load the 3D image and label
    image = sitk.ReadImage(str(image_file_path))
    label = sitk.ReadImage(str(label_file_path))

    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)

    if preprocess_fn:
        # import pdb; pdb.set_trace()
        image_array = preprocess_fn(image_array)

    num_slices = image_array.shape[0]
    
    no_tumor_slices = []
    tumor_slices = []
    
    for slice_idx in range(num_slices):
        image_slice = image_array[slice_idx, :, :]
        label_slice = label_array[slice_idx, :, :]

        if np.sum(label_slice) == 0:
            no_tumor_slices.append((image_slice, label_slice, slice_idx))
        else:
            tumor_slices.append((image_slice, label_slice, slice_idx))

    save_name = label_file.replace('.nii.gz','')
    
    if len(tumor_slices) != 0:
    
        last_tumor_slice = tumor_slices[-1][-1]
        
        
        # import pdb; pdb.set_trace()
        # Check if the files already exist
        if os.path.exists(os.path.join(output_image_dir, split, f"{save_name}_slice_{last_tumor_slice:03d}.png")) \
            and os.path.exists(os.path.join(output_label_dir, split, f"{save_name}_slice_{last_tumor_slice:03d}.png")):
            return None
    
    # Randomly sample a proportion of no tumor slices
    random.shuffle(no_tumor_slices)
    retained_no_tumor_slices = no_tumor_slices[:int(len(no_tumor_slices) * retain_no_tumor_ratio)]

    # Combine tumor slices and retained no tumor slices
    combined_slices = tumor_slices + retained_no_tumor_slices

    

    for image_slice, label_slice, slice_idx in combined_slices:
        os.makedirs(os.path.join(output_image_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_label_dir, split), exist_ok=True)
        
        # Save image slice
        image_slice_path = os.path.join(output_image_dir, split, f"{save_name}_slice_{slice_idx:03d}.png")
        label_slice_path = os.path.join(output_label_dir, split, f"{save_name}_slice_{slice_idx:03d}.png")

        
        plt.imsave(image_slice_path, image_slice, cmap='gray')
        plt.imsave(label_slice_path, label_slice, cmap='gray')

def preprocess_fn(image_array):
    mean_intensity = foreground_intensity_properties_per_channel["0"]["mean"]
    std_intensity = foreground_intensity_properties_per_channel["0"]["std"]
    lower_bound = foreground_intensity_properties_per_channel["0"]["percentile_00_5"]
    upper_bound = foreground_intensity_properties_per_channel["0"]["percentile_99_5"]

    image_array = image_array.astype(np.float32, copy=False)
    np.clip(image_array, lower_bound, upper_bound, out=image_array)
    image_array -= mean_intensity
    image_array /= max(std_intensity, 1e-8)

    return image_array

if __name__ == "__main__":
    input_dir = "xxx/yujunxuan/MICCAI_challenge_data/labeled_volumes_0514"
    output_dir = "xxx/yujunxuan/MICCAI_challenge_data/train_yolov5_data_labeled_from_volume"

    preprocess_and_slice_3d_to_2d(input_dir, output_dir, preprocess_fn=preprocess_fn)
