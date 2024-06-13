import cv2
import os
import numpy as np
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

lock = Lock()

def process_images(image_folder, output_path, summary_csv, max_workers=16):
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # with open(summary_csv, 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(['Image', 'Number of Masks'])  # Writing header
        
    def process_file(filename):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare the output file path
        output_file = os.path.join(output_path, os.path.splitext(filename)[0] + '.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as file:
            if len(contours) == 0:
                file.write('')
            else:
                for contour in contours:
                    # Calculate normalized coordinates
                    normalized_contour = contour / np.array([[image.shape[1], image.shape[0]]])
                    # Flatten the array and convert to a single line string
                    contour_str = '1 ' + ' '.join(map(str, normalized_contour.flatten()))
                    # Write to file
                    file.write(contour_str + '\n')

            # with lock:
            #     # Write the number of masks to the CSV file
            #     csv_writer.writerow([filename, len(contours)])

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(image_files)) as pbar:
        futures = {executor.submit(process_file, filename): filename for filename in image_files}
        for future in as_completed(futures):
            future.result()
            with lock:
                pbar.update(1)

def process_folders(label_path, save_label_txt_path, summary_csv, max_workers=16):
    subfolders = ['train', 'test', 'val']
    tasks = []
    
    for subfolder in subfolders:
        image_folder = os.path.join(label_path, subfolder)
        output_folder = os.path.join(save_label_txt_path, subfolder)
        summary_csv_subfolder = os.path.join(os.path.dirname(summary_csv), f'{subfolder}_summary.csv')
        
        tasks.append((image_folder, output_folder, summary_csv_subfolder))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(tasks)) as pbar:
        futures = {executor.submit(process_images, *task, max_workers=max_workers): task for task in tasks}
        for future in as_completed(futures):
            future.result()
            with lock:
                pbar.update(1)

if __name__ == "__main__":
    label_path = 'xxx/yujunxuan/MICCAI_challenge_data/train_yolov5_data_labeled_from_volume/labels_images'
    save_label_txt_path = 'xxx/yujunxuan/MICCAI_challenge_data/train_yolov5_data_labeled_from_volume/labels'
    os.makedirs(save_label_txt_path, exist_ok=True)
    summary_csv = 'xxx/yujunxuan/MICCAI_challenge_data/mask_summary.csv'
    
    process_folders(label_path, save_label_txt_path, summary_csv)
