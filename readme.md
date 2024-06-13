# Yolov5-SAM-Med3D


## Install
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

git clone xxx  # clone
cd Yolov5-SAM-Med3D
pip install -r requirements.txt  # install
```

## 🔨 Usage
### Training / Fine-tuning
(we recommend fine-tuning with Yolov5-SAM-Med3D pre-trained weights from [link](https://github.com/uni-medical/SAM-Med3D#-checkpoint) and )

To train the Yolov5-SAM-Med3D model on your own data, follow these steps:

#### 0. **(Recommend) Prepare the Pre-trained Weights**

Download the checkpoint from [ckpt section](https://github.com/uni-medical/SAM-Med3D#-checkpoint) and [ckpt section](https://github.com/ultralytics/yolov5?tab=readme-ov-file#pretrained-checkpoints). And move the pth file into `Yolov5-SAM-Med3D/ckpt/` (We recommand to use `SAM-Med3D-turbo.pth` and `yolov5m-seg.pt`.).


#### 1. Prepare Your Training Data for Stage2 (from nnU-Net-style dataset): 

> If the original data are in the **nnU-Net style**, follow these steps:
> 
> For a nnU-Net style dataset, the original file structure should be:
> ```
> Dataset130_MICCAI_challenge
>      ├── imagesTr
>      │ ├── 000001_02_01_008-023_0000.nii.gz
>      │ ├── ...
>      ├── labelsTr
>      │ ├── 000001_02_01_008-023.nii.gz
>      │ ├── ...

> Then you should resample and convert the masks into binary.You should change the dataset_root as the `nnunetv2/Dataset/nnUNet_raw`, select the dataset name in `dataset_list`, and modify the `target_dir`.

```bash
python utils/handle_data/prepare_data_from_nnUNet.py
```

Ensure that your training data is organized according to the structure shown in the `data/medical_preprocessed` directories. The target file structures should be like the following:
```
data/medical_preprocessed
    |—— Dataset130_MICCAI_challenge
    |   ├── Tumor
    |   │ ├── MICCAI_challenge
    |   │ │ ├── imagesTr
    |   │ │ │ ├── 000001_02_01_008-023.nii.gz
    |   │ │ │ ├── ...
    |   │ │ ├── labelsTr
    |   │ │ │ ├── 000001_02_01_008-023.nii.gz
    |   │ │ │ ├── ...
    |   ├── ...
    |—— ...
```

Then, modify the `utils/data_paths.py` according to your own data.
```
img_datas = [
'xxx/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/data/medical_preprocessed/Dataset130_MICCAI_challenge/Tumor/MICCAI_challenge',
...
]
```

#### 2. Prepare Your dataset_fingerprint.json for Stage2 (use nnU-Net dataset raw data): 

First, you should set up your data path in `utils/nnunet/paths.py`.

```bash
nnUNet_raw = 'xxx/yujunxuan/MICCAI_challenge_method/nnUNet-master/nnunetv2/Dataset/nnUNet_raw'
nnUNet_preprocessed = 'xxx/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/config'
```

Second,extract the dataset_fingerprint and save in `nnUNet_preprocessed`.
```bash
python utils/plan_and_preprocess_entrypoints.py -d 130(your dataset id)
```

#### 3. Prepare Your Training Data for Stage1 (from raw data): 


First, split the 3D volume into 'train,val,test', you should change the `base_dir` and `output_dir` in `split_3D_volume.py`.

```bash
python utils/handle_data/split_3D_volume.py
```

Second, save 3D data as 2D slices, you should change the  `json_path`(`dataset_fingerprint.json` from last step)  `input_dir` and `output_dir` in `save_3D_data_as_2D_data.py`.

```bash
python utils/handle_data/save_3D_data_as_2D_data.py
```

Third, make label_txt for Stage1, you should change the  `label_path` `summary_csv` and `save_label_txt_path` in `make_label_txt.py`.

```bash
python utils/handle_data/make_label_txt.py
```

Finally, the target file structures should be like the following:
```
> train_yolov5_data_labeled_from_volume
>      ├── images(.png)
>      │ ├── test
>      │ ├── train
>      | |—— val
>      ├── labels(.txt)
>      │ ├── test
>      │ ├── train
>      | |—— val
>      ├── labels_images(.png)
>      │ ├── test
>      │ ├── train
>      | |—— val
```



#### 4. **Run the Training Script**: 

First, if you want to train Stage1, the project will be saved in `MICCAI_challenge_method/yolov5-SAM-Med3D/runs`.

Importantly, please remember to change `path` in 
`config/MICCAI_data_seg.yaml`

```bash
python Stage1/train.py
```

if you want to use multi GPU, you can modify

```bash
parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
```

if you want to resume from latest checkpoint, you can modify

```bash
parser.add_argument("--resume", nargs="?", const=False, default=False, help="resume most recent training")
```

Second, if you want to train Stage2, the project will be saved in `MICCAI_challenge_method/yolov5-SAM-Med3D/work_dir`.

```bash
python Stage1/train_stage2.py
```

Please remember to change the task name and ensure the dataset ID is correct.


### Evaluation & Inference(Stage1)

Validate YOLOv5s-seg mask mAP on your own dataset:

```bash
python Stage1/val.py 
```


### Evaluation & Inference(Stage2)
Prepare your own dataset and refer to the samples in `data/validation` to replace them according to your specific scenario. 

Make sure the masks are processed into the one-hot format (have only two values: the main image (foreground) and the background). We highly recommend using the spacing of `1.5mm` for the best experience.

> Then you should resample and convert the masks into binary.You should change the dataset_root as the `nnunetv2/Dataset/nnUNet_raw`, select the dataset name in `dataset_list`, and modify the `target_dir`.

```bash
python utils/handle_data/prepare_data_from_nnUNet_validation.py
```

Ensure that your training data is organized according to the structure shown in the `data/validation` directories. The target file structures should be like the following:

```
data/validation
    |—— Dataset130_MICCAI_challenge
    |   ├── Tumor
    |   │ ├── MICCAI_challenge
    |   │ │ ├── imagesTs
    |   │ │ │ ├── 000010_01_01_078-090.nii.gz
    |   │ │ │ ├── ...
    |   │ │ ├── labelsTs
    |   │ │ │ ├── 000010_01_01_078-090.nii.gz
    |   │ │ │ ├── ...
    |   ├── ...
    |—— ...
```

Then,you can run validation,and the predicted nii will be saved in `./visualization`.The result json will be saved in `./results`.

```bash
python Stage2/validation.py
```
- (optional) skip_existing_pred: skip and not predict if output file is found existing


### Evaluation & Inference(Stage1 + Stage2)

First, if you want to use only gt_box as prompt to evaluate the performance bottleneck of this model,you can run as follow:

```bash
python union_inference_for_3D_case_use_gt_3Dbox_sliding_widow.py
```

if you want to use click point and gt_box as prompt to evaluate the performance bottleneck of this model,you can run as follow:

```bash
python union_inference_for_3D_case_use_gt_3Dbox_and_mask_sliding_widow.py
```

Second, if you want to use only pred_boxes from stage1 as prompt to evaluate the true performance of this model,you can run as follow:

```bash
python union_inference_for_3D_case_use_pred_3Dbox_yolov5.py 

or

python union_inference_for_3D_case_use_pred_3Dbox_yolov5_weighted.py 
```
(just different on postprocess of sliding window)


## 🔗 Checkpoint
**the most recommended version is SAM-Med3D-turbo**

| Model | Google Drive | Baidu NetDisk |
|----------|----------|----------|
| SAM-Med3D| [Download](https://drive.google.com/file/d/1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub/view?usp=drive_link) | [Download (pwd:r5o3)](https://pan.baidu.com/s/18uhMXy_XO0yy3ODj66N8GQ?pwd=r5o3) |
| SAM-Med3D-organ    | [Download](https://drive.google.com/file/d/1kKpjIwCsUWQI-mYZ2Lww9WZXuJxc3FvU/view?usp=sharing) | [Download (pwd:5t7v)](https://pan.baidu.com/s/1Dermdr-ZN8NMWELejF1p1w?pwd=5t7v) |
| SAM-Med3D-brain    | [Download](https://drive.google.com/file/d/1otbhZs9uugSWkAbcQLLSmPB8jo5rzFL2/view?usp=sharing) | [Download (pwd:yp42)](https://pan.baidu.com/s/1S2-buTga9D4Nbrt6fevo8Q?pwd=yp42) |
| SAM-Med3D-turbo    | [Download](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) | [Download (pwd:l6ol)](https://pan.baidu.com/s/1OEVtiDc6osG0l9HkQN4hEg?pwd=l6ol) |

Other checkpoints are available with their official link: [SAM](https://drive.google.com/file/d/1_U26MIJhWnWVwmI5JkGg2cd2J6MvkqU-/view?usp=drive_link) and [SAM-Med2D](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link).

## 🏆 Results
### 💡 Overall Performance
| **Model**    | **Prompt**   | **Resolution**                 | **Inference Time (s)** | **Overall Dice** |
|--------------|--------------|--------------------------------|------------------|------------------|
| nnUnet          | w/o     | 160×160×96                  | --              | 0.127            |
| Yolov5-SAM-Med3D   | pred_boxes      | 128×128×128                    | --               | 0.152           |
| Yolov5-SAM-Med3D   | gt_boxes    | 128×128×128                    | --                | 0.323           |



## 🎫 License
This project includes components from multiple sources, each with its own licensing terms. Please read and comply with the terms of each license.

### AGPL-3.0 License
Some components of this project are licensed under the AGPL-3.0 License. This license is ideal for non-commercial use, promoting open collaboration and knowledge sharing. If you use these components under the AGPL-3.0 License, you must comply with its terms, including the requirement to disclose your source code. See the [LICENSE-AGPL-3.0](link-to-agpl-license-file) file for more details.

### Ultralytics Enterprise License
For commercial use of Ultralytics components, please obtain an Enterprise License. This license allows you to integrate Ultralytics software and AI models into commercial products without the open-source obligations of the AGPL-3.0 License. Contact Ultralytics Licensing for more information.

### Apache 2.0 License
Some components of this project are licensed under the Apache 2.0 License. You are free to use, modify, and distribute these components, provided that you comply with the terms of the Apache 2.0 License. See the [LICENSE-Apache-2.0](link-to-apache-license-file) file for more details.

### Summary of Licenses
- AGPL-3.0 License: [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
- Ultralytics Enterprise License: [Ultralytics Licensing](https://ultralytics.com/license)
- Apache 2.0 License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

By using this project, you agree to comply with the terms of the applicable licenses.


## 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects:
  - [Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194;
  - [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/main)
  - [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)
  - [Yolov5](https://github.com/ultralytics/yolov5)
