import os
import os.path as osp
join = osp.join
from glob import glob
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
import torchio as tio
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import json
import pickle
from itertools import product
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.dataloaders import Dataset_Union_ALL_Val, SliceDataset
from models.common import DetectMultiBackend
from utils.nms_2to3D import nms_2to3D_with_merging
from utils.torch_utils import select_device
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2, get_next_click3D_torch_ritm_with_box, get_next_click3D_torch_2_with_box

click_methods = {
    'default': get_next_click3D_torch_ritm_with_box,
    'ritm': get_next_click3D_torch_ritm_with_box,
    'random': get_next_click3D_torch_2_with_box,
}

import sys
sys.path.append('/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/detection_models')


def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw boxes on an image.
    
    Parameters:
    - image: numpy array of the image.
    - boxes: list of boxes, each box is in the format [x1, y1, x2, y2].
    - color: box color.
    - thickness: box line thickness.
    """

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        
        

def run_stage_1(model, dataloader, output_dir, gt_boxes_ori, img_name):
  
    case_name = img_name.split('/')[-1].split('.')[0]
    save_img_path = f'{output_dir}/{case_name}/img'
    save_mask_path = f'{output_dir}/{case_name}/mask'
    os.makedirs(save_img_path,exist_ok=True)
    os.makedirs(save_mask_path,exist_ok=True)
    

    model.eval()
    all_detections = []
    for im, ori_im, ori_gt_masks, slice_indexs, ratio, pad in tqdm(dataloader, desc="Processing Slices"):

        im = im.to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # -------------------inference-------------------
        with torch.no_grad():
            pred, proto = model(im, augment=False, visualize=False)[:2]
        # -------------------NMS-------------------
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det, nm=32)
        

        for i, det in enumerate(pred):
            # --------------------Convert to numpy image------------------------
            image = ori_im[i].cpu().numpy().transpose(1, 2, 0) 
            gt_mask = ori_gt_masks[i].cpu().numpy().transpose(1, 2, 0)
            
            image = np.ascontiguousarray(image, dtype=np.uint8)
            gt_mask = np.ascontiguousarray(gt_mask, dtype=np.uint8)
            
            # -------------------concat pred_box and slice_idx-------------------
            if det.shape[0] == 0 :
                save_det = torch.cat((torch.zeros((1,5), device=det.device), torch.tensor([[slice_indexs[i]]] * 1, device=det.device)), dim=1)   
            else:
                ori_pred_boxes = inverse_letterbox(det.clone().cpu().numpy()[:,:4], (ratio[0][0].cpu().numpy(), ratio[1][0].cpu().numpy()), (pad[0][0].cpu().numpy(),pad[1][0].cpu().numpy()))
                save_det = torch.cat((torch.from_numpy(ori_pred_boxes).to(det.device), det[:,4:5], torch.tensor([[slice_indexs[i]]] * det.size(0), device=det.device)), dim=1)
                # Draw and save image with boxes
                draw_boxes_on_image(image, ori_pred_boxes, color=(0, 0, 255))
                draw_boxes_on_image(gt_mask, ori_pred_boxes, color=(0, 0, 255))
            
            all_detections.append(save_det)
            
            # -------------------select box with slice_idx-------------------
            slice_idx = slice_indexs[i].item()
            gt_box_slices = []
            for gt_box in gt_boxes_ori:
                if not np.all(gt_box==0):
                    if slice_idx <= gt_box[-1] and slice_idx >= gt_box[2]:
                        gt_box_slice = np.array([gt_box[1], gt_box[0], gt_box[4], gt_box[3]]).astype(int).tolist()
                        gt_box_slices.append(gt_box_slice)
            
            if len(gt_box_slices) != 0:
                image = np.ascontiguousarray(image, dtype=np.uint8)
                gt_mask = np.ascontiguousarray(gt_mask, dtype=np.uint8)
                try:
                    draw_boxes_on_image(image, gt_box_slices, color=(0, 255, 0))  # Red for ground truth boxes
                    draw_boxes_on_image(gt_mask, gt_box_slices, color=(0, 255, 0))  # Red for ground truth boxes
                except:
                    import pdb; pdb.set_trace()

                output_img_path = os.path.join(save_img_path, f"slice_{int(slice_idx):03d}.png")
                output_mask_path = os.path.join(save_mask_path, f"slice_{int(slice_idx):03d}.png")
                cv2.imwrite(output_img_path, image)
                cv2.imwrite(output_mask_path, gt_mask)
    # -------------------Merges 2D boxes to 3D cubes-------------------
    all_detections = torch.cat(all_detections, dim=0)
    keep_boxes = nms_2to3D_with_merging(all_detections.cpu().numpy(), nms_thresh=0.5, merge_thresh=20, max_boxes=20)
    
    return keep_boxes

def select_boxes(pred_3D_boxes):
    # 将有较多零值的 box 置为全零
    for i, box in enumerate(pred_3D_boxes):
        zero_count = box.count(0.0)
        if zero_count >= 3:
            pred_3D_boxes[i] = [0.0] * len(box)

    # 分离全零和非零的 box
    non_zero_boxes = [box for box in pred_3D_boxes if any(val != 0.0 for val in box)]
    zero_boxes = [box for box in pred_3D_boxes if all(val == 0.0 for val in box)]

    # 从非零的 box 中选择 20 个，不足则补充全零的 box
    if len(non_zero_boxes) >= 20:
        pred_3D_boxes = non_zero_boxes[:20]
    else:
        pred_3D_boxes = non_zero_boxes + zero_boxes[:20 - len(non_zero_boxes)]

    # 如果仍不足 20 个，则补充全为 0 的 box
    while len(pred_3D_boxes) < 20:
        pred_3D_boxes.append([0.0] * len(pred_3D_boxes[0]))
    
    return pred_3D_boxes

def expand_boxes(boxes, expansion_factor=1.2):
    expanded_boxes = []
    for box in boxes:
        x_min, y_min, z_min, x_max, y_max, z_max = box
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min
        
        x_min = int(x_min - (expansion_factor - 1) * width / 2)
        y_min = int(y_min - (expansion_factor - 1) * height / 2)
        z_min = int(z_min - (expansion_factor - 1) * depth / 2)
        x_max = int(x_max + (expansion_factor - 1) * width / 2)
        y_max = int(y_max + (expansion_factor - 1) * height / 2)
        z_max = int(z_max + (expansion_factor - 1) * depth / 2)
        
        expanded_boxes.append([x_min, y_min, z_min, x_max, y_max, z_max])
    return expanded_boxes

def merge_boxes(boxes):
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    z_min = min(box[2] for box in boxes)
    x_max = max(box[3] for box in boxes)
    y_max = max(box[4] for box in boxes)
    z_max = max(box[5] for box in boxes)
    return [x_min, y_min, z_min, x_max, y_max, z_max]

def expand_z_axis(box, z_min, z_max):
    box[2] = z_min
    box[5] = z_max
    return box


def draw_3d_box(ax, box, color='r'):
    """绘制3D框"""
    xmin, ymin, zmin, xmax, ymax, zmax = box
    x = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
    z = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]
    verts = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]], [x[3], y[3], z[3]],
             [x[4], y[4], z[4]], [x[5], y[5], z[5]], [x[6], y[6], z[6]], [x[7], y[7], z[7]]]
    verts = [[verts[j] for j in [0, 1, 2, 3]], [verts[j] for j in [4, 5, 6, 7]],
             [verts[j] for j in [0, 3, 7, 4]], [verts[j] for j in [1, 2, 6, 5]],
             [verts[j] for j in [0, 1, 5, 4]], [verts[j] for j in [2, 3, 7, 6]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='r', alpha=.25))


def compute_iou(box1, box2):
    xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = box1
    xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = box2

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_zmin = max(zmin1, zmin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    inter_zmax = min(zmax1, zmax2)

    # import pdb; pdb.set_trace()
    inter_volume = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin) * max(0, inter_zmax - inter_zmin)

    volume1 = (xmax1 - xmin1) * (ymax1 - ymin1) * (zmax1 - zmin1)
    volume2 = (xmax2 - xmin2) * (ymax2 - ymin2) * (zmax2 - zmin2)

    union_volume = volume1 + volume2 - inter_volume

    return inter_volume / union_volume if union_volume != 0 else 0


def correct_3D_boxes(pred_3D_boxes, original_shape, resized_shape):
    """
    Corrects the 3D boxes to fit the original 3D data size.

    Parameters:
    - pred_3D_boxes: List of predicted 3D boxes.
    - original_shape: Tuple representing the original 3D data shape.
    - resized_shape: Tuple representing the resized 2D slice shape.

    Returns:
    - corrected_boxes: List of corrected 3D boxes.
    """
    z_scale = original_shape[2] / resized_shape[2]
    y_scale = original_shape[0] / resized_shape[0]
    x_scale = original_shape[1] / resized_shape[1]

    # import pdb; pdb.set_trace()
    
    corrected_boxes = []
    for box in pred_3D_boxes:
        xmin, ymin, zmin, xmax, ymax, zmax = box
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        zmin = int(zmin * z_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)
        zmax = int(zmax * z_scale)
        corrected_boxes.append([xmin, ymin, zmin, xmax, ymax, zmax])

    return corrected_boxes


def adjust_boxes(boxes, ratio, pad):
    # Adjust boxes with the given ratio and padding
    # import pdb; pdb.set_trace()
    boxes = np.array(boxes)
    nonzero_box = np.array([box for box in boxes if not np.all(box==0)])
    boxes[:, [0, 3]] = boxes[:, [0, 3]] * ratio[0] + pad[0]  # x coordinates
    boxes[:, [1, 4]] = boxes[:, [1, 4]] * ratio[1] + pad[1]  # y coordinates
    # boxes[:, [0, 3]] = boxes[:, [0, 3]] * ratio[0] + pad[0]  # x coordinates
    # boxes[:, [1, 4]] = boxes[:, [1, 4]] * ratio[1] + pad[1]  # y coordinates
    return boxes

def inverse_letterbox(boxes, ratio, pad):
    """
    Reverse the letterbox operation on bounding boxes.
    
    :param boxes: numpy array of shape (num_boxes, 4) with box coordinates [x1, y1, x2, y2]
    :param ratio: tuple, (width_ratio, height_ratio)
    :param pad: tuple, (width_pad, height_pad)
    :return: numpy array of shape (num_boxes, 4) with box coordinates in original image size
    """
    # Extract ratios and padding
    r_w, r_h = ratio
    pad_w, pad_h = pad

    # Reverse padding and scaling
    boxes[:, [0, 2]] -= pad_w  # x padding
    boxes[:, [1, 3]] -= pad_h  # y padding
    boxes[:, [0, 2]] /= r_w    # width scaling
    boxes[:, [1, 3]] /= r_h    # height scaling

    return boxes


def run_inference(args):
    all_dataset_paths = glob(join(args.test_data_path, "*", "*"))
    all_dataset_paths = list(filter(osp.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    crop_transform = tio.CropOrPad(
        mask_name='label', 
        target_shape=(args.crop_size, args.crop_size, args.crop_size))
    
    infer_transform = [
        tio.ToCanonical(),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Test", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
        get_all_meta_info=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=False
    )

    # Stage2_weights = args.Stage2_weights

    device = args.device
    print("device:", device)

    device = torch.device(device)
    Stage1_model = DetectMultiBackend(args.Stage1_weights, device=device, dnn=args.dnn, data=args.data_config, fp16=args.half)

    os.makedirs(args.task_name,exist_ok=True)
    os.makedirs(args.vis_stage1_output_dir,exist_ok=True)

    for id, batch_data in enumerate(tqdm(test_dataloader)):
        image3D, gt3D, gt_3Dbox, meta_info = batch_data
        # 获取原始3D数据的形状和调整后的形状
        original_shape = image3D.shape[2:]  # 排除批次维度
        resized_shape = (512, 512, original_shape[-1])  # 假设只调整xy维度
        
        img_name = meta_info["image_path"][0]
        modality = osp.basename(osp.dirname(osp.dirname(osp.dirname(img_name))))
        dataset = osp.basename(osp.dirname(osp.dirname(img_name)))
        vis_root = osp.join(args.output_dir, modality, dataset)
        pred_path = osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"))
        if(args.skip_existing_pred and osp.exists(pred_path)):
            continue  # if the pred existed, skip the inference
        else: 
            # Stage1 -------------Slice 3D Data into 2D for Detection-------------
            detection_2d_dataset = SliceDataset(image3D.clone().cpu().numpy(), gt3D.clone().cpu().numpy())
            detection_2d_dataloader = DataLoader(detection_2d_dataset, batch_size=args.Stage1_batch_size, shuffle=False)
            pred_3D_boxes = run_stage_1(Stage1_model, detection_2d_dataloader, args.vis_stage1_output_dir, gt_3Dbox[0,0].reshape(-1,6).cpu().numpy(), img_name)
            # pred_3D_boxes = select_boxes(pred_3D_boxes)
        
        print('pred_3D_boxes', np.array(pred_3D_boxes).shape)
        print('gt_3Dbox', gt_3Dbox.shape)
        

        # 修正3D盒子
        # corrected_pred_3D_boxes = correct_3D_boxes(pred_3D_boxes, original_shape, resized_shape)
        corrected_pred_3D_boxes = pred_3D_boxes
        
        # 融合所有3Dbox
        merged_box = merge_boxes(corrected_pred_3D_boxes)
        
        # 获取z轴的最小值和最大值
        z_min = merged_box[2]
        z_max = merged_box[5]
        
        # 将预测的box的z轴拉满
        # expanded_pred_3D_boxes = [expand_z_axis(box, z_min, z_max) for box in corrected_pred_3D_boxes]
        
        # 过滤出不为0的真实3D盒子
        valid_gt_boxes = [box for box in gt_3Dbox[0, 0].reshape(-1, 6).cpu().numpy() if not np.all(box == 0)]
        
        # 可视化预测的 3D 盒子和真实标注的 3D 盒子
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for box in corrected_pred_3D_boxes:
            draw_3d_box(ax, box, color='r')

        for box in valid_gt_boxes:
            draw_3d_box(ax, box, color='g')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, original_shape[1])
        ax.set_ylim(0, original_shape[0])
        ax.set_zlim(0, original_shape[-1])
        # plt.show()
        plt.savefig(f'/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/{args.task_name}/{id}.png')
        plt.close()

        # 计算并输出 IoU
        for i, pred_box in enumerate(corrected_pred_3D_boxes):
            for j, gt_box in enumerate(valid_gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                print(f"Prediction Box {i} with Ground Truth Box {j}: IoU = {iou:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tdp', '--test_data_path', type=str, default='/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/data/validation/Dataset131_MICCAI_challenge_small')
    parser.add_argument('--output_dir', type=str, default='./visualization')
    parser.add_argument('--task_name', type=str, default='vis_pred_3Dbox_with_gt_3Dbox_conf_015_transpose')
    parser.add_argument('--vis_stage1_output_dir', type=str, default='./vis_box_with_img_transpose')
    parser.add_argument('--skip_existing_pred', action='store_true', default=False)
    parser.add_argument('--save_image_and_gt', action='store_true', default=True)
    parser.add_argument('--sliding_window', action='store_true', default=True)
    
    
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--Stage1_batch_size', type=int, default=32)
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--Stage1_weights", nargs="+", type=str, default="/mnt/data1/yujunxuan/MICCAI_challenge_method/YOLOV5/runs/train-seg/exp4/weights/best.pt", help="model path(s)")
    parser.add_argument("--data_config", type=str, default="/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/config/MICCAI_data_seg.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="confidence threshold(Stage1)")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold(Stage1)")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image(Stage1)")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")

    parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
    parser.add_argument('-nc', '--num_clicks', type=int, default=5)
    parser.add_argument('-pm', '--point_method', type=str, default='default')
    parser.add_argument('-dt', '--data_type', type=str, default='Ts')

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--split_num', type=int, default=1)
    parser.add_argument('--ft2d', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2023)
    
    args = parser.parse_args()
    run_inference(args)
