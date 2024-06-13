import os
import os.path as osp
join = osp.join
from glob import glob
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
import torchio as tio
from itertools import product
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import json
import pickle
from itertools import product
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.dataloaders import Dataset_Union_ALL_Val, SliceDataset

from models.common import DetectMultiBackend
from utils.nms_2to3D import nms_2to3D
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
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2, get_next_click3D_torch_ritm_with_box,get_next_click3D_torch_2_with_box

click_methods = {
    'default': get_next_click3D_torch_ritm_with_box,
    'ritm': get_next_click3D_torch_ritm_with_box,
    'random': get_next_click3D_torch_2_with_box,
}

def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def compute_dice(mask_gt, mask_pred, dtype=np.uint8):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt.astype(dtype) & mask_pred.astype(dtype)).sum()
    return 2*volume_intersect / volume_sum

def finetune_model_predict3D(img3D, gt3D, pred_3D_boxes, sam_model_tune, norm_transform, device='cuda:4', click_method='random', num_clicks=10, prev_masks=None):
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    
    pred_3D_boxes = torch.tensor(pred_3D_boxes,device=device,dtype=torch.float32)

    click_points = []
    click_labels = []

    pred_list = []

    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

    for click_idx in range(num_clicks):
        with torch.no_grad():
            if(click_idx>1):
                click_method = "random"
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device), pred_3D_boxes)

            points_co = torch.cat(batch_points, dim=0).to(device)  
            points_la = torch.cat(batch_labels, dim=0).to(device)  

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=pred_3D_boxes,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            # import pdb; pdb.set_trace()
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

    return pred_list, click_points, click_labels




def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> List[List[int]]:
    assert all(i >= j for i, j in zip(image_size, tile_size)), "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps

def pad_and_crop_with_sliding_window(img3D, gt3D, boxes_3D, crop_transform, tile_step_size=0.5):
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img3D.squeeze(0)),
        label=tio.LabelMap(tensor=gt3D.squeeze(0))
    )

    # Define target shape and volume shape
    roi_shape = crop_transform.target_shape
    vol_shape = img3D.shape[2:]
    
    # Calculate padding if needed
    padding_params = [0, 0, 0, 0, 0, 0]
    for i in range(3):
        if vol_shape[i] < roi_shape[i]:
            pad = roi_shape[i] - vol_shape[i]
            padding_params[2 * i + 1] = pad

    if any(padding_params):
        pad_transform = tio.Pad(padding_params)
        subject = pad_transform(subject)

    img3D_padded = subject.image.data.unsqueeze(0)  # Add batch dimension back
    vol_shape_padded = img3D_padded.shape[2:]

    window_list = []
    # import pdb; pdb.set_trace()
    steps = compute_steps_for_sliding_window(vol_shape_padded, roi_shape, tile_step_size)

    for start_x, start_y, start_z in product(*steps):
        end_x = min(start_x + roi_shape[0], vol_shape_padded[0])
        end_y = min(start_y + roi_shape[1], vol_shape_padded[1])
        end_z = min(start_z + roi_shape[2], vol_shape_padded[2])
        
        crop_params = (
            start_x, vol_shape_padded[0] - end_x,
            start_y, vol_shape_padded[1] - end_y,
            start_z, vol_shape_padded[2] - end_z
        )
        
        pad_and_crop = tio.Compose([
            tio.Crop(crop_params),
        ])
        
        subject_roi = pad_and_crop(subject)
        img3D_roi = subject_roi.image.data.clone().detach().unsqueeze(1)
        gt3D_roi = subject_roi.label.data.clone().detach().unsqueeze(1)
        
        pos3D_roi = dict(
            ori_roi=(start_x, end_x, start_y, end_y, start_z, end_z),
            pred_roi=(0, roi_shape[0], 0, roi_shape[1], 0, roi_shape[2])
        )

        # Map boxes to the sliding window coordinate system
        mapped_boxes = []
        contains_box = False
        for box in boxes_3D[0, 0].reshape(-1, 6):
            
            mapped_box = [
                max(box[0] - pos3D_roi['ori_roi'][0], 0), max(box[1] - pos3D_roi['ori_roi'][2], 0),
                max(box[2] - pos3D_roi['ori_roi'][4], 0), max(box[3] - pos3D_roi['ori_roi'][0], 0),
                max(box[4] - pos3D_roi['ori_roi'][2], 0), max(box[5] - pos3D_roi['ori_roi'][4], 0)
            ]
            # import pdb; pdb.set_trace()
            # Check if the box is completely within the current window
            if not torch.all(box==0):
                if (box[0] >= pos3D_roi['ori_roi'][0] and box[3] <= pos3D_roi['ori_roi'][1] and
                    box[1] >= pos3D_roi['ori_roi'][2] and box[4] <= pos3D_roi['ori_roi'][3] and
                    box[2] >= pos3D_roi['ori_roi'][4] and box[5] <= pos3D_roi['ori_roi'][5]):
                    contains_box = True
            
            mapped_boxes.append(mapped_box)

        if contains_box:
            window_list.append((img3D_roi, gt3D_roi, pos3D_roi, mapped_boxes))
    
    return window_list


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)
    

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

    Stage2_weights = args.Stage2_weights

    device = args.device
    print("device:", device)

    if(args.dim==3):
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        if Stage2_weights is not None:
            model_dict = torch.load(Stage2_weights, map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model_tune.load_state_dict(state_dict)
    else:
        raise NotImplementedError("this script is designed for 3D sliding-window inference, not support other dims")
    
    device = torch.device(device)
    # print(device)
    Stage1_model = DetectMultiBackend(args.Stage1_weights, device=device, dnn=args.dnn, data=args.data_config, fp16=args.half)

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    all_iou_list = []
    all_dice_list = []  

    out_dice = dict()
    out_dice_all = OrderedDict()

    for batch_data in tqdm(test_dataloader):
        image3D, gt3D, gt_3Dbox, meta_info = batch_data
        
        img_name = meta_info["image_path"][0]
        modality = osp.basename(osp.dirname(osp.dirname(osp.dirname(img_name))))
        dataset = osp.basename(osp.dirname(osp.dirname(img_name)))
        vis_root = osp.join(args.output_dir,args.task_name, modality, dataset)
        pred_path = osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"))
        if(args.skip_existing_pred and osp.exists(pred_path)):
            pass # if the pred existed, skip the inference
        else: 
            # Stage2 -------------Union 3D Data and pred_3D_boxes for SAM-Med3D-------------
            ''' inference '''
            iou_list, dice_list = [], []
            image3D_full, gt3D_full = image3D, gt3D
            pred3D_full_dict  = {click_idx:torch.zeros_like(gt3D_full).numpy() for click_idx in range(args.num_clicks)}
            offset_mode = "center" if(not args.sliding_window) else "rounded"
            sliding_window_list = pad_and_crop_with_sliding_window(image3D_full, gt3D_full, gt_3Dbox, crop_transform)
            
            for (image3D, gt3D, pos3D, mapped_boxes) in sliding_window_list:
                # import pdb; pdb.set_trace()
                gt3D = torch.zeros_like(image3D,device=image3D.device, dtype=image3D.dtype)
                seg_mask_list, points, labels = finetune_model_predict3D(
                    image3D, gt3D, mapped_boxes, sam_model_tune, norm_transform, device=device, 
                    click_method=args.point_method, num_clicks=args.num_clicks, 
                    prev_masks=None)
                ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"] 
                # print('ori_roi',ori_roi)
                # print('pred_roi',pred_roi)

                for idx, seg_mask in enumerate(seg_mask_list):
                    
                    seg_mask_roi = seg_mask[..., pred_roi[0]:pred_roi[1], pred_roi[2]:pred_roi[3], pred_roi[4]:pred_roi[5]]

                    if seg_mask_roi.shape[-1] > gt3D_full.shape[-1]:
                        seg_mask_roi = seg_mask_roi[:,:,:gt3D_full.shape[-1]]
        
                    if pred3D_full_dict[idx][..., ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3], ori_roi[4]:ori_roi[5]].shape[2:] == seg_mask_roi.shape:
                        pred3D_full_dict[idx][..., ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3], ori_roi[4]:ori_roi[5]] = seg_mask_roi
                    else:
                        import pdb; pdb.set_trace()
                        print('error!')
            os.makedirs(vis_root, exist_ok=True)
            # Get the coordinate information of the last sliding window
            last_pos3D_roi = sliding_window_list[-1][1]
            # Calculate the point offset
            ori_roi = last_pos3D_roi["ori_roi"]
            pred_roi = last_pos3D_roi["pred_roi"]

            point_offset = np.array([ori_roi[0], ori_roi[2], ori_roi[4]])
            points = [p.cpu().numpy() + point_offset for p in points]
            labels = [l.cpu().numpy() for l in labels]
            pt_info = dict(points=points, labels=labels)
            pt_path = osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_pt.pkl"))
            pickle.dump(pt_info, open(pt_path, "wb"))

            if args.save_image_and_gt:
                save_numpy_to_nifti(image3D_full, osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_img.nii.gz")), meta_info)
                save_numpy_to_nifti(gt3D_full, osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_gt.nii.gz")), meta_info)
            for idx, pred3D_full in pred3D_full_dict.items():
                save_numpy_to_nifti(pred3D_full, osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{idx}.nii.gz")), meta_info)
                radius = 2
                for pt in points[:idx+1]:
                    pt = pt.astype(int)  # Convert point coordinates to integers
                    x, y, z = pt[0, 0, 0], pt[0, 0, 1], pt[0, 0, 2]
                    pred3D_full[..., max(x-radius, 0):min(x+radius, pred3D_full.shape[2]),
                                max(y-radius, 0):min(y+radius, pred3D_full.shape[3]),
                                max(z-radius, 0):min(z+radius, pred3D_full.shape[4])] = 10
                save_numpy_to_nifti(pred3D_full, osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{idx}_wPt.nii.gz")), meta_info)

            ''' metric computation '''
            for click_idx in range(args.num_clicks):
                reorient_tensor = lambda in_arr : np.transpose(in_arr.squeeze().detach().cpu().numpy(), (2, 1, 0))
                curr_pred_path = osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", f"_pred{click_idx}.nii.gz"))
                medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                iou_list.append(round(compute_iou(medsam_seg, reorient_tensor(gt3D_full)), 4))
                dice_list.append(round(compute_dice(reorient_tensor(gt3D_full), medsam_seg), 4))

            per_iou = max(iou_list)
            all_iou_list.append(per_iou)
            all_dice_list.append(max(dice_list))
            print(dice_list)
            out_dice[img_name] = max(dice_list)
            cur_dice_dict = OrderedDict()
            for i, dice in enumerate(dice_list):
                cur_dice_dict[f'{i}'] = dice
            out_dice_all[img_name] = cur_dice_dict

    print('Mean IoU : ', sum(all_iou_list)/len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list)/len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if(args.split_num > 1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    results_dir = os.path.join(*args.save_name.split('/')[:-1])
    os.makedirs(results_dir, exist_ok=True) 
    
    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
        f.writelines('}')

    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")


if __name__ == "__main__":
    # print(list(product((-32, +32, 0), repeat=3)))
    parser = argparse.ArgumentParser()
    parser.add_argument('-tdp', '--test_data_path', type=str, default='/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/data/validation/Dataset131_MICCAI_challenge_small')
    
    parser.add_argument('--output_dir', type=str, default='./visualization')
    parser.add_argument('--task_name', type=str, default='MICCAI_Challenge_2024_use_gt_3Dbox')
    parser.add_argument('--save_name', type=str, default='./results/union_out_dice_use_gt_3Dbox_and_mask_step_0_5.py')
    parser.add_argument('--skip_existing_pred', action='store_true', default=False)
    parser.add_argument('--save_image_and_gt', action='store_true', default=True)
    parser.add_argument('--sliding_window', action='store_true', default=True)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--Stage1_batch_size', type=int, default=32)
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--Stage1_weights", nargs="+", type=str, default="/mnt/data1/yujunxuan/MICCAI_challenge_method/YOLOV5/runs/train-seg/exp3/weights/best.pt", help="model path(s)")
    parser.add_argument("--data_config", type=str, default="/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/config/MICCAI_data_seg.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold(Stage1)")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold(Stage1)")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image(Stage1)")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    
    parser.add_argument('--Stage2_weights', type=str, default='/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/work_dir/Dataset131_MICCAI_challenge_small_v2/sam_model_latest.pth')

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
