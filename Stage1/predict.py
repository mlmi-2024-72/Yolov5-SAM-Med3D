# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import nibabel as nib
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, LoadImages3D
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
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
import matplotlib.pyplot as plt
import numpy as np

def visualize_protos(proto,im_name):
    """
    Visualizes each channel in the proto tensor.
    
    Args:
    - proto (torch.Tensor): The proto tensor of shape (1, C, H, W), where
      C is the number of channels (e.g., 32), and H, W are the dimensions
      of each feature map (e.g., 160x160).
    """
    # 确保输入是正确的形状
    if proto.dim() == 4 and proto.size(0) == 1:
        num_channels = proto.size(1)
        fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 10))  # 根据通道数调整这些参数
        for i, ax in enumerate(axes.flatten()):
            if i < num_channels:
                ax.imshow(proto[0, i].detach().cpu().numpy(), cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {i+1}')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'/data/rs/yjx/MICCAI_challenge_method/YOLOV5/debug_img/{im_name}')

def nms_2to3D(dets, thresh):
    """
    Merges 2D boxes to 3D cubes. Therefore, boxes of all slices are projected into one slices. An adaptation of Non-maximum surpression
    is applied, where clusters are found (like in NMS) with an extra constrained, that surpressed boxes have to have 'connected'
    z-coordinates w.r.t the core slice (cluster center, highest scoring box). 'connected' z-coordinates are determined
    as the z-coordinates with predictions until the first coordinate, where no prediction was found.

    example: a cluster of predictions was found overlap > iou thresh in xy (like NMS). The z-coordinate of the highest
    scoring box is 50. Other predictions have 23, 46, 48, 49, 51, 52, 53, 56, 57.
    Only the coordinates connected with 50 are clustered to one cube: 48, 49, 51, 52, 53. (46 not because nothing was
    found in 47, so 47 is a 'hole', which interrupts the connection). Only the boxes corresponding to these coordinates
    are surpressed. All others are kept for building of further clusters.

    This algorithm works better with a certain min_confidence of predictions, because low confidence (e.g. noisy/cluttery)
    predictions can break the relatively strong assumption of defining cubes' z-boundaries at the first 'hole' in the cluster.

    :param dets: (n_detections, (y1, x1, y2, x2, scores, slice_id)
    :param thresh: iou matchin threshold (like in NMS).
    :return: keep: (n_keep) 1D tensor of indices to be kept.
    :return: keep_z: (n_keep, [z1, z2]) z-coordinates to be added to boxes, which are kept in order to form cubes.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, -2]
    slice_id = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep_boxes = []

    keep = []
    keep_z = []

    while order.size > 0:  # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
        i = order[0]  # pop higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order] - inter)
        matches = np.argwhere(ovr > thresh)  # get all the elements that match the current box and have a lower score

        slice_ids = slice_id[order[matches]]
        core_slice = slice_id[int(i)]
        upper_wholes = [ii for ii in np.arange(core_slice, np.max(slice_ids)) if ii not in slice_ids]
        lower_wholes = [ii for ii in np.arange(np.min(slice_ids), core_slice) if ii not in slice_ids]
        max_valid_slice_id = np.min(upper_wholes) if len(upper_wholes) > 0 else np.max(slice_ids)
        min_valid_slice_id = np.max(lower_wholes) if len(lower_wholes) > 0 else np.min(slice_ids)
        z_matches = matches[(slice_ids <= max_valid_slice_id) & (slice_ids >= min_valid_slice_id)]

        xmin = np.min(x1[order[z_matches]])
        ymin = np.min(y1[order[z_matches]])
        zmin = np.min(slice_id[order[z_matches]]) - 1
        xmax = np.max(x2[order[z_matches]])
        ymax = np.max(y2[order[z_matches]])
        zmax = np.max(slice_id[order[z_matches]]) + 1

        keep_boxes.append([xmin, ymin, zmin, xmax, ymax, zmax])
        order = np.delete(order, z_matches, axis=0)

    return keep_boxes



def dice_coefficient(pred, truth):
    """计算两个样本的Dice系数。"""
    pred = np.asarray(pred).astype(np.bool_)
    truth = np.asarray(truth).astype(np.bool_)

    if pred.shape != truth.shape:
        raise ValueError("Shape mismatch: pred and truth must have the same shape.")

    intersection = np.logical_and(pred, truth)

    return 2. * intersection.sum() / (pred.sum() + truth.sum())

# 在模型预测之后计算Dice系数
def evaluate_segmentation(pred_path, gt_path):
    # 假设已经保存了预测和真实标签为NIfTI格式
    predicted_nifti = nib.load(gt_path)
    truth_nifti = nib.load(pred_path)
    
    predicted_data = predicted_nifti.get_fdata()
    truth_data = truth_nifti.get_fdata()
    
    dice_score = dice_coefficient(predicted_data, truth_data)
    print(f"Dice Score: {dice_score}")
    
    return dice_score

def cal_iou(boxes1, boxes2):
    """
    计算两组boxes之间的IoU矩阵。
    :param boxes1: 第一组boxes, 形状为 (N, 6) [xmin, ymin, zmin, xmax, ymax, zmax]
    :param boxes2: 第二组boxes, 形状为 (M, 6)
    :return: IoU矩阵, 形状为 (N, M)
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    # 扩展boxes维度以支持广播
    boxes1 = np.expand_dims(boxes1, axis=1)  # 形状变为 (N, 1, 6)
    boxes2 = np.expand_dims(boxes2, axis=0)  # 形状变为 (1, M, 6)

    # 计算交集
    inter_min = np.maximum(boxes1[:, :, :3], boxes2[:, :, :3])
    inter_max = np.minimum(boxes1[:, :, 3:], boxes2[:, :, 3:])
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]

    # 计算每个box的体积
    vol1 = np.prod(boxes1[:, :, 3:] - boxes1[:, :, :3], axis=2)
    vol2 = np.prod(boxes2[:, :, 3:] - boxes2[:, :, :3], axis=2)

    # 计算并集体积
    union_vol = vol1 + vol2 - inter_vol

    # 计算IoU
    iou = inter_vol / union_vol
    return iou

def match_boxes_using_iou_matrix(iou_matrix, iou_threshold=0.5):
    """
    使用IoU矩阵进行boxes匹配。
    :param iou_matrix: IoU矩阵
    :param iou_threshold: IoU阈值
    :return: 匹配的列表
    """
    matches = []
    # 选择满足阈值的IoU，并且确保一一匹配（即一个预测只能匹配一个真实box，反之亦然）
    while True:
        # 找到最大IoU及其索引
        iou_max = np.max(iou_matrix)
        if iou_max < iou_threshold:
            break
        idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        matches.append((idx[0], idx[1], iou_max))
        # 将已匹配的行和列设置为0，防止重复匹配
        iou_matrix[idx[0], :] = 0
        iou_matrix[:, idx[1]] = 0

    return matches

def calculate_precision_recall_ap(matches, num_gt, num_pred):
    if not matches:
        return 0

    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    tp = np.zeros(len(matches) + 1)
    fp = np.zeros(len(matches) + 1)

    for i, match in enumerate(matches, start=1):  # 从1开始索引
        tp[i] = 1
        fp[i] = 0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / float(num_gt)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

    # 包括从0开始的点
    recalls = np.concatenate(([0], recalls))
    precisions = np.concatenate(([1], precisions))  # 假设没有FP的情况下精度为1

    # 计算AP
    return np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])



import json
def default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Not serializable")

def save_predictions_to_json(keep_boxes, slice_preds, save_dir):
    results = keep_boxes
   
    with open(save_dir, 'w') as file:
        json.dump(results, file, indent=4,default=default)
    print(f"Saved 3D predictions to {save_dir}")


@smart_inference_mode()
def run(
    weights=ROOT / "runs/train-seg/exp25/weights/best.pt",  # model.pt path(s)
    source="/data/rs/yjx/MICCAI_challenge_data/train_yolov5_data_with_label/images/test",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/MICCAI_data_seg.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="5",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-seg",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    source = str(source)
    save_img = True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    # 直接加载3D数据
    case_3D = True
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name / "images", exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    save_dir_3D = increment_path(Path(project) / name / "volumes", exist_ok=exist_ok)  # increment run
    (save_dir_3D / "labels" if save_txt else save_dir_3D).mkdir(parents=True, exist_ok=True)  # make dir
    
    save_dir_3D_boxes = increment_path(Path(project) / name / "boxes", exist_ok=exist_ok)  # increment run
    (save_dir_3D_boxes / "labels" if save_txt else save_dir_3D_boxes).mkdir(parents=True, exist_ok=True)  # make dir


    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    elif case_3D:
        dataset = LoadImages3D(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    
    all_detections = []  # 用于收集所有切片的box预测
    dice_score_list = []
    mAP_list = []
    
    for path, im, im0s, vid_cap, s, ori_shape in dataset:
        if dataset.get_current_index() == 1:
            volume = np.zeros((ori_shape[0],ori_shape[1],dataset.get_slice_num()), dtype=np.uint8) # 用于收集所有切片的mask预测
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # 收集当前切片的预测结果，附加切片索引
        slice_index = dataset.get_current_index() - 1

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # 构建一个张量，分别为x,y,x,y,confidence,slice_index
            # save_det = torch.cat((det[:,:5], torch.tensor([[slice_index]] * det.size(0), device=det.device)), dim=1)
            # print(det[:,:5])
            if det.shape[0] == 0 :
                save_det = torch.cat((torch.zeros((1,5), device=det.device), torch.tensor([[slice_index]] * 1, device=det.device)), dim=1)
            else:
                save_det = torch.cat((det[:,:5], torch.tensor([[slice_index]] * det.size(0), device=det.device)), dim=1)
            # print('save_det',save_det.shape)
            all_detections.append(save_det)
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            # 可视化protos
            # visualize_protos(proto,p.name)
            save_path = str(save_dir / p.name.replace('nii.gz','png'))  # im.jpg
            save_path_3D = str(save_dir_3D / p.name)  # case.nii
            save_path_3D_box = str(save_dir_3D_boxes / p.name.replace('nii.gz','json'))
            
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            flag = 0
            
            if len(det):
                
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], ori_shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], ori_shape)  # HWC
                else:
                    masks, ori_shape_mask = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], ori_shape, upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    
                # 将这一层的mask都放到对应的3D层中，|=是或操作
                # print("slice_index",slice_index)
                # print(masks)
                if masks.shape[0] == 1:
                    volume[:,:,slice_index] |= masks.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                else:
                    volume[:,:,slice_index] |= masks.sum(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))
                    ]
                
                gt_mask_name = p.name.replace('_0000.nii.gz','')
                slice_index_new = slice_index+1
            
                gt_mask_path = f'/data/rs/yjx/MICCAI_challenge_data/Release-FLARE24-T1/Train_Labels_v1/{gt_mask_name}_slice_{slice_index_new}.png'

                try:
                    gt_mask = cv2.imread(gt_mask_path,0) 
                    gt_mask = torch.as_tensor(gt_mask,device=det.device) 
                    gt_mask = gt_mask.unsqueeze(0) / 255.0
                    flag = 1
                except:
                    gt_mask = torch.zeros_like(masks)
                    gt_mask = torch.as_tensor(gt_mask,device=det.device)
                
                if gt_mask.max() != 0:
                    print(gt_mask.max())
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.pred_gt_masks(
                    masks,
                    pred_colors=[colors(x, True) for x in det[:, 5]],
                    gt_masks=gt_mask,
                    gt_colors=[(0,255,0)],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
                    / 255
                    if retina_masks
                    else im[i],
                )
                
                
                # annotator.masks(
                #     masks,
                #     colors=[colors(x, True) for x in det[:, 5]],
                #     im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
                #     / 255
                #     if retina_masks
                #     else im[i],
                # )

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image" :
                    if flag == 1:
                        case_name = p.name.replace('.nii.gz','')
                        save_img_path = f'{save_dir}/{case_name}_{slice_index}.png'
                        cv2.imwrite(save_img_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        
        if dataset.is_last_slice():  # 检查是否为当前3D数据集的最后一切片
            # 将所有检测结果合并为一个大数组
            all_detections = torch.cat(all_detections, dim=0)
            keep_boxes = nms_2to3D(all_detections.cpu().numpy(), thresh=0.5)
            # 处理保留的检测
            # for idx in keep:
            #     # 此处可以根据需要处理和使用保留的检测，例如打印或进一步分析
            #     print(f"Kept detection {idx} with z-range {keep_z[idx]}")
                
            save_predictions_to_json(keep_boxes, all_detections, save_path_3D_box)
            all_detections = []  # 重置列表以准备下一个3D数据集
            
            # Create a NIfTI image
            nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
            
            # Save the NIfTI image
            nib.save(nifti_img, save_path_3D)
            print(f"Saved 3D volume to {save_path_3D}")
            
            nii_name = p.name.replace('_0000.nii.gz','.nii.gz')
            gt_nii_path = f'/data/rs/yjx/MICCAI_challenge_data/labeled_volumes/labels/test_labels/{nii_name}'
            dice_score = evaluate_segmentation(save_path_3D,gt_nii_path)
            dice_score_list.append(dice_score)
            
            box_json_name = p.name.replace('_0000.nii.gz','.json')
            gt_3D_box_json_path = f'/data/rs/yjx/MICCAI_challenge_data/labeled_volumes/box_json/test/{box_json_name}'
            with open(gt_3D_box_json_path, 'r') as f:
                gt_boxes = json.load(f)
            
            gt_boxes = np.array(gt_boxes)
            keep_boxes = np.array(keep_boxes)
            
            iou = cal_iou(keep_boxes,gt_boxes)
            matches = match_boxes_using_iou_matrix(iou,iou_threshold=0.5)
            mAP = calculate_precision_recall_ap(matches,gt_boxes.shape[0],keep_boxes.shape[0])
            print(mAP)
            mAP_list.append(mAP)
        
        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    
    LOGGER.info(f"Dice Score: %.2" % np.mean(dice_score_list))
    LOGGER.info(f"mAP: %.2" % np.mean(mAP_list))
    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line options for YOLOv5 inference including model paths, data sources, inference settings, and
    output preferences.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-seg/exp28/weights/best.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default="/data/rs/yjx/MICCAI_challenge_data/labeled_volumes/images/test", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/MICCAI_data_seg.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="0,1,2,3", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true",default=False, help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true",default=True, help="whether to plot masks in native resolution")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking for requirements before launching."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
