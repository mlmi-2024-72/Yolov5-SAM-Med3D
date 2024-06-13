import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_next_click3D_torch_no_gt(prev_seg, img3D, threshold=170):
    """Selects prompt clicks from thresholded image (img3D) based on the previous segmentation (prev_seg).

    Args:
        prev_seg (torch.tensor): segmentation masks from previous iteration
        img3D (torch.tensor): input images
        threshold (int, optional): threshold value to apply to image for selecting point click. Defaults to 170.

    Returns:
        batch_points (list of torch.tensor): list of points to click
        batch_labels (list of torch.tensor): list of labels corresponding to the points
        NOTE: In this case, the labels are based on the thresholded image and not the ground truth.
    """

    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    likely_masks = img3D > threshold  # NOTE: Empirical threshold
    try:
        fn_masks = torch.logical_and(likely_masks, torch.logical_not(pred_masks))
    except:
        import pdb; pdb.set_trace()
    # NOTE: Given a strict/high threshold, the false positives are not going to be very useful (at least in my case)
    # fp_masks = torch.logical_and(torch.logical_not(likely_masks), pred_masks)

    for i in range(prev_seg.shape[0]):  # , desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        point = None
        if len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        # if no mask is given, random click a negative point
        if point is None:
            point = torch.Tensor(
                [np.random.randint(sz) for sz in fn_masks[i].size()]
            ).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1, 1, -1).to(pred_masks.device)
        bl = (
            torch.tensor(
                [
                    int(is_positive),
                ]
            )
            .reshape(1, 1)
            .to(pred_masks.device)
        )

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click3D_torch_no_gt_naive(prev_seg):
    """Selects prompt clicks from the area outside predicted masks based on previous segmentation (prev_seg). 

    Args:
        prev_seg (torch.tensor): segmentation masks from previous iteration

    Returns:
        batch_points (list of torch.tensor): list of points to click
        batch_labels (list of torch.tensor): list of labels corresponding to the points
        NOTE: In this case, the labels are based on the predicted masks and not the ground truth.
    """
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    uncertain_masks = torch.logical_xor(
        pred_masks, pred_masks
    )  # init with all False

    for i in range(prev_seg.shape[0]):
        uncertain_region = torch.logical_or(uncertain_masks[i, 0], pred_masks[i, 0])
        points = torch.argwhere(uncertain_region) # select outside of pred mask

        if len(points) > 0:
            point = points[np.random.randint(len(points))]
            is_positive = pred_masks[i, 0, point[1], point[2], point[3]]

            bp = point[1:].clone().detach().reshape(1, 1, 3)
            bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
            batch_points.append(bp)
            batch_labels.append(bl)
        else:
            point = torch.Tensor(
                [np.random.randint(sz) for sz in pred_masks[i, 0].size()]
            ).to(torch.int64)
            is_positive = pred_masks[i, 0, point[1], point[2], point[3]]

            bp = point[1:].clone().detach().reshape(1, 1, 3)
            bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
            batch_points.append(bp)
            batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click3D_torch(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):  # , desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        point = None
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # if no mask is given, random click a negative point
        if point is None:
            point = torch.Tensor(
                [np.random.randint(sz) for sz in fn_masks[i].size()]
            ).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1, 1, -1).to(pred_masks.device)
        bl = (
            torch.tensor(
                [
                    int(is_positive),
                ]
            )
            .reshape(1, 1)
            .to(pred_masks.device)
        )

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


import edt


def safe_random_int(min_val, max_val):
    if max_val - min_val <= 1:
        return min_val
    else:
        return np.random.randint(min_val, max_val)

def get_next_click3D_torch_ritm_with_box(prev_seg, gt_semantic_seg, pred_3D_boxes):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0

    if true_masks.sum() == 0:  
        
        non_zero_boxes = [j for j in range(pred_3D_boxes.shape[0]) if not torch.all(pred_3D_boxes[j] == 0)]
        if non_zero_boxes:
            box_index = np.random.choice(non_zero_boxes)
            box = pred_3D_boxes[box_index]
            
            x_min, y_min, z_min, x_max, y_max, z_max = box.clone().cpu().numpy()
            
         
            z_point = safe_random_int(z_min, z_max)
            y_point = safe_random_int(y_min, y_max)
            x_point = safe_random_int(x_min, x_max)
          
            point = torch.tensor([x_point, y_point, z_point], device=pred_3D_boxes.device).reshape(1, 1, 3)
            batch_points.append(point)
            batch_labels.append(torch.tensor([[1]], device=pred_3D_boxes.device, dtype=torch.int64))
        return batch_points, batch_labels
    


    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1, 1, 1, 1, 1, 1), "constant", value=0).to(
        torch.uint8
    )[0, 0]
    fp_mask_single = F.pad(fp_masks, (1, 1, 1, 1, 1, 1), "constant", value=0).to(
        torch.uint8
    )[0, 0]
    fn_mask_dt = torch.tensor(
        edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4)
    )[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(
        edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4)
    )[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = (
        fn_max_dist > fp_max_dist
    )  # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0)  # use a erosion area
    to_point_mask = to_point_mask[None, None]

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        try:
            point = points[np.random.randint(len(points))]
        except:
            import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()

def get_next_click3D_torch_with_gt_3Dbox(prev_seg, gt_3D_boxes):
    """Selects prompt clicks from the area inside predicted 3D boxes based on previous segmentation (prev_seg). 

    Args:
        prev_seg (torch.tensor): segmentation masks from previous iteration
        gt_3D_boxes (torch.tensor): gt 3D bounding boxes

    Returns:
        batch_points (list of torch.tensor): list of points to click
        batch_labels (list of torch.tensor): list of labels corresponding to the points
        NOTE: In this case, the labels are based on the predicted masks and not the ground truth.
    """
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold

    

    for i in range(prev_seg.shape[0]):
        try:
            non_zero_boxes = [j for j in range(gt_3D_boxes.shape[1]) if not torch.all(gt_3D_boxes[i, j] == 0)]
        except:
            print(non_zero_boxes)
            import pdb; pdb.set_trace()
        if non_zero_boxes:
            box_index = np.random.choice(non_zero_boxes)
            # import pdb; pdb.set_trace()
            if gt_3D_boxes.shape[0] == prev_seg.shape[0]:
                box = gt_3D_boxes[i][box_index]
            else:
                box = gt_3D_boxes[box_index]
            
            x_min, y_min, z_min, x_max, y_max, z_max = box.clone().cpu().numpy()
            
            z_point = safe_random_int(z_min, z_max)
            y_point = safe_random_int(y_min, y_max)
            x_point = safe_random_int(x_min, x_max)
            
            # # Compute the center point of the box
            # x_point = int((x_min + x_max) / 2)
            # y_point = int((y_min + y_max) / 2)
            # z_point = int((z_min + z_max) / 2)
            
            # Ensure the points are within the bounds of the pred_masks dimensions
            z_point = np.clip(z_point, 0, pred_masks.shape[4] - 1)
            y_point = np.clip(y_point, 0, pred_masks.shape[3] - 1)
            x_point = np.clip(x_point, 0, pred_masks.shape[2] - 1)
            
            # import pdb; pdb.set_trace()   
            point = torch.tensor([x_point, y_point, z_point], device=gt_3D_boxes.device).reshape(1, 1, 3)
            batch_points.append(point)
            batch_labels.append(torch.tensor([[1]], device=gt_3D_boxes.device, dtype=torch.int64))
        else:
            uncertain_masks = torch.logical_xor(pred_masks, pred_masks)  # init with all False
            uncertain_region = torch.logical_or(uncertain_masks[i, 0], pred_masks[i, 0])
            points = torch.argwhere(uncertain_region)  # select outside of pred mask

            if len(points) > 0:
                point = points[np.random.randint(len(points))]
                is_positive = pred_masks[i, 0, point[0], point[1], point[2]]

                bp = point.clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)
            else:
                point = torch.Tensor(
                    [np.random.randint(sz) for sz in pred_masks[i, 0].size()]
                ).to(torch.int64).to(gt_3D_boxes.device)
                is_positive = pred_masks[i, 0, point[0], point[1], point[2]]

                bp = point.clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click3D_torch_with_pred_3Dbox(prev_seg, pred_3D_boxes):
    """Selects prompt clicks from the area inside predicted 3D boxes based on previous segmentation (prev_seg). 

    Args:
        prev_seg (torch.tensor): segmentation masks from previous iteration
        pred_3D_boxes (torch.tensor): predicted 3D bounding boxes

    Returns:
        batch_points (list of torch.tensor): list of points to click
        batch_labels (list of torch.tensor): list of labels corresponding to the points
        NOTE: In this case, the labels are based on the predicted masks and not the ground truth.
    """
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold

    

    for i in range(prev_seg.shape[0]):
        non_zero_boxes = [j for j in range(pred_3D_boxes.shape[1]) if not torch.all(pred_3D_boxes[i, j] == 0)]
        if non_zero_boxes:
            box_index = np.random.choice(non_zero_boxes)
            # import pdb; pdb.set_trace()
            if pred_3D_boxes.shape[0] == prev_seg.shape[0]:
                box = pred_3D_boxes[i][box_index]
            else:
                box = pred_3D_boxes[box_index]
            
            x_min, y_min, z_min, x_max, y_max, z_max = box.clone().cpu().numpy()
            
            # z_point = safe_random_int(z_min, z_max)
            # y_point = safe_random_int(y_min, y_max)
            # x_point = safe_random_int(x_min, x_max)
            
            # Compute the center point of the box
            x_point = int((x_min + x_max) / 2)
            y_point = int((y_min + y_max) / 2)
            z_point = int((z_min + z_max) / 2)
            
            # Ensure the points are within the bounds of the pred_masks dimensions
            z_point = np.clip(z_point, 0, pred_masks.shape[4] - 1)
            y_point = np.clip(y_point, 0, pred_masks.shape[3] - 1)
            x_point = np.clip(x_point, 0, pred_masks.shape[2] - 1)
            
            # import pdb; pdb.set_trace()   
            point = torch.tensor([x_point, y_point, z_point], device=pred_3D_boxes.device).reshape(1, 1, 3)
            batch_points.append(point)
            batch_labels.append(torch.tensor([[1]], device=pred_3D_boxes.device, dtype=torch.int64))
        else:
            uncertain_masks = torch.logical_xor(pred_masks, pred_masks)  # init with all False
            uncertain_region = torch.logical_or(uncertain_masks[i, 0], pred_masks[i, 0])
            points = torch.argwhere(uncertain_region)  # select outside of pred mask

            if len(points) > 0:
                point = points[np.random.randint(len(points))]
                is_positive = pred_masks[i, 0, point[0], point[1], point[2]]

                bp = point.clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)
            else:
                point = torch.Tensor(
                    [np.random.randint(sz) for sz in pred_masks[i, 0].size()]
                ).to(torch.int64).to(pred_3D_boxes.device)
                is_positive = pred_masks[i, 0, point[0], point[1], point[2]]

                bp = point.clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click3D_torch_2_with_box(prev_seg, gt_semantic_seg, pred_3D_boxes):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    
    if true_masks.sum() == 0:  
        non_zero_boxes = [j for j in range(pred_3D_boxes.shape[0]) if not torch.all(pred_3D_boxes[j] == 0)]
        if non_zero_boxes:
            box_index = np.random.choice(non_zero_boxes)
            box = pred_3D_boxes[box_index]
            x_min, y_min, z_min, x_max, y_max, z_max = box.clone().cpu().numpy()
            z_point = safe_random_int(z_min, z_max)
            y_point = safe_random_int(y_min, y_max)
            x_point = safe_random_int(x_min, x_max)
            point = torch.tensor([x_point, y_point, z_point], device=pred_3D_boxes.device).reshape(1, 1, 3)
            batch_points.append(point)
            batch_labels.append(torch.tensor([[1]], device=pred_3D_boxes.device, dtype=torch.int64))
        return batch_points, batch_labels
    
    
    
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        # import pdb; pdb.set_trace()
        points = torch.argwhere(to_point_mask[i])

        point = points[np.random.randint(len(points))]

        # import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()

def get_next_click3D_torch_ritm(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1, 1, 1, 1, 1, 1), "constant", value=0).to(
        torch.uint8
    )[0, 0]
    fp_mask_single = F.pad(fp_masks, (1, 1, 1, 1, 1, 1), "constant", value=0).to(
        torch.uint8
    )[0, 0]
    fn_mask_dt = torch.tensor(
        edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4)
    )[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(
        edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4)
    )[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = (
        fn_max_dist > fp_max_dist
    )  # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0)  # use a erosion area
    to_point_mask = to_point_mask[None, None]
    # import pdb; pdb.set_trace()

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        try:
            point = points[np.random.randint(len(points))]
        except:
            import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        # import pdb; pdb.set_trace()
        points = torch.argwhere(to_point_mask[i])
        # print(points.shape)
        # if len(points) == 0:
        #     # 如果没有要处理的点，跳过当前迭代
        #     continue

        
        point = points[np.random.randint(len(points))]

        # import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


def get_next_click3D_torch_with_dice(prev_seg, gt_semantic_seg):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = mask_pred > mask_threshold
        # mask_gt = mask_gt.astype(bool)
        mask_gt = mask_gt > 0

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # bp = torch.tensor(point[1:]).reshape(1,1,3)
        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    return batch_points, batch_labels, (sum(dice_list) / len(dice_list)).item()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_point(point, label, ax):
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color="red"))
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color="green"))
    # plt.scatter(point[0], point[1], label=label)


if __name__ == "__main__":
    gt2D = torch.randn((2, 1, 256, 256)).cuda()
    prev_masks = torch.zeros_like(gt2D).to(gt2D.device)
    batch_points, batch_labels = get_next_click3D_torch(
        prev_masks.to(gt2D.device), gt2D
    )
    print(batch_points)
