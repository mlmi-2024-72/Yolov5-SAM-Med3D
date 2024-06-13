import numpy as np

def merge_boxes(boxes, distance_threshold):
    """
    Merge 3D boxes that are close to each other into larger boxes.

    Parameters:
    - boxes: List of 3D boxes [xmin, ymin, zmin, xmax, ymax, zmax]
    - distance_threshold: Distance threshold for merging boxes

    Returns:
    - merged_boxes: List of merged 3D boxes
    """
    merged_boxes = []
    while boxes:
        box = boxes.pop(0)
        to_merge = [box]

        for other_box in boxes[:]:
            if is_close(box, other_box, distance_threshold):
                to_merge.append(other_box)
                boxes.remove(other_box)
        if len(merged_boxes) != 0:
            for other_box in merged_boxes[:]:
                if is_close(box, other_box, distance_threshold):
                    to_merge.append(other_box)
                    merged_boxes.remove(other_box)

        merged_box = merge(to_merge)
        merged_boxes.append(merged_box)

    return merged_boxes

def is_close(box1, box2, threshold):
    """
    Check if two 3D boxes are close to each other.

    Parameters:
    - box1, box2: 3D boxes [xmin, ymin, zmin, xmax, ymax, zmax]
    - threshold: Distance threshold

    Returns:
    - close: Boolean indicating if boxes are close
    """
    center1 = np.array([(box1[0] + box1[3]) / 2, (box1[1] + box1[4]) / 2, (box1[2] + box1[5]) / 2])
    center2 = np.array([(box2[0] + box2[3]) / 2, (box2[1] + box2[4]) / 2, (box2[2] + box2[5]) / 2])
    distance = np.linalg.norm(center1 - center2)
    return distance < threshold

def merge(boxes):
    """
    Merge multiple 3D boxes into a single box.

    Parameters:
    - boxes: List of 3D boxes [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
    - merged_box: Single merged 3D box
    """
    xmin = min(box[0] for box in boxes)
    ymin = min(box[1] for box in boxes)
    zmin = min(box[2] for box in boxes)
    xmax = max(box[3] for box in boxes)
    ymax = max(box[4] for box in boxes)
    zmax = max(box[5] for box in boxes)
    return [xmin, ymin, zmin, xmax, ymax, zmax]

def nms_2to3D_with_merging(dets, nms_thresh=0.5, merge_thresh=10, max_boxes=20):
    """
    Merges 2D boxes to 3D cubes and then merges close 3D boxes into larger boxes.

    Parameters:
    - dets: (n_detections, (y1, x1, y2, x2, scores, slice_id))
    - nms_thresh: IoU matching threshold for NMS
    - merge_thresh: Distance threshold for merging boxes
    - max_boxes: Maximum number of boxes after merging

    Returns:
    - final_boxes: List of final merged 3D boxes
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

    while order.size > 0:  # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
        i = order[0]  # pop highest scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order] - inter)
        matches = np.argwhere(ovr > nms_thresh)  # get all the elements that match the current box and have a lower score

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

    print('Before merging:', len(keep_boxes))

    # Merge close boxes into larger boxes until we have fewer than max_boxes
    while len(keep_boxes) > max_boxes:
        keep_boxes = merge_boxes(keep_boxes, merge_thresh)
        merge_thresh += 5  # Increase the merge threshold to encourage more merging

    print('After merging:', len(keep_boxes))
    return keep_boxes

if __name__ == "__main__":
    # 示例使用
    dets = np.random.rand(100, 6)  # 假设你有一些2D检测框数据
    nms_thresh = 0.5  # 非极大值抑制阈值
    merge_thresh = 10  # 3D框融合的距离阈值

    final_boxes = nms_2to3D_with_merging(dets, nms_thresh, merge_thresh)
    print(final_boxes)
