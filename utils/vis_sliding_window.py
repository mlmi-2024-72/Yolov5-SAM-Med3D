import numpy as np
import matplotlib.pyplot as plt

def visualize_roi(img3D, ori_roi, pred_roi, slice_idx):
    """
    可视化图像切片上的ROI框
    :param img3D: 3D图像数组，形状为(depth, height, width)
    :param ori_roi: 原始ROI，格式为(z_min, z_max, y_min, y_max, x_min, x_max)
    :param pred_roi: 预测ROI，格式为(z_min, z_max, y_min, y_max, x_min, x_max)
    :param slice_idx: 要可视化的z轴切片索引
    """
    img_slice = img3D[slice_idx, :, :]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_slice, cmap="gray")
    
    # 绘制原始ROI框
    y_min, y_max, x_min, x_max = ori_roi[2], ori_roi[3], ori_roi[4], ori_roi[5]
    ori_rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             edgecolor='r', facecolor='none', linewidth=2, label='Original ROI')
    ax.add_patch(ori_rect)
    
    # 绘制预测ROI框
    y_min, y_max, x_min, x_max = pred_roi[2], pred_roi[3], pred_roi[4], pred_roi[5]
    pred_rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              edgecolor='b', facecolor='none', linewidth=2, linestyle='--', label='Predicted ROI')
    ax.add_patch(pred_rect)
    
    ax.legend()
    ax.set_title(f'Visualization of ROI at z-slice {slice_idx}')
    # plt.show()
    plt.savefig('/mnt/data1/yujunxuan/MICCAI_challenge_method/yolov5-SAM-Med3D/1.png')

# 示例数据
img3D = np.random.rand(100, 256, 256)  # 生成随机3D图像
ori_roi = (30, 70, 100, 150, 50, 100)  # 示例原始ROI
pred_roi = (32, 72, 102, 152, 52, 102)  # 示例预测ROI

visualize_roi(img3D, ori_roi, pred_roi, slice_idx=50)
