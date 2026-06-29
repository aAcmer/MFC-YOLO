import torch

def bbox_iou(box, gt, name):
    """
    计算两个xywh格式边界框的IoU（交并比）
    
    参数:
        box (Tensor): 预测框，格式为[x_center, y_center, width, height]（归一化坐标）
        gt (Tensor): 真实框，格式同预测框（归一化坐标）
        H (int): 图像高度（用于归一化坐标转换）
        W (int): 图像宽度（用于归一化坐标转换）
    
    返回:
        iou (float): 交并比，范围[0,1]
    """
    # 将归一化的xywh转换为绝对坐标xyxy
    def xywh_to_xyxy(x, y, w, h):
        x1 = (x - w/2)
        y1 = (y - h/2)
        x2 = (x + w/2)
        y2 = (y + h/2)
        return x1, y1, x2, y2

    # 转换预测框和真实框
    box_x1, box_y1, box_x2, box_y2 = xywh_to_xyxy(box[0][0], box[0][1], box[0][2], box[0][3])
    gt_x1, gt_y1, gt_x2, gt_y2 = xywh_to_xyxy(gt[0][0],gt[0][1], gt[0][2], gt[0][3])


    # 计算交集区域坐标
    inter_x1 = torch.max(box_x1, gt_x1)
    inter_y1 = torch.max(box_y1, gt_y1)
    inter_x2 = torch.min(box_x2, gt_x2)
    inter_y2 = torch.min(box_y2, gt_y2)

    # 计算交集面积（处理无重叠情况）
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_width * inter_height

    # 计算各自面积
    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    # 计算并集面积
    union = box_area + gt_area - intersection + 1e-6  # 避免除零

    # 计算IoU
    iou = intersection / union

    # if name == 'scratches_90':
    #     print(box_x1, box_y1, box_x2, box_y2)
    #     print(gt_x1, gt_y1, gt_x2, gt_y2)
    #     print(intersection, union)
    #     print('#'*50)
    return iou