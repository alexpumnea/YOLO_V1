import torch
#import torchvision
#from torchvision.io import read_image
#from torchvision.utils import draw_bounding_boxes
#from PIL import Image


def iou_pytorch(boxes_predictions, boxes_labels, boxes_format="midpoint"):
    """
    Compute intersection over union https://giou.stanford.edu/
    Parameters:
        boxes_predictions (tensor): Predictions of BBs (Bounding Boxes) (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of BBs (BATCH_SIZE, 4)
        boxes_format (str): centre/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union inputs
    """
    eps =  1e-6
    if boxes_format == "midpoint":
            box1_x1 = boxes_predictions[..., 0:1] - boxes_predictions[..., 2:3] / 2
            box1_y1 = boxes_predictions[..., 1:2] - boxes_predictions[..., 3:4] / 2
            box1_x2 = boxes_predictions[..., 0:1] + boxes_predictions[..., 2:3] / 2
            box1_y2 = boxes_predictions[..., 1:2] + boxes_predictions[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if boxes_format == "corners":
            box1_x1 = boxes_predictions[..., 0:1]
            box1_y1 = boxes_predictions[..., 1:2]
            box1_x2 = boxes_predictions[..., 2:3]
            box1_y2 = boxes_predictions[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) when there’s no intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + eps)