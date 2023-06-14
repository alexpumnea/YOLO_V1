import torch
from IoU import iou_pytorch


def nms(bboxes, iou_threshold, probability_threshold, boxes_format="corners"):
    """
    Given the BBs computes the NMS
    Parameters:
        bboxes (list): list of lists containing all BBs with each BBs
        specified as [class_pred, prob_score, x1, y1, x2, y2]

        iou_threshold (float): threshold where predicted BBs is correct

        threshold (float): threshold to remove predicted BBs (independent of IoU)

        box_format (str): "centre" or "corners" used to specify BBs
    Returns:
        list: BBs after performing NMS given a specific IoU threshold
    """


    bboxes = [box for box in bboxes if box[1] > probability_threshold]
    # Sort decreasing probability order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    nms_out = []

    while bboxes:
        bbox_selected = bboxes.pop(0) # Get the Highest score BB

        bboxes = [box for box in bboxes
            if box[0] != bbox_selected[0]
            or iou_pytorch(
                torch.tensor(bbox_selected[2:]),
                torch.tensor(box[2:]),
                boxes_format=boxes_format,
            )
            < iou_threshold
        ]

        nms_out.append(bbox_selected)

    return nms_out

bbs = [
        torch.tensor([1, 1, 0.5, 0.45, 0.4, 0.5]),
        torch.tensor([1, 0.8, 0.5, 0.5, 0.2, 0.4]),
        torch.tensor([1, 0.7, 0.25, 0.35, 0.3, 0.1]),
        torch.tensor([1, 0.05, 0.1, 0.1, 0.1, 0.1]),
    ]

#print(nms(bbs, probability_threshold=0.2, iou_threshold=7 / 20, boxes_format="midpoint"))
