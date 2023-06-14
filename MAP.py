import torch
import numpy as np
from collections import Counter
from IoU import iou_pytorch

def map_pytorch(
        pred_boxes, true_boxes, iou_threshold=0.5, boxes_format="midpoint", num_classes=3
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all BBs (bboxes / Bounding Boxes) with each BBs
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted BBs is correct
        box_format (str): "midpoint" or "corners" used to specify BBs
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6
    # go through all the image classes
    for c in range(num_classes):
        detections = []
        ground_truths = []  # ground truths

# Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection) # get a list of detections for the class

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # create a dict amount_bboxes = {0:2, 1:3} where:
        #   0 is the image 0 with 2 ground truth bboxes
        #   1 is the image 0 with 3 ground truth bboxes
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # add a list of 0s of size(ground_truths)
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        # sorting over the probability score
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none  for a class then skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = iou_pytorch(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    boxes_format=boxes_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #print('Precision')
        average_precisions.append(torch.trapz(precisions, recalls))
        map = sum(average_precisions) / len(average_precisions)
    return map