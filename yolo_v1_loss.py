
import torch
import torch.nn as nn
from IoU import iou_pytorch
from utils import class_size


class YoloLoss(nn.Module):
    """
    loss for yolo (v1)
    """

    def __init__(self, S=7, B=2, C=class_size):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # mean squared error

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (for the face mask detection is we use 3, check the mask decoder file for classification)
        """
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = iou_pytorch(predictions[..., 4:8], target[..., 4:8])
        # 0-2 class probability
        # 3 class score
        # 4-8 BB coordinates for the first box

        iou_b2 = iou_pytorch(predictions[..., 9:13], target[..., 4:8])
        # 9-13 BB coordinates for the second box

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that best_box will be indices of 0, 1 for which bbox was best
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., class_size].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 9:13]
                + (1 - best_box) * predictions[..., 4:8]
            )
        )

        box_targets = exists_box * target[..., 4:8]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        #  (N,S,S,4) -> (N*S*S,4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2), # -2 convert to a format compatible to mean squared error (mse)
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            best_box * predictions[..., 8:9] + (1 - best_box) * predictions[..., 3:4]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 3:4]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # For Both Boxes
        #1
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 3:4], start_dim=1), # start_dim=1 => (N,S,S,1) -> (N,S*S)
            torch.flatten((1 - exists_box) * target[..., 3:4], start_dim=1),
        )
        #2
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 8:9], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 3:4], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :class_size], end_dim=-2,),
            torch.flatten(exists_box * target[..., :class_size], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first component of the loss function in the paper
            + object_loss  # third component of the loss function in the paper
            + self.lambda_noobj * no_object_loss  # forth component of the loss function in the paper
            + class_loss  # fifth component of the loss function in the paper
        )

        return loss