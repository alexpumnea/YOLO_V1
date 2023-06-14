import sys
import unittest
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os
import re
from NMS import nms
import numpy as np



class TestNonMaxSuppression(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_boxes = [
            np.multiply([1, 1, 0.5, 0.45, 0.4, 0.5], 1000),
            np.multiply([1, 0.8, 0.5, 0.5, 0.2, 0.4], 1000),
            np.multiply([1, 0.7, 0.25, 0.35, 0.3, 0.1], 1000),
            np.multiply([1, 0.05, 0.1, 0.1, 0.1, 0.1], 1000),
        ]

        self.c1_boxes = [np.multiply([1, 1, 0.5, 0.45, 0.4, 0.5], 1000), np.multiply([1, 0.7, 0.25, 0.35, 0.3, 0.1], 1000)]
        self.s1_boxes = "midpoint"

        self.t2_boxes = [
            np.multiply([1, 1, 0.5, 0.45, 0.4, 0.5], 1000),
            np.multiply([2, 0.9, 0.5, 0.5, 0.2, 0.4], 1000),
            np.multiply([1, 0.8, 0.25, 0.35, 0.3, 0.1], 1000),
            np.multiply([1, 0.05, 0.1, 0.1, 0.1, 0.1], 1000),
        ]

        self.c2_boxes = [
            np.multiply([1, 1, 0.5, 0.45, 0.4, 0.5], 1000),
            np.multiply([2, 0.9, 0.5, 0.5, 0.2, 0.4], 1000),
            np.multiply([1, 0.8, 0.25, 0.35, 0.3, 0.1], 1000),
        ]
        self.s2_boxes = "midpoint"

        self.t3_boxes = [
            np.multiply([1, 0.9, 0.5, 0.45, 0.4, 0.5], 1000),
            np.multiply([1, 1, 0.5, 0.5, 0.2, 0.4], 1000),
            np.multiply([2, 0.8, 0.25, 0.35, 0.3, 0.1], 1000),
            np.multiply([1, 0.05, 0.1, 0.1, 0.1, 0.1], 1000),
        ]

        self.c3_boxes = [np.multiply([1, 1, 0.5, 0.5, 0.2, 0.4], 1000), np.multiply([2, 0.8, 0.25, 0.35, 0.3, 0.1], 1000)]
        self.s3_boxes = "midpoint"

        self.t4_boxes = [
            np.multiply([1, 0.9, 0.5, 0.45, 0.4, 0.5], 1000),
            np.multiply([1, 1, 0.5, 0.5, 0.2, 0.4], 1000),
            np.multiply([1, 0.8, 0.25, 0.35, 0.3, 0.1], 1000),
            np.multiply([1, 0.05, 0.1, 0.1, 0.1, 0.1], 1000),
        ]

        self.c4_boxes = [
            np.multiply([1, 0.9, 0.5, 0.45, 0.4, 0.5], 1000),
            np.multiply([1, 1, 0.5, 0.5, 0.2, 0.4], 1000),
            np.multiply([1, 0.8, 0.25, 0.35, 0.3, 0.1], 1000),
        ]
        self.s4_boxes = "midpoint"

        self.t5_boxes = [
            np.multiply([1, 1, 1.5, 0.45, 1.2, 0.3], 1000),
            np.multiply([1, 0.8, 1.5, 0.8, 1.2, 0.6], 1000),
            np.multiply([1, 0.7, 1.25, 1.35, 1.1, 0.1], 1000),
            np.multiply([1, 0.05, 0.1, 0.1, 0.1, 0.1], 1000),
        ]

        self.c5_boxes = [ np.multiply([1, 1, 1.5, 0.45, 1.2, 0.3], 1000),
                         np.multiply([1, 0.7, 1.25, 1.35, 1.1, 0.1], 1000)]
        self.s5_boxes = "midpoint"

    def test_remove_on_iou(self):
        bboxes = nms(
            self.t1_boxes,
            probability_threshold=0.2,
            iou_threshold=7 / 20,
            boxes_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c1_boxes))

    def test_keep_on_class(self):
        bboxes = nms(
            self.t2_boxes,
            probability_threshold=0.2,
            iou_threshold=7 / 20,
            boxes_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c2_boxes))

    def test_remove_on_iou_and_class(self):
        bboxes = nms(
            self.t3_boxes,
            probability_threshold=0.2,
            iou_threshold=7 / 20,
            boxes_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c3_boxes))

    def test_keep_on_iou(self):
        bboxes = nms(
            self.t4_boxes,
            probability_threshold=0.2,
            iou_threshold=9 / 20,
            boxes_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c4_boxes))

    def test_6(self):
        # select the keys used to obtain the test parameters for drawing
        collection_boxes = list()
        collection_nms_boxes = list()
        collection_boxes_shape = list()
        for item in list(self.__dict__.keys()):
            # Regex to get the keys for the test
            if re.match('t[0-9]+', item):
                collection_boxes.append(item)
            if re.match('c[0-9]+', item):
                collection_nms_boxes.append(item)
            if re.match('s[0-9]+', item):
                collection_boxes_shape.append(item)
        output_img_path = os.getcwd() + "\\nms_results_img"
        if os.getcwd() is not output_img_path:
            if not os.path.isdir(output_img_path):
                os.mkdir(output_img_path)
            os.chdir(output_img_path)

        for index, item in enumerate (collection_boxes):
            # create a PNG file
            image_width = 2000
            image_height = 2000
            white_image = torch.zeros(image_width, image_height)
            image_name = item + ".png"
            torchvision.utils.save_image(white_image, image_name)
            img = read_image(image_name)

            # draw bounding box on the input image
            bboxes_tensor =[torch.tensor(item) for item in self.__dict__.get(collection_boxes[index])]
            bboxes_tensor = [torchvision.ops.box_convert(box[2:], 'cxcywh', 'xyxy').numpy() for box in bboxes_tensor]
            bboxes_tensor = torch.tensor(bboxes_tensor)
            img = draw_bounding_boxes(img, bboxes_tensor, width=1, colors=(255, 104, 26))
            # transform it to PIL image and display
            img = torchvision.transforms.ToPILImage()(img)
            img.save(image_name)

            img = read_image(image_name)
            bboxes_nms_tensor = [torch.tensor(item) for item in self.__dict__.get(collection_nms_boxes[index])]
            bboxes_nms_tensor = [torchvision.ops.box_convert(box[2:], 'cxcywh', 'xyxy').numpy() for box in bboxes_nms_tensor]
            bboxes_nms_tensor = torch.tensor(bboxes_nms_tensor)
            img = draw_bounding_boxes(img, bboxes_nms_tensor, width=1, colors=(0, 255, 0))
            # transform it to PIL image and display
            img = torchvision.transforms.ToPILImage()(img)
            img.save(image_name)
            img.close()


if __name__ == "__main__":
    print("NMS test run")
    unittest.main()