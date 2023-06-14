import sys
import unittest
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from typing import NamedTuple
import re
from PIL import Image
import os

from IoU import iou_pytorch

class iou_TestScenario(NamedTuple):
    box1: torch.FloatTensor
    box2: torch.FloatTensor
    correct_iou: float
    coordinates_type: str

class TestIntersectionOverUnion(unittest.TestCase):
    def setUp(self):
        """
        Test several intersection over union scenarios
        :return:
        """
        self.t1 = iou_TestScenario(torch.tensor([800, 100, 200, 200]), torch.tensor([900, 200, 200, 200]), 1 / 7, "midpoint")
        self.t2 = iou_TestScenario(torch.tensor([950, 600, 500, 200]), torch.tensor([950, 700, 300, 200]), 3 / 13, "midpoint")
        self.t3 = iou_TestScenario(torch.tensor([250, 150, 300, 100]), torch.tensor([250, 350, 300, 100]), 0, "midpoint")
        self.t4 = iou_TestScenario(torch.tensor([700, 950, 600, 100]), torch.tensor([500, 1150, 400, 700]), 3 / 31, "midpoint")
        self.t5 = iou_TestScenario(torch.tensor([500, 500, 200, 200]), torch.tensor([500, 500, 200, 200]), 1, "midpoint")

        # Prevent convergence issues
        self.epsilon = 0.001

    def test_1(self):
        iou = iou_pytorch(self.t1.box1, self.t1.box2, boxes_format=self.t1.coordinates_type)
        self.assertTrue((torch.abs(iou - self.t1.correct_iou) < self.epsilon))
    def test_2(self):
        iou = iou_pytorch(self.t2.box1, self.t2.box2, boxes_format=self.t2.coordinates_type)
        self.assertTrue((torch.abs(iou - self.t2.correct_iou) < self.epsilon))
    def test_3(self):
        iou = iou_pytorch(self.t3.box1, self.t3.box2, boxes_format=self.t3.coordinates_type)
        self.assertTrue((torch.abs(iou - self.t3.correct_iou) < self.epsilon))
    def test_4(self):
        iou = iou_pytorch(self.t4.box1, self.t4.box2, boxes_format=self.t4.coordinates_type)
        self.assertTrue((torch.abs(iou - self.t4.correct_iou) < self.epsilon))
    def test_5(self):
        iou = iou_pytorch(self.t5.box1, self.t5.box2, boxes_format=self.t5.coordinates_type)
        self.assertTrue((torch.abs(iou - self.t5.correct_iou) < self.epsilon))

    def test_6(self):
        # select the keys used to obtain the test parameters for drawing
        collection = list()
        for item in list(self.__dict__.keys()):
            # Regex to get the keys for the test
            if re.match('t[0-9]+', item):
                collection.append(item)
        output_img_path = os.getcwd() + "\\iou_results_img"
        if os.getcwd() is not output_img_path:
            if not os.path.isdir(output_img_path):
                os.mkdir(output_img_path)
            os.chdir(output_img_path)

        for item in collection:


            # create a PNG file
            image_width = 2000
            image_height = 2000
            white_image = torch.zeros(image_width, image_height)
            image_name = item + ".png"
            torchvision.utils.save_image(white_image, image_name)
            img = read_image(image_name)

            # draw bounding box on the input image
            bboxes = torch.tensor(
                [torchvision.ops.box_convert(self.__dict__.get(item).box1, 'cxcywh', 'xyxy').numpy(),
                 torchvision.ops.box_convert(self.__dict__.get(item).box2, 'cxcywh', 'xyxy').numpy()])
            img = draw_bounding_boxes(img, bboxes, width=10, colors=(255, 104, 26))

            # transform it to PIL image and display
            img = torchvision.transforms.ToPILImage()(img)

            img.save(image_name)

if __name__ == "__main__":
    print("IoU Unit Test run")
    unittest.main()


