import sys
import unittest
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os
import re
from typing import NamedTuple
import numpy as np

from MAP import map_pytorch

class map_TestScenario(NamedTuple):
    predictions: list
    targets: list
    correct_map: float

class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):
        # test cases we want to run

        self.t1 = map_TestScenario(predictions=[
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],

        targets=[
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        correct_map=1)



        self.t2 = map_TestScenario(predictions=[
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],
        targets=[
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],
        correct_map=1)

        self.t3 = map_TestScenario(predictions=[
            [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],
        targets=[
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],
        correct_map = 0)

        self.t4 = map_TestScenario(predictions=[
            [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],
        targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ],
        correct_map= 5 / 18)


        self.epsilon = 1e-4

    def test_1(self):
        mean_avg_prec = map_pytorch(
            self.__dict__.get('t1').predictions,
            self.__dict__.get('t1').targets,
            iou_threshold=0.5,
            boxes_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.__dict__.get('t1').correct_map - mean_avg_prec) < self.epsilon)


    def test_2(self):
        mean_avg_prec = map_pytorch(
            self.__dict__.get('t2').predictions,
            self.__dict__.get('t2').targets,
            iou_threshold=0.5,
            boxes_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.__dict__.get('t2').correct_map - mean_avg_prec) < self.epsilon)

    def test_3(self):
        mean_avg_prec = map_pytorch(
            self.__dict__.get('t3').predictions,
            self.__dict__.get('t3').targets,
            iou_threshold=0.5,
            boxes_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(self.__dict__.get('t3').correct_map - mean_avg_prec) < self.epsilon)

    def test_4(self):
        mean_avg_prec = map_pytorch(
            self.__dict__.get('t4').predictions,
            self.__dict__.get('t4').targets,
            iou_threshold=0.5,
            boxes_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.__dict__.get('t4').correct_map - mean_avg_prec) < self.epsilon)

    def test_5(self):
        mean_avg_prec = map_pytorch(
            self.__dict__.get('t3').predictions,
            self.__dict__.get('t3').targets,
            iou_threshold=0.5,
            boxes_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(self.__dict__.get('t3').correct_map - mean_avg_prec) < self.epsilon)


    def test_6(self):
        # select the keys used to obtain the test parameters for drawing
        collection_tests = list()
        for item in list(self.__dict__.keys()):
            # Regex to get the keys for the test
            if re.match('t[0-9]+', item):
                collection_tests.append(item)
        output_img_path = os.getcwd() + "\\map_results_img"
        if os.getcwd() is not output_img_path:
            if not os.path.isdir(output_img_path):
                os.mkdir(output_img_path)
            os.chdir(output_img_path)

        for index, item in enumerate (collection_tests):
            # create a PNG file
            image_width = 2000
            image_height = 2000
            white_image = torch.zeros(image_width, image_height)
            image_name = item + ".png"
            torchvision.utils.save_image(white_image, image_name)
            img = read_image(image_name)

            # draw bounding box on the input image
            predictions_tensor = [torch.tensor(np.multiply(i[3:], 1000)) for i in self.__dict__.get(item).predictions]
            predictions_tensor = [torchvision.ops.box_convert(box, 'cxcywh', 'xyxy').numpy() for box in predictions_tensor]
            predictions_tensor = torch.tensor(predictions_tensor)
            img = draw_bounding_boxes(img, predictions_tensor, width=3, colors=(255, 104, 26))
            # have different widths to distinguish between target and prediction
            # transform it to PIL image and display
            img = torchvision.transforms.ToPILImage()(img)
            img.save(image_name)


            img = read_image(image_name)
            target_tensor = [torch.tensor(np.multiply(i[3:], 1000)) for i in self.__dict__.get(item).targets]
            target_tensor = [torchvision.ops.box_convert(box, 'cxcywh', 'xyxy').numpy() for box in target_tensor]
            target_tensor = torch.tensor(target_tensor)
            img = draw_bounding_boxes(img, target_tensor, width=1, colors=(0, 255, 0))
            # transform it to PIL image and display
            img = torchvision.transforms.ToPILImage()(img)
            img.save(image_name)
            img.close()




if __name__ == "__main__":
    print("Running Mean Average Precisions Tests:")
    unittest.main()