"""
Creates a Pytorch dataset to load the dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
from utils import class_size

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=class_size, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        #print(label_path)
        with open(label_path) as f:
            for label in f.readlines():
              #  try:
                if label.strip():
                    #print(label)
                    class_label, x, y, width, height = [
                        float(x) if float(x) != int(float(x)) else int(x)   # if int but float convert to int
                        for x in label.replace("\n", "").split()]
                    boxes.append([class_label, x, y, width, height])



        img_path = os.path.normpath(os.path.join(self.img_dir, self.annotations.iloc[index, 0]))
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) #
        for box in boxes:
            class_label, x, y, width, height = box.tolist() # list to tensor to list since in tensor shape we can apply
                                                            # transformation for data augmentation
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            # x, y,w, h _cell = all are relative to the cell.
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i


            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # the algorithm is restricted to ONE object per cell!
            if label_matrix[i, j, class_size] == 0:
                # Set that there exists an object
                label_matrix[i, j, class_size] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                ## box coord start after the number of classes
                label_matrix[i, j, 4:8] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1



        return image, label_matrix