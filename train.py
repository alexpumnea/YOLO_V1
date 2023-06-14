"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from yolo_v1 import Yolo_v1
from in_process import VOCDataset
from IoU import iou_pytorch
from NMS import nms
from MAP import map_pytorch
from utils import (
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    decode_label
)
from yolo_v1_loss import YoloLoss
import time
import sys
import datetime
import os
import subprocess

# open a file for writing
with open('output.txt', 'w') as f:
    # redirect standard output to the file
    sys.stdout = f


# restore standard output to the console
sys.stdout = sys.__stdout__

start_time = time.time()


# When      LOAD_MODEL = True => TEST
#           LOAD_MODEL = False => TRAIN
global LOAD_MODEL, scenario_type, sim_env

LOAD_MODEL = False#
scenario_type = "mask" #"general"
sim_env = "local"




seed = 123
torch.manual_seed(seed) # get the same data sate loading

# Hyperparameters
LEARNING_RATE = 2e-5 #
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 64 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0    # We are not retraining the entire model | how will this work here:
                    # Overfit a single batch and check if it makes sense, mAP ~ 1
                    # Overfit 100 examples
                    # To verfit easily we set in a way that we have no regularisation

EPOCHS = 1000
NUM_WORKERS = 2 #
PIN_MEMORY = True
UNIVERSAL_STAMP = str(datetime.datetime.now()).replace("-","_",).replace(" ", "_").replace(":", "_").replace(".", "_")[:20]


hyper_IoU_th = 0.7
hyper_pobability_th = 0.7

# Step by step save of the weights
# map thresholds fo intermediary save of the weights
#map_th = [[0.55, False],[0.6, False],[0.65, False],[0.7, False],[0.75, False], [0.80, False], [0.85, False], [0.9, False], [0.95, False],[0.975, False],[0.98, False]]

map_th = [[0.80, False], [0.85, False], [0.9, False], [0.95, False],[0.975, False],[0.98, False]]



#global LOAD_MODEL_FILE, IMG_DIR, LABEL_DIR, LABEL_DECODER_PATH, TRAIN_DATASET



if sim_env == "local":
    if scenario_type == "general":
        LOAD_MODEL_FILE = "overfit.pth.tar"
        IMG_DIR = "data/images"
        LABEL_DIR = "data/labels"
        LABEL_DECODER_PATH = "data/label_decoder.txt"
        TRAIN_DATASET = "data/100examples.csv"
        TEST_DATASET = TRAIN_DATASET

    elif scenario_type == "mask":
        LOAD_MODEL_FILE = UNIVERSAL_STAMP + "mask_overfit.pth.tar"
        IMG_DIR ="data_mask/images_jpg_channels_fixed" #"data_mask/image_jpg_reshaped"
        LABEL_DIR = "data_mask/labels"
        LABEL_DECODER_PATH = "data_mask/label_decoder.txt"
        TRAIN_DATASET = "data_mask/100examples.csv"
        TEST_DATASET = "data_mask/test.csv"

    else:
        print("Scenario not SPECIFIED")
        sys.exit()
elif sim_env == "colab":
    if scenario_type == "general":
        LOAD_MODEL_FILE = "overfit.pth.tar"
        IMG_DIR = "data/images"
        LABEL_DIR = "data/labels"
        LABEL_DECODER_PATH = "data/label_decoder.txt"
        TRAIN_DATASET = "data/100examples.csv"
    elif scenario_type == "mask":
        LOAD_MODEL_FILE = UNIVERSAL_STAMP + "mask_overfit.pth.tar"
        IMG_DIR = "data_mask/images_jpg_channels_fixed" #"data_mask/image_jpg_reshaped"
        LABEL_DIR = "data_mask/labels"
        LABEL_DECODER_PATH = "data_mask/label_decoder.txt"
        TRAIN_DATASET = "data_mask/100examples.csv"
        TEST_DATASET = "data_mask/test.csv"
else:
    print("Simulation environment NOT SPECIFIED")
    sys.exit()

class Compose(object):
    # chain multiple img transformations if needed
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes    # the transform that we send in is going to operate on the image.

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])    # to improve do a normalization
                                                                                # => mean = 0 and stdev = 1


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    print(torch.cuda.is_available())
    model = Yolo_v1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        PASCAL_VOC_labels_decoded = decode_label(LABEL_DECODER_PATH)

    train_dataset = VOCDataset(
        TRAIN_DATASET,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        TEST_DATASET, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
        # drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
        #drop_last=True
    )

    print("\n The content of the Loader is:")
    if LOAD_MODEL:
        print("Test loader:")
        print(test_loader.dataset.annotations['image'].values)
    else:
        print("Train loader:")
        print(train_loader.dataset.annotations['image'].values)


    print(LOAD_MODEL)
    for epoch in range(EPOCHS):
        print('in epochs')
        if LOAD_MODEL:
            print('in load model')
            #print('\n%%%%%%  ' + str(epoch) + '  %%%%%%\n')
            for x, y in test_loader:
                print('in test loader')
                x = x.to(DEVICE)
                for idx in range(test_loader.dataset.annotations['image'].shape[0]):  # iterate through the entire test data set

                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = nms(bboxes[idx], iou_threshold=hyper_IoU_th, probability_threshold=hyper_pobability_th, boxes_format="midpoint")
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes, PASCAL_VOC_labels_decoded)


                sys.exit()

        print("%%% Epoch = " + str(epoch) + " %%%")



        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=hyper_IoU_th, threshold=hyper_pobability_th
        )

        mean_avg_prec = map_pytorch(
            pred_boxes, target_boxes, iou_threshold=hyper_IoU_th, boxes_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")



        for th in map_th:
            if mean_avg_prec >= th[0] and th[1] is False:
                th[1] = True
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                stop_time = time.time()
                save_checkpoint(checkpoint, filename="epoch_" + str(epoch) + "_map_" +
                                                     str(mean_avg_prec.item()).replace('.','') +"_" +LOAD_MODEL_FILE)
                time.sleep(10)
                """
                with open("epoch_" + str(epoch) + "_map_" +str(mean_avg_prec.item()).replace('.','') +"_" +
                          LOAD_MODEL_FILE.replace('mask_overfit.pth.tar', '.txt'), "w") as file:
                    file.write("\n\n             >>>>  Configuration   <<<           "+
                              "\n ---> LEARNING_RATE = " + str(LEARNING_RATE) +
                      "\n ---> DEVICE = " + str(DEVICE) +r
                      "\n ---> BATCH_SIZE = " + str(BATCH_SIZE) +
                      "\n ---> WEIGHT_DECAY = " + str(WEIGHT_DECAY) +
                      "\n ---> EPOCHS = " + str(EPOCHS) +
                      "\n ---> EPOCHS RUN= " + str(epoch) +
                      "\n ---> NUM_WORKERS = " + str(NUM_WORKERS) +
                      "\n ---> PIN_MEMORY = " + str(PIN_MEMORY) +
                      "\n ---> LOAD_MODEL = " + str(LOAD_MODEL) +
                      "\n ---> LOAD_MODEL_FILE = " + "epoch_" + str(epoch) + "_map_" +
                      str(mean_avg_prec.item()).replace('.', '') + "_" + LOAD_MODEL_FILE +
                      "\n ---> IMG_DIR = " + str(IMG_DIR) +
                      "\n ---> LABEL_DIR = " + str(LABEL_DIR) +
                      "\n ---> LABEL_DECODER_PATH = " + str(LABEL_DECODER_PATH) +
                      "\n ---> TRAIN_DATASET = " + str(TRAIN_DATASET)+
                        "\n\n\n ### Time span = " + str(stop_time - start_time) + "   ###")
                """
                print("\n\n             >>>>  Configuration   <<<           ")

                print("\n ---> LEARNING_RATE = " + str(LEARNING_RATE) +
                      "\n ---> DEVICE = " + str(DEVICE) +
                      "\n ---> BATCH_SIZE = " + str(BATCH_SIZE) +
                      "\n ---> WEIGHT_DECAY = " + str(WEIGHT_DECAY) +
                      "\n ---> EPOCHS = " + str(EPOCHS) +
                      "\n ---> EPOCHS RUN= " + str(epoch) +
                      "\n ---> NUM_WORKERS = " + str(NUM_WORKERS) +
                      "\n ---> PIN_MEMORY = " + str(PIN_MEMORY) +
                      "\n ---> LOAD_MODEL = " + str(LOAD_MODEL) +
                      "\n ---> LOAD_MODEL_FILE = " + "epoch_" + str(epoch) + "_map_" +
                                                     str(mean_avg_prec.item()).replace('.','') +"_" +LOAD_MODEL_FILE +
                      "\n ---> IMG_DIR = " + str(IMG_DIR) +
                      "\n ---> LABEL_DIR = " + str(LABEL_DIR) +
                      "\n ---> LABEL_DECODER_PATH = " + str(LABEL_DECODER_PATH) +
                      "\n ---> TRAIN_DATASET = " + str(TRAIN_DATASET))

                print("\n\n\n ### Time span = " + str(stop_time - start_time) + "   ###")

                # For each
                source_file = open('logs\\log_out.txt', 'r')
                dest_file = open("epoch_" + str(epoch) + "_map_" +str(mean_avg_prec.item()).replace('.','') +"_" +
                          LOAD_MODEL_FILE.replace('mask_overfit.pth.tar', '.txt'), "w")

                content = source_file.read()
                dest_file.write(content)

                source_file.close()
                dest_file.close()


        if epoch == EPOCHS-1:
            stop_time = time.time()
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="epoch_FINAL_" + str(epoch) + "_map_" +
                                                     str(mean_avg_prec.item()).replace('.','') +"_" +LOAD_MODEL_FILE)

            time.sleep(10)

            with open("epoch_FINAL_" + str(epoch) + "_map_" + str(mean_avg_prec.item()).replace('.', '') + "_" +
                      LOAD_MODEL_FILE.replace('mask_overfit.pth.tar', '.txt'), "w") as file:

                file.write("\n\n             >>>>  Configuration   <<<           "+
                      "\n ---> LEARNING_RATE = " + str(LEARNING_RATE) +
                      "\n ---> DEVICE = " + str(DEVICE) +
                      "\n ---> BATCH_SIZE = " + str(BATCH_SIZE) +
                      "\n ---> WEIGHT_DECAY = " + str(WEIGHT_DECAY) +
                      "\n ---> EPOCHS = " + str(EPOCHS) +
                      "\n ---> EPOCHS RUN= " + str(epoch) +
                      "\n ---> NUM_WORKERS = " + str(NUM_WORKERS) +
                      "\n ---> PIN_MEMORY = " + str(PIN_MEMORY) +
                      "\n ---> LOAD_MODEL = " + str(LOAD_MODEL) +
                      "\n ---> LOAD_MODEL_FILE = " + str(LOAD_MODEL_FILE) +
                      "\n ---> IMG_DIR = " + str(IMG_DIR) +
                      "\n ---> LABEL_DIR = " + str(LABEL_DIR) +
                      "\n ---> LABEL_DECODER_PATH = " + str(LABEL_DECODER_PATH) +
                      "\n ---> TRAIN_DATASET = " + str(TRAIN_DATASET)+
                      "\n\n\n ### Time span = " + str(stop_time - start_time) + "   ###")

            print("\n\n             >>>>  Configuration   <<<           ")

            print("\n ---> LEARNING_RATE = " + str(LEARNING_RATE) +
                  "\n ---> DEVICE = " + str(DEVICE) +
                  "\n ---> BATCH_SIZE = " + str(BATCH_SIZE) +
                  "\n ---> WEIGHT_DECAY = " + str(WEIGHT_DECAY) +
                  "\n ---> EPOCHS = " + str(EPOCHS) +
                  "\n ---> EPOCHS RUN= " + str(epoch) +
                  "\n ---> NUM_WORKERS = " + str(NUM_WORKERS) +
                  "\n ---> PIN_MEMORY = " + str(PIN_MEMORY) +
                  "\n ---> LOAD_MODEL = " + str(LOAD_MODEL) +
                  "\n ---> LOAD_MODEL_FILE = " + str(LOAD_MODEL_FILE) +
                  "\n ---> IMG_DIR = " + str(IMG_DIR) +
                  "\n ---> LABEL_DIR = " + str(LABEL_DIR) +
                  "\n ---> LABEL_DECODER_PATH = " + str(LABEL_DECODER_PATH) +
                  "\n ---> TRAIN_DATASET = " + str(TRAIN_DATASET))

            print("\n\n\n ### Time span = " + str(stop_time - start_time) + "   ###")

            sys.exit()

        train_fn(train_loader, model, optimizer, loss_fn)

    #stop_time = time.time()
    """
    print("\n\n             >>>>  Configuration   <<<           ")

    print("\n ---> LEARNING_RATE = " + str(LEARNING_RATE) +
          "\n ---> DEVICE = " + str(DEVICE) +
          "\n ---> BATCH_SIZE = " + str(BATCH_SIZE) +
          "\n ---> WEIGHT_DECAY = " + str(WEIGHT_DECAY) +
          "\n ---> EPOCHS = " + str(EPOCHS) +
          "\n ---> NUM_WORKERS = " + str(NUM_WORKERS) +
          "\n ---> PIN_MEMORY = " + str(PIN_MEMORY) +
          "\n ---> LOAD_MODEL = " + str(LOAD_MODEL) +
          "\n ---> LOAD_MODEL_FILE = " + str(LOAD_MODEL_FILE) +
          "\n ---> IMG_DIR = " + str(IMG_DIR) +
          "\n ---> LABEL_DIR = " + str(LABEL_DIR) +
          "\n ---> LABEL_DECODER_PATH = " + str(LABEL_DECODER_PATH) +
          "\n ---> TRAIN_DATASET = " + str(TRAIN_DATASET) )


    print("\n\n\n ### Time span = " + str(stop_time-start_time)+ "   ###")
    #with open("log_time.txt", "w") as f:
    #    f.write("Time Span = " + str(stop_time-start_time))
    """



if __name__ == "__main__":
    main()