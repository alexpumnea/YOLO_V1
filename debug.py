#"C:\Users\Alex\Google Drive (not syncing)\Thesis_2023\data_mask\labels\Mask_1.txt"

#"C:\Users\Alex\Google Drive (not syncing)\Thesis_2023\data_mask\images\Mask_1.jpg"

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import time
#'C:\Users\Alex\Google Drive (not syncing)\Thesis_2023\data'

data_location = 'C:\\Users\\Alex\\Google Drive (not syncing)\\Thesis_2023\\data_mask'
for i in range(650)[1:]:
    print('\n')
    print(i)

    if '_mask' in data_location:
        file_content = open("C:\\Users\Alex\\Google Drive (not syncing)\\Thesis_2023\\data_mask\\labels\\Mask_"+str(i)+".txt", 'r').read()
        img = Image.open("C:\\Users\\Alex\\Google Drive (not syncing)\\Thesis_2023\\data_mask\\images_jpg_channels_fixed\\Mask_"+str(i)+".jpg")
        label = open("C:\\Users\\Alex\\Google Drive (not syncing)\\Thesis_2023\\data_mask\\label_decoder.txt", 'r').read().split()
        label_list = [label[1], label[3], label[5]]
    else:
        file_content = open("C:\\Users\\Alex\\Google Drive (not syncing)\\Thesis_2023\\data\\labels\\000004.txt").read()
        img = Image.open("C:\\Users\\Alex\\Google Drive (not syncing)\Thesis_2023\data\images\\000004.jpg")


    file_content_split = file_content.split()
    file_content_split_sublist = [file_content_split[i:i+5] for i in range(0, len(file_content_split), 5)]
    #print(file_content_split_sublist)

    transform = T.ToTensor()
    img_tensor = transform(img)
    draw = ImageDraw.Draw(img)
    outline  = (255, 0, 0)  # red
    fill = (255, 0, 0)
    width = 2
    font = ImageFont.truetype('arial.ttf', size=16)


    for bbx in file_content_split_sublist:
        cx, cy, w, h = float(bbx[1:][0]), float(bbx[1:][1]), float(bbx[1:][2]), float(bbx[1:][3])

        # Convert to x-y coordinates
        x1 = (cx - w / 2)*img.width
        y1 = (cy - h / 2)*img.height
        x2 = (cx + w / 2)*img.width
        y2 = (cy + h / 2)*img.height

        text = label_list[int(bbx[0])]
        text_pos = (x2 + 10, y1 + 10)

        #x1, y1 = float(bbx[1:][0]) * img.width, float(bbx[1:][1]) * img.height
        #x2, y2 = float(bbx[1:][2]) * img.width, float(bbx[1:][3]) * img.height
        xy = ((x1, y1), (x2, y2))
        draw.rectangle(xy, outline=outline)
        draw.text(text_pos, text, fill=fill, font=font)


    #draw = ImageDraw.Draw(img)
    #draw.rectangle(file_content_split_sublist[0][1:], outline="red", width=2)


    # Draw the rectangle on the image



    # Save the modified image
    img.save( 'C:\\Users\Alex\\Google Drive (not syncing)\\Thesis_2023\\data_mask\\labeled_imgs\\output_' +str(i) +  '.jpg')

stop =1