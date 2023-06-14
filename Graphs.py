import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

plt.close('all')

#file = 'C:\\Users\\Alex\\Google Drive (not syncing)\\Thesis_2023\\weights\\2000\\out - Copy.txt'

#file = "C:\\Users\\Alex\\Google Drive (not syncing)\\Thesis_2023\\out - Copy.txt" # 5000

#file = r"D:\Uni\YOLO_CODE\output_logs\log_out.txt"#


file = r"D:\Uni\YOLO_CODE\output_logs\8 examples _400 runs\ALL_epoch_112_map_08565109372138977_2023_05_07_14_16_17_.txt"

metrics = pd.DataFrame(columns = ['epoch', 'map'])
map_string = 'Train mAP: '
mean_loss_string = 'Mean loss was '
epoch_string = '%%% Epoch = '
metrisc_list = []
with open(file, 'r') as file:
    #for line in file:
    #    if line.startswith(('Train mAP:')):
    #        metrisc_list.append(line.strip())
    lines = file.readlines()
    map_lines = [line for line in lines if map_string in line]
    mean_loss_lines = [line for line in lines if mean_loss_string in line]
    epoch_lines=[line for line in lines if epoch_string in line]
        #new_row_df = pd.DataFrame({'epoch':ls_epoch,'map':ls_map})

map_lines_trim = [round(float(line[line.index(map_string) + len(map_string):line.index('\n')]), 3) for line in map_lines]
mean_loss_lines_trim = [round(float(line[line.index(mean_loss_string) + len(mean_loss_string):line.index('\n')]), 3) for line in mean_loss_lines]
epoch_lines_trim = [round(float(line[line.index(epoch_string) + len(epoch_string):line.index(' %%%')]), 3) for line in epoch_lines]




#y = [float(re.findall('\d+\.\d+', my_string)[0]) for my_string in metrisc_list]

individual = True


if individual == True:
    # First figure
    plt.figure()  # Create a new figure

    plt.plot(epoch_lines_trim, map_lines_trim, label='mAP')
    plt.title('Mean average precision')
    plt.ylabel('mAP')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(epoch_lines_trim[:-1], mean_loss_lines_trim, label='Mean Loss')
    plt.title('Mean Loss')
    plt.ylabel('Mean Loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(epoch_lines_trim, np.divide(map_lines_trim, max(map_lines_trim)), color='red', label='mAP')
    plt.plot(epoch_lines_trim[:-1], np.divide(mean_loss_lines_trim, max(mean_loss_lines_trim)), color='blue',
               label='Mean Loss')
    plt.title('Normed mAP and Mean Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Normed mAP and Mean Loss')
    plt.grid()
    plt.legend()

    # Display the figures
    plt.show()
else:
    fig, ax = plt.subplots(1, 3)

    ax[0].plot(epoch_lines_trim, map_lines_trim, label='mAP')
    ax[0].set_title('Mean average precision')
    ax[0].set_ylabel('mAP')
    ax[0].set_xlabel('Epochs')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(epoch_lines_trim[:-1], mean_loss_lines_trim, label='Mean Loss')
    ax[1].set_title('Mean Loss')
    ax[1].set_ylabel('Mean Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(epoch_lines_trim, np.divide(map_lines_trim,max(map_lines_trim)), color='red', label='mAP')
    ax[2].plot(epoch_lines_trim[:-1], np.divide(mean_loss_lines_trim,max(mean_loss_lines_trim)), color='blue', label='Mean Loss')
    ax[2].set_title('Normed mAP and Mean Loss')
    #ax[2].text(0, -0.15, "mAP", color="red", fontsize=14)
    #ax[2].text(0, -0.17, "Mean Loss", color="blue", fontsize=14)
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Normed mAP and Mean Loss')
    ax[2].grid()
    ax[2].legend()
    #for element in y:
    #    if element > 1:
    #        y.remove(element)

    #x = list(range(0, len(y), 1))
    #plt.plot(x,y)
    plt.show()







stop = 1