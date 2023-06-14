
import torch
import torch.nn as nn

""" 
Below the YOLO V1 architecture is constructed as a list of either tuples or  list of tuples.
-> tuples for the convolutional layers structured as follows: (kernel_size, filters, stride, padding) 
-> "Maxpooling" for a Maxpool with stride 2x2 and kernel size 2x2
-> List containing first a tuple with the convolutional layer structure followed by the number of time the layer should 
            be repeated
"""

architecture_config = [
    (7, 64, 2, 3),

    "Maxpooling",

    (3, 192, 1, 1),

    "Maxpooling",

    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),

    "Maxpooling",

    [(1, 256, 1, 0), (3, 512, 1, 1), 4],

    (1, 512, 1, 0),
    (3, 1024, 1, 1),

    "Maxpooling",

    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],

    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)   # try w/ and w/o
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolo_v1(nn.Module):
    def __init__(self, in_channels=3, **kwargs): # 3 =  RGB
        super(Yolo_v1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs) # fully connected

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:  # here we add a CNN block
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3], #kernel_size = out_channels
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:    # here we add a MaxPool 2d
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:   # here he add the conv layers repeated a number of times
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],   # the in channels are the out channels from conv1
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes


        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # 4096 in the original paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),    # S x S times ( nb_classes + bounding boxes * 5 );
                                                    # 5 - probability score and x1 y1 x2 y2
        )

def test (S=7, B=2, C=20):
    model = Yolo_v1(split_size=S, num_boxes=B, num_classes=C)
    x=torch.randn((2, 3, 448, 448))
