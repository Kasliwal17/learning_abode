import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )        # fully connected layer, output 10 classes
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
            nn.Softmax(),
                                 )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization
model = CNN()
optimizer = optim.Adam(model.parameters(),lr=0.001)
loss_function = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(('/home/aditya/Downloads/finalmodel'),map_location='cpu'))
model.eval()
load_from_sys = True

if load_from_sys:
    hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('cannot open camera')
    exit()

cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)

canvas = None

x1 = 0
y1 = 0

noise_thresh = 800

while True:
    ret, frame = cap.read()
    if not ret:
        print('cant recieve video')
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if load_from_sys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask1 = cv2.resize(mask,(28,28))
    mask1 = torch.from_numpy(mask1)
    mask_out = str(torch.argmax(model(mask1)))
    cv2.imshow(mask_out, cv2.resize(mask, None, fx=0.6, fy=0.6))

    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if cv2.contourArea(max(contours, key= cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), [130, 255, 255], 4)

        x1, y1 = x2, y2

    else:
        x1, y1 = 0, 0

    frame = cv2.add(frame, canvas)

    stacked = np.hstack((canvas, frame))
    cv2.imshow('screen_pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    if cv2.waitKey(1) == ord('e'):
        break

    # Clear the canvas when 'c' is pressed
    if cv2.waitKey(1) == ord('c'):
        canvas = None

cv2.destroyAllWindows()
cap.release()
