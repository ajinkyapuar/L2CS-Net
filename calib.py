import argparse
import numpy as np
import cv2
import time
import os
import tkinter as tk

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS


points = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
          (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
          (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]

class CalibrationApp:
    def __init__(self, master):
        self.master = master
        master.title("Gaze Calibration App")

        # Get the screen size with scaling factor
        screen_width = master.winfo_screenwidth() * 1.5
        screen_height = master.winfo_screenheight() * 1.5

        # Set the window size to full screen
        master.geometry("%dx%d+0+0" % (screen_width, screen_height))
        master.attributes("-fullscreen", True)

        # Create a canvas to display the grid of points
        self.canvas = tk.Canvas(master, width=1920, height=1080)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Create a list to store the ID of each target point on the canvas
        self.targets = []

        # Create the target points on the canvas
        for x, y in points:
            x_px = x * 1920
            y_px = y * 1080
            target = self.canvas.create_oval(x_px - 20, y_px - 20, x_px + 20, y_px + 20, fill="red")
            self.targets.append(target)

        # Create a button to start the calibration process
        self.button = tk.Button(master, text="Calibrate", command=self.start_calibration)
        self.button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        # Create an exit button to close the app
        self.exit_button = tk.Button(master, text="Exit", command=self.exit_app)
        self.exit_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        # Initialize the gaze direction model
        self.model = None
        self.gpu = None

        # Define the transformations for image preprocessing
        self.transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def start_calibration(self):
        # Start the calibration process
        self.current_target = 0
        self.capture_count = 0
        self.dataset = []
        self.capture_folder = "capture_%d" % self.current_target
        os.makedirs(self.capture_folder, exist_ok=True)

        # Load the gaze direction model
        self.load_gaze_direction_model()

        self.display_next_target()

    def load_gaze_direction_model(self):
        snapshot_path = "models/L2CSNet/Gaze360/L2CSNet_gaze360.pkl"
        arch = "ResNet50"
        bins = 90

        # Create the gaze direction model with the specified architecture
        if arch == 'ResNet18':
            model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        elif arch == 'ResNet34':
            model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        elif arch == 'ResNet101':
            model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        elif arch == 'ResNet152':
            model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
        else:
            if arch != 'ResNet50':
                print('Invalid value for architecture is passed! '
                    'The default value of ResNet50 will be used instead!')
            model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)

        # model = nn.DataParallel(model)
        checkpoint = torch.load(snapshot_path, map_location=self.gpu)
        # model.load_state_dict(checkpoint['state_dict'], strict=True)
        # saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(checkpoint)

        model.eval()

        self.model = model.to(self.gpu)

    def display_next_target(self):
        # Hide the previous target point (if any)
        if self.current_target > 0:
            prev_target = self.targets[self.current_target - 1]
            self.canvas.itemconfig(prev_target, fill="red")

        # Display the next target point and wait for the specified duration
        target = self.targets[self.current_target]
        self.canvas.itemconfig(target, fill="green")
        self.master.after(3000, self.capture_image, target)

    def capture_image(self, target):
        # Capture an image from the webcam and save it to the current capture folder
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        cv2.imwrite(os.path.join(self.capture_folder, "%d.jpg" % self.capture_count), frame)

        # Preprocess the image for the gaze direction model
        image_path = os.path.join(self.capture_folder, "%d.jpg" % self.capture_count)
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = self.transformations(image).unsqueeze(0).to(self.gpu)

        # Run the gaze direction model on the image
        with torch.no_grad():
            prediction = self.model(image)

        # Get the gaze direction and screen coordinates of the target point
        gaze_direction = prediction.cpu().numpy()[0]
        screen_coordinates = self.get_screen_coordinates(target)

        # Append the gaze direction and screen coordinates to the dataset
        self.dataset.append((gaze_direction, screen_coordinates))

        # Increase the capture count
        self.capture_count += 1

        # Display a message when all images have been captured for the current target
        if self.capture_count == 10:
            self.current_target += 1
            self.capture_count = 0

            # Check if all targets have been displayed and captured
            if self.current_target == len(self.targets):
                self.calibration_complete()
                self.save_dataset()
            else:
                self.capture_folder = "capture_%d" % self.current_target
                os.makedirs(self.capture_folder, exist_ok=True)
                self.display_next_target()
        else:
            # Display the next image capture for the current target
            self.display_next_image(target)

    def display_next_image(self, target):
        # Display the next image capture for the current target and wait for the specified duration
        self.canvas.itemconfig(target, fill="yellow")
        self.master.after(500, self.display_next_target)

    def calibration_complete(self):
        # Display a message when the calibration is complete
        self.canvas.delete("all")
        message = tk.Label(self.master, text="Calibration complete!")
        message.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def save_dataset(self):
        # Save the collected dataset to a file
        dataset_path = "dataset.npy"
        np.save(dataset_path, self.dataset)
        print("Dataset saved to", dataset_path)

    def get_screen_coordinates(self, target):
        # Calculate the screen coordinates of the target point
        x = points[target][0]
        y = points[target][1]
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        screen_x = int(x * screen_width)
        screen_y = int(y * screen_height)
        return (screen_x, screen_y)

    def exit_app(self):
        # Exit the app
        self.master.destroy()


def main():
    root = tk.Tk()
    my_gui = CalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()