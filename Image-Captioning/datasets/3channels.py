import cv2
import os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('image_dir', type=str)
args = parser.parse_args()

images = os.listdir(args.image_dir)
image_paths = [os.path.join(args.image_dir, image) for image in images]

for path in image_paths:
    img = cv2.imread(path)
    if img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:,:,0] = gray
    img2[:,:,1] = gray
    img2[:,:,2] = gray
    cv2.imwrite(path, img2)