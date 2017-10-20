from __future__ import print_function
from PIL import Image
from matplotlib import pyplot as plt
import cv2 
import os, sys, csv

ofile = open('histograms-clean.csv', 'wb')
writer = csv.writer(ofile, delimiter=',')

path = "data - Copy/"

folders = os.listdir(path)

#iterate through each image in each folder
for folder in folders:
    print(folder)
    if folder != '.DS_Store':
        files = os.listdir(os.path.join(path, folder))
        for f in files:
            features = [0] * 769
            if f != '.DS_Store':
                im = Image.open(os.path.join(path, folder, f))

                #calculate color histograms of rgb values
                for i in range(im.size[0]):
                    for j in range(im.size[1]):
                        r,g,b = im.getpixel((i,j))
                        features[r] += 1
                        features[g + 256] += 1
                        features[b + 512] += 1
                features[768] = folder

                #write to csv
                writer.writerow(features)

ofile.close()
