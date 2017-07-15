import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.misc import imresize

from skimage import io
from skimage import exposure

from skimage import color
import os.path as path

import os

def rgb2gray(rgb):
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    img_gray = R * .299 + G * .587 + B * .114
    return  img_gray

if not path.exists('./dataset'):
  os.mkdir('./dataset')

raw_data_path = "./data/"
dirs = os.listdir(raw_data_path)
dirs = sorted(dirs,key=str.lower)
dirstring = '\n'.join(dirs)
f = open("labels.txt","w")
f.write(dirstring)
f.close()
filenames = []
labels = []
label = 0
for dirc in dirs:
  if not os.path.isfile(dirc):
    files = os.listdir(raw_data_path + dirc)
    for file in files:
      if os.path.isfile(raw_data_path+dirc+"/"+file):
        filenames.append(raw_data_path+dirc+"/"+file)
        labels.append(label)
    label +=1

num_examples = len(filenames)
dataset = np.zeros(num_examples, dtype='int8, object')

index = 0
for file in filenames:
  image = img.imread(file)
  image = rgb2gray(image)
  image = imresize(image,(48,48),interp='lanczos')
  # plt.imshow(image,cmap='gray')
  # plt.show()
  flatten = image.flatten()
  dataset[index][0] = labels[index]
  dataset[index][1] = ' '.join(map(str, flatten.astype(float)))
  index += 1


with open('./dataset/dataset.csv', 'wb') as f:
  f.write(b'label,raw_data\n')
  np.savetxt(f, dataset, delimiter=' ',fmt='%i,%s')



