import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.misc import imresize

from skimage import io
from skimage import exposure

from skimage import color


import os

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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
  image = imresize(image,(28,28),interp='lanczos')
  image = rgb2gray(image)
  # plt.imshow(image,cmap='gray')
  # plt.show()
  flatten = image.flatten()
  dataset[index][0] = labels[index]
  dataset[index][1] = ' '.join(map(str, flatten.astype(float)))
  index += 1

with open('dataset.csv', 'wb') as f:
  f.write(b'label,raw_data\n')
  np.savetxt(f, dataset, delimiter=' ',fmt='%i,%s')



