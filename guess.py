import tensorflow as tf
from matplotlib import image as img
from skimage.transform import resize
from scipy.misc import imresize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_dataset():
    print("Dataset loading...")
    df_train = pd.read_csv('./dataset/dataset.csv')
    trainY = df_train.loc[:,"label"].as_matrix()
    trainY = np.asarray(trainY, dtype=np.int32)
    trainY = np.eye(NUM_CLASSES)[trainY]
    trainX = np.zeros((trainY.shape[0], 48*48),dtype=np.float32)
    for i in range(trainX.shape[0]):
        trainX[i] = np.fromstring(df_train.loc[:,"raw_data"].loc[i],dtype=np.float32,sep=' ')
    # df_test = pd.read_csv('test.csv')
    # testY = df_test.loc[:, "label"].as_matrix()
    # testY = np.eye(NUM_CLASSES)[testY]
    # testY = testY[:1000]
    # testX = np.asarray(testY, dtype=np.int32)
    # testX = np.zeros((testX.shape[0], 48 * 48), dtype=np.float32)
    # testX = testX[:1000]
    # for i in range(testX.shape[0]):
    #     testX[i] = np.fromstring(df_test.loc[:, "raw_data"].loc[i], dtype=np.float32,sep=' ')
    testY = trainY[:50]
    testX = trainX[:50]
    return trainX,trainY,testX,testY

IMAGE_SIZE = 48
NUM_CLASSES = 125
file = "apple.png"
image = img.imread(file)
image = imresize(image, (48, 48), interp='lanczos')
image = rgb2gray(image)
# plt.imshow(image,cmap='gray')
# plt.show()
image = image.flatten()
image = image.astype(float)
label = [1] + [0] * (NUM_CLASSES-1)
# trainX,trainY,testX,testY = load_dataset()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./output/ai.chkp.meta')
    saver.restore(sess,'./output/ai.chkp')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input:0")
    y_ = graph.get_tensor_by_name("labels:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    output = graph.get_tensor_by_name("output:0")
    output = tf.argmax(output, 1)
    feed_dict = {x:[image],y_:[label],keep_prob:1.0}
    print(output.eval(feed_dict))
