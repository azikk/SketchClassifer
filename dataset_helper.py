import numpy as np
import pandas as pd

class Dataset:
    epoch = 0
    def __init__(self,dataset_path,num_classes,image_size,batch_size):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size

    def load_dataset(self,shuffle=True):
        print("Dataset loading...")
        df_train = pd.read_csv(self.dataset_path)
        self.trainY = df_train.loc[:, "label"].as_matrix()
        self.trainY = np.asarray(self.trainY, dtype=np.int32)
        self.trainY = np.eye(self.num_classes)[self.trainY]
        self.trainX = np.zeros((self.trainY.shape[0], self.image_size * self.image_size), dtype=np.float32)
        for i in range(self.trainX.shape[0]):
            self.trainX[i] = np.fromstring(df_train.loc[:, "raw_data"].loc[i], dtype=np.float32, sep=' ')

        if shuffle:
            perm = np.arange(self.trainX.shape[0])
            np.random.shuffle(perm)
            self.trainX = self.trainX[perm]
            self.trainY = self.trainY[perm]
        self.testY = self.trainY[:500]
        self.testX = self.trainX[:500]
        self.trainY = self.trainY[500:]
        self.trainX = self.trainX[500:]
        print('Number of training examples:'+str(self.trainX.shape[0]))
        print('Number of test examples:' + str(self.testX.shape[0]))

    def next_batch(self,shuffle=False):
        if shuffle:
            perm = np.arange(self.trainX.shape[0])
            np.random.shuffle(perm)
            self.trainX = self.trainX[perm]
            self.trainY = self.trainY[perm]
        # Go to the next epoch
        if self.epoch*self.batch_size + self.batch_size > self.trainX.shape[0]:
            # Finished epoch
            # Get the rest examples in this epoch
            rest_num_examples = self.trainX.shape[0] - self.epoch*self.batch_size
            images_rest_part = self.trainX[self.trainX.shape[0]-rest_num_examples:self.trainX.shape[0]]
            labels_rest_part = self.trainY[self.trainX.shape[0]-rest_num_examples:self.trainX.shape[0]]
            # Start next epoch
            self.epoch = 0
            self.start = self.epoch * self.batch_size
            self.end = self.start + self.batch_size
            images_new_part = self.trainX[self.start:self.end]
            labels_new_part = self.trainY[self.start:self.end]
            self.epoch += 1
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self.start = self.epoch*self.batch_size
            self.end = self.start+self.batch_size
            self.epoch += 1
            return self.trainX[self.start:self.end], self.trainY[self.start:self.end]

