import numpy as np
import pandas as pd

class Dataset:
    def __init__(self,dataset_path,num_classes,image_size,batch_size):
        self.datset_path = dataset_path
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size

    def load_dataset(self):
        print("Dataset loading...")
        df_train = pd.read_csv(self.dataset_path)
        trainY = df_train.loc[:, "label"].as_matrix()
        trainY = np.asarray(trainY, dtype=np.int32)
        trainY = np.eye(self.num_classes)[trainY]
        trainX = np.zeros((trainY.shape[0], self.image_size * self.image_size), dtype=np.float32)
        for i in range(trainX.shape[0]):
            trainX[i] = np.fromstring(df_train.loc[:, "raw_data"].loc[i], dtype=np.float32, sep=' ')

        perm = np.arange(trainX.shape[0])
        np.random.shuffle(perm)
        self.trainX = trainX[perm]
        self.trainY = trainY[perm]
        self.testY = trainY[:500]
        self.testX = trainX[:500]
        self.trainY = trainY[500:]
        self.trainX = trainX[500:]

    def next_batch(epoch,self,fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * (self.image_size * self.image_size)
            fake_label = [1] + [0] * self.num_classes

            return [fake_image for _ in range(self.batch_size)], [
                fake_label for _ in range(self.batch_size)
            ]

        # Go to the next epoch
        if epoch*self.batch_size + self.batch_size > self.trainX.shape[0]:
            # Finished epoch
            # Get the rest examples in this epoch
            rest_num_examples = self.trainX.shape[0] - self.batch_index
            images_rest_part = self.trainX[epoch*self.batch_size:self.trainX.shape[0]]
            labels_rest_part = self.trainY[epoch*self.batch_size:self.trainX.shape[0]]
            # Start next epoch
            start = 0
            end = self.batch_size
            images_new_part = self.trainX[start:end]
            labels_new_part = self.trainY[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            start = epoch*self.batch_size
            end = start+self.batch_size
            return self.trainX[start:end], self.trainY[start:end]

