# Python 3.6.0
# tensorflow 1.1.0

import os
import os.path as path

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import  numpy as np
import sys,getopt

output_path = "./output/"
MODEL_NAME = 'ai'
NUM_STEPS = 10000
BATCH_SIZE = 100
DISPLAY_SIZE = 1
NUM_CLASSES = 125
batch_index = 0

def load_dataset():
    print("Dataset loading...")
    df_train = pd.read_csv('dataset.csv')
    trainY = df_train.loc[:,"label"].as_matrix()
    trainY = np.asarray(trainY, dtype=np.int32)
    trainY = np.eye(NUM_CLASSES)[trainY]
    trainX = np.zeros((trainY.shape[0], 28*28),dtype=np.float32)
    for i in range(trainX.shape[0]):
        trainX[i] = np.fromstring(df_train.loc[:,"raw_data"].loc[i],dtype=np.float32,sep=' ')
    perm = np.arange(trainX.shape[0])
    np.random.shuffle(perm)
    trainX = trainX[perm]
    trainY = trainY[perm]
    testY = trainY[:1000]
    testX = trainX[:1000]
    trainY = trainY[1000:]
    trainX = trainX[1000:]
    return trainX,trainY,testX,testY

def next_batch(epoch,fake_data=False, shuffle=True):
    global trainX,trainY,batch_index
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * (28*28)
      fake_label = [1] + [0] * NUM_CLASSES

      return [fake_image for _ in range(BATCH_SIZE)], [
          fake_label for _ in range(BATCH_SIZE)
      ]

    # Shuffle for the first epoch
    if batch_index==0 and epoch == 0 and shuffle:
      perm0 = np.arange(trainX.shape[0])
      np.random.shuffle(perm0)
      trainX = trainX[perm0]
      trainY = trainY[perm0]

    # Go to the next epoch
    if batch_index + BATCH_SIZE > trainX.shape[0]:
      # Finished epoch
      # Get the rest examples in this epoch
      rest_num_examples = trainX.shape[0] - batch_index
      images_rest_part = trainX[batch_index:trainX.shape[0]]
      labels_rest_part = trainY[batch_index:trainX.shape[0]]
      # Shuffle the data
      if shuffle:
        perm = np.arange(trainX.shape[0])
        np.random.shuffle(perm)
        trainX = trainX[perm]
        trainY = trainY[perm]
      # Start next epoch
      start = 0
      batch_index = BATCH_SIZE - rest_num_examples
      end = batch_index
      images_new_part = trainX[start:end]
      labels_new_part = trainY[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      batch_index += BATCH_SIZE
      end = batch_index
      return trainX[end-BATCH_SIZE:end], trainY[end-BATCH_SIZE:end]

trainX, trainY, testX, testY = load_dataset()

def model_input(input_node_name, keep_prob_node_name):
    x = tf.placeholder(tf.float32, shape=[None, trainX.shape[1]], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y_ = tf.placeholder(tf.float32, shape=[None,NUM_CLASSES],name="labels")
    return x, keep_prob, y_

def build_model(x, keep_prob, y_, output_node_name):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 48*48*1

    conv1 = tf.layers.conv2d(x_image, 64, 5, 1, 'same', activation=tf.nn.relu)
    # 48*48*64
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same')
    # 24*24*64

    conv2 = tf.layers.conv2d(pool1, 128, 5, 1, 'same', activation=tf.nn.relu)
    # 24*24*128
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'same')
    # 12*12*128

    conv3 = tf.layers.conv2d(pool2, 256, 5, 1, 'same', activation=tf.nn.relu)
    # 12*12*256
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'same')
    # 6*6*256

    flatten = tf.reshape(pool3, [-1, 4*4*256])
    fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
    dropout = tf.nn.dropout(fc, keep_prob)
    logits = tf.layers.dense(dropout, NUM_CLASSES)
    outputs = tf.nn.softmax(logits, name=output_node_name)
    # loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op


def train(x, keep_prob, y_, train_step, loss, accuracy,
        merged_summary_op, saver):
    print("training start...")

    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.write_graph(sess.graph_def, output_path,
            MODEL_NAME + '.pbtxt', True)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/',
            graph=tf.get_default_graph())
        batch_index = 0;
        for step in range(NUM_STEPS):
            batch = next_batch(step)
            if step % DISPLAY_SIZE == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %f' % (step, train_accuracy))
            _,summary = sess.run([train_step, merged_summary_op],
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary, step)
        print("Saving model...")
        saver.save(sess, output_path + MODEL_NAME + '.chkp',global_step=NUM_STEPS)
        print("Model saved!")
        test_accuracy = accuracy.eval(feed_dict={x: testX,
                                    y_: testY,
                                    keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

    print("training finished!")

def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph(output_path + MODEL_NAME + '.pbtxt', None, False,
        output_path + MODEL_NAME+ '.chkp' + "-"+ str(NUM_STEPS), output_node_name, "save/restore_all",
        "save/Const:0", output_path+'frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_path+'frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(output_path+'/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def main():
    global output_path
    ops,args = getopt.getopt(sys.argv,"f",["floyd"])

    for  arg in args:
        if arg == '-f':
            print("Running on floyd mode")
            output_path = "/output/"

    if not path.exists('./output'):
        os.mkdir('./output')

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    x, keep_prob, y_ = model_input(input_node_name, keep_prob_node_name)

    train_step, loss, accuracy, merged_summary_op = build_model(x, keep_prob,
        y_, output_node_name)
    saver = tf.train.Saver()

    train(x, keep_prob, y_, train_step, loss, accuracy,
        merged_summary_op, saver)

    export_model([input_node_name, keep_prob_node_name], output_node_name)

if __name__ == "__main__":
    main()
