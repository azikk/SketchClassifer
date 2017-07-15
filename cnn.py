# Python 3.6.0
# tensorflow 1.1.0

import os
import os.path as path

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import sys,getopt,time
from dataset_helper import Dataset

output_path = "./output/"
dataset_path = "./dataset/dataset.csv"
model_path =  "./output/"
MODEL_NAME = 'ai'
NUM_STEPS = 1000
BATCH_SIZE = 128
DISPLAY_SIZE = 1
NUM_CLASSES = 125
IMAGE_SIZE = 48

def model_input(dataset,input_node_name, keep_prob_node_name):
    x = tf.placeholder(tf.float32, shape=[None, dataset.trainX.shape[1]], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y_ = tf.placeholder(tf.float32, shape=[None,NUM_CLASSES],name="labels")
    return x, keep_prob, y_

def build_model(x, keep_prob, y_, output_node_name):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
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

    conv4 = tf.layers.conv2d(pool3, 512, 5, 1, 'same', activation=tf.nn.relu)
    # 6*6*512
    pool4 = tf.layers.max_pooling2d(conv4, 2, 2, 'same')
    # 3*3*512

    conv5 = tf.layers.conv2d(pool4, 1024, 5, 1, 'same', activation=tf.nn.relu)
    # 3*3*1024
    pool5 = tf.layers.max_pooling2d(conv5, 2, 2, 'same')
    # 2*2*1024

    flatten = tf.reshape(pool5, [-1, 2*2*1024])
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


def train(dataset,x, keep_prob, y_, train_step, loss, accuracy,
        merged_summary_op, saver, save, restore):

    print("training start...")

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        if restore:
            saver = tf.train.import_meta_graph(model_path + 'ai.chkp.meta')
            saver.restore(sess, '/model/ai.chkp')
        tf.train.write_graph(sess.graph_def, output_path,
            MODEL_NAME + '.pbtxt', True)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/',
            graph=tf.get_default_graph())
        start_time = time.time()
        for step in range(NUM_STEPS):
            batch = dataset.next_batch(shuffle=True)
            if step % DISPLAY_SIZE == 0:
                elapsed_time = time.time()-start_time
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %f elapsed time %s' % (step, train_accuracy,time.strftime('%H:%M:%S',time.gmtime(elapsed_time))))
            _,summary = sess.run([train_step, merged_summary_op],
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary, step)
            if step>0 and step % 500 == 0:
                test_accuracy = accuracy.eval(feed_dict={x: dataset.testX,
                                                         y_: dataset.testY,
                                                         keep_prob: 1.0})
                print('test accuracy %g' % test_accuracy)
        print('Total elaped time: %s Average step time: %f sec.'%(time.strftime('%H:%M:%S',time.gmtime(elapsed_time)),elapsed_time/NUM_STEPS))
        if save:
            print("Saving model...")
            saver.save(sess, output_path + MODEL_NAME + '.chkp')
            print("Model saved!")
        test_accuracy = accuracy.eval(feed_dict={x: dataset.testX,
                                    y_: dataset.testY,
                                    keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

    print("training finished!")

def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph(output_path + MODEL_NAME + '.pbtxt', None, False,
        output_path + MODEL_NAME+ '.chkp', output_node_name, "save/restore_all",
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
    global output_path,save,restore,model_path
    dataset = Dataset('./dataset/dataset.csv',125,48,BATCH_SIZE)
    ops,args = getopt.getopt(sys.argv,"f:s:r",["floyd,save,restore"])
    save = False
    restore = False

    for  arg in args:
        if arg == '-f':
            print("Running on floyd mode")
            output_path = "/output/"
            dataset.dataset_path = "/dataset/dataset.csv"
            model_path = "/model/"
        if arg =='-s':
            save = True
        if arg == '-r':
            restore = True

    if not path.exists('./output'):
        os.mkdir('./output')

    dataset.load_dataset(shuffle=True)

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    x, keep_prob, y_ = model_input(dataset,input_node_name, keep_prob_node_name)

    train_step, loss, accuracy, merged_summary_op = build_model(x, keep_prob,
        y_, output_node_name)
    saver = tf.train.Saver()

    train(dataset,x, keep_prob, y_, train_step, loss, accuracy,
        merged_summary_op, saver,save,restore)

    export_model([input_node_name, keep_prob_node_name], output_node_name)

if __name__ == "__main__":
    main()
