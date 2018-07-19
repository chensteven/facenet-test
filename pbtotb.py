"""Imports a protobuf model as a graph in Tensoboard."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def import_to_tensorboard(model_dir, log_dir):
    with tf.Session(graph=tf.Graph()) as sess:
        with tf.gfile.FastGFile(model_dir, "rb") as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
        pb_visual_writer = tf.summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
        print("Model Imported. Visualize by running: tensorboard --logdir={}".format(log_dir))


def main(unused_args):
    import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", required=True, help="Protobuf location")
    parser.add_argument("--log_dir", type=str, default="", required=True, help="Location of log files for Tensorboard")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)