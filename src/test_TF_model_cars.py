# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Script for running inference on TF model
for car image files
'''

# pylint: disable=W0612 E1129

import argparse
import tensorflow as tf
from utils.plot_images import plot_image_single
from utils.image_processing import create_boundary

def main(FLAGS):
    modelfile = FLAGS.modelfile
    
    # open the original image as grayscale
    image_file_name = FLAGS.carfile
    padded_image = create_boundary(image_file_name)
    padded_image = padded_image.reshape(1, *padded_image.shape, 1)
    
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(modelfile, "rb") as f:  # noqa: F841
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())                 
                sess.graph.as_default()
                tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                                    name="", producer_op_list=None)
                l_input = graph.get_tensor_by_name('x:0')  # Input Tensor
                l_output = graph.get_tensor_by_name('Identity:0')  # Output Tensor
                tf.compat.v1.global_variables_initializer()  # initialize_all_variables
                
                Session_out = sess.run(l_output, feed_dict={l_input: padded_image})
    plot_image_single(padded_image, Session_out, image_file_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--modelfile',
                        type=str,
                        default="modelfile.pb",
                        help="modelfile to store the results to")
                        
    parser.add_argument('-c',
                        '--carfile',
                        type=str,
                        default="data/cars/car_001.png",
                        help="input file for car profile")

    parser.add_argument('-p',
                        '--plot_images',
                        default=False,
                        action="store_true",
                        help="output comparative images")

    FLAGS = parser.parse_args()
    main(FLAGS)
