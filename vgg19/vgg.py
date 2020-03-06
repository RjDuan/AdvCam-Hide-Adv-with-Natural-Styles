import os
import tensorflow as tf

import numpy as np
import time
import inspect
VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

    def fprop(self, rgb, include_top = True):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        self.input = rgb
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.input)
        self.input = tf.concat(axis=3, values=[
                                       blue - VGG_MEAN[0],
                                       green - VGG_MEAN[1],
                                       red - VGG_MEAN[2],
                                       ])
#        assert self.input.get_shape().as_list()[1:] == [224, 224, 3]
        self.conv1_1 = self.conv_layer(self.input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        if include_top:
            self.fc6 = self.fc_layer(self.pool5, "fc6")
            assert self.fc6.get_shape().as_list()[1:] == [4096]
            self.relu6 = tf.nn.relu(self.fc6)

            self.fc7 = self.fc_layer(self.relu6, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)

            self.logits = self.fc_layer(self.relu7, "fc8")

            self.prob = tf.nn.softmax(self.logits, name="prob")

            return {'Logits':self.logits,
                    'Prob':self.prob}

    def get_logits(self, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output logits
        (i.e., the values fed as inputs to the softmax layer).
        """
        return self.logits
#        outputs = self.fprop(x, **kwargs)
#        if self.logits in outputs:
#            return outputs['Logits']
#        raise NotImplementedError(str(type(self)) + "must implement `get_logits`"
#                                                    " or must define a " + self.logits +
#                                  " output in `fprop`")

    def get_predicted_class(self, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the predicted label
        """
        pred = tf.argmax(self.get_logits(**kwargs), axis=1)
        return pred[0]

    def get_all_layers(self):
        return [self.conv1_1, self.conv1_2, self.pool1,\
                self.conv2_1, self.conv2_2, self.pool2, \
                self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, self.pool3, \
                self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, self.pool4, \
                self.conv5_1]

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def get_params(self):
        """
        Provides access to the model's parameters.
        :return: A list of all Variables defining the model parameters.
        """

        if hasattr(self, 'params'):
            return list(self.params)

        # Catch eager execution and assert function overload.
        try:
            if tf.executing_eagerly():
                raise NotImplementedError("For Eager execution - get_params "
                                          "must be overridden.")
        except AttributeError:
            pass

        # For graph-based execution
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       self.scope + "/")

        if len(scope_vars) == 0:
            self.make_params()
            scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           self.scope + "/")
            assert len(scope_vars) > 0

        # Make sure no parameters have been added or removed
        if hasattr(self, "num_params"):
            if self.num_params != len(scope_vars):
                print("Scope: ", self.scope)
                print("Expected " + str(self.num_params) + " variables")
                print("Got " + str(len(scope_vars)))
                for var in scope_vars:
                    print("\t" + str(var))
                assert False
        else:
            self.num_params = len(scope_vars)

        return scope_vars
