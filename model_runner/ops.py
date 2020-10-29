import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow_core.python.keras.layers import LeakyReLU

# SOME CODE DUPLICATED FROM https://github.com/thunil/TecoGAN

def maxpool(inputs, scope='maxpool'):
    return slim.max_pool2d(inputs, [2, 2], scope=scope)


# Define our Lrelu
def lrelu(inputs, alpha):
    return LeakyReLU(alpha=alpha).call(inputs)


# Define the convolution transpose building block
def conv2_tran(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME',
                                         data_format='NHWC',
                                         activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME',
                                         data_format='NHWC',
                                         activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                         biases_initializer=None)


# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               biases_initializer=None)


# Define our Lrelu
def lrelu(inputs, alpha):
    return LeakyReLU(alpha=alpha).call(inputs)


def maxpool(inputs, scope='maxpool'):
    return slim.max_pool2d(inputs, [2, 2], scope=scope)


def upscale_four(inputs, scope='upscale_four'):  # mimic the tensorflow bilinear-upscaling for a fix ratio of 4
    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]

        p_inputs = tf.concat((inputs, inputs[:, -1:, :, :]), axis=1)  # pad bottom
        p_inputs = tf.concat((p_inputs, p_inputs[:, :, -1:, :]), axis=2)  # pad right

        hi_res_bin = [
            [
                inputs,  # top-left
                p_inputs[:, :-1, 1:, :]  # top-right
            ],
            [
                p_inputs[:, 1:, :-1, :],  # bottom-left
                p_inputs[:, 1:, 1:, :]  # bottom-right
            ]
        ]

        hi_res_array = []
        for hi in range(4):
            for wj in range(4):
                hi_res_array.append(
                    hi_res_bin[0][0] * (1.0 - 0.25 * hi) * (1.0 - 0.25 * wj)
                    + hi_res_bin[0][1] * (1.0 - 0.25 * hi) * (0.25 * wj)
                    + hi_res_bin[1][0] * (0.25 * hi) * (1.0 - 0.25 * wj)
                    + hi_res_bin[1][1] * (0.25 * hi) * (0.25 * wj)
                )

        hi_res = tf.stack(hi_res_array, axis=3)  # shape (b,h,w,16,c)
        hi_res_reshape = tf.reshape(hi_res, (b, h, w, 4, 4, c))
        hi_res_reshape = tf.transpose(hi_res_reshape, (0, 1, 3, 2, 4, 5))
        hi_res_reshape = tf.reshape(hi_res_reshape, (b, h * 4, w * 4, c))

    return hi_res_reshape


def bicubic_four(inputs, scope='bicubic_four'):
    '''
        equivalent to tf.image.resize_bicubic( inputs, (h*4, w*4) ) for a fix ratio of 4 FOR API <=1.13
        For API 2.0, tf.image.resize_bicubic will be different, old version is tf.compat.v1.image.resize_bicubic
        **Parallel Catmull-Rom Spline Interpolation Algorithm for Image Zooming Based on CUDA*[Wu et. al.]**
    '''

    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]

        p_inputs = tf.concat((inputs[:, :1, :, :], inputs), axis=1)  # pad top
        p_inputs = tf.concat((p_inputs[:, :, :1, :], p_inputs), axis=2)  # pad left
        p_inputs = tf.concat((p_inputs, p_inputs[:, -1:, :, :], p_inputs[:, -1:, :, :]), axis=1)  # pad bottom
        p_inputs = tf.concat((p_inputs, p_inputs[:, :, -1:, :], p_inputs[:, :, -1:, :]), axis=2)  # pad right

        hi_res_bin = [p_inputs[:, bi:bi + h, :, :] for bi in range(4)]
        r = 0.75
        mat = np.float32([[0, 1, 0, 0], [-r, 0, r, 0], [2 * r, r - 3, 3 - 2 * r, -r], [-r, 2 - r, r - 2, r]])
        weights = [np.float32([1.0, t, t * t, t * t * t]).dot(mat) for t in [0.0, 0.25, 0.5, 0.75]]

        hi_res_array = []  # [hi_res_bin[1]]
        for hi in range(4):
            cur_wei = weights[hi]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[
                3] * hi_res_bin[3]

            hi_res_array.append(cur_data)

        hi_res_y = tf.stack(hi_res_array, axis=2)  # shape (b,h,4,w,c)
        hi_res_y = tf.reshape(hi_res_y, (b, h * 4, w + 3, c))

        hi_res_bin = [hi_res_y[:, :, bj:bj + w, :] for bj in range(4)]

        hi_res_array = []  # [hi_res_bin[1]]
        for hj in range(4):
            cur_wei = weights[hj]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[
                3] * hi_res_bin[3]

            hi_res_array.append(cur_data)

        hi_res = tf.stack(hi_res_array, axis=3)  # shape (b,h*4,w,4,c)
        hi_res = tf.reshape(hi_res, (b, h * 4, w * 4, c))

    return hi_res


def save_img(out_path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img[:, :, ::-1])


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


# Define our Lrelu
def lrelu(inputs, alpha):
    return tf.keras.layers.LeakyReLU(alpha=alpha).call(inputs)
