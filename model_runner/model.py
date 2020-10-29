import os
import time
from io import BytesIO
from typing import Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import tensorflow as tf

# SOME CODE DUPLICATED FROM https://github.com/thunil/TecoGAN
from model_runner.dataloader import inference_data_loader
from model_runner.frvsr import generator_F, fnet
from model_runner.ops import deprocess, upscale_four, save_img


class TecoGAN:
    def __init__(self, input_shape: Tuple[int, int, int], model_input: str, output_extension='png',
                 num_resblock=16) -> None:
        super().__init__()

        model_dir = os.path.dirname(model_input)

        if not os.path.exists(model_dir):
            self.download_model(model_dir)

        self.num_resblock = num_resblock
        self.output_extension = output_extension
        self.model_checkpoint = model_input
        self.input_shape = input_shape
        self.outputs = None
        self.inputs_raw = None
        self.before_ops = None
        self.build_network()

    model_location = 'https://ge.in.tum.de/download/data/TecoGAN/model.zip'

    def download_model(self, location):
        if not os.path.exists(location):
            os.mkdir(location)

        print("Downloading Model")

        with urlopen(TecoGAN.model_location) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(location)

    def build_network(self):
        input_shape = [1, ] + list(self.input_shape)
        output_shape = [1, input_shape[1] * 4, input_shape[2] * 4, 3]
        oh = input_shape[1] - input_shape[1] // 8 * 8
        ow = input_shape[2] - input_shape[2] // 8 * 8
        paddings = tf.constant([[0, 0], [0, oh], [0, ow], [0, 0]])
        print("input shape:", input_shape)
        print("output shape:", output_shape)

        # build the graph
        self.inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')

        pre_inputs = tf.Variable(tf.zeros(input_shape), trainable=False, name='pre_inputs')
        pre_gen = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_gen')
        pre_warp = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_warp')

        transpose_pre = tf.space_to_depth(pre_warp, 4)
        inputs_all = tf.concat((self.inputs_raw, transpose_pre), axis=-1)
        with tf.variable_scope('generator'):
            gen_output = generator_F(inputs_all, 3, reuse=False, num_resblock=self.num_resblock)
            # Deprocess the images outputed from the model, and assign things for next frame
            with tf.control_dependencies([tf.assign(pre_inputs, self.inputs_raw)]):
                self.outputs = tf.assign(pre_gen, deprocess(gen_output))

        inputs_frames = tf.concat((pre_inputs, self.inputs_raw), axis=-1)
        with tf.variable_scope('fnet'):
            gen_flow_lr = fnet(inputs_frames, reuse=False)
            gen_flow_lr = tf.pad(gen_flow_lr, paddings, "SYMMETRIC")
            gen_flow = upscale_four(gen_flow_lr * 4.0)
            gen_flow.set_shape(output_shape[:-1] + [2])
        pre_warp_hi = tf.contrib.image.dense_image_warp(pre_gen, gen_flow)
        self.before_ops = tf.assign(pre_warp, pre_warp_hi)

        print('Finish building the network')

    def run(self, input_folder, output_folder, number_outputs=-1, output_name='output'):
        output_folder = os.path.join(output_folder, '')

        # Declare the test data reader
        inference_data = inference_data_loader(input_folder, number_outputs)  ## FIXME put in correct parameters

        # In inference time, we only need to restore the weight of the generator
        var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='generator')
        var_list = var_list + tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')

        weight_initiallizer = tf.train.Saver(var_list)

        # Define the initialization operation
        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with tf.Session(config=config) as sess:
            # Load the pretrained model
            sess.run(init_op)
            sess.run(local_init_op)

            print('Loading weights from ckpt model')
            print(os.path.exists(self.model_checkpoint))

            weight_initiallizer.restore(sess, self.model_checkpoint)
            max_iter = len(inference_data.inputs)

            srtime = 0
            print('Frame evaluation starts!!')
            for i in range(max_iter):
                input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
                feed_dict = {self.inputs_raw: input_im}
                t0 = time.time()
                if (i != 0):
                    sess.run(self.before_ops, feed_dict=feed_dict)
                output_frame = sess.run(self.outputs, feed_dict=feed_dict)
                srtime += time.time() - t0

                if (i >= 5):
                    name, _ = os.path.splitext(os.path.basename(str(inference_data.paths_LR[i])))
                    filename = output_name + '_' + name
                    print('saving image %s' % filename)
                    out_path = os.path.join(output_folder, "%s.%s" % (filename, self.output_extension))
                    save_img(out_path, output_frame[0])
                else:  # First 5 is a hard-coded symmetric frame padding, ignored but time added!
                    print("Warming up %d" % (5 - i))
        print("total time " + str(srtime) + ", frame number " + str(max_iter))


if __name__ == '__main__':
    teco_gan = TecoGAN((360, 480, 3), './model/TecoGAN/TecoGAN')

    # teco_gan.run('../input', '../output', 10)
