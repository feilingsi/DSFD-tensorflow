import tensorflow as tf

import tensorflow.contrib.slim as slim
from train_config import config as cfg

from net.resnet.basemodel import resnet_arg_scope

def cpm(product,dim,scope):

    with tf.variable_scope(scope):

        eyes_1=slim.conv2d(product, 256, [3, 3], stride=1,rate=1, activation_fn=tf.nn.relu, scope='eyes_1')

        eyes_2_1=slim.conv2d(product, 256, [3, 3], stride=1,rate=2,  activation_fn=tf.nn.relu, scope='eyes_2_1')
        eyes_2=slim.conv2d(eyes_2_1, 128, [3, 3], stride=1,rate=1,  activation_fn=tf.nn.relu, scope='eyes_2')

        eyes_3_1 = slim.conv2d(eyes_2_1, 128, [3, 3], stride=1, rate=2, activation_fn=tf.nn.relu, scope='eyes_3_1')
        eyes_3 = slim.conv2d(eyes_3_1, 128, [3, 3], stride=1,rate=1,  activation_fn=tf.nn.relu, scope='eyes_3')

        fme_res = tf.concat([eyes_1, eyes_2,eyes_3], axis=3)

    return fme_res


if 'MobilenetV1' in cfg.MODEL.net_structure:
    resnet_dims=[64,128,256,512,512,256]
elif 'resnet' in cfg.MODEL.net_structure:
    resnet_dims=[256,512,1024,2048,512,256]
else:
    ssd_backbne = None



def create_fem_net(blocks, L2_reg,is_training, trainable=True,data_format='NHWC'):

    of0,of1,of2,of3,of4,of5=blocks

    initializer = tf.contrib.layers.xavier_initializer()

    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg,bn_is_training=is_training)):

        lateral = slim.conv2d(of2, resnet_dims[2], [1, 1],
                                trainable=trainable, weights_initializer=initializer,
                                padding='SAME',
                                scope='lateral/res{}'.format(2))

        upsample = slim.conv2d(of3, resnet_dims[2], [1, 1],
                               trainable=trainable, weights_initializer=initializer,
                               padding='SAME',
                               scope='merge/res{}'.format(2), data_format=data_format)
        upsample = tf.keras.layers.UpSampling2D(data_format='channels_last' if data_format=='NHWC' else 'channels_first')(upsample)

        fem_2 = lateral* upsample

        lateral = slim.conv2d(of1, resnet_dims[1], [1, 1],
                              trainable=trainable, weights_initializer=initializer,
                              padding='SAME',
                              scope='lateral/res{}'.format(1))

        upsample = slim.conv2d(fem_2, resnet_dims[1], [1, 1],
                               trainable=trainable, weights_initializer=initializer,
                               padding='SAME',
                               scope='merge/res{}'.format(1), data_format=data_format)
        upsample = tf.keras.layers.UpSampling2D(data_format='channels_last' if data_format == 'NHWC' else 'channels_first')(upsample)

        fem_1 = lateral * upsample

        lateral = slim.conv2d(of0, resnet_dims[0], [1, 1],
                              trainable=trainable, weights_initializer=initializer,
                              padding='SAME',
                              scope='lateral/res{}'.format(0))

        upsample = slim.conv2d(fem_1, resnet_dims[0], [1, 1],
                               trainable=trainable, weights_initializer=initializer,
                               padding='SAME',
                               scope='merge/res{}'.format(0), data_format=data_format)
        upsample = tf.keras.layers.UpSampling2D(data_format='channels_last' if data_format == 'NHWC' else 'channels_first')(upsample)

        fem_0 = lateral * upsample



        #####enhance model
        fpn_fms=[fem_0,fem_1,fem_2,of3,of4,of5]
        global_fems_fms=[]

        for i, fem in enumerate(fpn_fms):
            tmp_res=cpm(fem,dim=resnet_dims[i],scope='fems%d'%i)
            global_fems_fms.append(tmp_res)

    return global_fems_fms

