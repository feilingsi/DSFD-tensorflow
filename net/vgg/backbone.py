import tensorflow as tf
import tensorflow.contrib.slim as slim
from net.resnet.basemodel import resnet50, resnet101,resnet_arg_scope

from net.vgg.vgg import vgg_16


def l2_normalization(x, scale, name):
    with tf.variable_scope(name):
        x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x

def extra_feature(x):

    extra_fm=[]
    with tf.variable_scope('extra'):
        x=slim.conv2d(x, 256, [1, 1],stride=1,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    scope='extra_conv1')
        x = slim.conv2d(x, 512, [3, 3],stride=2,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope='extra_conv2')
        extra_fm.append(x)

        x = slim.conv2d(x, 128, [1, 1], stride=1,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope='extra_conv3')
        x = slim.conv2d(x, 256, [3, 3], stride=2,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope='extra_conv4')
        extra_fm.append(x)

    return extra_fm


def vgg_ssd(image,L2_reg,is_training=True):
    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training, bn_trainable=True)):

        net,end_points=vgg_16(image,num_classes=None,global_pool=False,spatial_squeeze=False,fc_conv_padding='SAME')

        for k,v in end_points.items():
            print('mobile backbone output:',k,v)

        ###add conv6, conv7
        conv6 = slim.conv2d(net, 1024, [3, 3],
                          activation_fn=tf.nn.relu,
                          normalizer_fn=None,
                          scope='fc6')
        conv7 = slim.conv2d(conv6, 1024, [1, 1],
                          activation_fn=tf.nn.relu,
                          normalizer_fn=None,
                          scope='fc7')




        extra_fms=extra_feature(conv7)

        vgg_fms = [end_points['vgg_16/conv3/conv3_3'], end_points['vgg_16/conv4/conv4_3'],
                   end_points['vgg_16/conv5/conv5_3'],    conv7]+extra_fms


        print(vgg_fms)



    return vgg_fms
