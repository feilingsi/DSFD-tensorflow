# coding: utf-8
import tensorflow as tf
import os.path
import argparse
from tensorflow.python.framework import graph_util
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_DIR = './model'
MODEL_NAME = 'detector.pb'
pb_path = os.path.join(MODEL_DIR , MODEL_NAME)
model_save_path = './model'

if not tf.gfile.Exists(MODEL_DIR):  # 创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

output_node_names = "tower_0/images,tower_0/boxes,tower_0/scores,tower_0/labels,tower_0/num_detections,training_flag"  # 原模型输出操作节点的名字



def freeze_graph(model_folder):
    print("start")
    checkpoint = tf.train.get_checkpoint_state(model_folder)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)  # 得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        #         print("predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]})) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字
        # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("~~~~")



if __name__ == '__main__':

    freeze_graph(model_save_path)
    print("Well done!")
