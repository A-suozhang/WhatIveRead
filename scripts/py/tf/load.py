import tensorflow as tf
import cv2
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.summary import FileWriter


def convert_ckpt2pb():
    # with tf.Session() as sess:
    sess = tf.Session()
    saver = tf.train.import_meta_graph(DIR+sub+".meta")
    FileWriter("__tb", sess.graph)
    saver.restore(sess, tf.train.latest_checkpoint(DIR))
    val_names = [v.name for v in tf.global_variables()]
    # Save the graph for tensorboard
    g = tf.get_default_graph()
    ops = g.get_operations()
    ops_ = [op.name for op in g.get_operations()]
    graph_def = tf.get_default_graph().as_graph_def()
    possible_io_nodes = [n.name + '=>' +  n.op for n in graph_def.node if n.op in ( 'Softmax','Placeholder')]
    output_nodes = []
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_nodes)
    with tf.gfile.GFile("./test.pb", "wb") as fid:
        serialized_graph = graph_def.SerializeToString()
        fid.write(serialized_graph)

def read_nodes():
    reader = pywrap_tensorflow.NewCheckpointReader(DIR+sub)
    d = reader.get_variable_to_shape_map()
    return d

def load_pb(pb_path):
    g = tf.Graph()
    with g.as_default():
        gd = tf.GraphDef()
        with tf.gfile.GFile(pb_path,"rb") as fid:
            serialized_graph = fid.read()
            gd.ParseFromString(serialized_graph)
            tf.import_graph_def(gd,name='')
    import ipdb; ipdb.set_trace()
    g.get_tensor_by_name("box_encodings")

    return g, gd


if __name__ == "__main__":
    global DIR
    DIR = "./"
    sub = "model.ckpt"
    # convert_ckpt2pb()
    # d = read_nodes()
    # g, gd = load_pb(DIR+"test.pb")
    img_0 = cv2.imread("/home/tianchen/apks/test_image_224.jpg")
    img_1 = cv2.resize(img_0,(300,300))
    img_1 = img_1.reshape([1,300,300,3])
    saver = tf.train.import_meta_graph("./model.ckpt.meta")
    with tf.Session() as sess:
        saver.restore(sess, "./model.ckpt")
        # input_data = np.random.randn(1,300,300,3)
        input_data = img_1
        target = sess.graph.get_tensor_by_name("concat_1:0")
        classes = sess.run(target, feed_dict={"image_tensor:0":input_data})
    print(classes)
