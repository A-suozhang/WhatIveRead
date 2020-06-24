import tensorflow as tf
import numpy as np

# a = tf.constant(1.0, name="alpha")
a = tf.placeholder(tf.float32, name="alpha")
b = tf.placeholder(tf.float32, name="beta")
c = tf.multiply(a, b, name="mult")
# d = sum([a,b,c])
# d = tf.div(a,b)
d = a / b
# d = tf.add_n([a,b,c],name="add")


sess = tf.Session()
print(sess.run(d, feed_dict = {a:3.3, b:2.2}))


gd = tf.get_default_graph().as_graph_def()
with open("./test.pb","wb") as f:
    f.write(gd.SerializeToString())
