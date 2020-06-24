import tensorflow as tf
import numpy as np

from tensorflow.keras import *
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Some test for tf-1.0 grpah


# ---- test for simple graph ----

# # a = tf.constant(1.0, name="alpha")
# a = tf.placeholder(tf.float32, name="alpha")
# b = tf.placeholder(tf.float32, name="beta")
# c = tf.multiply(a, b, name="mult")
# d = tf.add(c,a,name="add")
# 
# 
# sess = tf.Session()
# print(sess.run(d, feed_dict = {a:3.3, b:2.2}))






# ---- Load the minst dataset --------
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

trainset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
testset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)



# ---- test for simple net training ----
class MyNet(Model):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = Conv2D(32,1,activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128,activation="relu")
        self.d2 = Dense(10)
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

# The alternative way of defining the network
def build_net():
    model = models.Sequential()
    model.add(Conv2D(32,1,activation="relu"))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(10))
    return model

# net = MyNet()
net = build_net()


def compile_model(model):
    model.compile(optimizer=optimizers.Nadam(),
                loss=losses.SparseCategoricalCrossentropy(),
                metrics=[metrics.SparseCategoricalAccuracy(),metrics.SparseTopKCategoricalAccuracy(5)]) 
    return(model)

net = compile_model(net)

# Compile and train the Model - Plan A
# hist = net.fit(trainset, validation_data = testset, epochs=10)

# Train on batch method - Plan b
# Buggy could not work in TF1.15
@tf.function
def train_model(model,ds_train,ds_valid,epoches):

    for epoch in tf.range(1,epoches+1):
        model.reset_metrics()
        # if epoch == 5:
        #     model.optimizer.lr.assign(model.optimizer.lr/2.0)
        #     tf.print("Lowering optimizer Learning Rate...\n\n")
        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)
        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y,reset_metrics=False)
        if epoch%1 ==0:
            # printbar()
            tf.print("epoch = ",epoch)
            print("train:",dict(zip(model.metrics_names,train_result)))
            print("valid:",dict(zip(model.metrics_names,valid_result)))
            print("")

train_model(net,trainset,testset,epoches=10)


# ------ The traininig components -----

# loss = tf.keras.losses.SparseCategoricalCrossentropy()
# optim = tf.keras.optimizers.Adam()

pred = net(x_train[:1])

# ---- Quantizing the graph ----------

g = tf.get_default_graph()
# tf.contrib.quantize.create_training_graph(input_graph=g,quant_delay=0)
# tf.contrib.quantize.create_eval_graph(input_graph=g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(pred))


gd = tf.get_default_graph().as_graph_def()
with open("./test.pb","wb") as f:
    f.write(gd.SerializeToString())
