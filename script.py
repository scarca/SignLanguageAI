
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split 

# load data 
x = np.load("./Sign-language-digits-dataset 2/X.npy")
y = np.load("./Sign-language-digits-dataset 2/Y.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y)

# lets create a CNN 
def get_fn(features, labels, mode): 
    # cnn 
    outputs = labels
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])
    layer1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=32,
        kernel_size=[8, 8], 
        padding="same", 
        activation=tf.nn.relu
    )
    
    layer2 = tf.layers.max_pooling2d(
        inputs=layer1, 
        pool_size = [2, 2], 
        strides=2,
    )
    
    layer3 = tf.layers.conv2d(
        inputs=layer2, 
        filters=64, 
        kernel_size=[16, 16], 
        padding="same", 
        activation=tf.nn.relu
    )
    layer4 = tf.layers.max_pooling2d(inputs=layer3, pool_size=[4, 4], strides=2)
    
    # Flatten 
    flat = tf.reshape(layer4, [-1,14400]) 
    # dropout inputs 
    dropout = tf.layers.dropout(
        inputs=flat, 
        rate=0.25, 
        training = mode == tf.estimator.ModeKeys.TRAIN
    )
    
    layer5 = tf.layers.dense(
        inputs = dropout, 
        units=128*128, 
        activation = tf.nn.relu
    )
    
    dropout2 = tf.layers.dropout(
        inputs=layer5, 
        rate=0.25, 
        training = mode == tf.estimator.ModeKeys.TRAIN 
    )
    
    layer6 = tf.layers.dense( 
        inputs = dropout2, 
        activation = tf.nn.sigmoid,
        units = 100
    )
    
    layer7 = tf.layers.dense ( 
        inputs = layer6, 
        activation = tf.nn.sigmoid, 
        units=10
    )
    
    predictions = {
        "classes": tf.argmax(input=layer7, axis=1), 
        "probabilities": tf.nn.softmax(layer7, name="softmax_tensor") 
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT: 
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions) 
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels = outputs, logits=layer7)
    if mode == tf.estimator.ModeKeys.TRAIN: 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001) 
        train_op = optimizer.minimize( 
            loss = loss, 
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)  
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy( 
            labels=outputs, 
            predictions = predictions["classes"], 
        ), 
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops = eval_metric_ops) 

train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": x_train}, 
                                                   y = y_train, 
                                                   batch_size = 100, 
                                                   num_epochs = None, 
                                                   shuffle=True)

estimator = tf.estimator.Estimator(model_fn = get_fn, model_dir = "./tmp")
estimator.train(
    input_fn = train_input_fn, 
    steps = 2000
)

