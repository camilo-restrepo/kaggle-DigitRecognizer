import tensorflow as tf
import pandas as pd
import numpy as np

training_dataset = pd.read_csv("data/train.csv")

labels = training_dataset[[0]].values.ravel()
labels = labels.astype(np.float32)
labels_one_hot = np.zeros((len(labels), 10))

i = 0
for num in labels:
    labels_one_hot[i, num] = 1.0
    i += 1

train = training_dataset.iloc[:, 1:].values
train = np.array(train).astype(np.float32)
train = np.multiply(train, 1.0 / 255.0)

training = train[0:37000]
training_labels = labels_one_hot[0:37000]

validation = train[37000:42000]
validation_labels = labels_one_hot[37000:42000]

index_in_epoch = 0
epochs_completed = 0


def next_batch(batch_size):
    global index_in_epoch
    global training
    global epochs_completed
    global training_labels

    start = index_in_epoch
    index_in_epoch += batch_size
    num_examples = training.shape[0]
    if index_in_epoch > num_examples:
        # Finished epoch
        epochs_completed += 1
        # Shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        training = training[perm]
        training_labels = training_labels[perm]
        # Start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return training[start:end], training_labels[start:end]


# ---------------------------------------------------------------------------------------------------------------------
# MULTILAYER CONVOLUTIONAL NETWORK
# ---------------------------------------------------------------------------------------------------------------------


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_varialbe(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# FIRST LAYER
# ---------------------------------------------------------------------------------------------------

# Patch de 5x5, 1 input channel, 32 channels de salida
W_conv1 = weight_variable([5, 5, 1, 32])

# One component for each output channel
b_conv1 = bias_varialbe([32])

# Reshape x to 4d tensor: 2nd an 3rd image width and height and 4th: number of color channels
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x with weight and bias apply ReLU and maxpool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND LAYER: 64 features for each 5x5 patch
# ----------------------------------------------------------------------------------------------------
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_varialbe([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# DENSELY CONNECTED LAYER
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_varialbe([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER: SOFTMAX LAYER
# -----------------------------------------------------------------------------------------------------
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_varialbe([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

session = tf.Session()
session.run(tf.initialize_all_variables())

for i in range(20000):
    batch_xs, batch_ys = next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}, session=session)
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}, session=session)

print "test accuracy %g" % accuracy.eval(feed_dict={x: validation, y_: validation_labels, keep_prob: 1.0}, session=session)

# Prediction and submission
test_dataset = pd.read_csv("data/test.csv")
test = test_dataset.iloc[:, :].values
test = np.array(test).astype(np.float32)
test = np.multiply(test, 1.0 / 255.0)

prediction = tf.argmax(y_conv, 1)
y_test = prediction.eval(feed_dict={x: test, keep_prob: 1.0}, session=session)
np.savetxt('submission/multilayer_cnn.csv', np.c_[range(1, len(test)+1), y_test], delimiter=',', header='ImageId,Label',
           comments='', fmt='%d')
session.close()
