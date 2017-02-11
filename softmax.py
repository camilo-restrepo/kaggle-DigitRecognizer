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
# SOFTMAX NN
# ---------------------------------------------------------------------------------------------------------------------

# Softmax to get probability of image class
# evidence = sum over j of W(i,j)*x(j) + b
# y = softmax(evidence)

# Model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross entropy cost function: Hy'(y) = - sum over i of y'(i) * log(y(i)), y' is true and y is predicted
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Training
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Stochatic gradient descent
for i in range(1000):
    batch_xs, batch_ys = next_batch(100)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: training, y_: training_labels}, session=session)
        print "step %d, training accuracy %g" % (i, train_accuracy)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Performance
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print session.run(accuracy, feed_dict={x: validation, y_: validation_labels})

# Prediction and submission
test_dataset = pd.read_csv("data/test.csv")
test = test_dataset.iloc[:, :].values
test = np.array(test).astype(np.float32)
test = np.multiply(test, 1.0 / 255.0)

prediction = tf.argmax(y, 1)
y_test = prediction.eval(feed_dict={x: test}, session=session)
np.savetxt('submission/softmax_nn.csv', np.c_[range(1, len(test)+1), y_test], delimiter=',', header='ImageId,Label',
           comments='', fmt='%d')
session.close()
