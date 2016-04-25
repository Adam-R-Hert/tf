'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng = numpy.random
data = numpy.genfromtxt('input.csv',delimiter=',')
data = data.T
print(data.shape)

# Parameters
learning_rate = 0.01
training_epochs = 20000
display_step = 50

# Training Data
train_X = data[1:][50:]
train_Y = data[0][50:]
print(train_Y.shape)
data = numpy.genfromtxt('input.csv')

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.mul(X, W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'


    # Testing example, as requested (Issue #2)
    test_X = data[1:][:50]
    test_Y = data[0][:50]

    print "Testing... (L2 loss Comparison)"
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation-Y, 2))/(2*test_X.shape[0]),
                            feed_dict={X: test_X, Y: test_Y}) #same function as cost above
    print "Testing cost=", testing_cost
    print "Absolute l2 loss difference:", abs(training_cost - testing_cost)

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig('plot.png')
