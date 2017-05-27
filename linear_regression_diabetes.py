# Tensor flow example for linear regression (Y = WX + b)

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import numpy as np

# get_ipython().magic('matplotlib inline')

# Load the dataset
data = load_diabetes()

# Get the X and Y data
X_ = data['data']
y_ = data['target']

# Just take one feature from the data
diabetes_X = X_[:, 2] # Take the 3rd feature

# Take the train data
train_diabetes_X = diabetes_X[:-20]
train_diabetes_y = y_[:-20]

# print the shapes of the datasets
print (train_diabetes_X.shape, train_diabetes_y.shape)

# Take the test data
test_diabetes_X = diabetes_X[-20:]
test_diabetes_y = y_[-20:]


# Now take the hyperparameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
# Number of samples
n_samples = train_diabetes_X.shape[0]
batch_size = 20
steps_per_epoch = n_samples//batch_size


# Inputs to the graph
X = tf.placeholder("float")
y = tf.placeholder("float")


# Set the weights and biases
weights = tf.Variable(np.random.randn()) # Since we are optimizing the weights and biases these are set as variable. 
bias = tf.Variable(np.random.randn())
# Prediction output
pred = tf.add(tf.multiply(weights, X), bias)

# Cost node (MSE)
cost = tf.reduce_sum(tf.square(pred-y))*1./(2*n_samples)

# Optimizer (Gradient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all the variable
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        
        for (x_sample, y_sample) in zip(train_diabetes_X, train_diabetes_y):
            sess.run(optimizer, feed_dict={X: x_sample, y: y_sample}) # Feeding X and Y and run optimizer for every sample point

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_diabetes_X, y: train_diabetes_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),             "W=", sess.run(weights), "b=", sess.run(bias))
    
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_diabetes_X, y: train_diabetes_y})
    print("Training cost=", training_cost, "W=", sess.run(weights), "b=", sess.run(bias), '\n')

    # Graphic display
    plt.plot(train_diabetes_X, train_diabetes_y, 'ro', label='Original data')
    plt.plot(train_diabetes_X, sess.run(weights) * train_diabetes_X + sess.run(bias), label='Fitted line')
    plt.legend()
    plt.show()
    
    # Now test the model
    
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * test_diabetes_X.shape[0]),
        feed_dict={X: test_diabetes_X, y: test_diabetes_y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_diabetes_X, test_diabetes_y, 'bo', label='Testing data')
    plt.plot(test_diabetes_X, sess.run(weights) * test_diabetes_X + sess.run(bias), label='Fitted line')
    plt.legend()
    plt.show()

	

# Now use batch gradient descent instead of all training samples
from sklearn.utils import resample

# Cost node (MSE)
cost = tf.reduce_sum(tf.square(pred-y))*1./(2*batch_size)

# Initialize all the variable
init = tf.global_variables_initializer()
print ('Starting batch based gradient-descent optimization.....')

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        loss = 0
        for i in range(steps_per_epoch):
            # Take a mini batch
            batch_x, batch_y = resample(train_diabetes_X, train_diabetes_y,n_samples=batch_size)
            
            # run the optimizer
            _, c = sess.run([optimizer, cost], feed_dict={X:batch_x, y: batch_y})
            
            loss += c
        
        avg_loss = loss*1./steps_per_epoch
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_loss),             "W=", sess.run(weights), "b=", sess.run(bias))
        
    print("Optimization Finished!")
       
    training_cost = sess.run(tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * n_samples), 
                             feed_dict={X: train_diabetes_X, y: train_diabetes_y})
    # Normalization for training cost is need, as our cost function is divided by batch_size
    #training_cost = sess.run(cost, feed_dict={X: train_diabetes_X, y: train_diabetes_y})*(batch_size*1.)/n_samples
    
    print("Training cost=", training_cost, "W=", sess.run(weights), "b=", sess.run(bias), '\n')

    # Graphic display
    plt.plot(train_diabetes_X, train_diabetes_y, 'ro', label='Original data')
    plt.plot(train_diabetes_X, sess.run(weights) * train_diabetes_X + sess.run(bias), label='Fitted line')
    plt.legend()
    plt.show()

    # Now test the model

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * test_diabetes_X.shape[0]),
        feed_dict={X: test_diabetes_X, y: test_diabetes_y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_diabetes_X, test_diabetes_y, 'bo', label='Testing data')
    plt.plot(test_diabetes_X, sess.run(weights) * test_diabetes_X + sess.run(bias), label='Fitted line')
    plt.legend()
    plt.show()



