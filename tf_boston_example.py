# Let's build a linear regression using tensorflow

from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample 

data = load_boston() # Load the boston data set

X_ = data['data']
y_ = data['target']

# Normalize the data tensorflow has L2 normalization
X_norm_ = (X_ - np.mean(X_, axis=0))/np.std(X_, axis=0)

# Define the hyperparameters
learning_rate = 0.01
epochs = 1000
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

# Now build the tensorflow graph
X = tf.placeholder(tf.float32, shape=[batch_size, X_.shape[1]]) # These shapes should be in size of batches
Y = tf.placeholder(tf.float32, shape=(batch_size,))

n_hidden = 10
# Assign weights and bias as variables that needs to be optimized
W_1 = tf.Variable(tf.random_normal(shape=[X_.shape[1], n_hidden])) # truncated normal removes the tail numbers
bias_1 = tf.Variable(tf.zeros(shape=[n_hidden])) # Number of bias units sould be same as the number of hidden units

# Now do the dot product of the 
layer_1 = tf.nn.bias_add(tf.matmul(X, W_1), bias_1) # Make sure you use bias add instead of just add

# Now pass this through the sigmoid unit

layer_1_sigmoid = tf.nn.sigmoid(layer_1)

# Now add another layer with the same hidden units as variables that need to be optimized

W_2 = tf.Variable(tf.random_normal(shape=(n_hidden, 1)))
bias_2 = tf.Variable(tf.zeros(shape=(1,)))

# Now compute the prediction

#y_pred = tf.nn.bias_add(tf.matmul(layer_1_sigmoid, W_2), bias_2)
y_pred = tf.add(tf.matmul(layer_1_sigmoid, W_2), bias_2)
# reshape is necessary to make sure the output from NN and ground truth are of same size
y_pred = tf.reshape(y_pred, shape=(batch_size,)) 
print (y_pred)
# Now define the cost function (MSE)
cost = tf.reduce_sum(tf.square(y_pred-Y))*1./(2*X_.shape[0]) # reduce_sum summs across all the dimensions

# Now define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Initialize all the variables
init = tf.global_variables_initializer()



# Now lauch the computational graph
with tf.Session() as sess:
    sess.run(init) # Initialize the session
    
    for epoch in range(epochs):
        loss = 0
        for step in range(steps_per_epoch):
            # Take the batch of data and feed it to the network graph
            batch_x, batch_y = resample(X_norm_, y_, n_samples=batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            
            loss += c
        
        avg_loss = loss/steps_per_epoch
        #print (avg_loss)
        print("Epoch: {}, Loss: {:.3f}".format(epoch+1, avg_loss))
       
