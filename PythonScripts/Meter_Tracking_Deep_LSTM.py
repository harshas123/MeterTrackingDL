#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:35:26 2017

@author: Harsha
"""
#%% 
#This code is to create:
#    Input --->LSTM*num_layers ---> Softmax Layer
#Import the required packages
import numpy as np
import math
import h5py
import scipy.io as sio
import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
import matplotlib as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')




#%%
# Define all the required Functions
def mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    n_input, m = X.shape                  # number of training examples
    n_output = Y.shape[0]
#    np.random.seed(seed)
    

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    batch_X = np.zeros((num_complete_minibatches,mini_batch_size, n_input))
    batch_Y = np.zeros((num_complete_minibatches,mini_batch_size, n_output))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        batch_X[k,:,:] = mini_batch_X.T
        batch_Y[k,:,:] = mini_batch_Y.T
   
    #Shuffle the batches before you send it across
    permutation = list(np.random.permutation(num_complete_minibatches))
    shuffled_batch_X = batch_X[permutation,:,:]
    shuffled_batch_Y = batch_Y[permutation,:,:]

    return shuffled_batch_X, shuffled_batch_Y
    
def read_from_h5py_mat(dataset_type):
    # Inputs: fname - Filename with Path
    #         var_names - a list of names of the variables required
    fname = "../Data/" + dataset_type + ".mat"
    varname = dataset_type.replace('-','_').lower()
    f = h5py.File(fname, 'r')
    data_X = np.array(f.get(varname +"_X")).T
    data_Y = np.array(f.get(varname +"_Y")).T
    print data_Y.shape
    return data_X, data_Y
    
def load_music_dataset(mini_batch_size):
    
    
    train_X, train_Y = read_from_h5py_mat("Train")
    train_dev_X, train_dev_Y = read_from_h5py_mat("Train-Dev")
    dev_X, dev_Y = read_from_h5py_mat("Dev")
    test_X, test_Y = read_from_h5py_mat("Test")
    
    
    #Try taking the difference of successive frames to get better features for beat detection
#    train_X = np.diff(train_X, axis=1)
#    train_Y = np.diff(train_Y, axis=1)
#    
#    train_dev_X = np.diff(train_dev_X, axis=1)
#    train_dev_Y = np.diff(train_dev_Y, axis=1)
#    
#    dev_X = np.diff(dev_X, axis=1)
#    dev_Y = np.diff(dev_Y, axis=1)
#    
#    test_X = np.diff(test_X, axis=1)
#    test_Y = np.diff(test_Y, axis=1)
    
#    # Normalize Data to have zero mean and unit variance along all dimensions    
#    mean_vec = np.mean(train_X, axis=1, keepdims=True)
#    train_X = train_X - mean_vec
#    std_vec = np.std(train_X, axis=1, keepdims=True)
#    print std_vec
#    train_X /= std_vec
#    train_Y = np.reshape(train_Y[1,:],(1,-1))
    #Reshape data in train_X to fit Tensorflow RNN cell requirement to be of 
    #size - (batch_size, n_steps, n_input)
    #
    train_X, train_Y = mini_batches(train_X, train_Y, mini_batch_size)

    #Apply Training normalization to Train-Dev dataset
#    train_dev_X = train_dev_X - mean_vec
#    train_dev_X /= std_vec
#    train_dev_Y = np.reshape(train_dev_Y[1,:],(1,-1))
    #Reshape data for Tensorflow and divide into batches
    train_dev_X, train_dev_Y = mini_batches(train_dev_X, train_dev_Y, mini_batch_size)

    #Apply Training normalization to Dev dataset
#    dev_X = dev_X - mean_vec
#    dev_X /= std_vec
#    dev_Y = np.reshape(dev_Y[1,:],(1,-1))
    #Reshape data for Tensorflow and divide into batches
    dev_X, dev_Y = mini_batches(dev_X, dev_Y, mini_batch_size)

    #Apply Training normalization to Dev dataset
#    test_X = test_X - mean_vec
#    test_X /= std_vec
#    test_Y = np.reshape(test_Y[1,:],(1,-1))
    #Reshape data for Tensorflow and divide into batches
    test_X, test_Y = mini_batches(test_X, test_Y, mini_batch_size)

    
    return train_X, train_Y, train_dev_X, train_dev_Y, dev_X, dev_Y, test_X, test_Y

def create_placeholders(n_input, n_classes, n_steps):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=(None, n_steps, n_input))
    Y = tf.placeholder(tf.float32, shape=(None, n_steps, n_classes))
    ### END CODE HERE ###
    
    return X, Y

def initialize_parameters(num_hidden, n_classes):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        
    Input: layers_dims  - a python list which contains [I/P Dim, L1 dim, L2 Dim , ..., O/P Dim]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
#    tf.set_random_seed(1)                   # so that your "random" numbers match ours
    #Initialize the Weights and Biases of the final Softmax Layer
#    weights = {
#        'out': tf.Variable(tf.random_normal([num_hidden, n_classes]))
#    }
#    biases = {
#        'out': tf.Variable(tf.random_normal([n_classes]))
#    }
    parameters = {}
    parameters["W-out"] = tf.get_variable("W-out", shape=[2*num_hidden, n_classes], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    parameters["b-out"] = tf.get_variable("b-out", shape=[n_classes], initializer=tf.constant_initializer(0.0))
                                                     
    return parameters
    
def RNN(x, parameters, n_steps, num_hidden, number_of_layers, n_classes):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (num_batches, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (num_batches, n_input)
    #n_steps = x.shape[1]
    # Permuting num_batches and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*num_batches, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (num_batches, n_input)
    x = tf.split(x, n_steps, axis=0)
    
    
    # Define a lstm cell with tensorflow
    ########NOTE to Self: Re-Check stat_is_tuple and confirm concretely
    
    #lstm_fw = rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
    #lstm_bw = rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
    stacked_lstm_fw=[]
    stacked_lstm_bw=[]
    for layer_id in range(number_of_layers):
        stacked_lstm_fw.append(rnn.LSTMCell(num_hidden, state_is_tuple=True))
        stacked_lstm_bw.append(rnn.LSTMCell(num_hidden, state_is_tuple=True))
    multi_fw = rnn.MultiRNNCell(stacked_lstm_fw, state_is_tuple=True)
    multi_bw = rnn.MultiRNNCell(stacked_lstm_bw, state_is_tuple=True)
#    stacked_lstm_fw = rnn.MultiRNNCell([lstm_fw] * number_of_layers, \
#    state_is_tuple=True)
#    stacked_lstm_bw = rnn.MultiRNNCell([lstm_bw] * number_of_layers, \
#    state_is_tuple=True)
    outputs,_,_  = rnn.static_bidirectional_rnn(multi_fw, multi_bw, x, dtype=tf.float32)

    # Get lstm cell output
    #outputs, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)
    #outputs - n_steps X num_batches X n_input
    #Permuting the first and second dimensions - num_batches X n_steps X n_input
    op2 = tf.transpose(outputs,[1,0,2])
    # Reshaping to 2D tensor for multiplication (num_batches*n_steps) X num_hidden
    op3 = tf.reshape(op2,[-1,2*num_hidden])
    #Return the logits of shape num_batches X n_steps X n_classes
    logits = tf.reshape(tf.matmul(op3, parameters["W-out"]) + parameters["b-out"],[-1,n_steps,n_classes])
    
    #Return the logits of shape (num_batches * n_steps) X n_classes
    #logits = tf.matmul(op3, parameters["W-out"]) + parameters["b-out"]
    
#    op2 = tf.reshape(outputs,[-1, n_hidden_3])
#    return tf.split(0,n_steps,tf.matmul(op2, weights['out']) + biases['out']),op2, states
    #return tf.matmul(op2, weights['out']) + biases['out'],op2,states
    #NOTE: Size of the returned Tensor is batch_size X n_steps X n_classes
    
   
    return logits

def compute_cost(logits,Y):
    
    #Both logits and Y are of shape batch_size X n_steps X n_classes. So, convert them to be 
    # of shape (batch_size X n_steps) X n_classes
    logits = tf.reshape(logits, [-1, n_classes])
    labels = tf.reshape(Y, [-1, n_classes])
    #weights = tf.reshape(tf.constant([0.1,0.9]),[1,2])
    #logits = logits*weights
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits\
                          (logits=logits, targets=labels,pos_weight=4000.0))
    
    return cost
    
def model(train_X, train_Y, train_dev_X, train_dev_Y, dev_X, dev_Y, test_X, test_Y,\
          num_hidden, num_layers, learning_rate = 0.0001, num_epochs = 1500, \
          minibatch_size = 32, print_cost = True):
    """
    Implements a Multi-layer tensorflow LSTM Recurrent neural network: 
        LSTM * num_layers ->SOFTMAX.
    
    Arguments:
    train_X -- training set, of shape (input size = 128, number of training examples = ?)
    train_Y -- training labels, of shape (output size = 1, number of training examples = ?)
    train_dev_X -- train-dev set, of shape (input size = 128, number of examples = ?)
    train_dev_Y -- train-dev labels, of shape (output size = 1, number of examples = ?)
    dev_X -- dev set, of shape (input size = 128, number of examples = ?)
    dev_Y -- dev labels, of shape (output size = 1, number of examples = ?)
    test_X -- test set, of shape (input size = 128, number of examples = ?)
    test_Y -- test labels, of shape (output size = 1, number of examples = ?)
    num_layers -- Number of LSTM layers
    num_hidden -- Number of LSTM cells per layer
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch == Number of steps = n_steps == 
    Number of steps in a truncated back prop algo
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    tf.reset_default_graph()
    _, n_steps, n_input = train_X.shape
    n_classes = train_Y.shape[2]
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders for X and Y
    X, Y = create_placeholders(n_input, n_classes, n_steps)


    # Initialize parameters
    parameters = initialize_parameters(num_hidden, n_classes)

    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    logits = RNN(X, parameters, n_steps, num_hidden, number_of_layers, n_classes)

    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(logits, Y)

    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

     # Calculate the correct predictions: Compute the average Hamming Score
#    nn_op =  
    logits = tf.reshape(logits,[-1, n_classes])
    labels = tf.reshape(Y,[-1, n_classes])
    pred = tf.nn.softmax(logits)
    #correct_prediction = tf.equal(tf.argmax(pred,axis=1), tf.argmax(labels,axis=1))
    #correct_prediction = tf.equal(pred >= 0.5, labels>0)

    # Calculate accuracy on the test set
    #accuracy = tf.reduce_mean(tf.reduce_mean(tf.cast(correct_prediction, "float"),axis=1))*100.0
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100.0
    #Do not use Accuracy use F-measure or F1 score
    _,prec = tf.metrics.precision(predictions=pred, labels=labels)
    _,recall = tf.metrics.recall(predictions=pred, labels=labels)
    accuracy = 2.0 * tf.multiply(prec,recall)/(tf.add(prec, recall))#Actually F-score but just called accuracy
#    accuracy,_ = tf.metrics.auc(predictions=pred, labels=labels)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            
            # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
            
            _ , batch_cost = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
            
            
            epoch_cost += batch_cost

            # Print the cost every epoch 
            if print_cost == True :
                #parameters = sess.run(parameters)
                train_acc = sess.run(accuracy, feed_dict={X: train_X, Y: train_Y})
                train_dev_acc = sess.run(accuracy, feed_dict={X: train_dev_X, Y: train_dev_Y})
                dev_acc = sess.run(accuracy, feed_dict={X: dev_X, Y: dev_Y})
                print ("%i\t %.5f\t %.2f\t %.2f\t %.2f\n " % (epoch, epoch_cost, train_acc, train_dev_acc, dev_acc))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        #plt.plot(np.squeeze(costs))
        #plt.ylabel('cost')
        #plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        train_acc = sess.run(accuracy, feed_dict={X: train_X, Y: train_Y})
        train_dev_acc = sess.run(accuracy, feed_dict={X: train_dev_X, Y: train_dev_Y})
        dev_acc = sess.run(accuracy, feed_dict={X: dev_X, Y: dev_Y})
        test_acc = sess.run(accuracy, feed_dict={X: test_X, Y: test_Y})
        
        train_pred = pred.eval({X: train_X, Y: train_Y})
        train_dev_pred = pred.eval({X: train_dev_X, Y: train_dev_Y})
        dev_pred = pred.eval({X: dev_X, Y: dev_Y})
        test_pred = pred.eval({X: test_X, Y: test_Y})

        
        
        return parameters, costs, train_acc, train_dev_acc, dev_acc, test_acc,\
                                train_pred, train_dev_pred, dev_pred, test_pred
    
#%%
## Load the dataset
n_steps = 100
train_X, train_Y, train_dev_X, train_dev_Y, dev_X, dev_Y, test_X, test_Y = load_music_dataset(n_steps)
print("Train Data shape = " + str(train_X.shape))
print("Train-Dev Data shape = " + str(train_dev_X.shape))
print("Dev Data shape = " + str(dev_X.shape))
print("Test Data shape = " + str(test_X.shape))
print("Train Labels shape = " + str(train_Y.shape))
print("Train-Dev Labels shape = " + str(train_dev_Y.shape))
print("Dev Labels shape = " + str(dev_Y.shape))
print("Test Labels shape = " + str(test_Y.shape))
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")

#%%
# Tweak-able Parameters
n_input = 128
num_hidden = 25
number_of_layers = 3
n_classes = 2
num_iter = 50
nn_config = "NumLayers_" + str(number_of_layers) + "num_iter_" + str(num_iter)

new_params_fname = '../Results/Parameters_NN_config_' + nn_config + ".mat"
print("Epoch\tEpoch-Cost\tTrain-Acc\tTrainDev-Acc\tDev-Acc")
parameters, costs, train_acc, train_dev_acc, dev_acc, test_acc,\
                    train_pred, train_dev_pred, dev_pred, test_pred =\
                 model(train_X, train_Y, train_dev_X, train_dev_Y, dev_X, dev_Y,\
                       test_X, test_Y,\
                       num_hidden= num_hidden,\
                       num_layers = number_of_layers,\
                       learning_rate = 0.0001, \
                       num_epochs = num_iter, \
                       minibatch_size = n_steps)

sio.savemat(new_params_fname,\
            {"parameters":parameters, 'costs':costs, 'train_acc': train_acc, \
            'train_dev_acc': train_dev_acc, 'dev_acc': dev_acc, 'test_acc': test_acc,\
            'train_pred':train_pred, 'train_dev_pred':train_dev_pred, 'dev_pred':dev_pred,\
            'test_pred':test_pred, 'train_Y':train_Y, 'train_dev_Y':train_dev_Y, 'dev_Y':dev_Y,\
            'test_Y':test_Y,})