from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        n_0 = D # D i.e. Units in input layer. 
        n_1, n_2 = W2.shape # n_1 (H) i.e. Units in first hidden layer. 
                            # n_2 (C) i.e. Units in second layer/output layer. 

        # Names and shapes consitent with Andrew Ng's notations. 
        b1 = b1.reshape(n_1,1) # convert to row vector from rank array. Original shape was (n_1,)
        b2 = b2.reshape(n_2,1) # convert to row vector from rank array. Original shape was (n_2,)
        W1 = W1.T # shape(n_1, n_0)
        W2 = W2.T # shape(n_2, n_1)
        A0 = X.T # A0 shape (N, n_0)


        Z1 = W1.dot(A0) + b1
        # assert(Z1.shape == (n_1,N))
        A1 = np.maximum(0, Z1)
        # assert(A1.shape == (n_1,N))
        Z2 = W2.dot(A1) + b2 
        # assert(Z2.shape == (n_2,N))
      
        scores = Z2.T

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Softmax Interpertaion

        # Exp in next line may overflow otherwise
        Z2 -= np.max(Z2) 
        unnorm_probs = np.exp(Z2) # (n_2, N).

        # Each example is divided by the sum of all of its classes. 
        norm_probs = unnorm_probs / (unnorm_probs.sum(axis=0, \
                                                keepdims=True)) # (n_2, N).
        correct_classes_ind = y, np.arange(N)
        data_loss = -np.log(norm_probs[correct_classes_ind])
        mean_data_loss = np.sum(data_loss) / N
        reg_loss = .5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = mean_data_loss + reg_loss 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****s

        '''
            Line 164
            Remember. Element-wise Multiplication for Activation Function. 
            Derivate of a function applied element-wise is also 
            multiplied element-wise instead of dot-product. The reason is in
            this case Jacobian turns out to be a diagonal matrix. A diagnol matrix
            multiplied element-wise is same as its dot-product. Refer to Linear Algebra
            notes for further clarification.    
        '''

        dZ2 = norm_probs # (n_2, N).
        dZ2[correct_classes_ind] -= 1
        dZ2 /= N 

        # assert(dZ2.shape == (n_2, N))
        dA1 = W2.T.dot(dZ2)
        # assert(dA1.shape == (n_1,N))

        dZ1 = dA1 * (Z1>0)
        # assert(dZ1.shape == (n_1,N))

        dW1 = A0.dot(dZ1.T)
        # assert(dW1.shape == (n_0,n_1))

        dW2 = A1.dot(dZ2.T) 
        # assert(dW2.shape == (n_1,n_2))

        db1 = np.sum(dZ1, axis=1)
        db2 = np.sum(dZ2, axis=1)
        dW1 += reg * W1.T
        dW2 += reg * W2.T 

        grads['W1'], grads['W2'], grads['b1'], grads['b2'] = dW1, \
                                                    dW2, db1, db2
       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            random_indices = np.random.choice(a=num_train, size=batch_size)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W1'] += -learning_rate * grads['W1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['b2'] += -learning_rate * grads['b2']

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        predicted_scores = self.loss(X)
        y_pred = np.argmax(predicted_scores, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
