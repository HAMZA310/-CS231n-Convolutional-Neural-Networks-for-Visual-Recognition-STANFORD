from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        # ------------ SOFTMAX INTERPERTATION ------------------ # 
        # 'i' suffix corresponds to ith example

        unnormalized_log_prob_i = X[i].dot(W) # dim: (1,C)
        unnormalized_log_prob_i -= np.max(unnormalized_log_prob_i) # to prevent overflow
        unnormalized_prob_i = np.exp(unnormalized_log_prob_i) # (1,C)
        correct_class_index_i = y[i]

        col_sum_of_unnorm_prob_i = 0
        for j in range(num_classes):
            col_sum_of_unnorm_prob_i += unnormalized_prob_i[j] 

        col_sum_of_unnorm_prob_i = [unnormalized_prob_i[j] + col_sum_of_unnorm_prob_i]
        normalized_prob_i = unnormalized_prob_i / col_sum_of_unnorm_prob_i  # (1,C)
        neg_log_prob_of_correct_class_i = -np.log( \
                                    normalized_prob_i[correct_class_index_i])
        loss += neg_log_prob_of_correct_class_i

        for j in range(num_classes): # gradient computation 
            grad_j = normalized_prob_i[j] * X[i]  # (1) by (1, D)
            if j != correct_class_index_i:
                dW[:, j] += grad_j
            else:
                grad_j = -X[i] + grad_j # (C, ) - (D)
                dW[:, j] += grad_j
           
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W

    # # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    unnormalized_log_probs = np.dot(X, W)   # dim: (N,C). Softmax Interpertation 

    # exp in the next to next line may overflow otherwise
    unnormalized_log_probs -= np.max(unnormalized_log_probs) # (N,C).                          
    unnormalized_probs = np.exp(unnormalized_log_probs) # (N,C).

    # (C,N). Each example is divided by the sum of all of its classes
    normalized_probs = unnormalized_probs.T / (unnormalized_probs.sum(axis=1)) 
    normalized_probs = normalized_probs.T #(N, C)
    correct_classes_indexes = np.arange((unnormalized_log_probs.shape[0])), y
    loss = -np.log(normalized_probs[correct_classes_indexes])
    loss = np.sum(loss)

    # Algebra. Instead of subtracting X[i], subtract 1 and then multiply in the 
    # next line to get equivalent expression as that in naive version. 
    normalized_probs[correct_classes_indexes] -= 1 
    dW = np.dot(X.T, normalized_probs) # (D,C)
 
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    num_train = X.shape[0]
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
