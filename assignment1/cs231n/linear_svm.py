from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
          if j == y[i]:
            continue
          margin = scores[j] - correct_class_score + 1 
          if margin > 0:
            loss += margin
            dW[:,y[i]] -= X[i,:] # this one position was modified O(num_train * num_classes) times 
            dW[:,j] += X[i,:] # this one position was modified O(num_train) times  

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    delta = 1.0
    loss = 0.0

    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

 	# ------------------------------- HL ------------------------------- #
    # W.T has shape (C, D)
	# X.T has shape (D, N)
    # for each col (example) of scores, subtract an 
    # integer (true label score) from that col. In other words,
    # y[i] gives index that corresponds to ith "col" (or example) of scores. 
 	# ------------------------------- HL ------------------------------- #

    scores = (W.T).dot(X.T) # scores has shape (C, N) i.e. one col for each example
    correct_classes_indexes = y, np.arange((scores.shape[1]))
    margins = np.maximum(0, scores - scores[correct_classes_indexes] + delta)
    margins[correct_classes_indexes] = 0
    loss = np.sum(margins)
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    num_train = X.shape[0]
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)


    # # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # #############################################################################
    # # TODO:                                                                     #
    # # Implement a vectorized version of the gradient for the structured SVM     #
    # # loss, storing the result in dW.                                           #
    # #                                                                           #
    # # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # # to reuse some of the intermediate values that you used to compute the     #
    # # loss.                                                                     #
    # #############################################################################
    # # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins_bin = margins
    margins_bin[margins > 0] = 1 
    out_of_margin_classes_count = np.sum(margins_bin, axis=0)
    margins_bin[correct_classes_indexes] = -out_of_margin_classes_count
    # shape needed (D, C) from (C, N) and (N, D). Thus:
    dW = (np.dot(margins, X)).T
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW








