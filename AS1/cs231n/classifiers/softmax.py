import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  S = X.dot(W)
  exps = np.exp(S)
  dens = np.sum(exps, axis=1)
  Ps = exps / dens.reshape(num_train, 1)
  for i in xrange(num_train):
      loss -= np.log(Ps[i, y[i]])
      for j in xrange(num_classes):
          if j != y[i]:
             dW[:, j] += X[i] * Ps[i, j]
          else:
             dW[:, j] += X[i]*(Ps[i, j]-1)
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  S = X.dot(W)
  exps = np.exp(S)
  Ps = exps / np.sum(exps, axis=1).reshape(num_train, 1)
  loss -= np.sum(np.log(Ps)[range(num_train), y])
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  
  Ps[range(num_train), y] -= 1
  dW += X.T.dot(Ps)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

