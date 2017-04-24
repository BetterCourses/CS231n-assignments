from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache

def affine_activate_forward(x, w, b, activate=relu_forward):
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = activate(a)
  cache = (fc_cache, relu_cache)
  return out, cache

def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_activate_backward(dout, cache, activate_backward=relu_backward):
  fc_cache, relu_cache = cache
  da = activate_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def affine_batchnorm_relu(x, w, b, gamma, beta, bn_param):
  aout, acache = affine_forward(x, w, b)
  bout, bcache = batchnorm_forward(aout, gamma, beta, bn_param)
  result, rcache = relu_forward(bout)
  return result, (acache, bcache, rcache)
    
def affine_batchnorm_rulu_backward(dout, cache):
    acache, bcache, rcache = cache
    dout = relu_backward(dout, rcache)
    dx, dgamma, dbeta = batchnorm_backward(dout, bcache)
    dx, dw, db = affine_backward(dx, acache)
    return dx, dw, db, dgamma, dbeta

def affine_batchnorm_activate_forward(x, w, b, gamma, beta, bn_param, activate=relu_forward):
  aout, acache = affine_forward(x, w, b)
  bout, bcache = batchnorm_forward(aout, gamma, beta, bn_param)
  result, rcache = activate(bout)
  return result, (acache, bcache, rcache)

def affine_batchnorm_activate_backward(dout, cache, activate_backward=relu_backward):
    acache, bcache, rcache = cache
    dout = activate_backward(dout, rcache)
    dx, dgamma, dbeta = batchnorm_backward(dout, bcache)
    dx, dw, db = affine_backward(dx, acache)
    return dx, dw, db, dgamma, dbeta

def conv_activate_forward(x, w, b, conv_param, activate=relu_forward):
  conv_out, cache = conv_forward_fast(x, w, b, conv_param)
  result, rcache = activate(conv_out)
  return result, (cache, rcache)

def conv_batchnorm_activate_forward(x, w, b, conv_param, gamma, beta, bn_param, activate=relu_forward):
  conv_out, cache = conv_forward_fast(x, w, b, conv_param)
  bout, bcache = spatial_batchnorm_forward(conv_out, gamma, beta, bn_param)
  result, rcache = activate(bout)
  return result, (cache, bcache, rcache)

def conv_activate_backward(dout, cache, activate_backward=relu_backward):
  cache, rcache = cache
  dout = activate_backward(dout, rcache)
  dx, dw, db = conv_backward_fast(dout, cache)
  return dx, dw, db

def conv_batchnorm_activate_backward(dout, cache, activate_backward=relu_backward):
  cache, bcache, rcache = cache
  dout = activate_backward(dout, rcache)
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout, bcache)
  dx, dw, db = conv_backward_fast(dout, cache)
  return dx, dw, db, dgamma, db

    
    
    
    
    