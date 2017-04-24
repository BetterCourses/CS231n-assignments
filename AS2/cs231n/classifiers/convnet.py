import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MyConvNet(object):
  def __init__(self, input_dim=(3, 32, 32), filter_params=[{'num_filters':32, 'filter_size':3, 'stride':1, 'repeat':2},
                                                            {'num_filters':64, 'filter_size':3, 'stride':1, 'repeat':3}], 
               pool_size=2, activate_pair=(relu_forward, relu_backward), hidden_dims=[128], num_classes=10, 
               weight_scale=0.01, reg=0.0, dtype=np.float32,
               dropout=0, use_spatial_batchnorm=False, use_batchnorm=False, seed=None):
    
    
    self.filter_params = filter_params
    self.params = {}
    self.pool_size = pool_size
    self.activate, self.activate_backward = activate_pair
    self.num_hidden_layers = len(hidden_dims)
    self.dropout = dropout
    self.reg = reg
    self.dtype = dtype
    self.use_dropout = dropout>0
    self.use_spatial_batchnorm = use_spatial_batchnorm
    self.use_batchnorm = use_batchnorm
    
    self.num_filter_set = len(self.filter_params)
    self.num_conv_layers = 0
    for p in self.filter_params:
      self.num_conv_layers += p['repeat']
    
    volume = input_dim[0]
    for i, params in enumerate(self.filter_params):
      num_filters = params['num_filters']
      filter_size = params['filter_size']
      for j in xrange(params['repeat']):
        sub = str(i+1)+str(j+1)
        self.params['W'+sub] = weight_scale * np.random.randn(num_filters, volume, filter_size, filter_size)
        self.params['b'+sub] = np.zeros(num_filters)
        volume = num_filters
        
        if self.use_spatial_batchnorm:
          self.params['gamma'+sub] = np.ones(volume)
          self.params['beta'+sub] = np.zeros(volume)
         
    
    pre = self.filter_params[-1]['num_filters']*(input_dim[1]/(pool_size**self.num_filter_set))**2
    for i, dim in enumerate(hidden_dims):
      sub = str(self.num_filter_set+i+1)
      self.params['W'+sub] = weight_scale * np.random.randn(pre, dim)
      self.params['b'+sub] = np.zeros(dim)
      pre = dim
                                                             
      if self.use_batchnorm:
        self.params['gamma'+sub] = np.ones(dim)
        self.params['beta'+sub] = np.zeros(dim) 
        
    sub = str(self.num_filter_set+len(hidden_dims)+1)
    self.params['W'+sub] = weight_scale * np.random.randn(pre, num_classes)
    self.params['b'+sub] = np.zeros(num_classes)
    
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
        
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(len(hidden_dims))]
    
    self.sp_bn_params = []
    if self.use_spatial_batchnorm:
      for param in self.filter_params:
        self.sp_bn_params.append([])
        for _ in xrange(param['repeat']):
          self.sp_bn_params[-1].append({'mode':'train'})
                                                             
  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    
    if self.use_spatial_batchnorm:
      for p in self.sp_bn_params:
        for spatial_bn_param in p:
          spatial_bn_param[mode] = mode
        
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    pool_param = {'pool_height': self.pool_size, 'pool_width': self.pool_size, 'stride': self.pool_size}
    
    #forward pass for conv layers
    scores = None                                                     
    conv_caches = []
    out = X
    for i, param in enumerate(self.filter_params):
      conv_caches.append([])
      stride = self.filter_params[i]['stride']
      filter_size = self.filter_params[i]['filter_size']
      pad = (filter_size-1)/2
      conv_param = {}
      conv_param['stride'] = stride
      conv_param['pad'] = pad
      for j in xrange(param['repeat']):
        sub = str(i+1) + str(j+1)
        W = self.params['W'+sub]
        b = self.params['b'+sub]
        if self.use_spatial_batchnorm:
          gamma = self.params['gamma'+sub]
          beta = self.params['beta'+sub]
          out, cache = conv_batchnorm_activate_forward(out, W, b, conv_param, gamma, beta, self.sp_bn_params[i][j], self.activate)
        else:
          out, cache = conv_activate_forward(out, W, b, conv_param, self.activate)
        conv_caches[-1].append(cache)
      out, p_cache = max_pool_forward_fast(out, pool_param)
      conv_caches[-1].append(p_cache)
    
    #forward pass for fc layers
    fc_caches = []    
    for i in xrange(self.num_hidden_layers):
      sub = str(self.num_filter_set+i+1)
      W = self.params['W'+sub]
      b = self.params['b'+sub]
      if self.use_batchnorm:
        gamma = self.params['gamma'+sub]
        beta = self.params['beta'+sub]  
        out, cache = affine_batchnorm_activate_forward(out, W, b, gamma, beta, self.bn_params[i], self.activate)
      else:
        out, cache = affine_activate_forward(out, W, b, self.activate)
      if self.use_dropout:
        out, dcache = dropout_forward(out, self.dropout_param)
        fc_caches.append((cache, dcache))
      else:
        fc_caches.append(cache)
        
    sub = str(self.num_filter_set+self.num_hidden_layers+1)
    W = self.params['W'+sub]
    b = self.params['b'+sub]
    scores, cache = affine_forward(out, W, b)
    fc_caches.append(cache)
    
    if mode == 'test':
      return scores
    
    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores, y)
    for k, v in self.params.iteritems():
      if k.startswith('W'):
        loss += 0.5*self.reg*np.sum(v**2)
        
    #backward pass for fc layers
    dout, dw, db = affine_backward(dscores, fc_caches[-1])
    sub = str(self.num_filter_set+self.num_hidden_layers+1)
    grads['W'+sub] = dw
    grads['b'+sub] = db
    for i in xrange(self.num_hidden_layers-1, -1, -1):
      sub = str(self.num_filter_set+i+1)
      cache = None
      if not self.use_dropout:
        cache = fc_caches[i]
      else: 
        cache, dcache = fc_caches[i]
        dout = dropout_backward(dout, dcache)
      if self.use_batchnorm:
        dout, dw, db, dgamma, dbeta = affine_batchnorm_activate_backward(dout, cache, self.activate_backward)
        grads['gamma'+sub] = dgamma
        grads['beta'+sub] = dbeta
      else: 
        dout, dw, db = affine_activate_backward(dout, cache, self.activate_backward)
      grads['W'+sub] = dw
      grads['b'+sub] = db
    
    
    #backward pass for conv layers
    for i in xrange(self.num_filter_set-1, -1, -1):
      p_cache = conv_caches[i][-1]
      dout = max_pool_backward_fast(dout, p_cache)
      for j in xrange(self.filter_params[i]['repeat']-1, -1, -1):
        sub = str(i+1) + str(j+1)
        if self.use_spatial_batchnorm:
          dout, dw, db, dgamma, dbeta = conv_batchnorm_activate_backward(dout, conv_caches[i][j])
          grads['gamma'+sub] = dgamma
          grads['beta'+sub] = dbeta
        else:
          dout, dw, db = conv_activate_backward(dout, conv_caches[i][j])
        grads['W'+sub] = dw
        grads['b'+sub] = db
    
    for k in grads:
      if k.startswith('W'):
        grads[k] += self.reg * self.params[k]
    return loss, grads
  
    
  def scores(self, X, batch_size=100):
    num_batches = X.shape[0] / batch_size
    scores = []
    for i in xrange(num_batches):
      start = i*batch_size
      end = start + batch_size
      scores.append(self.loss(X[start:end]))
    scores = np.vstack(scores)
    return scores
        
        
        
        
       
      
    
      