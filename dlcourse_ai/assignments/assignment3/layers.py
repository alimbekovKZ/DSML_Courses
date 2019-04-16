import numpy as np
import itertools
from math import exp, sqrt, log

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength*sum(sum(W**2));
    grad = reg_strength*2*W;
    
    return loss, grad

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    #print("probs:", probs);
    
    return -log(probs[target_index - 1]);

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    if predictions.ndim == 1:
        predictions_ = predictions - np.max(predictions);
        dprediction = np.array(list(map(exp, predictions_)));
        summ = sum(dprediction);
        dprediction /= summ;
        
        loss = cross_entropy_loss(dprediction, target_index);
        dprediction[target_index - 1] -= 1;
        
        return loss, dprediction;
    else:
    
        predictions_ = predictions - np.max(predictions, axis = 1)[:, np.newaxis];
        exp_vec = np.vectorize(exp);
        #print("predictions_:", predictions_);
        
        dprediction = np.apply_along_axis(exp_vec, 1, predictions_);
        #print("dprediction before division: ", dprediction);
    
        summ = sum(dprediction.T);
        #print("summ: ", summ);
        dprediction /= summ[:, np.newaxis];
            
        #print("dprediction after division: ", dprediction);
    
        loss = np.array([cross_entropy_loss(x,y) for x,y in zip(dprediction, target_index)]);
        #print("loss: ", loss);
        
        #print("target_index - 1:", target_index - 1);
        it = np.nditer(target_index - 1, flags = ['c_index'] )
        while not it.finished:
            #print("it[0] = ", it[0]);
            dprediction[it.index, it[0]] -= 1
            it.iternext()
        
        dprediction /= len(target_index);
        #print("dprediction after subtraction: ", dprediction);
    
        return loss.mean(), dprediction;


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value, grad = None):
        self.value = value
        if grad is None:
            self.grad = np.zeros_like(value)
        else:
            self.grad = grad
        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X;
        return (X > 0)*X;

    def backward(self, d_out):
        # TODO copy from the previous assignment
        return (self.X > 0)*d_out;

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output, isfull = True):
        if isfull:
            self.W = Param(0.01 * np.random.randn(n_input, n_output))
            self.B = Param(0.01 * np.random.randn(1, n_output))
        else:
            self.W = None;
            self.B = None;
        self.X = None;
        self.isfull = isfull;

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X;
        #if np.any(self.W.init != self.W.value) or np.any(self.B.init != self.B.value):
        if self.isfull:
            self.W.grad = np.zeros_like(self.W.value);
            self.B.grad = np.zeros_like(self.B.value);
        #    self.W.init = self.W.value;
        #    self.B.init = self.B.value;
        return np.dot(self.X, self.W.value) + self.B.value;

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        dW = np.dot(self.X.T, d_out);
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out);
        
        d_input = np.dot(d_out, self.W.value.T);
        #print("self.X = ", self.X);
        #print("self.W.grad.T = ", self.W.grad.T);
        #print("dW.T = ", dW.T);
        
        self.W.grad += dW;
        self.B.grad += dB;
        
        return d_input;

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size;
        self.in_channels = in_channels;
        self.out_channels = out_channels;
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        );
        
        self.B = Param(np.zeros(out_channels));
        self.X = None;
        self.FCLs = None;
        self.padding = padding;


    def forward(self, X):
        
        batch_size, height, width, channels = (X.shape[0], X.shape[1] + 2*self.padding, X.shape[2] + 2*self.padding, X.shape[3]);
        self.X = np.zeros((batch_size, height, width, channels));
        self.X[:, self.padding : X.shape[1] + self.padding, self.padding : X.shape[2] + self.padding, :] = X;
        out_height = height - self.filter_size + 1;
        out_width = width - self.filter_size + 1;
        
        self.W.grad = np.zeros_like(self.W.value);
        self.B.grad = np.zeros_like(self.B.value);
        
        self.FCLs = np.tile(FullyConnectedLayer(self.in_channels*self.filter_size**2, self.out_channels, isfull = False), (out_height, out_width));
        
        for y in range(out_height):
            for x in range(out_width):
                self.FCLs[y, x].W = Param(self.W.value.reshape((-1, self.out_channels)), self.W.grad.reshape((-1, self.out_channels)));
                self.FCLs[y, x].B = Param(self.B.value.reshape((-1, self.out_channels)), self.B.grad.reshape((-1, self.out_channels)));
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels));
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                result[:, y, x, :] = self.FCLs[y, x].forward(self.X[:, y:y+self.filter_size, x:x+self.filter_size, :].reshape((batch_size, -1)));
                
        return result;

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        result = np.zeros_like(self.X);
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                result[:, y : y + self.filter_size, x : x + self.filter_size, :] += self.FCLs[y, x].backward(d_out[:, y, x, :]).reshape(batch_size, self.filter_size, self.filter_size, channels);
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
        #print(result[:, 0 : 0 + self.filter_size, 0 : 0 + self.filter_size, :]);
        
        return result[:, self.padding : height - self.padding, self.padding : width - self.padding, :];
        
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.argmax = None
        
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        assert (height - self.pool_size) % self.stride == 0, "height = %i, self.pool_size = %i, self.stride = %i" % (height, self.pool_size, self.stride)
        
        out_height = int((height - self.pool_size)/self.stride + 1);
        out_width = int((width - self.pool_size)/self.stride + 1);
        self.X = X;
        
        #self.argmax = [np.argmax(self.X[:, i : i + self.pool_size, j : j + self.pool_size, :],                  axis = (1, 2)) for i, j in zip(range(0, height, self.stride), range(0, width, self.stride)) ];
        
        result = np.zeros((batch_size, out_height, out_width, channels));
        i = 0;
        for y in range(out_height):
            j = 0;
            for x in range(out_width):
                result[:, y, x, :] += np.amax(self.X[:, i : i + self.pool_size, j : j + self.pool_size, :],                  (1, 2));
                j += self.stride;
            i += self.stride;    
    
        return result;
    
    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape;
        _, out_height, out_width, _ = d_out.shape
        result = np.zeros_like(self.X);
        #print("d_out = ", d_out);
        
        for m in range(batch_size):
            for k in range(channels):
                i = 0;
                for y in range(out_height):
                    j = 0;
                    for x in range(out_width):
                        ind = np.unravel_index(np.argmax(self.X[m, i : i + self.pool_size, j : j +                                  self.pool_size, k], axis=None), (self.pool_size, self.pool_size));
                        result[m, i : i + self.pool_size, j : j + self.pool_size, k][ind] += d_out[m, y, x, k];
                        #print(result[m, i : i + self.pool_size, j : j + self.pool_size, k][ind]);
                        #ind = np.where(abs(self.X[m, i : i + self.pool_size, j : j + self.pool_size, k] - np.max(self.X[m, i : i + self.pool_size, j : j + self.pool_size, k])) <= 1e-5);
                        #print(ind);
                        #if len(ind[0]) > 1 or len(ind[1]) > 1:
                        #    result[m, i : i + self.pool_size, j : j + self.pool_size, k][ind] += 0.5*d_out[m, y, x, k];
                        #else:
                        #        result[m, i : i + self.pool_size, j : j + self.pool_size, k][ind] += d_out[m, y, x, k];
                        j += self.stride;
                    i += self.stride; 
        #print("result = ", result);
        return result;
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape;
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height*width*channels);

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape);

    def params(self):
        # No params!
        return {}
