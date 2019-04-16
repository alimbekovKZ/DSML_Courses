import numpy as np
import math

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(np.square(W))
    grad = 2*np.array(W)*reg_strength

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    exp_vect = np.vectorize(math.exp)
    exponents = np.array(preds.shape)
    probs = np.zeros(exponents.shape)
    if (len(preds.shape) == 2):
        rowmax = np.amax(preds, axis = 1, keepdims=True)
        new_predictions = np.subtract(preds, rowmax)
        exponents = exp_vect(new_predictions)
        exp_sum = np.sum(exponents, axis = 1, keepdims=True)
        probs = exponents/exp_sum
    else:
        new_predictions = preds - np.max(preds)
        exponents = exp_vect(new_predictions)
        exp_sum = np.sum(exponents)
        probs = exponents/exp_sum
    #print(probs, 'Probs')
    dprediction = np.array(probs)
    if (type(target_index) != int):
        m = len(target_index)
        #print(probs[range(m), target_index], 'Probs vector for true class')
        loss = np.sum(-1*np.log(probs[range(m), target_index]))/m
        dprediction[range(m), target_index] -= 1
        dprediction /= m
    else:
        loss = -1*np.log(probs[target_index])
        dprediction[target_index] -= 1
        #print(probs[target_index], 'Probs value for true class')
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        d_input = np.ones_like(self.X)
        d_input[np.where(self.X < 0)] = 0
        d_result = d_out * d_input
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        self.X = X
        out = np.dot(X, self.W.value) + self.B.value
        return out

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
        d_result = np.dot(d_out, (self.W.value).T)
        dW = np.dot(self.X.T, d_out)
        dB = np.sum(d_out, axis=0, keepdims=True) # Суммируем градиент выхода по строкам
        self.W.grad = dW
        self.B.grad = dB
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
