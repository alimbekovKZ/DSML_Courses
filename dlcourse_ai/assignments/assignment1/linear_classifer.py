import numpy as np
import math

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    exp_vect = np.vectorize(math.exp)
    exponents = np.array(predictions.shape)
    probs = np.zeros(exponents.shape)
    if (len(predictions.shape) == 2):
        rowmax = np.amax(predictions, axis = 1, keepdims=True)
        new_predictions = np.subtract(predictions, rowmax)
        exponents = exp_vect(new_predictions)
        exp_sum = np.sum(exponents, axis = 1, keepdims=True)
        probs = exponents/exp_sum
    else:
        new_predictions = predictions - np.max(predictions)
        exponents = exp_vect(new_predictions)
        exp_sum = np.sum(exponents)
        probs = exponents/exp_sum
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (N, batch_size) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    if (type(target_index) != int):
        m = len(target_index)
        #print(probs[range(m), target_index], 'Probs vector for true class')
        loss = np.sum(-1*np.log(probs[range(m), target_index]))/m
    else:
        loss = -1*np.log(probs[target_index])
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    exp_vect = np.vectorize(math.exp)
    exponents = np.array(predictions.shape)
    probs = np.zeros(exponents.shape)
    if (len(predictions.shape) == 2):
        rowmax = np.amax(predictions, axis = 1, keepdims=True)
        new_predictions = np.subtract(predictions, rowmax)
        exponents = exp_vect(new_predictions)
        exp_sum = np.sum(exponents, axis = 1, keepdims=True)
        probs = exponents/exp_sum
    else:
        new_predictions = predictions - np.max(predictions)
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

    # TODO: implement l2 regularization and gradient
    loss = reg_strength*np.sum(np.square(W))
    grad = 2*np.array(W)*reg_strength

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W
    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes
    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss
    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        self.batch_size = batch_size
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #print(num_train, num_features, num_classes)
            #print(sections)
            for batch in batches_indices:
                #print(len(X[batch]))
                predictions = np.dot(X[batch], self.W)
                loss, dprediction = softmax_with_cross_entropy(predictions, y[batch])
                dW = np.dot(X[batch].T, dprediction)
                reg_loss = reg*np.sum(np.square(self.W))
                grad = 2*np.array(self.W)*reg
                loss += reg_loss
                dW += grad
                self.W -= learning_rate*dW
            # end
            loss_history.append(loss)
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis = 1)
        return y_pred