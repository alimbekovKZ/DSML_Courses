import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.hidden_layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.hidden_layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu_layer1 = ReLULayer()
        self.relu_layer2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()	
        for param_key in params:
            #print(param_key, ' shape is ', params[param_key].value.shape)
            params[param_key].grad = 0
		
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        z1 = self.hidden_layer1.forward(X)
        a1 = self.relu_layer1.forward(z1)
        #print(a1.shape, ' - a1')
        a2 = self.hidden_layer2.forward(a1)
        #print(a2.shape, ' - a2')
        '''
        output = self.relu_layer2.forward(a2)
        '''
        #print(output, ' - ReLULayer output')
        #print(output.shape, ' - output')
        loss, dprediction = softmax_with_cross_entropy(a2, y)
        #print(dprediction.shape, ' - dpred')
        '''
        d_out_hidden2 = self.relu_layer2.backward(dprediction)
        '''
        #print(d_out_hidden2.shape, ' - d_out_hidden2')
        d_out_hidden1 = self.hidden_layer2.backward(dprediction)
        #print(d_out_hidden1.shape, ' - d_out_hidden1')
        d_out_relu1 = self.relu_layer1.backward(d_out_hidden1)
        self.hidden_layer1.backward(d_out_relu1)

        # After that, implement l2 regularization on all params
        # Hint: use self.params()
        for param_key in params:
            reg_loss, reg_grad = l2_regularization(params[param_key].value, self.reg)
            loss += reg_loss
            #print(param_key, ' grad before ', params[param_key].grad)
            params[param_key].grad += reg_grad
            #loss += self.reg*np.sum(np.square(params[param_key].value))
            #params[param_key].grad += 2*np.array(params[param_key].value)*self.reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

		
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int) 
        z1 = self.hidden_layer1.forward(X)
        a1 = self.relu_layer1.forward(z1)
        #print(a1.shape, ' - a1')
        a2 = self.hidden_layer2.forward(a1)
        #print(a2.shape, ' - a2')
        output = self.relu_layer2.forward(a2)
        #print(output.shape, ' - output')
        pred = np.argmax(output, axis = 1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        hidden_layer_params1 = self.hidden_layer1.params()
        for param_key in hidden_layer_params1:
            result[param_key + '-1'] = hidden_layer_params1[param_key]
			
        hidden_layer_params2 = self.hidden_layer2.params()
        for param_key in hidden_layer_params2:
            result[param_key + '-2'] = hidden_layer_params2[param_key]

        return result
