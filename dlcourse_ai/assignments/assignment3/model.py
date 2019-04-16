import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )
from gradient_check import check_layer_gradient, check_layer_param_gradient

class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels, reg = 0):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
        Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        image_width, image_height, n_channels = input_shape;
        padding_1 = 1;
        padding_2 = 1;
        filter_size_1 = 3;
        filter_size_2 = 3;
        pooling_size_1 = 4;
        pooling_size_2 = 4;
        stride_1 = 4;
        stride_2 = 4;
        
        height = image_height + 2*padding_1;
        width = image_width + 2*padding_1;
        
        out_height = height - filter_size_1 + 1;
        out_width = width - filter_size_1 + 1;
        #print(height, width, filter_size_1, out_height, out_width);
        
        assert (out_height - pooling_size_1)%stride_1 == 0;
        assert (out_width - pooling_size_1)%stride_1 == 0;
        
        height = out_height;
        width = out_width;
        
        out_height = int((height - pooling_size_1)/stride_1 + 1);
        out_width = int((width - pooling_size_1)/stride_1 + 1);
        #print(height, width, pooling_size_1, out_height, out_width);
        
        height = out_height + 2*padding_2;
        width = out_width + 2*padding_2;
        
        out_height = height - filter_size_2 + 1;
        out_width = width - filter_size_2 + 1;
        #print(height, width, filter_size_2, out_height, out_width);
        
        assert (out_height - pooling_size_2)%stride_2 == 0;
        assert (out_width - pooling_size_2)%stride_2 == 0;
        
        
        height = out_height;
        width = out_width;
        
        out_height = int((height - pooling_size_2)/stride_2 + 1);
        out_width = int((width - pooling_size_2)/stride_2 + 1);
        #print(height, width, pooling_size_2, out_height, out_width);
        
        # TODO Create necessary layers
        self.Conv_first = ConvolutionalLayer(n_channels, conv1_channels, filter_size_1, padding_1);
        self.Relu_first = ReLULayer();
        self.Maxpool_first = MaxPoolingLayer(pooling_size_1, stride_1);
        self.Conv_second = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size_2, padding_2);
        self.Relu_second = ReLULayer();
        self.Maxpool_second = MaxPoolingLayer(pooling_size_2, stride_2);
        self.Flattener = Flattener();
        self.FC = FullyConnectedLayer(out_height*out_width*conv2_channels, n_output_classes);
        self.n_output = n_output_classes;
        self.reg = reg;
        #print(out_height*out_width*conv2_channels, n_output_classes);

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param in self.params():
            self.params()[param].grad = np.zeros_like(self.params()[param].grad);
        
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        #assert check_layer_gradient(self.Conv_first, X);
        #assert check_layer_param_gradient(self.Conv_first, X, 'W')
        #print("X = ", X);
        #print("X_end = ", X[:, 25:, 25:, :]);
        X1 = self.Conv_first.forward(X);
        #print(self.Conv_first.W.value);
        #print("X1 = ", X1);
        #print("W = ", self.Conv_first.params()['W'].value);
        
        #assert check_layer_gradient(self.Relu_first, X1);
        X1_Relu = self.Relu_first.forward(X1);
        #print("X1_Relu = ", X1_Relu);
        
        #assert check_layer_gradient(self.Maxpool_first, X1_Relu);
        X1_Max = self.Maxpool_first.forward(X1_Relu);
        
        #assert check_layer_gradient(self.Conv_second, X1_Max);
        X2 = self.Conv_second.forward(X1_Max);
        
        #assert check_layer_gradient(self.Relu_second, X2);
        X2_Relu = self.Relu_second.forward(X2);
        
        #assert check_layer_gradient(self.Maxpool_second, X2_Relu);
        X2_Max = self.Maxpool_second.forward(X2_Relu);
        
        #assert check_layer_gradient(self.Flattener, X2_Max);
        X3 = self.Flattener.forward(X2_Max);
        
        #assert check_layer_gradient(self.FC, X3);
        X3_FC = self.FC.forward(X3);
        
        loss, dX3_FC = softmax_with_cross_entropy(X3_FC, y + 1);
        dX3 = self.FC.backward(dX3_FC);
        dX2_Max = self.Flattener.backward(dX3);
        dX2_Relu = self.Maxpool_second.backward(dX2_Max);
        #print("dX2_Max = ", dX2_Max);
        #print("dX2_Relu = ", dX2_Relu);
        
        dX2 = self.Relu_second.backward(dX2_Relu);
        dX1_Max = self.Conv_second.backward(dX2);
        dX1_Relu = self.Maxpool_first.backward(dX1_Max);
        dX1 = self.Relu_first.backward(dX1_Relu);
        
        dX = self.Conv_first.backward(dX1);
        
        reg_loss_w, reg_grad_w = l2_regularization(self.FC.W.value, self.reg);
        reg_loss_b, reg_grad_b = l2_regularization(self.FC.B.value, self.reg)

        loss += (reg_loss_w + reg_loss_b)


        self.FC.W.grad += reg_grad_w;
        
        return loss;


    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int);
        predictions = self.FC.forward(self.Flattener.forward(self.Maxpool_second.forward(self.Relu_second.forward(self.Conv_second.forward(self.Maxpool_first.forward(self.Relu_first.forward(self.Conv_first.forward(X))))))));
        #print("predictions = ", predictions);
        
        i=0;
        for predict in predictions:
            values = [softmax_with_cross_entropy(predict, target_index + 1)[0] \
                        for target_index in range(self.n_output)];
            pred[i] = min(range(len(values)), key=values.__getitem__);
            i += 1;
        #print("pred = ", pred);
        return pred;

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {}
        dict_first = self.Conv_first.params();
        dict_second = self.Conv_second.params();
        dict_FC = self.FC.params();
        
        # TODO Implement aggregating all of the params
        
        for key in dict_first.keys():
            result[key + 'C1'] = dict_first[key];
        
        for key in dict_second.keys():
            result[key + 'C2'] = dict_second[key];
        
        for key in dict_FC.keys():
            result[key + 'F1'] = dict_FC[key];
        
        return result
