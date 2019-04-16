import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
	
    print(np.sum(prediction == ground_truth), ' out of ', len(ground_truth))
    accuracy = np.sum(prediction == ground_truth)/len(prediction)
    return accuracy
