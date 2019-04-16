import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = np.sum(np.logical_and(prediction, ground_truth))
    fp = np.sum(ground_truth[np.logical_not(prediction)])
    fn = np.sum(np.logical_not(prediction[ground_truth]))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = (2*precision*recall)/(precision + recall)
    accuracy = np.sum(np.logical_not(np.logical_xor(prediction, ground_truth)))/len(prediction)
    return accuracy, precision, recall, f1


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
    print(np.sum(prediction == ground_truth))
    accuracy = np.sum(prediction == ground_truth)/len(prediction)
    return accuracy