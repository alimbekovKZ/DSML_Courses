def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return sum([int(prediction[i] == ground_truth[i]) for i in range(len(prediction))])/len(prediction);

