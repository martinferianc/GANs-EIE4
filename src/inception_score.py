import numpy as np
import tensorflow as tf

def inception_score(model, images, labels):
    predicted_labels = model.predict(images)
    accuracy = 0
    for i in range(len(predicted_labels)):
        golden_label = np.argmax(labels[i])
        predicted_label = np.argmax(predicted_labels[i])
        if golden_label == predicted_label:
            accuracy+=1
    return accuracy/len(labels)
