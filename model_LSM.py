"""
Description: Create the model for the Mexican Sign Language 
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
"""

# Import standar libraries
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import time

# Import user-defined libraries
import LSM_utils as utils

# Input the arguments and parse them
parser = argparse.ArgumentParser(description='Enter the arguments')
parser.add_argument('-e','--epochs', type=int, help='epochs for training')
parser.add_argument('-ts','--test_size', type=float, help='test size')
parser.add_argument('-tr','--train', type=int, help='retrain model')
parser.add_argument('-m','--model', type=str, help='location of the pretrained model')
args = parser.parse_args()

# Import the arrays of keypoints sequences and labels
labels = np.load("../saved_data/labels_2023-03-13 11:50:04.132165.npy")
sequences = np.load("../saved_data/keypoints_2023-03-13 11:50:04.132003.npy")
print("sequences: {}, labels: {}".format(sequences.shape,labels.shape))

# Split the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=args.test_size, stratify=labels)
print("X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(X_train.shape,X_test.shape, y_train.shape,y_test.shape))
# Start the timer to get how long it takes to create the model
start = time.time()

# Create the model 
model = utils.model_creation(labels.shape[1], sequences, labels,X_test, y_test, epochs=args.epochs, train=args.train, pretrained_model=args.model)

# Print the time it takes to create the model
print("model time: {}".format(time.time()-start))

