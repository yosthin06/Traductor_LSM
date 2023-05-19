"""
Description: Create the model for the Mexican Sign Language 
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
"""

# Import standar libraries
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse
import time
import pandas as pd

# Import user-defined libraries
import LSM_utils as utils

# Input the arguments and parse them
parser = argparse.ArgumentParser(description='Enter the arguments')
parser.add_argument('-e','--epochs', type=int, help='epochs for training')
parser.add_argument('-ts','--test_size', type=float, help='test size')
parser.add_argument('-tr','--train', type=int, help='retrain model')
parser.add_argument('-m','--model', type=str, help='location of the pretrained model')
args = parser.parse_args()

# Initialize the csv file path
csv_path = "data_folder/LSM_database.csv"

# Read CSV file for Training the model using Pandas
df = pd.read_csv(csv_path, header=0)

# Obtain the labels of the signs
labels = np.unique(df["Sign"])
df["Sign"] = pd.Categorical(df["Sign"])
df["Sign"] = df.Sign.cat.codes

# Copy Label and Feature for training
y = df.pop("Sign")
x = df.copy()

# Copied Features turn to Array by using NumPy
x = np.array(x)
x = np.reshape(x, (x.shape[0],1, x.shape[1]))
num_classes=len(labels)
y = to_categorical(y, num_classes)

# Split the data into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y)

# Start the timer to get how long it takes to create the model
start = time.time()

# Create the model 
model = utils.model_creation(x_train, y_train,x_test, y_test, epochs=args.epochs, train=args.train, pretrained_model=args.model)

# Print the time it takes to create the model
print("model time: {}".format(time.time()-start))

