import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from k_nearest_neighbor import KNN

# Load the "BankNote_Authentication.csv" Dataset:
# -----------------------------------------------
BankNote_Authentication = pd.read_csv("BankNote_Authentication.csv")

# Preprocess The Data:
# --------------------
# i) remove records with missing values from the original "BankNote_Authentication" DataFrame directly and returns None
BankNote_Authentication.dropna(inplace=True)

# ii) separate features and target(s)
X = BankNote_Authentication.drop(columns=['class'])  # Features (variance, skew, curtosis and entropy)
y = BankNote_Authentication['class']                 # Target   (class attribute)

# iii) split the data into training and testing sets (70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# iv) normalize Training Data and Testing Data using mean and standard deviation from the training data
means = X_train.mean()  # calculate Mean for each feature column in the training data
stds = X_train.std()    # Calculate Standard Deviation for each feature column in the training data

X_train_normalized = (X_train - means) / stds  # normalize Training Data
X_test_normalized = (X_test - means) / stds    # normalize Testing Data

# Experiment with different values of k=1,2,3, …., 9
# --------------------------------------------------
for k in range(1, 20):
    # create a k-NN classifier
    knn = KNN(k=k)
    # train the model using the training set
    knn.fit(X_train_normalized.values, y_train.values)
    # predict
    y_predicted = knn.predict(X_test_normalized.values)

    correctly_classified_test_instances = np.sum(y_predicted == y_test.values)
    total_test_set_instances = len(y_test)
    accuracy = correctly_classified_test_instances / total_test_set_instances

    print("k =", k)
    print("Number of Correctly Classified Test Instances:", correctly_classified_test_instances)
    print("Total Number of Instances in The Test Set    :", total_test_set_instances)
    print("Accuracy                                     :", int(accuracy*100), "%", "\n")
