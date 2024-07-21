import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# a) Load the "loan_old.csv" dataset:
# -----------------------------------
loan_old = pd.read_csv("loan_old.csv")

# h) Load "loan_new.csv" dataset:
# -----------------------------
loan_new = pd.read_csv("loan_new.csv")

# b) Perform analysis on the dataset:
# -----------------------------------
# i) check whether there are missing values in "loan_old.csv" dataset
print("Missing Values in loan_old Dataset:")
print(loan_old.isnull().sum(), "\n")
#    check whether there are missing values in "loan_new.csv" dataset
print("Missing Values in loan_new Dataset:")
print(loan_new.isnull().sum(), "\n")

# ii) check the type of each column (categorical or numerical)
print("Data Types of Each Column in loan_old Dataset:")
print("[float64, int64] means numerical")
print("[object]         means categorical")
print("----------------------------------")
print(loan_old.dtypes, "\n")

# iii) check whether numerical columns have the same scale
print("Summary Statistics for the Numerical Features:")
print(loan_old.describe(), "\n")

# iv) visualize a pairplot between numerical columns
sns.pairplot(loan_old.select_dtypes(include=['int64', 'float64']))
plt.show()

# c) Preprocess the data:
# -----------------------
# i) remove records with missing values from the original "loan_old" DataFrame directly and returns None
loan_old.dropna(inplace=True)
#    remove records with missing values from the original "loan_new" DataFrame directly and returns None
loan_new.dropna(inplace=True)

# ii) separate features and targets
X = loan_old.drop(columns=['Loan_ID', 'Max_Loan_Amount', 'Loan_Status'])
X_new = loan_new.drop(columns=['Loan_ID'])
y_amount = loan_old['Max_Loan_Amount']
y_status = loan_old['Loan_Status']

# iii) shuffle and split the data into training and testing sets using train_test_split() function from scikit-learn,
#      it randomly shuffles the data before splitting it into training and testing sets
# X, y_amount, y_status = shuffle(X, y_amount, y_status)
X_train, X_test, y_amount_train, y_amount_test, y_status_train, y_status_test = train_test_split(X, y_amount, y_status, test_size=0.2)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
X_new_encoded = X_new.copy()

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# iv) encode categorical features into numerical labels
encoder = LabelEncoder()
for column in categorical_columns:
    X_train_encoded[column] = encoder.fit_transform(X_train_encoded[column])
    X_test_encoded[column] = encoder.transform(X_test_encoded[column])
    X_new_encoded[column] = encoder.transform(X_new_encoded[column])

# v ) encode categorical targets into numerical labels
y_status_train = encoder.fit_transform(y_status_train)
y_status_test = encoder.transform(y_status_test)

# vi) standardize the features using the mean and standard deviation
mean_values = X_train_encoded[numerical_columns].mean()
std_values = X_train_encoded[numerical_columns].std()

X_train_encoded[numerical_columns] = (X_train_encoded[numerical_columns] - mean_values) / std_values
X_test_encoded[numerical_columns] = (X_test_encoded[numerical_columns] - mean_values) / std_values
X_new_encoded[numerical_columns] = (X_new_encoded[numerical_columns] - mean_values) / std_values

# d) Fit linear regression model:
# -------------------------------
linear_regression = LinearRegression()
linear_regression.fit(X_train_encoded, y_amount_train)

# e) Evaluate linear regression model:
# ------------------------------------
y_amount_predicted = linear_regression.predict(X_test_encoded)
r2 = r2_score(y_amount_test, y_amount_predicted)
print("R^2 Score of the Linear Regression Model: ", r2)

# f) Fit logistic regression model from scratch:
# ----------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_GD(X, y):
    m, n = X.shape

    # initialize theta array with ones and size n+1 (theta.T)
    theta = np.zeros(n + 1)
    # add a column of ones to X
    X = np.column_stack((np.ones(m), X))

    # set alpha and max_iterations
    alpha = 0.01
    # set max_iterations
    max_iterations = 1000

    # iterate over max_iterations
    for i in range(max_iterations):
        h = sigmoid(np.dot(X, theta))
        # compute the partial derivative of the error: ∂J(θ)/∂θj=(1/m) * X.T * (h - y)
        partial_derivative = (1 / m) * np.dot(X.T, (h - y))
        # update  theta_j according to the equation: θ = θ - α * ∂J(θ)/∂θ
        theta -= alpha * partial_derivative

    return theta

def predict(X, theta):
    return sigmoid(np.dot(X, theta))

X_train_logistic = X_train_encoded.copy()
theta = fit_GD(X_train_logistic.values, y_status_train)

# g) Function (from scratch) to calculate the accuracy of the logistic regression model:
# --------------------------------------------------------------------------------------
def accuracy(actual, predicted):
    total_samples = len(actual)
    correct_predictions = 0
    for i in range(total_samples):
        if predicted[i] >= 0.5:
            if actual[i] == 1:
                correct_predictions += 1
        else:
            if actual[i] == 0:
                correct_predictions += 1
    return correct_predictions / total_samples

# Calculate the accuracy of the model
X_test_logistic = X_test_encoded.copy()
X_test_logistic = np.column_stack((np.ones(X_test_logistic.shape[0]), X_test_logistic))
y_status_pred = predict(X_test_logistic, theta)
model_accuracy = accuracy(y_status_test, y_status_pred)
print("Accuracy of the Logistic Regression Model: ", model_accuracy, "\n")

# j) Predict using models:
# ------------------------
y_amount_new = linear_regression.predict(X_new_encoded)
X_predict_logistic = X_new_encoded.copy()
X_predict_logistic = np.column_stack((np.ones(X_predict_logistic.shape[0]), X_predict_logistic))
y_status_new = predict(X_predict_logistic, theta)
#y_status_new = predict(np.hstack((X_new_encoded.values, np.ones((X_new_encoded.shape[0], 1)))), theta)

# Print predictions for loan amounts and loan status for new data
print("Predicted loan details for new data:")
print("--------------------------------------")
for idx, (loan_id, amount, status) in enumerate(zip(loan_new['Loan_ID'], y_amount_new, y_status_new)):
    print(f"  - Loan ID: {loan_id}")
    print(f"  - Predicted Loan Amount: {amount:.2f}$")
    print(f"  - Predicted Loan Status: {'Approved (Y)' if status >= 0.5 else 'Rejected (N)'}")
    print("--------------------------------------")
