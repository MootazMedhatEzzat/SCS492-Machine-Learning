import pandas as pd
import matplotlib.pyplot as plt

from decision_tree_experiment import Experiment

# Load the "BankNote_Authentication.csv" Dataset:
# -----------------------------------------------
BankNote_Authentication = pd.read_csv("BankNote_Authentication.csv")

# Preprocess The Data:
# --------------------
# i) remove records with missing values from the original "BankNote_Authentication" DataFrame directly and returns None
BankNote_Authentication.dropna(inplace=True)

# ii) separate features and target(s)
X = BankNote_Authentication.drop(columns=['class'])  # Features
y = BankNote_Authentication['class']                 # Target

# Range of train_test Split Ratio [ (30%-70%), (40%-60%), (50%-50%), (60%-40%) and (70%-30%) ]:
# ---------------------------------------------------------------------------------------------
training_set_sizes = [30, 40, 50, 60, 70]

# Experiment With a Fixed train_test Split Ratio (25% For Training and 75% For Testing):
# --------------------------------------------------------------------------------------
experiment_1 = Experiment(X, y)
experiment_1_results = experiment_1.run_experiment(fixed=True)

print("----------------------------------------------------------------------------------")
print("| > Experiment with a fixed train_test split ratio (25% training and 75% test) < |")
print("|--------------------------------------------------------------------------------|")
for experiment_number in range(len(experiment_1_results)):
    print("| Experiment:", experiment_number+1, "                                                                 |")
    print("|---------------------------------------------------------------------------------")
    print("| Accuracy of The Tree                    : ", experiment_1_results[experiment_number][0])
    print("| Tree Depth                              : ", experiment_1_results[experiment_number][1])
    print("| Size of The Tree (Total Number of Nodes): ", experiment_1_results[experiment_number][2])
    print("| Number of Leaf Nodes                    : ", experiment_1_results[experiment_number][3])
    print("|---------------------------------------------------------------------------------")

# Experiment With Different Range of train_test Split Ratio:
# ----------------------------------------------------------
experiment_2 = Experiment(X, y)
experiment_2_results = experiment_2.run_experiment(training_set_sizes)
print()
print("---------------------------------------------------------------"
      "---------------------------------------------------------------")
print("| > Experiment with different range of train_test split ratio "
      "[ (30%-70%), (40%-60%), (50%-50%), (60%-40%) and (70%-30%) ] < |")
print("|--------------------------------------------------------------"
      "--------------------------------------------------------------|")
for i in range(len(experiment_2_results)):
    split_ratio = training_set_sizes[i]
    print("| Train-Test Split Ratio:(", split_ratio, "%", "-", 100-split_ratio, "%", ")",
          "                                                                                    |")
    print("|----------------------------------------------------------------------------------------"
          "-------------------------------------")
    print("| Mean Accuracy                   : ", experiment_2_results[i][0])
    print("| Max Accuracy                    : ", experiment_2_results[i][1])
    print("| Min Accuracy                    : ", experiment_2_results[i][2])
    print("| Mean Tree Size                  : ", experiment_2_results[i][3])
    print("| Max Tree Size                   : ", experiment_2_results[i][4])
    print("| Min Tree Size                   : ", experiment_2_results[i][5])
    print("|----------------------------------------------------------------------------------------"
          "-------------------------------------")

# Plotting The Mean Accuracy Against Training Set Size:
# -----------------------------------------------------
plt.figure(figsize=(10, 5))
plt.grid(True)
mean_accuracies = [result[0] for result in experiment_2_results]
plt.plot(training_set_sizes, mean_accuracies, marker='o')
plt.title('Mean Accuracy vs Training Set Size')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Mean Accuracy')
plt.show()

# Plotting The Mean Number of Nodes in The Final Tree Against Training Set Size:
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.grid(True)
mean_tree_sizes = [result[3] for result in experiment_2_results]
plt.plot(training_set_sizes, mean_tree_sizes, marker='o')
plt.title('Mean Number of Nodes in Final Tree vs Training Set Size')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Mean Number of Nodes in The Final Tree')
plt.show()
