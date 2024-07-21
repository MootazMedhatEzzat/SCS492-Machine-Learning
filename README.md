# SCS492-Selected-Topics-in-Software-Engineering-2

![image](https://github.com/user-attachments/assets/e106d083-19da-4296-96e5-da13c6187a7b)
---

<div align="center">
  <table width="100%">
    <tr>
      <td colspan="2" align="center"><strong>{ Assignment 1: Linear and Logistic Regression }</strong></td>
    </tr>
    <tr>
      <td align="left"><strong>Name</strong>: Mootaz Medhat Ezzat Abdelwahab</td>
      <td align="right"><strong>Id</strong>: 20206074</td>
    </tr>
    <tr>
      <td align="left"><strong>Program</strong>: Software Engineering</td>
      <td align="right"><strong>Group</strong>: B (S5)</td>
    </tr>
    <tr>
      <td align="center" colspan="2"><strong>Delivered To</strong>: DR. Hanaa Mobarez</td>
    </tr>
    <tr>
      <td align="center" colspan="2"><strong>Delivered To</strong>: TA. Abdalrahman Roshdi</td>
    </tr>
  </table>
</div>

---

## Assignment 1

Cairo University  
Faculty of Computers and Artificial Intelligence  
Machine Learning Course  

### Assignment 1: Linear and Logistic Regression

A housing finance company offers interest-free home loans to customers. When a customer applies for a home loan, the company validates the customer's eligibility for a loan before making a decision.

Now, the company wants to automate the customers' eligibility validation process based on the customers' details provided while filling the application form. These details include gender, education, income, credit history, and others. The company also wants to have a predictive model for the maximum loan amount that an applicant is authorized to borrow based on his details.

You are required to build a linear regression model and a logistic regression model for this company to predict loan decisions and amounts based on some features.

### Datasets

There are two attached datasets:

- **loan_old.csv**: Contains 614 records of applicants' data with 10 feature columns in addition to 2 target columns. The features are:
  - Loan application ID
  - Applicant's gender
  - Marital status
  - Number of dependents
  - Education and income
  - Co-applicant's income
  - Number of months until the loan is due
  - Applicant's credit history check
  - Property area

  The targets are:
  - Maximum loan amount (in thousands)
  - Loan acceptance status

- **loan_new.csv**: Contains 367 records of new applicants' data with the 10 feature columns.

Note: These datasets are modified versions of the "Loan Eligibility Dataset". The original datasets were obtained from Kaggle.

### Requirements

Write a Python program in which you do the following:

a) Load the "loan_old.csv" dataset.

b) Perform analysis on the dataset to:
   1. Check whether there are missing values
   2. Check the type of each feature (categorical or numerical)
   3. Check whether numerical features have the same scale
   4. Visualize a pairplot between numerical columns

c) Preprocess the data such that:
   1. Records containing missing values are removed
   2. The features and targets are separated
   3. The data is shuffled and split into training and testing sets
   4. Categorical features are encoded
   5. Categorical targets are encoded
   6. Numerical features are standardized

d) Fit a linear regression model to the data to predict the loan amount.
   - Use sklearn's linear regression.

e) Evaluate the linear regression model using sklearn's R2 score.

f) Fit a logistic regression model to the data to predict the loan status.
   - Implement logistic regression from scratch using gradient descent.

g) Write a function (from scratch) to calculate the accuracy of the model.

h) Load the "loan_new.csv" dataset.

i) Perform the same preprocessing on it (except shuffling and splitting).

j) Use your models on this data to predict the loan amounts and status.

### Remarks

- You can use functions from data analysis and computing libraries (e.g., Pandas and NumPy) as you please throughout the entire code.
- You can use machine learning libraries such as Scikit-learn for preprocessing and metrics but NOT for "from scratch" requirements.
- The train/test split has to be performed before the encoding and standardization steps.
- The categorical features of the test set (and of the new data) should be transformed (encoded) using the encoder fitted on the train set.
- The numerical features of the test set (and of the new data) should be standardized using the mean and standard deviation of the train set.
- We will use R2 score to evaluate the linear regression model as it provides a measure of how well observed outcomes are replicated by the model (based on the proportion of total variation of outcomes explained by the model). The best possible score is 1, but the score can be negative as the model can be arbitrarily worse.

### Deliverables

You are required to submit ONE zip file containing the following:
- Your code (.py) file. If you have a (.ipynb) file, you have to save/download it as (.py) before submitting.
- A report (.pdf) containing the team members' names and IDs, and the code with screenshots of the output of each part. If you have a (.ipynb) file, you can just convert it to pdf.

The zip file must follow this naming convention: ID1_ID2_ID3_ID4_ID5_Group

### Submission Remarks

- The maximum number of students in a team is 5 and the minimum is 4.
- Team members must be from the same lab (or have the same TA).
- No late submission is allowed.
- A penalty will be imposed for violating any of the assignment rules.
- Cheaters will get ZERO and no excuses will be accepted.

### Grading Criteria

Both the code and the report must include:
- Analysis: 2 marks
- Preprocessing: 6 marks
- Linear regression and R2 score: 2 marks
- Logistic regression (gradient descent): 6 marks
- Accuracy: 2 marks
- New predictions: 2 marks

The total is 20 marks (will be scaled to 5 marks).
