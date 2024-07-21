# SCS492-Selected-Topics-in-Software-Engineering-2

<div align="center">
  <table width="100%">
    <tr>
      <td colspan="2" align="center"><strong>{ Final Project: SVM - Neural Networks  - Convolutional Neural Networks }</strong></td>
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
      <td align="center" colspan="2"><strong>Delivered To:</strong><br>DR. Hanaa Mobarez<br>TA. Abdalrahman Roshdi</td>
    </tr>
  </table>
</div>

---

## Final Project

Cairo University  
Faculty of Computers and Artificial Intelligence  
Machine Learning Course   

# Machine Learning (Spring 2024)

## Project Summary

With access to the three datasets provided in the next section:

- Download, analyze and choose one of them as a team.
- Start working through the steps outlined in the requirements section.

Make sure to refer to the descriptions provided on Kaggle for each dataset before diving into the requirements.

## Dataset(s)

- Gender Classification Dataset [kaggle.com](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)
- Garbage Classification Dataset [kaggle.com](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- Bone Fracture Multi-Region X-Ray Dataset [kaggle.com](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)

## Requirements

- **Data exploration and preparation**:  
  - Reshape the RGB images, so that the dimension of each image is (64,64,3).
  - Convert the RGB images to greyscale.
  - Normalize each image.
  
- **Experiments and results**:  
  - Split the data into training and testing datasets (if there is no testing dataset).
  - **First experiment**:  
    - Train an SVM model on the grayscale images.
    - Test the model and provide the confusion matrix and the average f-1 scores for the suitable testing dataset.
  - Split the training dataset into training and validation datasets. (if there is no validation dataset)
  - **Second experiment**:  
    - Build 2 different Neural Networks (different number of hidden layers, neurons, activations, etc.)
    - Train each one of these models on the grayscale images and plot the error and accuracy curves for the training data and validation data.
    - Save the best model in a separated file, then reload it.
    - Test the best model and provide the confusion matrix and the average f-1 scores for the testing dataset.
  - **Third experiment**:  
    - Train a Convolutional Neural Network on the grayscale images and plot the error and accuracy curves for the training data and validation data.
    - Train another Convolutional Neural Network on the RGB images and plot the error and accuracy curves for the training data & validation data.
    - Save the best model in a separated file, then reload it.
    - Test the best model and provide the confusion matrix and the average f-1 score for the suitable testing dataset.
  - Compare the results of the models and suggest the best model.

## Deliverables

You are required to submit ONE zip file containing the following:
- Your code (.py) file. If you have a (.ipynb) file, you have to save/download it as (.py) before submitting.
- A report (.pdf) containing the team members' names and IDs, the dataset you chose, and the code with screenshots of the output of each part. If you have a (.ipynb) file, you can just convert it to pdf.

The zip file must follow this naming convention: ID1_ID2_ID3_ID4_ID5_ ID6_Group

## Instructions

1. The minimum number of students in a team is 3 and the maximum is 6.
2. No late submission is allowed.
3. Cheating students will take ZERO and no excuses will be accepted.
4. You can use any Python libraries.

## Grading Criteria

- Data Preparation: 2 marks
- SVM: 2 marks
- Neural Networks: 3 marks
- Convolutional Neural Networks: 3 marks

Total = 10 marks
