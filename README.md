# Salary_Prediction_with_DecisionTree
This repository contains Python code for analyzing salary data and building a Decision Tree Regression model for predicting total pay based on various features. The code includes data preprocessing steps, handling missing values, and using scikit-learn for machine learning. Explore the code to understand how to predict salaries with Decision Trees.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Machine Learning](#machine-learning)

## Introduction

In this project, we aim to predict the total pay of employees based on different attributes, including job titles, years of experience, and more. We use a Decision Tree Regression model for this purpose.

## Installation

To run the code in this repository, you need Python and the following libraries:

- pandas
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas scikit-learn
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone <https://github.com/prashantkparth/Salary_Prediction_with_DecisionTree>
```

2. Navigate to the project directory:

```bash
cd Salary_Prediction_with_DecisionTree
```

3. Run the Jupyter Notebook or Python script to analyze the salary data and train the Decision Tree Regression model.

## Data

We used the "Salaries.csv" dataset for this project. The dataset contains information about employee salaries, job titles, and more. Make sure to update the data source if you want to use a different dataset.

## Machine Learning

We utilized scikit-learn to build a Decision Tree Regression model. The code includes data preprocessing steps, handling missing values, and encoding categorical features.


# Example code for training the model and making predictions
from sklearn.tree import DecisionTreeRegressor

# Create and train the Decision Tree Regression model
model = DecisionTreeRegressor()
model.fit(inputs_new, target)

# Make predictions
predictions = model.predict([[3]])



### Feel free to explore the code in the Jupyter Notebook or Python script for more details.








# Code Explanation



Certainly, I'll explain each part of the provided code step by step:

```python
import pandas as pd

salaries = pd.read_csv(r"C:\Users\prash\machine learning\Salaries.csv")
```
This code snippet imports the Pandas library and uses it to read a CSV file named "Salaries.csv" located at the specified file path. It loads the data into a Pandas DataFrame named `salaries`.

```python
salaries = salaries[~(salaries[salaries.columns[7:]] == 0).any(axis=1)]
salaries = salaries[~(salaries[salaries.columns[7:]] == -618.13).any(axis=1)]
```
These lines of code filter the `salaries` DataFrame by removing rows where any of the columns from index 7 onwards contain either 0 or -618.13.

```python
salaries1 = salaries.drop(['Notes', 'OvertimePay', 'OtherPay', 'Benefits', 'BasePay', 'TotalPayBenefits', 'Status', 'EmployeeName', 'Agency', 'Year', 'Id'], axis=1)
```
This code creates a new DataFrame named `salaries1` by dropping specific columns from the `salaries` DataFrame. It removes columns such as 'Notes', 'OvertimePay', 'OtherPay', 'Benefits', and more.

```python
pd.isnull(salaries1).sum()
```
This line checks for missing (NaN) values in the `salaries1` DataFrame and calculates the sum of missing values for each column.

```python
salaries1.dropna(inplace=True)
```
Here, missing values in the `salaries1` DataFrame are removed by using the `dropna` method with `inplace=True`, which means the changes are applied to the DataFrame itself.

```python
inputs = salaries1.drop(['TotalPay'], axis=1)
target = salaries1['TotalPay']
```
These lines split the data into input features (`inputs`) and the target variable (`target`). The input features include all columns except 'TotalPay,' and the target variable is 'TotalPay' itself.

```python
from sklearn.preprocessing import LabelEncoder

le_JobTitle = LabelEncoder()

inputs['JobTitle_new'] = le_JobTitle.fit_transform(inputs['JobTitle'])
```
This code snippet imports `LabelEncoder` from scikit-learn and uses it to encode the 'JobTitle' column in the `inputs` DataFrame. The encoded values are stored in a new column called 'JobTitle_new.'

```python
inputs_new = inputs.drop(['JobTitle'], axis=1)
```
This line removes the original 'JobTitle' column from the `inputs` DataFrame, leaving only the encoded 'JobTitle_new' column.

```python
from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(inputs_new, target)
```
These lines import the DecisionTreeRegressor from scikit-learn and create a Decision Tree Regression model. The model is trained on the `inputs_new` data and the `target` variable (total pay).

```python
model.score(inputs_new, target)
```
This code calculates the R-squared score of the trained Decision Tree model, indicating how well the model fits the data.

```python
model.predict([[3]])
model.predict([[293]])
```
These lines make predictions using the trained Decision Tree model. They predict the total pay for employees with certain input features. The first line predicts the total pay for an employee with an encoded 'JobTitle_new' value of 3, and the second line predicts it for a value of 293.

The code essentially preprocesses salary data, encodes categorical features, trains a Decision Tree Regression model, and makes predictions based on the model. It's a basic example of using machine learning for salary prediction.

