# Ames Housing Dataset - House Price Prediction

A machine learning project using **ridge regression** to predict house prices in the Ames City of Iowa. This project focuses on building an end-to-end pipeline that automates data preprocessing, feature engineering, model training, and delivering accurate predictions.

## Project Structure
This repository contains three folders and a README.md file. 

data

- data_description.text --> description of all the variables in the dataset

- train.csv             --> training set

- test.csv              --> testing set

notebook

- EDA.ipynb                                  --> detailed exploration of the training set with visualizations

- Feature Engineering.ipynb                  --> handling of missing values and outliers; encoding and transformation of variables

- Model Training and Pipeline Building.ipynb --> model selection, tuning, and building complete model pipeline

- Predictions.ipynb                          --> loading model pipeline to make predictions on testing data

module

- imputation_and_feature_create.py --> custom transformers to integrate into the model pipeline

## Model Information
Model - Ridge Regression

Optimum alpha ($\alpha$) - 13.8489

R2 Score - 91.93%
