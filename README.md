# Kitchenwareclassification

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Kitchenware.jpg" width="900" height="350">
</p>

---
## Index

- 1.[Description of the problem](#1-description-of-the-problem)
- 2.[Objective](#2-objective)
- 3.[Data description](#3-data-description)
- 4.[Setting up the virtual environment](#4-setting-up-the-virtual-environment)
- 5.[Importing data](#5-importing-data)
- 6.[Notebook](#6-notebook)
  - 6.1.[Exploratory Data Analysis (EDA)](#62-exploratory-data-analysis-eda)
  - 6.2.[Model selection and parameter tuning](#64-model-selection-and-parameter-tuning)
- 7.[Instructions on how to run the project](#7-instructions-on-how-to-run-the-project)
- 8.[Locally deployment](#8-locally-deployment)
- 9.[Google Cloud deployment (GCP)](#9-google-cloud-deployment-gcp)
- 10.[References](#10-references)
---
## Structure of the repository

The repository contains the next files and folders:

- `150_Stability_045011`: directory of the stability with different test sizes of the 4 trained models 
- `4_Models_000123`: directory of models history and parameters
- `Date_Fruit_Datasets`: directory of the data set
- `images`: directory with images to README.md
- `Best_Model_3.h5`: archive of best chosen model 
- `Best_Model_3.tflite`: archive with extension tensorflow lite of best chosen model
- `Date_Fruit_Classification.ipynb`: python notebook where the analysis and modeling is done
- `Dockerfile`: archive to containerize the project
- `convert_to_tflite.py`: python script to convert a h5 file to tfile file
- `predict.py`: python script to make the web service with method 'POST' and upload the parameters of `Best_Model_3.tflite`
- `predict_test.py`: python script to make a request locally
- `predict_test_cloud.py`: python script to make a request on Google Cloud Platform (GCP)
- `requirements.txt`: archive with the dependencies and libraries
- `std_scaler.bin`: binary archive with the training normalization values 
- `train.py`: python script to train the model and get `150_Stability_045011`,`4_Models_000123`,`Date_Fruit_Datasets` and `std_scaler.bin`

## 1. Description of the problem

<p align="justify">
This project is part of the Kitchenware Classification from Kaggle, you can check it out here: 



-
-
-
-
-
-


</p>
