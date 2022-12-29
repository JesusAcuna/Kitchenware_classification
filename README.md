# Kitchenware classification

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Kitchenware.jpg" width="750" height="400">
</p>

---
## Index

- 1.[Description of the problem](#1-description-of-the-problem)
- 2.[Objective](#2-objective)
- 3.[Data description](#3-data-description)
- 4.[Setting up the virtual environment](#4-setting-up-the-virtual-environment)
- 5.[Importing data](#5-importing-data)
- 6.[Notebook](#6-notebook)
  - 6.1.[Exploratory Data Analysis (EDA)](#61-exploratory-data-analysis-eda)
  - 6.2.[Model selection and parameter tuning](#62-model-selection-and-parameter-tuning)
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
- `requirements.txt`: archive with the dependencies and libraries
- `train.py`: python script to train the model and get `150_Stability_045011`,`4_Models_000123`,`Date_Fruit_Datasets` and `std_scaler.bin`


## 1. Description of the problem

This project is part of the 'Kitchenware Classification' competition from Kaggle, you can check it out here: 

> https://www.kaggle.com/competitions/kitchenware-classification

<p align="justify">
With the increase of data, advanced algorithms and computational power, many computer vision applications have emerged to make decisios for us with a high precision. Convolutional Neuronal Networks is an algorithm of computer vision, which take images as inputs and passes them through convolutional and dense layers to classify objects, so based on this brief introduction the problem in this case is to classify objects among 6 categories such as :
</p>

- cups
- glasses
- plates
- spoons
- forks
- knives


## 2. Objective

<p align="justify">
The aim of this project is to classify the types of kitchenware among six categories, with a high accuracy by using <b>a convolutional neuronal network model</b>.
</p>


## 3. Data description

<b>Data set source:</b>
> https://www.kaggle.com/competitions/kitchenware-classification/data

<p align="justify">
In accordance with this purpose, 5559 images for training and 3808 images for testing of six different utensils were obtained by `Toloka` , which was used for collecting the images for this competition, you can check it out here:
</p>

> https://www.youtube.com/watch?v=POGiLFWxQWQ

## 4. Setting up the virtual environment

<p align="justify">
A virtual environment allows us to manage libraries or dependencies for different projects without having version compatibility problem by creating isolated virtual environments for them. There are many environments managment systems for python such as conda, pipenv, venv, virtualenv and so on, but for this project I used pipenv. 
</p>

<p align="justify">
Next, I'll explain how to install pipenv.
Before starting , first we need to install pip, which is a package-management system to install python packages. Run these codes in the console.
</p>

> For windows:

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
  
> For linux:

    sudo apt update
    sudo apt install python3-pip
  
Then after installing pip, we need to install pipenv
    
    pip install pipenv

    
<b> IMPORTANT </b> 
 
<b>Since the tf lite library doesn't support environments, in this project I didn't include a virtual environment, instead I made a `requirements.txt` file to install the necessary libraries. You can take a look at the page : https://www.tensorflow.org/lite/guide/python </b>


<p align="justify">
Once here, it's necessary to clone this repository from Github to your computer, so create a folder in the path of your choice, and give it a name.
</p>

    mkdir ./directory_name
    cd ./directory_name
    git clone git@github.com:JesusAcuna/Kitchenware_classification.git
    cd Kitchenware_classification

    
And install the content of this file `requirements.txt`, this one contains information about the libraries and dependencies I used.

    pip install -r requirements.txt

For this project I used these libraries:
- flask          : to build the web service framework
- tflite-runtime : lite tensorflow library for prediction
- pillow         : to create image objects
- waitress       : to build a production web service on Windows
- gunicorn       : to build a production web service on Linux
- numpy          : to create arrays from image objects

## 5. Importing data

<p align="justify">
We can download the data from the web : https://www.kaggle.com/competitions/kitchenware-classification/data, this data contains a directory named `images` with all the training and testing images, two csv files, which contain the ids for training and testing images, and a csv file for a submission example.
</p>

Another way to download the data is using the APi of Kaggle, this API allows us to download the data and makes submissions from scripts.
Follow the next steps to use the API:

1.First, login to your kaggle account and got to 'Account' settings

2.Second, click on 'Create New API Token' button to generate a `kaggle.json` file

<p align="center">
  <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/kaggle-API.png">
</p>

3.Type the code below:

    # Install Kaggle
    !pip install -q kaggle
    
If you are using google colab type this:

    # Import Kaggle.json file
    import os 
    if not os.path.exists("./kaggle.json"):
      from google.colab import files
      files.upload()

If you are using another python IDE, just copy the `kaggle.json` file in your environment.
    
4. Create the kaggle folder, copy the `kaggle.json` inside the folder, and give it permissions

    # Create a Kaggle folder
    import os
    if not os.path.exists(os.path.expanduser('~/.kaggle')):
      os.mkdir(os.path.expanduser('~/.kaggle'))
      #!mkdir '~/.kaggle'
    # Copy the Kaggle.json to folder created
    !cp kaggle.json ~/.kaggle
    # Permissions
    !chmod 600 ~/.kaggle/kaggle.json

## 6. Notebook

Data preparation, data cleaning, EDA, feature importance analysis, model selection and parameter tuning was performed in `Date_Fruit_Classification.ipynb`

### 6.1. Exploratory Data Analysis (EDA)

<p align="justify"> 
From the image below, we can see that there are about 200 examples where the target variable is 'DOKOL' or 'SAFAVI'. Both classes along with 'ROTANA' are the three largest classes, on the other hand 'BERHI','DEGLET','IRAQI' and 'SOGAY' have examples lower than 100. Also, I did some analysis on some features like area distribution and I found out that three classes had almost the same distribution range, and could be replaced by one, you can analyze all the features if you would like, but the idea is the same.
</p>

<p align="center">
  <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/target_distribution.png">
</p>

### 6.2. Model selection and parameter tuning

<p align="justify"> 
For model selection, I decided to choose a deep learning model tuned with Optuna library, for more information about optuna library you can check it out for more examples with keras in https://github.com/optuna/optuna-examples/tree/main/keras 

According to the notebook `Date_Fruit_Classification.ipynb` the steps to obtain the best model are the following:

  1. The function `MakeTrial` creates a trial with optuna library and based on the parameter ranges of my model, optuna evaluates the best accuracy result of my model according to these parameters.
  2. The function `Study_Statistics` shows the parameters of the best model such as number of hidden layers, activation function, learning rate, and so on.
  3. The function `MakeNeuralNetwork` creates a bigger model in epochs of the best model obtained, this is to see if the best model went into overfitting.
  4. The function `N_Models` puts all the previous steps together and creates a number of best models, this was done since optuna trial starts randomly and I wanted to have several models to analyze instead of one.
  5. The final step is the stability test, in that part I tested the stability of four models, giving them as input different test sets of different sizes.
  
The results show that in front of 150 different test sets the best model is the third one with a best accuracy value of `0.9333`, and the architecture of this model is:
- Number of hidden layers :2 
- Layer 1 number of neurons: 352
- Layer 1 activation function: elu
- Layer 1 dropout: 0.0
- Layer 2 number of neurons: 96
- Layer 2 activation function: selu
- Layer 2 dropout: 0.02
- Learning rate: 0.001760
- beta_1: 0.075162
- beta_2: 0.093541
- epsilon: 7.775386e-07

</p>
Model 3 Architecture: 
<p align="center">
  <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/model_3_architecture.png">
</p>



