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

- `Kitchenware_data`: directory with 4 subdirectories: 
    - `Full_train`: Image training set
    - `Test`      : Image testing set
    
    By modifying the train size of `Full_train`, you can get a small set for a fast training 
    
    - `Train`     : Small image training set from `Full_train`
    - `Val`       : Small image testing set from `Full_train`
- `data`: directory of image dataset
- `images`: directory of images to README.md
- `static`: directory of css, js files and images for frontend 
- `templates`: directory of html files for frontend 
- `Dockerfile`: archive to containerize the project
- `app.py`: python script to make the web service for classification with `Flask`  <b>FRONTEND</b>
- `best_model.h5`: file of best chosen model 
- `best_model.tflite`: file with extension tensorflow lite of best chosen model
- `convert_to_tflite.py`: python script to convert a 'h5' file to 'tfile' file
- `model.py`: python script to enter image, do normalization and prediction <b>BACKEND</b>
- `requirements.txt`: file with dependencies and libraries
- `train.py`: python script to tune parameters and train a best model, from this script you get the directory `Kitchenware_data` and the file `best_model.h5`
- `Kitchenware_Classificaion.ipynb`: python notebook where the analysis and modeling is done

## 1. Description of the problem

This project is part of the 'Kitchenware Classification' competition from Kaggle, you can check it out here: 

> https://www.kaggle.com/competitions/kitchenware-classification

<p align="justify">
With the increase of data, advanced algorithms and computational power, many computer vision applications have emerged to make decisions for us with a high precision. Convolutional Neuronal Networks is an algorithm of computer vision, which take images as inputs and passes them through convolutional and dense layers to classify objects, so based on this brief introduction, the problem in this case is to classify objects among 6 categories such as :
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
We can download the data from the web : https://www.kaggle.com/competitions/kitchenware-classification/data, this URL contains a directory called `images` with all the images for training and testing, two csv files `train.csv` and `test.csv`, which contain the 'ids' of the images for training and testing respectively , and a csv file `sample_submission.csv` for submission.
</p>

Another way to download the data is using the APi of Kaggle, this API allows us to download the data and makes submission of our trained model.
Follow the next steps to use the API:

1.First, login to your Kaggle account and go to 'Account' settings

2.Second, click on 'Create New API Token' button to generate a `kaggle.json` file

<p align="center">
  <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/kaggle-API.png">
</p>

3.Type the code below:

    # Install Kaggle
    !pip install -q kaggle
    
If you are using google colab type this to upload the `kaggle.json` file:

    # Import Kaggle.json file
    import os 
    if not os.path.exists("./kaggle.json"):
      from google.colab import files
      files.upload()

If you are using another python IDE, just copy the `kaggle.json` file in your environment.
    
4.Create a kaggle folder, copy the `kaggle.json` inside the folder, and give it permissions

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

Exploratory data analysis(EDA), model selection and parameter tuning was performed in `Kitchenware_Classificaion.ipynb`

### 6.1. Exploratory Data Analysis (EDA)

<p align="justify"> 
From the image below, we can see that there are over 1100 images where the target variable is 'cup' or 'plate'. Both classes along with 'spoon' are the three largest classes, on the other hand 'glass','knife', and 'fork' have images lower than 900. Also, we notice that 'fork' class has about 600 images, this could generate problems to generalize this class later, but for the moment I'm not going to duplicate images from this class, instead of it I will use data augmentation.
</p>

<p align="center">
  <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/target_distribution.png">
</p>

### 6.2. Model selection and parameter tuning

<p align="justify"> 
For model selection, I decided to choose a convolutional neuronal network, to do this I used a trained model, which is 'Xception' with 'imagenet' weights, with 299x299 pixel images and I just taken the convolutional part, then for for the multilayer part, I used Optuna library, which was used for tunning parameters for the last part of this architecture.

For more information about optuna library you can check it out for more examples with keras in https://github.com/optuna/optuna-examples/tree/main/keras 

According to the notebook `Kitchenware_Classificaion.ipynb` the steps to obtain the best model are the following:
  
  1. The function `Making_Directory` creates a directory `Kitchenware_data`, which contains two subdirectories: `Full_Train` and `Test` for training and testing respectively and other two subdirectories with lower images : `Train` and `Val` for doing a quick fit.
  2. The function `MakeTrial` creates a trial with optuna library and based on the parameter ranges of my model, optuna evaluates the best accuracy result of my model according to these parameters.
  3. The function `Study_Statistics` shows the parameters of the best model such as number of hidden layers, activation function, learning rate, and so on.
  4. The function `MakeCNN` creates a bigger model in epochs of the best model obtained, this is to see if the best model went into overfitting.
</p>

The results of the best model after 20 epochs, with an accuracy of `0.9703237414360046`.
- Number of hidden layers : 1
- Layer 1 number of neurons: 480 
- Layer 1 activation function: relu
- Layer 1 dropout: 0.0
- Learning rate: 0.007164294777909971
- Momentum: 0.8281463565323526
- Random state: 563


<b> Model Architecture: </b>

<p align="center">
  <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/model_architecture.png">
</p>

<b> Model History: </b>

<p align="center">
  <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/model_history.png">
</p>

## 7. Instructions on how to run the project

Steps:
   
  1. Do this part 4. [Setting up the virtual environment](#4-setting-up-the-virtual-environment), which is for setting up the environment.
   
  2. (Optional) Run the file `train.py`, this file will allow you to obtain a best model, it's configured to do a search for best parameters for 5 hours and a training with larger epochs. You can check the logs from the fit inside the file `Kitchenware_Classificaion.ipynb` to see how it was fitted. For that parameters tunning I used a virtual machine on Google Cloud with these features: a v8CPU and a NVIDIA V100. The output of `train.py` are  the directory `Kitchenware_data` and the `best_nodel.h5`, which contains all the parameters of the best model I trained. I'll   put them inside the repository to be able to do the next step.
   
  3. (Optional) Run the file `converter_to_tflite.py` to convert the model `best_model.h5` to `best_model.tflite`, since the tensorFlow library is big and we need to use a tensorFlow lite library, which is a lighter library to predict. The file is already uploaded, so you don't need to do this step.
    
  4. Run the file `app.py` to run the web service locally.
  
    winpty python app.py

  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/local_app.png">
  </p>
  
  4. This is the <b>frontend</b> of the application, I based myself on Sachin Pal's code to make the frontend, you can chek it out here:

 > https://github.com/Sachin-crypto/Flask_Image_Recognition
  
  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/frontend_app.png">
  </p>
  
  5. You can make predictions choosing a kitchenware image

  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/choose-image.png">
  </p>
  
  6. And press the button submit to predict

  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/prediction-image.png">
  </p>

## 8. Locally deployment 

Dockerfile contains all the specifications to build a container: python version, virtual environment, dependencies, scripts ,files, and so on. To do the cloud deployment we first need to configure locally and conteinerize it with Docker

Steps:

  1. Install docker: https://www.docker.com/, and if you're using WSL2 for running Linux as subsytem on Windows activate WSL Integration in Settings/Resources/WSL Integration.
  2. Open the console and locate in the repository where is the `Dockerfile` , if your using Windows there won't be any problem, but if you're using Linux
change two things in `Dockerfile`:

> first after the line `RUN pip install -r requirements.txt`:

     RUN pip install gunicorn  
  
> and second, change the entrypoint for this:
    
    ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9696","app:app"]
  
  3. Build the docker and enter this command:

    docker build -t kitchenware_classification .
  
  4. Once you build the container you can chek all the images you created running this command:  
  
    docker images
    
  5. (Optional) You can check the files that are in created docker running this command:

    docker exec -it kitchenware_classification bash
    
  6. Run the docker entering this command:
  
  > Windows
  
    winpty docker run -it --rm -p 9696:9696 kitchenware_classification:latest

  > Linux
  
    docker run -it --rm -p 9696:9696 kitchenware_classification:latest

  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/docker.png">
  </p>
  
## 9. Google Cloud deployment (GCP)

Steps:

  1. Create a Google Cloud Platform (GCP) account https://cloud.google.com/
  
  2. Install the gcloud CLI, you can follow the instrucctions here https://cloud.google.com/sdk/docs/install ,this is to be able to use gcloud console commands 
  
  3. Create a project:
    
    gcloud projects create Kitchenware-classification

  4. To see all the projects you've created run the following:
  
    gcloud projects list 
    
  5. To select a project:
  
    gcloud config set project Kitchenware-classification
    
    # To see what is the active project 
    gcloud config list project
    
  6. Create a tag to the image
  
    docker tag kitchenware_classification:latest gcr.io/kitchenware-classification/kitchenware-classification:latest
    
  7. Activate Google Container Registry API 

    gcloud services enable containerregistry.googleapis.com
    
  8. To configure docker authentication run, this is for the next step : 
  
    gcloud auth configure-docker

  9. Push the  image to Container Registry 
  
    docker push gcr.io/kitchenware-classification/kitchenware-classification:latest
    
  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/container_registry.png">
   </p>    
   
  10. Deploy the image
   
    gcloud run deploy kitchenware-classification --image gcr.io/kitchenware-classification/kitchenware-classification:latest --port 9696 --max-instances 15 --platform managed --region us-central1 --allow-unauthenticated --memory 1Gi
    
  <p align="center">
    <img src="https://github.com/JesusAcuna/Kitchenware_classification/blob/master/images/google_cloud.png">
   </p>
    
  For more information on how to deploy : https://cloud.google.com/sdk/gcloud/reference/run/deploy
  
    #To delete a service
    gcloud run services delete kitchenware-classification --region us-central1

  11. The web service was available on https://kitchenware-classification-oezkcy27ia-uc.a.run.app, I'll make a video how it works.
    

  https://user-images.githubusercontent.com/57125377/209990738-7818f06f-085a-4d6e-a7fe-d9e1a5c5eb40.mp4


  12. All the previous steps can be done within the interface offered by GCP
  
## 10. References


 Google cloud reference documentation
 
 https://cloud.google.com/sdk/gcloud/reference
    
 Docker run reference
 
 https://docs.docker.com/engine/reference/run/
    
 Flask micro web framework
 
 https://flask.palletsprojects.com/en/2.2.x/
 
 Flask Image Recognition
 
 https://github.com/Sachin-crypto/Flask_Image_Recognition
 
 Dataset Kaggle
 
 https://www.kaggle.com/competitions/kitchenware-classification/data
 
 Optuna library
 
 https://optuna.readthedocs.io/en/stable/index.html
 
 


