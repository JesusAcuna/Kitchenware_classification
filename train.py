#Kitchenware Classification.ipynb
import numpy as np
import pandas as pd 

import os 
import shutil

from sklearn.model_selection import train_test_split


df_train = pd.read_csv('data/train.csv', dtype={'Id': str})
df_train['filename'] = 'data/images/' + df_train['Id'] + '.jpg'


"""# Making directories"""

def Making_Directory(ProjectDirectory,
                     Directories,
                     DataFrame,
                     FullTrainSize,
                     RandomState):    
        
    df_Full_Train, df_Test = train_test_split(DataFrame,
                                              train_size = FullTrainSize,              
                                              random_state = RandomState,
                                              stratify = DataFrame['label']) 
    
    df_Train, df_Val = train_test_split(df_Full_Train,
                                        train_size = 0.75,
                                        random_state = RandomState,            
                                        stratify = df_Full_Train['label']) 
    
    df_Tuple = (df_Full_Train, df_Test, df_Train, df_Val)

    df_Dictionary = dict(zip(Directories, df_Tuple))

    CategoryArray = np.unique(DataFrame['label']) 

    # Directory   
    if not os.path.exists(ProjectDirectory):
        os.mkdir(ProjectDirectory) 

    # SubDirectory
    for k, v in df_Dictionary.items():

      if not os.path.exists(f"{ProjectDirectory}/{k}"):
        os.mkdir(f"{ProjectDirectory}/{k}")

      for j in CategoryArray:

        if not os.path.exists(f"{ProjectDirectory}/{k}/{j}"):
          os.mkdir(f"{ProjectDirectory}/{k}/{j}")

        df_Category = v[v['label']==j]

        for i in df_Category['filename'].to_list():
          shutil.copy(i, f"{ProjectDirectory}/{k}/{j}")

    return df_Full_Train, df_Test, df_Train, df_Val


import tensorflow as tf
from tensorflow import keras

import keras.applications

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input,GlobalAveragePooling2D,Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import SGD


# optuna library
#!pip install --quiet optuna
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

"""# Model

"""

""" Define objective function for the structure of the CNN """

def create_model(trial):

    BaseModel = tf.keras.applications.Xception(weights = "imagenet",
                                              input_shape=(299, 299, 3),
                                              include_top = False)
    BaseModel.trainable = False
    inputs = Input(shape=(299, 299, 3))
    x = BaseModel(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    n_layers = trial.suggest_int("n_layers", 1, 6,step=1)                 # Number of hidden layers
    # Number of neurons and activation function of each Hidden layer
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_L{i}',                   # Number of neurons for the ith hidden layer
                                       32, 512, step = 32)      
        # Activation function for the ith hidden layer
        func_activation = trial.suggest_categorical( f'f_activation_L{i}',
                                                    ['relu','sigmoid','tanh','selu','elu'])
        x = Dense(units=num_hidden,
                        activation=func_activation)(x)
        # Dropout Layer
        dropout = trial.suggest_categorical(f'dropout_L{i}',              
                                            [0.0,0.2,0.4,0.6,0.8])
        x = Dropout(rate=dropout)(x)

    # Output layer (Number of dependent variables)
    outputs = Dense(units=6, 
                    activation = "softmax")(x)                            # Activation function for the output layer

    model = Model(inputs, outputs)

    # Compile the model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate",                  # Range of learning rate values
                                        1e-5, 1e-2,
                                        log=True) 
    momentum = trial.suggest_float("momentum",
                                    1e-2, 1e0, log=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=SGD(learning_rate = learning_rate,
                                momentum=momentum),
                  metrics="accuracy")

    return model

"""  Define objective function for Optuna """

def objective(trial):
    
    # Generate our trial model
    model = create_model(trial)

    ####
    if os.path.exists("Kitchenware_data"):
      shutil.rmtree('Kitchenware_data')

    Dir_Data = "Kitchenware_data"
    Dir = ("Full_Train", "Test", "Train", "Val")

    Full_Train_Size = 0.8
    Random_State = trial.suggest_int("random_state", 1, 2000, step=1)             

    df_Full_Train, df_Test, df_Train, df_Val = Making_Directory(ProjectDirectory= Dir_Data,
                                                                Directories = Dir,
                                                                DataFrame = df_train,
                                                                FullTrainSize = Full_Train_Size,
                                                                RandomState = Random_State)
    ####
    TamImagen =  299
    batch_size = 32

    train_data_gen = ImageDataGenerator(zoom_range =   [0.80, 1.0],
                                        brightness_range =[0.8,1.2],
                                        vertical_flip = True,  
                                        horizontal_flip = True, 
                                        #shear_range = 30,
                                        #rotation_range = 90,                                
                                        preprocessing_function=keras.applications.xception.preprocess_input)

    test_data_gen = ImageDataGenerator(preprocessing_function=keras.applications.xception.preprocess_input)

    train_generator = train_data_gen.flow_from_directory("./Kitchenware_data/Full_Train", 
                                                        (TamImagen, TamImagen),
                                                        batch_size = batch_size,
                                                        class_mode = "categorical")

    valid_generator = test_data_gen.flow_from_directory("./Kitchenware_data/Test", 
                                                        (TamImagen, TamImagen),
                                                        batch_size = batch_size,
                                                        class_mode = "categorical")

    ####
    model.fit(train_generator,
              steps_per_epoch = train_generator.n//batch_size,
              validation_data = valid_generator,
              validation_steps = valid_generator.n//batch_size,
              epochs= trial.suggest_categorical("epochs", [20]), 
              callbacks=[TFKerasPruningCallback(trial, "val_accuracy")],
              workers=8,
              verbose = 1)

    # Evaluate the model accuracy on the validation set in each trial
    score = model.evaluate(valid_generator, verbose=0)
    
    return score[1]

def MakeTrial(study_name, n_trials=None,timeout=None):
  
  print(f"\nStarting Trial: {study_name}\n")

  study = optuna.create_study(study_name=study_name,
                              direction="maximize",
                              pruner=optuna.pruners.MedianPruner())
  
  # Set 'n_trials' and/or 'timeout' in seconds for optimization 
  study.optimize(objective, n_trials=n_trials, timeout=timeout)
  pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

  print(f"\nTrial Completed: {study_name}\n")

  return study, pruned_trials, complete_trials

study_name = f"Kitchenware_CNN" 
# Set 'n_trials' and/or 'timeout' in seconds for optimization 
study, pruned_trials, complete_trials = MakeTrial(study_name, timeout=18000 )


# This callback allows to get the best model when a model is being fitted
from keras.callbacks import ModelCheckpoint

# This function is similar to 'create_model(trial)' function 
# Also this function accepts as arguments the previous study 
def MakeCNN(study,
            verbose):
  #
  BaseModel = tf.keras.applications.Xception(weights = "imagenet",
                                            input_shape=(299, 299, 3),
                                            include_top = False)
  BaseModel.trainable = False
  inputs = Input(shape=(299, 299, 3))
  x = BaseModel(inputs, training=False)
  x = GlobalAveragePooling2D()(x)

  for i in range(study.best_trial.params['n_layers']):
    x = Dense(# Number of neurons for the ith hidden layer
              units= study.best_trial.params[f'n_units_L{i}'],
              # Activation function
              activation=study.best_trial.params[f'f_activation_L{i}'],
              #  We can add an alias to the ith hidden layer
              name= f'{i+1}HiddenLayer_'+str(study.best_trial.params[f'n_units_L{i}'])+'Neurons')(x)
    
    x = Dropout(rate= study.best_trial.params[f'dropout_L{i}'],        # Dropout layer rate
                name=f"{i+1}Dropout")(x)

  # At the end I add one last layer: Output Layer

  outputs = Dense(units=6, 
                  activation = "softmax")(x)                           # Activation function for the output layer

  model = Model(inputs, outputs)

  # Model compilation
  model.compile(loss="categorical_crossentropy",
                optimizer=SGD(learning_rate = study.best_trial.params['learning_rate'],
                              momentum = study.best_trial.params['momentum']),
                metrics="accuracy")

  # Path where our 'best_model.h5' file is going to be saved 
  checkpoint_filepath = './best_model.h5' 

  model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                              monitor='val_accuracy',
                                              mode='max',
                                              save_best_only=True)

  ####
  if os.path.exists("Kitchenware_data"):
    shutil.rmtree('Kitchenware_data')

  Dir_Data = "Kitchenware_data"
  Dir = ("Full_Train", "Test", "Train", "Val")

  Full_Train_Size = 0.8
  Random_State =study.best_trial.params['random_state']            

  df_Full_Train, df_Test, df_Train, df_Val = Making_Directory(ProjectDirectory= Dir_Data,
                                                              Directories = Dir,
                                                              DataFrame = df_train,
                                                              FullTrainSize = Full_Train_Size,
                                                              RandomState = Random_State)
  ####
  TamImagen =  299
  batch_size = 32

  train_data_gen = ImageDataGenerator(zoom_range =   [0.80, 1.0],
                                      brightness_range =[0.8,1.2],
                                      vertical_flip = True,  
                                      horizontal_flip = True, 
                                      #shear_range = 30,
                                      #rotation_range = 90,                                
                                      preprocessing_function=keras.applications.xception.preprocess_input)

  test_data_gen = ImageDataGenerator(preprocessing_function=keras.applications.xception.preprocess_input)

  train_generator = train_data_gen.flow_from_directory("./Kitchenware_data/Full_Train", 
                                                      (TamImagen, TamImagen),
                                                      batch_size = batch_size,
                                                      class_mode = "categorical")

  valid_generator = test_data_gen.flow_from_directory("./Kitchenware_data/Test", 
                                                      (TamImagen, TamImagen),
                                                      batch_size = batch_size,
                                                      class_mode = "categorical")
  
  history = model.fit(train_generator,
                      steps_per_epoch = train_generator.n//batch_size,
                      validation_data = valid_generator,
                      validation_steps = valid_generator.n//batch_size,
                      epochs= int(study.best_trial.params['epochs']*5),
                      callbacks=[model_checkpoint_callback],
                      workers=8,
                      verbose = 1)
  return model,history

model, model_history = MakeCNN(study,
                               verbose=1)