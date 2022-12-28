#!pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np


interpreter= tflite.Interpreter(model_path='best_model.tflite')
interpreter.allocate_tensors()

input_index=interpreter.get_input_details()[0]['index']
output_index=interpreter.get_output_details()[0]['index']

classes=['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']

def normalization(x):
    x /= 127.5
    x -= 1.
    return x

# Preparing and pre-processing the image
def preprocess_img(img_path):

    #response = requests.get(URL)
    #img = Image.open(BytesIO(response.content))
    img = Image.open(img_path)
    img_resize = img.resize((299, 299))
    img_array = np.array(img_resize, dtype='float32')
    img_array_norm = normalization(img_array)
    img_reshape = img_array_norm.reshape(1,299,299,3)
    
    return img_reshape

# Predicting function
def predict_result(img_reshape):
    
    interpreter.set_tensor(input_index, img_reshape)
    interpreter.invoke()
    predictions=interpreter.get_tensor(output_index)
    prediction=classes[np.argmax(predictions[0], axis=-1)]
    
    return prediction

