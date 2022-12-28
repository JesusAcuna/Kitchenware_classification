from flask import Flask ,render_template, request

from model import preprocess_img, predict_result

import os

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'images')


@app.route('/', methods=['GET'])
def main():
    IMG_LIST = os.listdir('static/images')
    IMG_LIST = ['images/' + i for i in IMG_LIST]

    return render_template('index.html', imagelist=IMG_LIST)

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    
  try:
      if request.method == 'POST':
          img = preprocess_img(request.files['file'].stream)
          pred = predict_result(img)
          return render_template("result.html", predictions=str(pred))

  except:
      error = "File cannot be processed."
      return render_template("result.html", err=error)
    


if __name__ == '__main__' :
    app.config['UPLOAD_FOLDER'] = IMG_FOLDER
    app.run(debug=True,host='0.0.0.0',port=9696)