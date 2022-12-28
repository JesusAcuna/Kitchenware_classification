FROM python:3.9-slim

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt


COPY ["best_model.tflite","model.py","app.py", "./"]


WORKDIR /app/templates


COPY ["templates/index.html","templates/layout.html","templates/result.html", "./"]


WORKDIR /app/static


WORKDIR /app/static/css


COPY ["static/css/custom.css", "./"]


WORKDIR /app/static/images


COPY ["static/images/cup.jpg","static/images/fork.jpg","static/images/glass.jpg","static/images/knife.jpg","static/images/plate.jpg","static/images/spoon.jpg", "./"]


WORKDIR /app/static/js


COPY ["static/js/image_upload.js", "./"]


WORKDIR /app


EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "app:app"]