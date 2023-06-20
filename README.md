<h1 align="center">Architecture Classifier</h1>
<hr>
This project is aimed at building a deep learning model that can classify the architectural style of a building by looking at its photo.

## Dataset
The dataset consists of photographs belonging to 25 different architectural style classes.
<br>**Dataset:** https://www.kaggle.com/datasets/wwymak/architecture-dataset

## Model Architecture
The model was trained using Convolutional Neural Networks. MobileNetV3Large was used as the pre-trained model. L2 Regularization was applied to the last layers to reduce overfitting.

## Script Usage
```commandline
python arch_classfier.py "path_of_photo"
```
