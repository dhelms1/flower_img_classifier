# Flower Classifier Project

<img src="/image/flower_img.jpeg" width="300">

This project comes from the Udacity Intro to Machine Learning with PyTorch NanoDegree. It is broken up into two parts: the first being to build an image classifier using PyTorch in Jupyter Notebooks, and the second being turning this classifier into an application that is run from the command line with arguments that can determine the build and structure of the network.

---

#### The focus of this project is:
- Loading data using transforms to augment the images as they are read in.
- Using transfer learning (choosing training parameters, creating new classifiers, importing models).
- Training a classifier using CUDA to achieve over 80% accuracy.
- Saving models to a checkpoint and re-loading them to be used for future predictions.
- Processing user images to Tensor's that can be used for predictions.
- Predicting flower classes and their corresponding probabilities.

---

#### Requirements to run the project:
- [Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower classes ([download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)).
- [PyTorch (1.7.1)](https://pytorch.org/get-started/locally/) and CUDA 10.2.
- NumPy, Seaborn, and Matplotlib.
- GPU for CUDA use (otherwise model is trained on CPU at a slower rate).

---

#### Command line application:
- functions.py contains all functions needed for both train.py and predict.py
- train.py will take a directory of images (along with other optional arguments) and train a Neural Network.
- predict.py will take a single image and a saved model checkpoint (along with other optional arguments) and return the top predictions and their probabilities.
