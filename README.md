# Image Classifier Project

![](/image/flower_img.jpeg)

This project comes from the Udacity Intro to Machine Learning with PyTorch NanoDegree. It is broken up into two parts: the first being to build an image classifier using PyTorch in Jupyter Notebooks, and the second being turning this classifier into an application that is run from the command line with arguements that can determine the build and structure of the network.

The focus of this project is:
- Loading data using transforms to augment the images as they are read in.
- Using transfer learning (choosing training parameters, creating new classifiers, importing models).
- Training a classifier using CUDA to achieve over 80% accuracy.
- Saving models to a checkpoint and re-loading them to be used for future predictions.
- Processing user images to Tensor's that can be used for predictions.
- Predicting flower classes and their corresponding probabilities.

Requirements to run the project:
- [Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower classes (needs to be split into Train/Test/Valid directories with labeled folders).
- [PyTorch (1.7.1)](https://pytorch.org/get-started/locally/) and CUDA 10.2.
- NumPy, Seaborn, and Matplotlib.
- GPU for CUDA use (otherwise model is trained on CPU at a slower rate).
