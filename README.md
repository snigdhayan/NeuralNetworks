# NeuralNetworks

This repository contains my experiments with neural networks. Currently there is a TF + Keras based deep neural network that learns to forecast from a handicrafted time series dataset. In the 'Ludwig' directory there is a simple Uber Ludwig based approach to analyze breast cancer. To run the script 'BreastCancerAnalysisViaLudwig.py' one must install Ludwig and associated dependencies (e.g., I created a separate conda environment with python=3.6); moreover, it assumes the existence of data (breast_cancer_dataset.csv) and model definition (LudwigModelDefinitionFile.yml) in the current directory. In the 'Tensorflow-Quantum' directory I have uploaded my jupyter notebook, where I compared the performance of a quantum neural network with that of a comparable classical neural network on MNIST dataset following this tutorial

https://www.tensorflow.org/quantum/tutorials/mnist

Since I could not manage to install tensorflow-quantum on my Windows 10 laptop, I ran all the experiments in a docker container. I will add the dockerfile later on.