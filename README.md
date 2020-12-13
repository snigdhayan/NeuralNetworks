# NeuralNetworks

This repository contains my experiments with neural networks. 

1. There is a TF + Keras based deep neural network that learns to forecast from a handicrafted time series dataset. 
2. In the 'Ludwig' directory there is a simple Uber Ludwig based approach to analyze breast cancer. To run the script 'BreastCancerAnalysisViaLudwig.py' one must install Ludwig and associated dependencies (e.g., I created a separate conda environment with python=3.6); moreover, it assumes the existence of data (breast_cancer_dataset.csv) and model definition (LudwigModelDefinitionFile.yml) in the current directory. 
3. In the 'Tensorflow-Quantum' directory I compared the performance (time + accuracy) of a quantum neural network with that of a comparable classical neural network on MNIST dataset after 2 epochs following this tutorial: https://www.tensorflow.org/quantum/tutorials/mnist

I conducted the experiment in a docker container on my Windows 10 Home laptop. I added the dockerfile based on which the docker image is built.