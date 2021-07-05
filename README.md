# NeuralNetworks

This repository contains my experiments with neural networks. 

1. In the `Tensorflow` directory there is a tensorflow + keras based deep neural network that learns to forecast from a handicrafted time series dataset. Moreover, there is an `autoencoder` based approach (consult `CreditCardFraudDetectionViaAutoencoder.ipynb`) to detect credit card fraud as anomalies among normal cases. It uses the kaggle dataset - https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv and the idea is borrowed from this tensorflow tutorial - https://www.tensorflow.org/tutorials/generative/autoencoder.
2. In the `Ludwig` directory there is a simple https://github.com/ludwig-ai/ludwig based approach to analyze the breast cancer dataset `breast_cancer_dataset.csv`. Ludwig is a low-code framework that uses tensorflow underneath. To run the script `BreastCancerAnalysisViaLudwig.py` one must install Ludwig and associated dependencies (e.g., I created a conda environment with python=3.6 for this purpose). Moreover, it assumes the existence of the dataset `breast_cancer_dataset.csv` and the model definition `LudwigModelDefinitionFile.yml` in the working directory.
3. In the `FastAI` directory there is a containerized application (jupyter notebook) that analyzes the breast cancer dataset `breast_cancer_dataset.csv` based on https://www.fast.ai/. Since it is a tabular dataset, I used the tabular learner that uses the PyTorch framework underneath. For convenience I have included the dockerfile with the requirements file and the associated script `dockerLaunchFastAI.sh` to launch the application in docker.
4. In the `Tensorflow-Quantum` directory there is a comparison between the performances (time + accuracy) of a quantum neural network and a comparable classical neural network following this tutorial: https://www.tensorflow.org/quantum/tutorials/mnist. Furthermore, the `tensorflow-quantum` framework is applied to analyze the dataset `breast_cancer_dataset.csv` using the ideas of amplitude encoding and binary encoding. The experiment was conducted in a docker container during a quantum computing hackathon that I had organized in December 2020. For transparency I have included the dockerfile and the associated script `dockerLaunchTFQ.sh` to launch the application in docker.