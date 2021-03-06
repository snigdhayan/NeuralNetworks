# coding: utf-8

# Read data from local directory
import pandas as pd
import time

cancer_dataset = pd.read_csv('./breast_cancer_dataset.csv')

# Split data for training and testing
from sklearn.model_selection import train_test_split

split = 0.3
cancer_dataset_train, cancer_dataset_test = train_test_split(cancer_dataset, test_size=split)
# X_train, Y_train = cancer_dataset_train.drop(columns='label'), cancer_dataset_train['label']
X_test, Y_test = cancer_dataset_test.drop(columns='label'), cancer_dataset_test['label']

# Train locally defined Ludwig model
from ludwig.api import LudwigModel

start_time = time.time()
model = LudwigModel(model_definition_file='./LudwigModelDefinitionFile.yml')
train_stats = model.train(data_df=cancer_dataset_train,
                          skip_save_model=True, 
                          skip_save_processed_input=True, 
                          skip_save_training_statistics=True, 
                          skip_save_training_description=True, 
                          skip_save_log=True, 
                          skip_save_progress=True)
training_time = time.time() - start_time

# Visualize training statistics
from ludwig.visualize import learning_curves

learning_curves(train_stats, output_feature_name='label')

# Predict and print statistics
pred = model.predict(data_df=X_test)
predictions = pred['label_predictions']
Y_test = Y_test == 1  # Change labels from 0/1 to False/True

pred_correct = []
for i in range(1, len(Y_test)):
    pred_correct.append(predictions.iloc[i-1] == Y_test.iloc[i-1])

print("No. of correct predictions = {}".format(sum(pred_correct)))
print("No. of incorrect predictions = {}".format(len(Y_test)-sum(pred_correct)))
print("Training time = {} seconds".format(round(training_time,2)))

model.close()
