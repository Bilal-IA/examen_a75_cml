import os
import warnings
import sys
 
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB 
import mlflow
import mlflow.sklearn
 
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
 

def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    recall_score = metrics.recall_score(actual, pred)
    f1_score = metrics.f1_score(actual, pred)
    return accuracy, recall_score, f1_score
 
# Read the csv file
data = pd.read_csv('pointure.data')

label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(data['Genre'])
data['Genre'] = encoded_labels 
    
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)
 
# The predicted column is "Genre"
train_x = train.drop(["Genre"], axis=1)
test_x = test.drop(["Genre"], axis=1)
train_y = train[["Genre"]]
test_y = test[["Genre"]]
 
gnb = GaussianNB()
gnb.fit(train_x, train_y)
 
predicted = gnb.predict(test_x)
 
(accuracy, recall_score, f1_score) = eval_metrics(test_y, predicted)
 
print("  accuracy: %s" % accuracy)
print("  recall_score: %s" % recall_score)
print("  f1_score: %s" % f1_score)
        
with open("metrics.txt", 'w') as outfile:
    outfile.write("accuracy: " + str(accuracy) + "\n")
    outfile.write("recall_score: " + str(recall_score) + "\n")
    outfile.write("f1_score: " + str(f1_score) + "\n")
      
with mlflow.start_run():
    mlflow.set_experiment(experiment_name="mlflow_demo")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall_score", recall_score)
    mlflow.log_metric("f1_score", f1_score)
 
    mlflow.sklearn.log_model(gnb, "model")
    #print(f"artifact_uri={mlflow.get_artifact_uri()}")
