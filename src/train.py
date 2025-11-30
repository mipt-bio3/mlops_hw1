import pandas as pd
import numpy as np
import yaml
import logging
import sys
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
logger = logging.getLogger(__name__)

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model():
    params = load_params()
    
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment('iris-classification')
    
    with mlflow.start_run():
        logger.info("Loading processed data...")
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        
        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        mlflow.log_params({
            'split_ratio': params['data']['split_ratio'],
            'random_state': params['data']['random_state'],
            'model_type': params['model']['model_type'],
            'n_estimators': params['model_params']['n_estimators'],
            'max_depth': params['model_params']['max_depth'],
            'min_samples_split': params['model_params']['min_samples_split'],
            'min_samples_leaf': params['model_params']['min_samples_leaf'],
        })
        
        logger.info(f"Training {params['model']['model_type']} model...")
        
        if params['model']['model_type'] == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=params['model_params']['n_estimators'],
                max_depth=params['model_params']['max_depth'],
                min_samples_split=params['model_params']['min_samples_split'],
                min_samples_leaf=params['model_params']['min_samples_leaf'],
                random_state=params['model']['random_state'],
                n_jobs=-1
            )
        else:
            model = LogisticRegression(
                random_state=params['model']['random_state'],
                max_iter=1000
            )
        
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        logger.info(f"Metrics - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        mlflow.log_metrics(metrics)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        mlflow.log_artifact('model.pkl')
        
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        mlflow.log_artifact('metrics.json')
        
        logger.info("Model saved to model.pkl and metrics to metrics.json")

if __name__ == '__main__':
    train_model()