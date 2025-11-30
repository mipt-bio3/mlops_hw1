import pandas as pd
import numpy as np
import yaml
import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
logger = logging.getLogger(__name__)

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def prepare_data():
    params = load_params()
    split_ratio = params['data']['split_ratio']
    random_state = params['data']['random_state']
    
    logger.info("Loading Iris dataset...")
    df = pd.read_csv('data/raw/iris.csv')
    logger.info(f"Dataset shape: {df.shape}")
    
    logger.info("Checking for missing values...")
    missing = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing}")
    
    logger.info("Removing duplicates...")
    df = df.drop_duplicates()
    logger.info(f"Dataset shape after deduplication: {df.shape}")
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    logger.info(f"Splitting data with ratio {split_ratio} and random_state {random_state}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-split_ratio, random_state=random_state, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    logger.info(f"Train set shape: {train_df.shape}, saved to data/processed/train.csv")
    logger.info(f"Test set shape: {test_df.shape}, saved to data/processed/test.csv")

if __name__ == '__main__':
    prepare_data()