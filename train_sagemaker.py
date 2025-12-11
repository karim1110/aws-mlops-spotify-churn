
import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
from flaml import AutoML
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_budget", type=int, default=120)
    args, _ = parser.parse_known_args()
    
    print("Loading training data...")
    # SageMaker provides data in /opt/ml/input/data/training/
    train_file = "/opt/ml/input/data/training/train_processed.csv"
    df_train = pd.read_csv(train_file)
    
    print(f"Data shape: {df_train.shape}")
    
    X_train = df_train.drop(columns=["is_churned"])
    y_train = df_train["is_churned"]
    
    print(f"Features: {X_train.shape}, Target: {y_train.shape}")
    
    # Train with FLAML
    print(f"Starting FLAML training (budget: {args.time_budget}s)...")
    automl = AutoML()
    automl.fit(
        X_train, 
        y_train,
        task="classification",
        time_budget=args.time_budget,
        metric="f1",
        eval_method="cv",
        n_splits=3,
        verbose=3
    )
    
    model = automl.model
    
    # Evaluate on training data (for console metrics)
    preds = model.predict(X_train)
    probs = model.predict_proba(X_train)[:, 1]
    acc = accuracy_score(y_train, preds)
    f1 = f1_score(y_train, preds)
    roc_auc = roc_auc_score(y_train, probs)
    
    # Additional metrics
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    precision = precision_score(y_train, preds)
    recall = recall_score(y_train, preds)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
    specificity = tn / (tn + fp)
    
    # Print all metrics (will be captured by SageMaker)
    print(f"ACCURACY: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL: {recall:.4f}")
    print(f"SPECIFICITY: {specificity:.4f}")
    print(f"ROC_AUC: {roc_auc:.4f}")
    print(f"TRUE_POSITIVES: {tp}")
    print(f"TRUE_NEGATIVES: {tn}")
    print(f"FALSE_POSITIVES: {fp}")
    print(f"FALSE_NEGATIVES: {fn}")
    
    # Save model and metadata to /opt/ml/model/
    model_dir = "/opt/ml/model"
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "features.pkl"))
    
    # Save metrics
    metrics = {
        "best_estimator": automl.best_estimator,
        "best_f1": float(1 - automl.best_loss),
        "train_accuracy": float(acc),
        "train_f1": float(f1),
        "train_roc_auc": float(roc_auc),
        "feature_count": len(X_train.columns),
        "train_samples": len(X_train)
    }
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    print(f"Training complete!")
    print(f"Best model: {metrics['best_estimator']}")
    print(f"Best F1 (from CV): {metrics['best_f1']:.4f}")
