import joblib
import pandas as pd
import numpy as np
from io import StringIO
import os
import sys
import traceback
import subprocess

def model_fn(model_dir):
    """Load model with dependency installation."""
    import os, sys, subprocess
    print(f"Loading model from: {model_dir}")
    print(f"Files: {os.listdir(model_dir)}")
    
    # Install if missing (safe, idempotent)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flaml", "lightgbm", "xgboost"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ Dependencies installed")
    except:
        print("Dependencies already exist or install skipped")
    
    import joblib
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))
    print(f"✓ Model loaded: {type(model).__name__}, {len(features)} features")
    
    return {"model": model, "features": features}

def input_fn(request_body, content_type="text/csv"):
    """Parse CSV input."""
    try:
        if content_type == "text/csv":
            df = pd.read_csv(StringIO(request_body), header=None)
            print(f"Parsed input shape: {df.shape}")
            return df
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        print(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_data, model_dict):
    """Make predictions."""
    try:
        model = model_dict["model"]
        features = model_dict["features"]
        
        # Set column names
        input_data.columns = features
        
        print(f"Making predictions for {len(input_data)} samples")
        
        # Predict
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        return pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities
        })
    except Exception as e:
        print(f"Error in predict_fn: {str(e)}")
        print(traceback.format_exc())
        raise

def output_fn(prediction, accept="text/csv"):
    """Return CSV output."""
    try:
        return prediction.to_csv(index=False, header=False)
    except Exception as e:
        print(f"Error in output_fn: {str(e)}")
        raise
