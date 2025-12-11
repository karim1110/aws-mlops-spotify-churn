import joblib
import pandas as pd
import numpy as np
from io import StringIO
import os
import sys
import subprocess

def model_fn(model_dir):
    """Load model with dependency installation."""
    print(f"Loading model from: {model_dir}")
    print(f"Files: {os.listdir(model_dir)}")
    
    # Install dependencies if needed
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
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions."""
    model = model_dict["model"]
    features = model_dict["features"]
    
    # Set column names
    input_data.columns = features
    
    # Predict
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    
    return pd.DataFrame({
        "prediction": predictions,
        "probability": probabilities
    })

def output_fn(prediction, accept="text/csv"):
    """Return CSV output."""
    return prediction.to_csv(index=False, header=False)
