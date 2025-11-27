# Spotify Churn Prediction – MLOps Pipeline with AutoML & AWS SageMaker

This project builds and deploys a **machine learning pipeline** to predict Spotify user churn using **FLAML**, **scikit-learn**, and the **AWS SageMaker** MLOps stack.  
It includes automated training, packaging, deployment to a real-time inference endpoint, evaluation, feature sensitivity testing, and automatic cleanup.

---

## 🚀 Project Overview

This pipeline performs the following steps:

1. **Load and preprocess data** from Amazon S3  
2. **Train a classifier using FLAML AutoML**  
3. **Generate a fully custom inference script (`inference.py`)**  
4. **Package the model + features into `model.tar.gz`**  
5. **Upload model artifacts to S3**  
6. **Deploy a SageMaker real-time endpoint**  
7. **Run inference on:**
   - The original test set  
   - A modified test set with changed features  
8. **Cleanup the endpoint** to avoid unnecessary costs  

---

## 🛠️ Technologies Used

### Machine Learning / Python
- **FLAML** (AutoML optimization)
- **scikit-learn**
- **pandas / numpy**
- **joblib** (model serialization)

### AWS
- **Amazon S3** (data + model storage)
- **SageMaker Real-Time Endpoints**
- **sagemaker SKLearnModel**
- **boto3** (AWS client SDK)

### Additional Concepts
- One-hot encoding for categorical variables  
- Feature alignment between train & test sets  
- Endpoint invocation via `sagemaker-runtime`  
- Automatic dependency installation inside inference script  

---

## 🧠 Model Training

The training workflow uses **FLAML**:

FLAML automatically selects the best algorithm (e.g., LightGBM, XGBoost, Random Forest, Logistic Regression) and hyperparameters.

The best model and feature list are saved using `joblib`:

```
model.pkl
features.pkl
```

---

## 🧾 Inference Script (SageMaker)

The file `inference.py` handles:

- Model loading  
- Missing dependency installation (FLAML, LightGBM, XGBoost)  
- CSV parsing  
- Prediction + probability scoring  
- CSV output  

SageMaker requires **four functions**:

| Function | Purpose |
|---------|---------|
| `model_fn` | Load model + features from disk |
| `input_fn` | Parse CSV input |
| `predict_fn` | Run inference |
| `output_fn` | Return predictions as CSV |

---

## 📦 Packaging the Model

Before deployment, the model artifacts are compressed into:

```
model.tar.gz
```

SageMaker requires this file to contain:

```
model.pkl
features.pkl
```

This archive is uploaded to:

```
s3://mlopsfinalprojectdata/model-artifacts/model.tar.gz
```

---

## ☁️ Deployment to SageMaker

The model is deployed as:

```
spotify-churn-endpoint-v7
```

Using:

```python
sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=ENDPOINT_NAME
)
```

The script also handles failed endpoints and deletes previous failed configurations.

---

## 🧪 Evaluation

### **1. Original test set**
Metrics computed:

- **Accuracy**
- **F1 score**
- **ROC AUC**
- **Full classification report**

Predictions are obtained using:

```python
runtime.invoke_endpoint()
```

### **2. Modified test set**
Two features are changed:

- `listening_time` → +50%
- `skip_rate` → +0.05

The goal is to measure **model sensitivity** to user behavior changes.

Outputs include:

- New accuracy / F1 / AUC  
- % of predictions that changed  

---

## 🧹 Automatic Cleanup

To avoid unnecessary AWS cost, the script deletes:

1. The SageMaker endpoint  
2. The SageMaker endpoint configuration  

Outputs instructions for manual deletion if needed.

---

## 📝 Requirements

Install dependencies:

```bash
pip install flaml scikit-learn pandas boto3 sagemaker joblib
```

AWS requirements:

- IAM Role with SageMaker permissions  
- S3 bucket containing `train.csv` and `test.csv`  
- AWS credentials configured locally  
