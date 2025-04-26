from fastapi import FastAPI
from schemas import HyperParams
from model import train_model
import joblib

X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("Y_train.pkl")
y_test = joblib.load("Y_test.pkl")

app = FastAPI()

@app.post("/train/")
def train(hyperparams: HyperParams):
    try:
        warnings = []
        recommendations = []

        # --- Smart Validation Rules --- #

        # n_neighbors check
        if hyperparams.n_neighbors > len(X_train):
            return {
                "status": "error",
                "message": f"n_neighbors ({hyperparams.n_neighbors}) is too high! Must be <= number of training samples ({len(X_train)})."
            }
        elif hyperparams.n_neighbors <= 3:
            warnings.append("⚠️ n_neighbors is very small, model may overfit (too sensitive to noise). Recommended: 5-15")
            recommendations.append("Increase n_neighbors to between 5 and 15 to reduce overfitting.")
        elif hyperparams.n_neighbors > 50:
            warnings.append("⚠️ n_neighbors is quite large, model may underfit (too generalized). Recommended: 5-30")
            recommendations.append("Decrease n_neighbors to between 5 and 30 for better balance.")

        # leaf_size check
        if hyperparams.leaf_size > 1000:
            warnings.append("⚠️ leaf_size is very large, tree search may be inefficient. Recommended: <= 50")
            recommendations.append("Set leaf_size to <= 50 for faster searches.")

        # p value check
        if hyperparams.p not in [1, 2]:
            warnings.append("⚠️ p value should ideally be 1 (Manhattan) or 2 (Euclidean). Others are less common.")
            recommendations.append("Use p=1 or p=2 for best results.")

        # --- Train Model --- #
        result = train_model(X_train, X_test, y_train, y_test, hyperparams.dict())

        return {
            "status": "success",
            "accuracy": result.get('accuracy'),
            "warnings": warnings,
            "recommendations": recommendations
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
