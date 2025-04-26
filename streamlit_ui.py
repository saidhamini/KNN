import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/train/"

st.title("🔍 KNN Loan Default Classifier")
st.markdown("Tune hyperparameters below and train your model:")

# Form for user input
with st.form("knn_form"):
    n_neighbors = st.number_input("Number of Neighbors (n_neighbors)", min_value=1, value=5, step=1)
    weights = st.selectbox("Weights", options=["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])
    leaf_size = st.number_input("Leaf Size", min_value=1, value=30, step=1)
    p = st.selectbox("Minkowski Power (p)", options=[1, 2])
    metric = st.selectbox("Metric", options=["minkowski", "euclidean", "manhattan"])

    submit_btn = st.form_submit_button("🚀 Train Model")

# On submit
if submit_btn:
    # Prepare the payload
    payload = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
        "metric": metric
    }

    st.write("Sending data to FastAPI backend...")

    # Send request to FastAPI
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        if "accuracy" in result:
            st.success(f"✅ Model trained successfully with accuracy: **{result['accuracy']:.4f}**")
        else:
            st.error(f"❌ Error: {result.get('error', result)}")

    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Could not connect to FastAPI backend: {e}")
