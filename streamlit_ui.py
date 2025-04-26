import streamlit as st
import requests

st.title("KNN Hyperparameter Trainer")

# Input form
with st.form("Hyperparameters"):
    n_neighbors = st.number_input("n_neighbors", min_value=1, value=5)
    weights = st.selectbox("weights", ["uniform", "distance"])
    algorithm = st.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    leaf_size = st.number_input("leaf_size", min_value=1, value=30)
    p = st.number_input("p (Minkowski power)", min_value=1, value=2)
    metric = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"])

    submit = st.form_submit_button("Train Model")

if submit:
    params = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
        "metric": metric
    }

    with st.spinner("Training your model..."):
        response = requests.post("http://localhost:8000/train/", json=params)
        result = response.json()

    if result["status"] == "success":
        st.success(f"✅ Model trained! Accuracy: {round(result['accuracy'] * 100, 2)}%")

        if result.get("warnings"):
            st.warning("⚠️ Warnings:")
            for warn in result["warnings"]:
                st.markdown(f"- {warn}")

        if result.get("recommendations"):
            st.info("💡 Recommendations:")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")

    else:
        st.error(f"❌ Error: {result['message']}")
