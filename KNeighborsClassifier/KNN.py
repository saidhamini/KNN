import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import base64

os.environ['STREAMLIT_SERVER_PORT'] = '8071'

st.set_page_config(page_title="KNN Classification UI", layout="wide")

# Create folders if not exist
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Sidebar navigation
menu = ["Input Data", "Fine-tune Parameters", "Run Model", "View & Download Output"]
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", menu)

# Shared session state
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'target_cols' not in st.session_state:
    st.session_state.target_cols = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_columns' not in st.session_state:
    st.session_state.train_columns = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Input Data
if choice == "Input Data":
    st.title("Step 1: Upload Training Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_train = df
        df.to_csv("data/input/train.csv", index=False)
        st.success("Training data uploaded successfully.")
        st.dataframe(df.head(10))

# Fine-tune Parameters
elif choice == "Fine-tune Parameters":
    st.title("Step 2: Fine-tune Parameters")
    if st.session_state.df_train is not None:
        df = st.session_state.df_train

        st.subheader("Set KNN Parameters")
        n_neighbors = st.slider("n_neighbors", 1, 20, 5)
        weights = st.selectbox("weights", ["uniform", "distance"])
        algorithm = st.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        leaf_size = st.slider("leaf_size", 10, 100, 30)
        p = st.selectbox("p (1=manhattan, 2=euclidean)", [1, 2])
        metric = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"])

        if st.button("Train Model"):
            # Separate features and multiple target columns
            target_cols = ["SpenderType", "OrganicPreference", "FoodTypePreference", "LifestyleCategory"]
            X = df.drop(columns=target_cols)
            y = df[target_cols]

            X = pd.get_dummies(X)
            st.session_state.train_columns = X.columns  # Save train column names

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Calculate accuracy for each target variable
            accuracies = {}
            for i, target in enumerate(target_cols):
                acc = accuracy_score(y_val.iloc[:, i], y_pred[:, i])
                accuracies[target] = acc

            st.session_state.model = model
            st.session_state.target_cols = target_cols
            st.success(f"Model trained with accuracies: {accuracies}")
    else:
        st.warning("Please upload training data in 'Input Data' section.")

# Run Model
elif choice == "Run Model":
    st.title("Step 3: Run Model on Test Data")
    if st.session_state.model is not None:
        test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")
        if test_file:
            df_test = pd.read_csv(test_file)
            X_test = pd.get_dummies(df_test)

            # Align test columns to match training columns
            for col in st.session_state.train_columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[st.session_state.train_columns]  # Ensure correct order

            preds = st.session_state.model.predict(X_test)
            df_test[st.session_state.target_cols] = preds
            st.session_state.predictions = df_test
            st.success("Prediction completed.")
            st.dataframe(df_test.head(10))

            # Add download link for full predictions
            csv = df_test.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f"""
                <a href="data:file/csv;base64,{b64}" download="predicted_output.csv"
                   style="color: blue; text-decoration: underline; font-size: 16px;">
                   📥 Download Full Predicted File
                </a>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Please train the model in 'Fine-tune Parameters' section.")

# View & Download Output
elif choice == "View & Download Output":
    st.title("Step 4: Visualize & Download Results")

    if st.session_state.predictions is not None:
        df_pred = st.session_state.predictions

        # Encode categorical variables using LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_pred_encoded = df_pred.copy()

        for col in ['SpenderType', 'OrganicPreference', 'FoodTypePreference', 'LifestyleCategory']:
            df_pred_encoded[col] = le.fit_transform(df_pred[col])

        # Display the distribution of each target variable
        st.subheader("Target Variable Distributions")

        fig_spender, ax_spender = plt.subplots(figsize=(6, 4))
        df_pred['SpenderType'].value_counts().plot(kind='bar', ax=ax_spender, color='skyblue', edgecolor='black')
        ax_spender.set_title("Distribution of SpenderType")
        ax_spender.set_ylabel('Count')
        ax_spender.set_xlabel('Spender Type')

        fig_organic, ax_organic = plt.subplots(figsize=(6, 4))
        df_pred['OrganicPreference'].value_counts().plot(kind='bar', ax=ax_organic, color='lightgreen', edgecolor='black')
        ax_organic.set_title("Distribution of OrganicPreference")
        ax_organic.set_ylabel('Count')
        ax_organic.set_xlabel('Organic Preference')

        fig_food, ax_food = plt.subplots(figsize=(6, 4))
        df_pred['FoodTypePreference'].value_counts().plot(kind='bar', ax=ax_food, color='salmon', edgecolor='black')
        ax_food.set_title("Distribution of FoodTypePreference")
        ax_food.set_ylabel('Count')
        ax_food.set_xlabel('Food Type Preference')

        fig_lifestyle, ax_lifestyle = plt.subplots(figsize=(6, 4))
        df_pred['LifestyleCategory'].value_counts().plot(kind='bar', ax=ax_lifestyle, color='lightcoral', edgecolor='black')
        ax_lifestyle.set_title("Distribution of LifestyleCategory")
        ax_lifestyle.set_ylabel('Count')
        ax_lifestyle.set_xlabel('Lifestyle Category')

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_spender)
            st.pyplot(fig_organic)
        with col2:
            st.pyplot(fig_food)
            st.pyplot(fig_lifestyle)

    else:
        st.warning("Please run the model in the 'Run Model' section.")