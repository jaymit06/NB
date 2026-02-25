import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

st.title("Naive Bayes Classifier")

# -------------------------------
# Dataset Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Target Selection
    # -------------------------------
    target_column = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X)

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    test_size = st.slider("Test Size (%)", 10, 50, 20)
    test_size = test_size / 100

    # -------------------------------
    # Model Selection
    # -------------------------------
    model_choice = st.selectbox(
        "Select Naive Bayes Model",
        ["GaussianNB", "MultinomialNB", "BernoulliNB"]
    )

    if model_choice == "GaussianNB":
        model = GaussianNB()
    elif model_choice == "MultinomialNB":
        model = MultinomialNB()
    else:
        model = BernoulliNB()

    # -------------------------------
    # Train Model
    # -------------------------------
    if st.button("Train Model"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -------------------------------
        # Accuracy
        # -------------------------------
        acc = accuracy_score(y_test, y_pred)
        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {acc:.4f}")

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)