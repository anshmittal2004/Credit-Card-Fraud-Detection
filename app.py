import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset with Path Check
file_path = "creditcard.csv"
if not os.path.exists(file_path):
    st.error("⚠️ Error: Dataset file 'creditcard.csv' not found! Check the path.")
else:
    df = pd.read_csv(file_path)
    st.write("✅ Dataset Loaded Successfully!")
    st.write(df.head())

    # Data Preprocessing
    df['NormalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    # Splitting Data
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Streamlit UI
    st.title("Credit Card Fraud Detection")
    st.write(f"🎯 **Model Accuracy:** `{accuracy:.4f}`")
    st.text("📊 Classification Report:")
    st.text(report)

    # Data Visualization
    st.subheader("Fraud vs Non-Fraud Transactions")
    fig, ax = plt.subplots()
    sns.countplot(x=df['Class'], ax=ax)
    plt.xticks(ticks=[0,1], labels=["Non-Fraud", "Fraud"])
    plt.title("Fraud vs Non-Fraud Transactions")
    st.pyplot(fig)
    plt.clf()

    st.subheader("Distribution of Normalized Transaction Amount")
    fig, ax = plt.subplots()
    sns.histplot(df['NormalizedAmount'], bins=50, kde=True, ax=ax)
    plt.title("Distribution of Normalized Amount")
    st.pyplot(fig)
    plt.clf()

    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importances, y=features, ax=ax)
    plt.title("Feature Importance")
    st.pyplot(fig)
    plt.clf()

    st.sidebar.header("Enter Transaction Details")
    input_features = {}
    for col in X.columns:
        dtype = float if X_train[col].dtype == 'float64' else int
        input_features[col] = st.sidebar.number_input(col, value=dtype(X_train[col].mean()), format="%.5f")

    if st.sidebar.button("Check Fraud Probability"):
        input_df = pd.DataFrame([input_features])
        prediction = model.predict(input_df)[0]
        st.success("✅ Transaction is Safe.") if prediction == 0 else st.error("🚨 Fraud Detected!")
