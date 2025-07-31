import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

SUPPORTED = {
    ".csv": "CSV",
    ".xlsx": "Excel (xlsx)",
    ".xls": "Excel (xls)"
}

st.set_page_config(page_title="Complete EDA + ML Streamlit App", layout="wide")
st.title("Complete Data Science Pipeline with EDA and ML")
st.markdown(
    """
    Upload your dataset (**CSV, XLSX, XLS only**) and perform full EDA, preprocessing,
    training, prediction, evaluation, and visualization directly in this app.
    """
)

def read_file(uploaded_file, ext):
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext == ".xlsx":
        return pd.read_excel(uploaded_file, engine="openpyxl")
    elif ext == ".xls":
        return pd.read_excel(uploaded_file, engine="xlrd")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def clean_data(df, drop_duplicates, missing_strategy):
    if drop_duplicates:
        df = df.drop_duplicates()
    if missing_strategy == "Drop rows":
        df = df.dropna()
    elif missing_strategy == "Fill with mean/mode":
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
    return df

def encode_features(df, cat_cols):
    if not cat_cols:
        return df
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6,5))
    str_labels = [str(l) for l in labels]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=str_labels, yticklabels=str_labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix')
    return fig

def plot_regression_performance(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    return fig

uploaded_file = st.file_uploader("Upload your data file", type=[k[1:] for k in SUPPORTED.keys()])

if uploaded_file:
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    st.success(f"Uploaded file: **{uploaded_file.name}** ({SUPPORTED.get(ext, ext)})")

    try:
        df = read_file(uploaded_file, ext)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # --- Load & Inspect Data ---
    st.header("Load & Inspect Data")
    st.subheader("Data Preview (.head())")
    st.dataframe(df.head())

    st.subheader("Data Info (.info())")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    buffer.close()

    st.subheader("Descriptive Statistics (.describe())")
    st.dataframe(df.describe(include="all").transpose())

    # --- Clean & Preprocess ---
    st.header("Clean & Preprocess Data")
    drop_dupes = st.checkbox("Drop duplicates", value=True)
    missing_strategy = st.selectbox(
        "Missing value handling strategy",
        ("Drop rows", "Fill with mean/mode", "Do nothing"),
        index=1
    )
    df_clean = clean_data(df.copy(), drop_dupes, missing_strategy)
    st.write(f"Data shape after cleaning: {df_clean.shape}")

    st.subheader("Missing Values After Cleaning")
    st.dataframe(df_clean.isnull().sum())

    # --- Feature Selection & Engineering ---
    st.header("Feature Selection & Engineering")
    all_columns = list(df_clean.columns)
    target_col = st.selectbox("Select target column", all_columns)

    feature_cols = st.multiselect(
        "Select feature columns",
        [col for col in all_columns if col != target_col],
        default=[col for col in all_columns if col != target_col]
    )
    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    # ENFORCE ALL FEATURE COLUMN NAMES TO STRING TYPE (preempt sklearn error)
    feature_cols = [str(col) for col in feature_cols]
    df_clean.columns = df_clean.columns.astype(str)

    cat_cols = [col for col in feature_cols if df_clean[col].dtype == 'object']
    df_features = encode_features(df_clean[feature_cols].copy(), cat_cols)
    df_features.columns = df_features.columns.astype(str)  # Ensures all columns as str

    # Optional feature engineering example
    st.subheader("Optional Feature Engineering")
    add_feature = st.checkbox("Add product of first two numeric features as new feature")
    if add_feature:
        numeric_feats = df_features.select_dtypes(include=np.number).columns
        if len(numeric_feats) >= 2:
            new_feat_name = f"{numeric_feats[0]}_x_{numeric_feats[1]}"
            df_features[new_feat_name] = df_features[numeric_feats[0]] * df_features[numeric_feats[1]]
            st.info(f"Added new feature '{new_feat_name}'")
        else:
            st.info("Not enough numeric features to create new feature")

    # --- Split Data ---
    st.header("Split Data")
    test_size = st.slider("Test data split ratio", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random seed", value=42, step=1, format='%d')

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df_features, df_clean[target_col],
            test_size=test_size, random_state=random_state
        )
        # Ensure all train/test sets have string columns (for sklearn > 1.2)
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
    except Exception as e:
        st.error(f"Error splitting data: {e}")
        st.stop()

    st.write(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # --- Select & Train ML Algorithm ---
    st.header("Select & Train ML Model")
    model_type = st.selectbox("Choose model type", ["Classification", "Regression"])

    model = None
    if model_type == "Classification":
        st.write("Using Logistic Regression")
        C = st.number_input("Inverse of regularization strength (C)", min_value=0.01, value=1.0)
        max_iter = st.number_input("Max iterations", min_value=10, value=100, step=10)
        model = LogisticRegression(C=C, max_iter=int(max_iter))
    else:
        st.write("Using Random Forest Regressor")
        n_estimators = st.number_input("Number of trees", min_value=10, value=100, step=10)
        max_depth = st.number_input("Max tree depth (0 = None)", min_value=0, value=0, step=1)
        max_depth = None if max_depth == 0 else int(max_depth)
        model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=max_depth, random_state=random_state)

    with st.spinner("Training model..."):
        try:
            model.fit(X_train, y_train)
            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Model training error: {e}")
            st.stop()

    # --- Make Predictions ---
    st.header("Make Predictions")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # --- Evaluate Accuracy ---
    st.header("Evaluate Model Performance")
    if model_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: **{acc:.4f}**")
        # All labels as strings to prevent mathtext issues
        labels = sorted([str(l) for l in pd.unique(y_test)])
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm = plot_confusion_matrix(cm, labels)
        st.pyplot(fig_cm)
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.write(f"R² score: **{r2:.4f}**")
        st.write(f"MSE: **{mse:.4f}**, RMSE: **{rmse:.4f}**")
        fig_reg = plot_regression_performance(y_test, y_pred)
        st.pyplot(fig_reg)

    # --- Optimize & Retrain ---
    st.header("Optimize & Retrain")
    if st.button("Retrain model with current parameters"):
        with st.spinner("Retraining..."):
            try:
                model.fit(X_train, y_train)
                st.success("Retrained successfully!")
            except Exception as e:
                st.error(f"Retraining error: {e}")

    # --- Automated EDA Profiling ---
    st.header("Automated EDA Profiling Report")
    profile = ProfileReport(df_clean, title="YData Profiling Report", explorative=True)
    st_profile_report(profile)

else:
    st.info("Upload a data file (CSV or Excel only) to start the analysis.")
