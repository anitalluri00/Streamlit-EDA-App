import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    r2_score, mean_squared_error
)

# PDF reading support
try:
    import tabula
except ImportError:
    st.error("Please install 'tabula-py' to enable PDF file support (`pip install tabula-py`).")
    st.stop()

SUPPORTED = {
    ".csv": "CSV",
    ".xlsx": "Excel (xlsx)",
    ".xls": "Excel (xls)",
    ".txt": "Text",
    ".tsv": "TSV",
    ".json": "JSON",
    ".pdf": "PDF"
}

st.set_page_config(page_title="Full EDA + ML Streamlit App", layout="wide")

st.title("Full Data Science Pipeline: EDA and ML in Streamlit")
st.markdown("""
This app performs comprehensive data loading, exploration, cleaning, feature engineering,  
model training (classification & regression), prediction, evaluation, and visualization — all interactively.
""")

# ---------- Helper functions ----------

def read_file(uploaded_file, ext):
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext == ".xlsx":
        return pd.read_excel(uploaded_file, engine="openpyxl")
    elif ext == ".xls":
        return pd.read_excel(uploaded_file, engine="xlrd")
    elif ext == ".tsv":
        return pd.read_csv(uploaded_file, sep="\t")
    elif ext == ".json":
        return pd.read_json(uploaded_file)
    elif ext == ".txt":
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except:
            uploaded_file.seek(0)
            return pd.read_table(uploaded_file)
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            dfs = tabula.read_pdf(tmp.name, pages="all", multiple_tables=True)
            if len(dfs) == 0:
                raise ValueError("No tables found in PDF file.")
            df_pdf = dfs[0]
            # convert object columns to str for Arrow compatibility
            for col in df_pdf.select_dtypes(include=['object']).columns:
                df_pdf[col] = df_pdf[col].astype(str)
            return df_pdf
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def display_df_info(df):
    st.write("### DataFrame Info")
    buffer = []
    df.info(buf=buffer)
    st.text('\n'.join(buffer))

def clean_data(df, drop_duplicates, missing_strategy):
    if drop_duplicates:
        df = df.drop_duplicates()
    if missing_strategy == "Drop rows":
        df = df.dropna()
    elif missing_strategy == "Fill with mean (numeric) and mode (categorical)":
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
    return df

def encode_features(df, cat_cols):
    if len(cat_cols) == 0:
        return df
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded,
                              columns=encoder.get_feature_names_out(cat_cols),
                              index=df.index)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def plot_confusion_matrix(cm, target_cols):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=target_cols, yticklabels=target_cols, ax=ax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    return fig

def plot_regression_performance(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Regression: Actual vs Predicted')
    return fig

# ---------- Main app logic ----------

uploaded_file = st.file_uploader(
    "Upload your data file",
    type=[k[1:] for k in SUPPORTED.keys()]
)

if uploaded_file:
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    st.success(f"Uploaded file: **{uploaded_file.name}** ({SUPPORTED.get(ext, ext)})")
    
    try:
        df = read_file(uploaded_file, ext)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    # Show Raw Data Inspection
    st.header("1. Load & Inspect Data")

    st.subheader("Data Preview (.head())")
    st.dataframe(df.head())

    st.subheader("DataFrame Info (.info())")
    buffer = []
    df.info(buf=buffer)
    st.text('\n'.join(buffer))

    st.subheader("Descriptive Statistics (.describe())")
    st.dataframe(df.describe(include="all").transpose())

    # 2. Clean & Preprocess
    st.header("2. Clean & Preprocess")

    drop_dupes = st.checkbox("Drop duplicates", value=True)
    missing_strategy = st.selectbox(
        "Missing values handling strategy",
        ["Drop rows", "Fill with mean (numeric) and mode (categorical)", "Do nothing"],
        index=1
    )
    df_clean = clean_data(df.copy(), drop_dupes, missing_strategy)
    
    st.write(f"Shape after cleaning: {df_clean.shape}")

    st.subheader("Missing values after cleaning")
    st.dataframe(df_clean.isnull().sum())

    # Identify target and features for ML
    st.header("3. Feature Selection & Engineering")

    all_columns = df_clean.columns.tolist()
    target_col = st.selectbox("Select Target Column", all_columns)

    feature_cols = st.multiselect(
        "Select Feature Columns (at least one)",
        [col for col in all_columns if col != target_col],
        default=[col for col in all_columns if col != target_col]
    )
    if not feature_cols:
        st.warning("Please select at least one feature column to proceed.")
        st.stop()

    # Check for categorical features to encode
    cat_cols = [col for col in feature_cols if df_clean[col].dtype == 'object']
    if cat_cols:
        st.write(f"Encoding categorical features: {cat_cols}")
        df_encoded = encode_features(df_clean[feature_cols].copy(), cat_cols)
    else:
        df_encoded = df_clean[feature_cols].copy()

    X = df_encoded
    y = df_clean[target_col]

    # Optional feature engineering example:
    st.subheader("Optional Feature Engineering")
    add_feature = st.checkbox("Add new feature (product of first two numeric features)")
    if add_feature:
        numeric_features = list(X.select_dtypes(include=np.number).columns)
        if len(numeric_features) >= 2:
            new_feat = f"{numeric_features[0]}_x_{numeric_features[1]}"
            X[new_feat] = X[numeric_features[0]] * X[numeric_features[1]]
            st.write(f"Added new feature '{new_feat}'")
        else:
            st.warning("Not enough numeric features for product; skipping feature engineering.")

    # 4. Split Data
    st.header("4. Split Data")

    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=25) / 100
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    except Exception as e:
        st.error(f"Error splitting data: {e}")
        st.stop()

    st.write(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 5. Select & Train ML Algorithm
    st.header("5. Select and Train Model")

    model_type = st.selectbox("Choose model type", ["Classification", "Regression"])

    model = None
    if model_type == "Classification":
        st.write("Model: Logistic Regression")
        c_val = st.number_input("Inverse of regularization strength (C)", value=1.0, min_value=0.01)
        max_iter = st.number_input("Max iterations", value=100, min_value=10)
        try:
            model = LogisticRegression(C=c_val, max_iter=int(max_iter))
        except Exception as e:
            st.error(f"Model init error: {e}")
    else:
        st.write("Model: Random Forest Regressor")
        n_estimators = st.number_input("Number of trees", value=100, min_value=10)
        max_depth = st.number_input("Max depth (0 = None)", min_value=0, value=0)
        max_depth = None if max_depth == 0 else int(max_depth)
        try:
            model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=max_depth, random_state=random_state)
        except Exception as e:
            st.error(f"Model init error: {e}")

    if model is not None:
        with st.spinner('Training model...'):
            try:
                model.fit(X_train, y_train)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Training error: {e}")
                st.stop()

        # 6. Predict
        st.header("6. Make Predictions")
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # 7. Evaluate Accuracy
        st.header("7. Evaluate Model Performance")

        if model_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: **{acc:.4f}**")

            cm = confusion_matrix(y_test, y_pred)
            unique_classes = sorted(y.unique())
            fig_cm = plot_confusion_matrix(cm, unique_classes)
            st.pyplot(fig_cm)

        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            st.write(f"R² score: **{r2:.4f}**")
            st.write(f"Mean Squared Error (MSE): **{mse:.4f}**")
            st.write(f"Root Mean Squared Error (RMSE): **{rmse:.4f}**")

            fig_reg = plot_regression_performance(y_test, y_pred)
            st.pyplot(fig_reg)

        # 8. Optimize & Retrain (Basic option using user input)
        st.header("8. Hyperparameter Tuning (Retraining)")

        if st.button("Retrain with current parameters"):
            with st.spinner("Retraining..."):
                try:
                    model.fit(X_train, y_train)
                    st.success("Retrain successful!")
                except Exception as e:
                    st.error(f"Retrain error: {e}")

        # 9. Automated EDA profiling report
        st.header("Exploratory Data Analysis Profile")

        profile = ProfileReport(df_clean, title="YData Profiling Report", explorative=True)
        st_profile_report(profile)

else:
    st.info("Upload a data file to start the analysis.")

# ---------- Additional helper plots ----------

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    return fig

def plot_regression_performance(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Regression: Actual vs Predicted')
    return fig
