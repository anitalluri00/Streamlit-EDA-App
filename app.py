import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

SUPPORTED = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel (xls)",
    ".txt": "Text",
    ".tsv": "TSV",
    ".json": "JSON"
}

st.set_page_config(page_title="Comprehensive EDA App", layout="wide")

st.title("Comprehensive EDA Web App")
st.markdown("""
Upload your data in any common format (CSV, XLSX, XLS, JSON, TXT, TSV).  
The app will display all EDA metrics and a profiling report.
""")

uploaded_file = st.file_uploader(
    "Choose a data file", type=[k[1:] for k in SUPPORTED.keys()])
    
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
        except Exception:
            uploaded_file.seek(0)
            return pd.read_table(uploaded_file)
    else:
        return None

if uploaded_file:
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    st.success(f"File uploaded: **{uploaded_file.name}** ({SUPPORTED.get(ext,ext)})")

    try:
        df = read_file(uploaded_file, ext)
        if df is None:
            st.error("File type not supported yet.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.write("## Head of Data")
    st.dataframe(df.head())

    st.write("## Shape & Columns")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(f"Columns: {list(df.columns)}")

    st.write("## Data Types")
    st.write(df.dtypes)

    st.write("## Missing Values per Column")
    st.write(df.isnull().sum())

    st.write("## Duplicate Rows")
    duplicates = df[df.duplicated()]
    if len(duplicates) > 0:
        st.dataframe(duplicates)
    else:
        st.write("No duplicate rows found.")

    st.write("## Descriptive Statistics")
    st.write(df.describe(include="all").transpose())

    st.write("## Unique Values per Column")
    st.write(df.nunique())

    st.write("## Correlation Matrix (Numeric Columns)")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 0:
        corr = numeric_df.corr()
        st.dataframe(corr)
        st.write("## Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation analysis.")

    st.write("## Univariate Analysis (Numeric Columns - first 5)")
    numeric_cols = numeric_df.columns.tolist()
    for col in numeric_cols[:5]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty:
                st.write(f"Skipping histogram for '{col}' (no valid numeric data).")
                continue
            fig, ax = plt.subplots()
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title(f'Histogram and KDE of {col}')
            st.pyplot(fig)
        except Exception as e:
            st.write(f"Skipping histogram for '{col}' due to error: {e}")

    st.write("## Boxplot (Outlier Detection - Numeric Columns - first 5)")
    for col in numeric_cols[:5]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty:
                st.write(f"Skipping boxplot for '{col}' (no valid numeric data).")
                continue
            fig, ax = plt.subplots()
            sns.boxplot(x=data, ax=ax)
            ax.set_title(f'Boxplot of {col}')
            st.pyplot(fig)
        except Exception as e:
            st.write(f"Skipping boxplot for '{col}' due to error: {e}")

    st.write("## Pairplot (First 5 Numeric Columns)")
    if len(numeric_cols) >= 2:
        try:
            sns.pairplot(df[numeric_cols[:5]].dropna())
            st.pyplot()
        except Exception as e:
            st.write(f"Could not create pairplot: {e}")
    else:
        st.write("Not enough numeric columns for pairplot.")

    st.write("## Value Counts (First 3 Categorical Columns)")
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    for col in cat_cols[:3]:
        st.write(f"### {col}")
        st.write(df[col].value_counts())

    st.write("## Automated EDA Profiling (ydata-profiling)")
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    st_profile_report(profile)

else:
    st.info("Upload a file to get started.")
