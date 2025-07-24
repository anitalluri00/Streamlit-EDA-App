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

if uploaded_file:
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    st.success(f"File uploaded: **{uploaded_file.name}** ({SUPPORTED.get(ext,ext)})")

    try:
        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif ext in [".xlsx"]:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif ext == ".xls":
            # xlrd required for .xls files
            df = pd.read_excel(uploaded_file, engine='xlrd')
        elif ext == ".tsv":
            df = pd.read_csv(uploaded_file, sep="\t")
        elif ext == ".json":
            df = pd.read_json(uploaded_file)
        elif ext == ".txt":
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
            except:
                uploaded_file.seek(0)
                df = pd.read_table(uploaded_file)
        else:
            st.error("File type not supported yet.")
            st.stop()

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Show head of data
    st.write("## Head of Data")
    st.dataframe(df.head())

    # Shape and columns
    st.write("## Shape & Columns")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(f"Columns: {list(df.columns)}")

    # Data types
    st.write("## Data Types")
    st.write(df.dtypes)

    # Missing values
    st.write("## Missing Values per Column")
    st.write(df.isnull().sum())

    # Duplicates
    st.write("## Duplicate Rows")
    duplicates = df[df.duplicated()]
    if len(duplicates) > 0:
        st.dataframe(duplicates)
    else:
        st.write("No duplicate rows found.")

    # Descriptive statistics
    st.write("## Descriptive Statistics")
    st.write(df.describe(include="all").transpose())

    # Unique values per column
    st.write("## Unique Values per Column")
    st.write(df.nunique())

    # Correlation matrix for numeric columns only
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

    # Univariate analysis (numeric columns)
    st.write("## Univariate Analysis (Numeric Columns - first 5)")
    numeric_cols = numeric_df.columns.tolist()
    for col in numeric_cols[:5]:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Histogram and KDE of {col}')
        st.pyplot(fig)

    # Boxplot for outlier detection
    st.write("## Boxplot (Outlier Detection - Numeric Columns - first 5)")
    for col in numeric_cols[:5]:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        st.pyplot(fig)

    # Pairplot for numeric columns
    st.write("## Pairplot (First 5 Numeric Columns)")
    if len(numeric_cols) >= 2:
        sns.pairplot(df[numeric_cols[:5]].dropna())
        st.pyplot()
    else:
        st.write("Not enough numeric columns for pairplot.")

    # Value counts for categorical columns (first 3)
    st.write("## Value Counts (First 3 Categorical Columns)")
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    for col in cat_cols[:3]:
        st.write(f"### {col}")
        st.write(df[col].value_counts())

    # Automated profiling report
    st.write("## Automated EDA Profiling (ydata-profiling)")
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    st_profile_report(profile)

else:
    st.info("Upload a file to get started.")
