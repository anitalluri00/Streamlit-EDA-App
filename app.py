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
    ".xls": "Excel",
    ".txt": "Text",
    ".tsv": "TSV",
    ".json": "JSON"
    # Optionally add PDF, DOCX parsing for tables with tabula/pypdf/pandas-docx
}

st.set_page_config(page_title="Comprehensive EDA App", layout="wide")

st.title("Comprehensive EDA Web App")
st.markdown("""
Upload your data in any common format (CSV, XLSX, JSON, TXT, TSV).  
The app will display all EDA metrics and a profiling report.
""")

uploaded_file = st.file_uploader("Choose a data file", type=list([k[1:] for k in SUPPORTED.keys()]))

if uploaded_file:
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    st.success(f"File uploaded: **{uploaded_file.name}** ({SUPPORTED.get(ext,ext)})")

    # Reading different formats
    if ext == ".csv":
        df = pd.read_csv(uploaded_file)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file)
    elif ext == ".tsv":
        df = pd.read_csv(uploaded_file, sep="\t")
    elif ext == ".json":
        df = pd.read_json(uploaded_file)
    elif ext == ".txt":
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        st.error("File type not supported yet.")
        st.stop()

    st.write("## Head of Data")
    st.dataframe(df.head())

    st.write("## Shape & Columns")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(f"Columns: {list(df.columns)}")

    st.write("## Data Types")
    st.write(df.dtypes)

    st.write("## Checking Missing Values")
    st.write(df.isnull().sum())

    st.write("## Duplicates")
    st.write(df[df.duplicated()])

    st.write("## Descriptive Statistics")
    st.write(df.describe(include="all"))

    st.write("## Unique values per column")
    st.write(df.nunique())

    st.write("## Correlation Matrix (Numeric Columns)")
    st.dataframe(df.corr())
    st.write("## Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    st.write("## Univariate Analysis")
    for col in df.select_dtypes(include=np.number).columns[:5]:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    
    st.write("## Boxplot (Outlier Detection: Numeric Columns)")
    for col in df.select_dtypes(include=np.number).columns[:5]:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    st.write("## Pairplot (First 5 Numeric Columns)")
    if len(df.select_dtypes(include=np.number).columns) >= 2:
        cols = df.select_dtypes(include=np.number).columns[:5]
        sns.pairplot(df[cols].dropna())
        st.pyplot()

    st.write("## Value Counts (First 3 Categorical Columns)")
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols[:3]:
        st.write(f"### {col}")
        st.write(df[col].value_counts())

    st.write("## Automated EDA Profiling (ydata-profiling)")
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    st_profile_report(profile)
else:
    st.info("Upload a file to get started.")
