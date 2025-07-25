import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Import tabula for PDF table extraction with error handling
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

st.set_page_config(page_title="Comprehensive EDA App with ydata-sdk", layout="wide")

st.title("Comprehensive EDA Web App with ydata-sdk")
st.markdown("""
Upload your dataset in one of the supported formats: CSV, XLSX, XLS, JSON, TXT, TSV, PDF (with tables).  
The app performs detailed EDA and generates an interactive profile report using **ydata-sdk**.
""")

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
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            dfs = tabula.read_pdf(tmp.name, pages="all", multiple_tables=True)
            if len(dfs) == 0:
                raise ValueError("No tables found in PDF file.")
            df_pdf = dfs[0]
            # Convert all object dtype columns to string to avoid pyarrow errors
            for col in df_pdf.select_dtypes(include=['object']).columns:
                df_pdf[col] = df_pdf[col].astype(str)
            return df_pdf
    else:
        raise ValueError(f"Unsupported file type: {ext}")

uploaded_file = st.file_uploader(
    "Choose a data file", type=[k[1:] for k in SUPPORTED.keys()])

if uploaded_file:
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    st.success(f"File uploaded: **{uploaded_file.name}** ({SUPPORTED.get(ext, ext)})")

    try:
        df = read_file(uploaded_file, ext)
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

    st.write("## Univariate Analysis (Numeric Columns — first 5)")
    numeric_cols = numeric_df.columns.tolist()
    for col in numeric_cols[:5]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty:
                st.write(f"Skipping histogram for '{col}' — no valid numeric data.")
                continue
            fig, ax = plt.subplots()
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title(f'Histogram & KDE of {col}')
            st.pyplot(fig)
        except Exception as e:
            st.write(f"Skipping histogram for '{col}' due to error: {e}")

    st.write("## Boxplot (Outlier Detection — Numeric Columns — first 5)")
    for col in numeric_cols[:5]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty:
                st.write(f"Skipping boxplot for '{col}' — no valid numeric data.")
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
            # FIX: capture returned PairGrid, pass its .fig to st.pyplot
            pair_grid = sns.pairplot(df[numeric_cols[:5]].dropna())
            st.pyplot(pair_grid.fig)
        except Exception as e:
            st.write(f"Could not create pairplot: {e}")
    else:
        st.write("Not enough numeric columns for pairplot.")

    st.write("## Value Counts (First 3 Categorical Columns)")
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    for col in cat_cols[:3]:
        st.write(f"### {col}")
        st.write(df[col].value_counts())

    st.write("## Automated EDA Profiling (ydata-sdk)")
    profile = ProfileReport(df, title="YData Profiling Report", explorative=True)
    st_profile_report(profile)

else:
    st.info("Upload a file to get started.")
