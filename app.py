"""
Streamlit EDA + Feature Engineering + Quick ML pipeline
Single-file app: app.py
Supports .csv, .xlsx, .xls uploads.

Usage: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import base64
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Full EDA Flow — single app", page_icon="📊")
st.title("📊 Full EDA + Feature Engineering + Quick ML — single `app.py`")
st.markdown("Upload your dataset (`.csv`, `.xlsx`, `.xls`) and walk through an end-to-end EDA process.")

# ---------------------------
# Utility helpers
# ---------------------------
def load_file(uploaded_file):
    fname = uploaded_file.name.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif fname.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Upload CSV/XLS/XLSX.")
        return None

def get_column_types(df, thresh=0.05):
    """Return lists of numeric and categorical columns. thresh is fraction unique cutoff for numeric->categorical"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # But numeric-like but with few unique values might be categorical
    cat_from_num = [c for c in num_cols if df[c].nunique() / len(df) < thresh]
    num_cols = [c for c in num_cols if c not in cat_from_num]
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist() + cat_from_num
    # remove duplicates and keep order
    cat_cols = list(dict.fromkeys(cat_cols))
    num_cols = list(dict.fromkeys(num_cols))
    return cat_cols, num_cols

def quick_download_link(df, filename="cleaned.csv"):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# ---------------------------
# Sidebar - file upload & options
# ---------------------------
st.sidebar.header("1) Upload & options")
uploaded_file = st.sidebar.file_uploader("Upload CSV / XLSX / XLS", type=['csv','xlsx','xls'])
sample_data = st.sidebar.checkbox("Use sample dataset (Iris)", value=False)

if sample_data and uploaded_file is None:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    if 'target' in df.columns:
        df.rename(columns={'target': 'species'}, inplace=True)
    st.sidebar.success("Loaded iris sample dataset.")
elif uploaded_file:
    df = load_file(uploaded_file)
else:
    st.info("Upload a dataset or check 'Use sample dataset'.")
    st.stop()

# Make a copy to operate on
df_original = df.copy()
st.write("### Quick data preview")
st.dataframe(df_original.head())

# ---------------------------
# 2) Column detection
# ---------------------------
st.sidebar.header("2) Column detection")
cat_cols, num_cols = get_column_types(df)
st.sidebar.write("Detected categorical columns:", cat_cols)
st.sidebar.write("Detected numerical columns:", num_cols)

# Allow user to override
st.sidebar.header("Override column types (optional)")
user_cat = st.sidebar.multiselect("Force categorical columns", options=df.columns.tolist(), default=cat_cols)
user_num = st.sidebar.multiselect("Force numerical columns", options=[c for c in df.columns.tolist() if c not in user_cat], default=num_cols)

cat_cols = user_cat
num_cols = user_num

# ---------------------------
# Quick checks
# ---------------------------
st.header("=========== EDA Steps ============")
st.subheader("----------- DATA cleaning and ANALYSIS -------------")
st.markdown("**Quick checks**")
c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Size (cells)", df.size)
with c4:
    st.write("Data types:")
    st.write(df.dtypes)

st.write("**Head (5 rows)**")
st.dataframe(df.head())
st.write("**Tail (5 rows)**")
st.dataframe(df.tail())

st.write("**Info and memory**")
buf = io.StringIO()
df.info(buf=buf)
s = buf.getvalue()
st.text(s)

# ---------------------------
# 4) Missing value analysis
# ---------------------------
st.subheader("4) Missing value analysis")
miss = df.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
if miss.empty:
    st.success("No missing values detected.")
else:
    st.write("Columns with missing values (count):")
    st.dataframe(miss)
    st.write("Missing value percentage:")
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    st.dataframe(miss_pct[miss_pct>0])

    st.write("Missing patterns (heatmap)")
    fig, ax = plt.subplots(figsize=(10,3))
    sns.heatmap(df.isnull(), cbar=False)
    st.pyplot(fig)

# ---------------------------
# 5) Data quality checks / cleaning heuristics
# ---------------------------
st.subheader("5) Data quality checks / Cleaning")
st.markdown("""
The app will attempt to:
- Trim whitespace from string columns
- Remove unprintable characters
- Attempt to coerce numeric columns where possible
- Report suspicious columns (mixed types / punctuation in numeric columns)
""")

if st.button("Run automatic cleaning heuristics"):
    df = df.copy()
    # Trim and clean strings
    for c in df.select_dtypes(include=['object','category']).columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].str.replace(r'[\r\n\t]+', ' ', regex=True)
        df[c] = df[c].replace({'': np.nan})
    # Try coercing numeric-like columns
    for c in df.columns:
        if c in num_cols:
            # attempt to remove commas and currency symbols
            df[c] = df[c].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
            df[c] = pd.to_numeric(df[c], errors='coerce')
    st.success("Automatic cleaning applied. Re-detect columns in the sidebar if desired.")
    st.write(df.head())

# Detect columns with mixed types or punctuation
st.write("Columns with mixed types or suspicious entries (sample):")
mixed = []
for c in df.columns:
    num = pd.to_numeric(df[c], errors='coerce')
    if num.isnull().any() and df[c].dtype == object:
        sample_bad = df[c].loc[num.isnull()].dropna().unique()[:5]
        if len(sample_bad)>0:
            mixed.append((c, sample_bad))
if mixed:
    for c, samp in mixed:
        st.write(f"- {c}: examples -> {samp}")
else:
    st.write("No obvious mixed-type columns detected.")

# ---------------------------
# 6) Categorical column analysis
# ---------------------------
st.subheader("6) Categorical column analysis")
if len(cat_cols)==0:
    st.info("No categorical columns detected. You can force columns in the sidebar.")
else:
    cat_to_analyze = st.selectbox("Choose categorical column to analyze", options=cat_cols)
    if cat_to_analyze:
        st.write("Frequency table")
        freq = df[cat_to_analyze].value_counts(dropna=False)
        st.dataframe(freq)
        st.write("Relative frequency (%)")
        rel = df[cat_to_analyze].value_counts(normalize=True, dropna=False)*100
        st.dataframe(rel.round(3))
        # Bar chart
        st.write("Bar chart (top 20)")
        fig = px.bar(x=freq.index[:20].astype(str), y=freq.values[:20], labels={'x':cat_to_analyze, 'y':'count'}, title=f"Bar chart of {cat_to_analyze}")
        st.plotly_chart(fig, use_container_width=True)
        # Pie chart
        st.write("Pie chart (top 10)")
        fig2 = px.pie(names=freq.index[:10].astype(str), values=freq.values[:10], title=f"Pie chart of {cat_to_analyze}")
        st.plotly_chart(fig2, use_container_width=True)
        st.write("Understanding / notes:")
        st.write(f"- Column `{cat_to_analyze}` has {df[cat_to_analyze].nunique()} unique values.")
        st.write("- Check for high-cardinality (many unique categories) which may require grouping/embedding rather than one-hot encoding.")
        st.write("- Check frequent vs rare categories: rare categories may need to be grouped as 'Other'.")

# ---------------------------
# 7) Numerical column analysis
# ---------------------------
st.subheader("7) Numerical column analysis")
if len(num_cols)==0:
    st.info("No numerical columns detected. You can force columns in the sidebar.")
else:
    num_to_analyze = st.selectbox("Choose numerical column to analyze", options=num_cols)
    if num_to_analyze:
        st.write("Statistical summary")
        st.dataframe(df[num_to_analyze].describe().to_frame().T)
        # Histogram + KDE
        st.write("Histogram and density")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[num_to_analyze].dropna(), kde=True)
        st.pyplot(fig)
        st.write("Data distribution notes:")
        s = df[num_to_analyze].dropna()
        skew = s.skew()
        kurt = s.kurt()
        st.write(f"- Skewness: {skew:.3f}  |  Kurtosis: {kurt:.3f}")
        if abs(skew) > 1:
            st.write("- Distribution is highly skewed; consider log / boxcox transformations.")
        elif abs(skew) > 0.5:
            st.write("- Moderate skewness.")
        else:
            st.write("- Approximately symmetric distribution.")

# ---------------------------
# 8) Outlier analysis
# ---------------------------
st.subheader("8) Outlier analysis")
if len(num_cols) > 0:
    cols_for_outlier = st.multiselect("Select numerical columns for outlier visualization", options=num_cols, default=num_cols[:3])
    if cols_for_outlier:
        fig, axes = plt.subplots(len(cols_for_outlier), 1, figsize=(10, 4*len(cols_for_outlier)))
        if len(cols_for_outlier)==1:
            axes = [axes]
        for ax, c in zip(axes, cols_for_outlier):
            sns.boxplot(x=df[c].dropna(), ax=ax)
            ax.set_title(f"Boxplot for {c}")
        st.pyplot(fig)

        st.write("Outlier treatment options:")
        outlier_method = st.selectbox("Treatment method", [
            "None",
            "Clip to percentiles (1st/99th)",
            "Replace with median",
            "Winsorize (1%)"
        ])
        if st.button("Apply outlier treatment"):
            df = df.copy()
            if outlier_method == "Clip to percentiles (1st/99th)":
                for c in cols_for_outlier:
                    low = df[c].quantile(0.01)
                    high = df[c].quantile(0.99)
                    df[c] = df[c].clip(lower=low, upper=high)
            elif outlier_method == "Replace with median":
                for c in cols_for_outlier:
                    med = df[c].median()
                    q1 = df[c].quantile(0.25)
                    q3 = df[c].quantile(0.75)
                    iqr = q3 - q1
                    out_mask = (df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)
                    df.loc[out_mask, c] = med
            elif outlier_method == "Winsorize (1%)":
                for c in cols_for_outlier:
                    low = df[c].quantile(0.01)
                    high = df[c].quantile(0.99)
                    df[c] = np.where(df[c] < low, low, df[c])
                    df[c] = np.where(df[c] > high, high, df[c])
            st.success("Outlier treatment applied.")
            st.write(df[cols_for_outlier].describe())

# ---------------------------
# 9) Bi-variate & Multi-variate analysis
# ---------------------------
st.subheader("9) Bivariate & Multivariate analysis")
st.write("Correlation matrix (numerical columns)")
if len(num_cols)>0:
    corr_method = st.selectbox("Correlation method", options=["pearson", "spearman"], index=0)
    corr = df[num_cols].corr(method=corr_method)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    st.pyplot(fig)
    st.write("Notes: correlation shows linear (pearson) or rank (spearman) relationships among numeric features.")

st.write("Pairwise scatter (select columns)")
pair_cols = st.multiselect("Select up to 4 columns for pairplot/scatter matrix", options=num_cols, max_selections=4)
if len(pair_cols)>=2:
    sns_pair = sns.pairplot(df[pair_cols].dropna().sample(min(500, len(df))), diag_kind="kde", plot_kws={'s':20, 'alpha':0.6})
    st.pyplot(sns_pair.fig)

# ---------------------------
# Feature Engineering
# ---------------------------
st.header("------------- Feature Engineering -----------------")
st.subheader("10) Encoding")
st.write("Choose encoding strategy for categorical columns")
encoding_choice = st.selectbox("Encoding", ["Label Encoding (good for ordinal or small cardinality)", "One-Hot Encoding (for nominal, beware high-cardinality)", "Leave as-is"], index=2)
encode_cols = st.multiselect("Columns to encode", options=cat_cols, default=cat_cols[:3])
if st.button("Apply encoding"):
    df = df.copy()
    if encoding_choice.startswith("Label"):
        for c in encode_cols:
            le = LabelEncoder()
            df[c] = df[c].astype(str).fillna("NA")
            df[c] = le.fit_transform(df[c])
    elif encoding_choice.startswith("One-Hot"):
        st.write("One-hot encoding may create many columns. Proceeding...")
        df = pd.get_dummies(df, columns=encode_cols, dummy_na=True, drop_first=False)
    st.success("Encoding applied.")
    st.write("Data shape now:", df.shape)

st.subheader("11) Scaling")
st.write("Choose scaling for numeric columns")
scaler_choice = st.selectbox("Scaling", ["None", "StandardScaler (zero mean unit variance)", "MinMaxScaler (0-1)"], index=0)
scale_cols = st.multiselect("Numeric columns to scale", options=num_cols, default=num_cols[:4])
if st.button("Apply scaling"):
    df = df.copy()
    if scaler_choice != "None":
        if scaler_choice.startswith("Standard"):
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        st.success("Scaling applied.")
        st.write(df[scale_cols].describe())
    else:
        st.info("No scaling applied.")

st.subheader("12) Transformation (optional)")
transform_choice = st.selectbox("Transformation", ["None", "Log (x+1)", "Box-Cox (only positive)", "Yeo-Johnson (scipy) - not implemented"], index=0)
transform_cols = st.multiselect("Numeric columns to transform", options=num_cols, default=num_cols[:2])
if st.button("Apply transformation"):
    df = df.copy()
    if transform_choice == "Log (x+1)":
        for c in transform_cols:
            df[c] = np.log1p(df[c].clip(lower=0))
    elif transform_choice == "Box-Cox (only positive)":
        from scipy.stats import boxcox
        for c in transform_cols:
            s = df[c].dropna()
            if (s <= 0).any():
                st.warning(f"{c} has non-positive values; skipping Box-Cox.")
                continue
            df[c] = pd.Series(boxcox(s)[0], index=s.index)
    st.success("Transform applied.")

# ---------------------------
# Feature Selection
# ---------------------------
st.header("------------- Feature Selection --------------")
st.subheader("13) Selecting Important features for ML model")
st.write("Choose a target column (the label) to run quick feature selection and a basic ML model.")
target = st.selectbox("Select target (for supervised feature selection / models). If none, select skip.", options=[None] + list(df.columns), index=0)
task_type = None
if target:
    if df[target].dtype in [np.int64, np.float64] and df[target].nunique() > 10:
        task_type = st.selectbox("Detected numeric target -> choose task", options=["Regression", "Classification"], index=0)
    else:
        task_type = st.selectbox("Detected categorical/low-card target -> choose task", options=["Classification", "Regression"], index=0)

    st.write("Handle missing target rows by dropping (for demo).")
    df_model = df.dropna(subset=[target]).copy()
    X = df_model.drop(columns=[target])
    y = df_model[target]

    # Basic automatic numeric/categorical split for features
    feat_cat, feat_num = get_column_types(X)
    st.write(f"Feature counts - categorical: {len(feat_cat)}, numeric: {len(feat_num)}")

    # Basic encoding for modeling
    X_proc = X.copy()
    for c in feat_cat:
        X_proc[c] = X_proc[c].astype(str).fillna("NA")
        X_proc[c] = LabelEncoder().fit_transform(X_proc[c])

    # Fill remaining missing numeric with median
    for c in X_proc.select_dtypes(include=[np.number]).columns:
        X_proc[c] = X_proc[c].fillna(X_proc[c].median())

    # Simple feature selection using SelectKBest
    k = st.slider("Select top-k features (SelectKBest)", min_value=1, max_value=min(30, X_proc.shape[1]), value=min(10, X_proc.shape[1]))
    if task_type == "Classification":
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    try:
        selector.fit(X_proc.fillna(0), y)
        scores = selector.scores_
        cols = X_proc.columns
        top_idx = np.argsort(scores)[-k:][::-1]
        top_features = cols[top_idx].tolist()
        st.write("Top features by univariate test:")
        st.write(pd.DataFrame({'feature': top_features, 'score': scores[top_idx]}))
    except Exception as e:
        st.warning("SelectKBest failed: " + str(e))
        top_features = X_proc.columns.tolist()[:k]

    # Feature importance via RandomForest
    if st.checkbox("Show tree-based feature importances (RandomForest)"):
        try:
            # Choose simple estimator
            if task_type == "Classification":
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_proc.fillna(0), y)
            imp = pd.Series(rf.feature_importances_, index=X_proc.columns).sort_values(ascending=False)
            st.dataframe(imp.head(30).to_frame("importance"))
            fig = px.bar(x=imp.head(20).index, y=imp.head(20).values, title="Top 20 Feature Importances (RF)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("RandomForest failed: " + str(e))

    st.write("Proceed to quick model training")
    if st.button("Train a simple model (train/test split 70/30)"):
        X_train, X_test, y_train, y_test = train_test_split(X_proc[top_features], y, test_size=0.3, random_state=42)
        if task_type == "Classification":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, preds))
            st.text(classification_report(y_test, preds))
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.write("RMSE:", mean_squared_error(y_test, preds, squared=False))
            st.write("R^2:", r2_score(y_test, preds))
        st.success("Model trained. This is a quick baseline — improve with CV, tuning, pipelines.")

# ---------------------------
# Dimension reduction
# ---------------------------
st.header("---------- Dimension Reduction methods ----------")
st.subheader("14) PCA & SVD (TruncatedSVD)")
dd_cols = st.multiselect("Choose numeric columns for PCA/SVD", options=num_cols, default=num_cols[:5])
n_components = st.slider("n components", min_value=1, max_value=min(10, max(1, len(dd_cols))), value=min(3, max(1, len(dd_cols))))
if st.button("Run PCA / SVD"):
    Xdr = df[dd_cols].dropna()
    # fill missing
    Xdr = Xdr.fillna(Xdr.median())
    pca = PCA(n_components=n_components, random_state=42)
    pca_res = pca.fit_transform(Xdr)
    st.write("Explained variance ratio per component:", pca.explained_variance_ratio_)
    df_pca = pd.DataFrame(pca_res, columns=[f"PC{i+1}" for i in range(n_components)])
    if n_components >= 2:
        fig = px.scatter(df_pca, x="PC1", y="PC2", title="PCA scatter (PC1 vs PC2)")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_pca.head())

    # SVD (works with sparse / non-centered data like TF-IDF)
    svd = TruncatedSVD(n_components=min(n_components, Xdr.shape[1]-1 or 1), random_state=42)
    try:
        svd_res = svd.fit_transform(Xdr)
        st.write("SVD explained variance ratio (approx):", svd.explained_variance_ratio_[:n_components])
    except Exception as e:
        st.warning("SVD failed: " + str(e))

# ---------------------------
# Save & download cleaned data
# ---------------------------
st.header("Save & Export")
if st.button("Download cleaned data CSV"):
    st.markdown(quick_download_link(df, filename="cleaned_data.csv"), unsafe_allow_html=True)

st.write("If you'd like a local copy of the app, download this repo or copy `app.py`, `Dockerfile`, and `requirements.txt`.")
