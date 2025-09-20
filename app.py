import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from scipy import stats
import base64
import os

st.set_page_config(page_title="Full EDA — Single app.py", layout="wide")

# --- Styling / neat background UI ---
st.markdown(
    """
    <style>
    .stApp { 
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      color: #0f172a;
    }
    .sidebar .sidebar-content {
      background: linear-gradient(180deg, #ffffffcc, #f1f5f9cc);
      backdrop-filter: blur(5px);
    }
    .big-font { font-size:18px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper utilities -----------------------------------------------------------
@st.cache_data
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError('Unsupported file type. Upload .csv, .xlsx or .xls')

def detect_columns(df, cat_threshold=0.05):
    """Return lists of categorical and numerical column names.
    cat_threshold: fraction of unique values threshold to treat numeric-like as categorical.
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Some numeric-looking columns might be categorical (e.g., small unique counts)
    for col in numerical[:]:
        num_unique = df[col].nunique(dropna=True)
        if num_unique / len(df) < cat_threshold or num_unique < 10:
            # treat as categorical
            numerical.remove(col)
            non_numeric.append(col)
    return sorted(non_numeric), sorted(numerical)  # categorical, numerical

def quick_checks(df):
    checks = {
        'shape': df.shape,
        'size': df.size,
        'len (rows)': len(df),
        'head': df.head(),
        'tail': df.tail(),
        'info': df.info(buf=None)
    }
    return checks

def missing_value_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mz = pd.concat([mis_val, mis_val_percent], axis=1)
    mz.columns = ['missing_count', 'missing_percent']
    mz = mz[mz['missing_count'] > 0].sort_values('missing_percent', ascending=False)
    return mz

def clean_punctuation_in_numeric(df, cols):
    """Attempt to remove commas, $ signs etc from numeric columns and coerce to numeric."""
    for c in cols:
        if df[c].dtype == object or df[c].dtype == 'O':
            df[c] = df[c].astype(str).str.replace(r'[,$%]', '', regex=True).str.strip()
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().sum() > 0:
                df[c] = coerced
    return df

def frequency_table(df, col):
    freq = df[col].value_counts(dropna=False)
    rel = df[col].value_counts(normalize=True, dropna=False).round(4)
    return pd.concat([freq, rel], axis=1).rename(columns={col: 'frequency', 0: 'relative'})

def statistical_summary(df, numerical_cols):
    return df[numerical_cols].describe().T

def plot_histogram(df, col, bins=30):
    fig = px.histogram(df, x=col, nbins=bins, marginal='box')
    return fig

def plot_box(df, col):
    fig = px.box(df, y=col, points='outliers')
    return fig

def iqr_treatment(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return lower, upper

def zscore_treatment(series, threshold=3):
    z = np.abs(stats.zscore(series.dropna()))
    return z > threshold

def download_link(df, filename='cleaned.csv'):
    towrite = BytesIO()
    if filename.endswith('.csv'):
        df.to_csv(towrite, index=False)
    else:
        df.to_excel(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f"data:application/octet-stream;base64,{b64}"
    return href


# --- Sidebar: file upload + options ----------------------------------------
st.sidebar.title('Upload & Options')
uploaded_file = st.sidebar.file_uploader('Upload .csv / .xlsx / .xls', type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)
    except Exception as e:
        st.sidebar.error(f'Error reading file: {e}')
        st.stop()

    st.sidebar.success('File loaded — nice.')
    st.sidebar.write('Rows:', df.shape[0], 'Columns:', df.shape[1])

    # allow user to set threshold for numeric->categorical
    cat_threshold = st.sidebar.slider('Categorical unique-value threshold (fraction of rows)', 0.01, 0.2, 0.05)
    cat_cols, num_cols = detect_columns(df, cat_threshold=cat_threshold)

    st.sidebar.write('Detected categorical columns:', len(cat_cols))
    st.sidebar.write('Detected numerical columns:', len(num_cols))

    # --- Main app body ------------------------------------------------------
    st.title('Full EDA pipeline — single-file Streamlit app')
    st.markdown('Follow the sections on the left; everything runs live.')

    # SECTION 1: Read the data ------------------------------------------------
    st.header('1) Read the data')
    st.markdown(f'**File name:** {uploaded_file.name} — **shape:** {df.shape}')
    if st.checkbox('Show raw data (first 200 rows)'):
        st.dataframe(df.head(200))

    # SECTION 2: Create categorical and numerical columns ---------------------
    st.header('2) Create categorical and numerical column lists')
    st.write('Categorical columns detected (sample):')
    st.write(cat_cols[:20])
    st.write('Numerical columns detected (sample):')
    st.write(num_cols[:20])

    # SECTION 3: Quick checks -------------------------------------------------
    st.header('3) Data quick checks')
    st.subheader('A) shape  B) size  C) len  D) head  E) tail  F) info')
    c1, c2 = st.columns(2)
    with c1:
        st.write('shape: ', df.shape)
        st.write('size: ', df.size)
        st.write('len (rows): ', len(df))
    with c2:
        st.write('head:')
        st.dataframe(df.head())
        st.write('tail:')
        st.dataframe(df.tail())

    buffer = BytesIO()
    df.info(buf=buffer)
    s = buffer.getvalue().decode()
    st.text('info:\n' + s)

    # SECTION 4: Missing value analysis -------------------------------------
    st.header('4) Missing value analysis')
    mis_table = missing_value_table(df)
    if mis_table.empty:
        st.success('No missing values detected!')
    else:
        st.dataframe(mis_table)
        fig = px.bar(mis_table.reset_index(), x='index', y='missing_percent', title='Missing % by column')
        st.plotly_chart(fig, use_container_width=True)

    # SECTION 5: Data quality checks / cleaning -------------------------------
    st.header('5) Data quality checks / cleaning')
    st.markdown('Common problems: punctuation in numeric columns, mixed types, stray whitespace, inconsistent categories.')

    if st.button('Run automatic cleaning attempts'):
        before = df.copy()
        # strip whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # try remove punctuation from numeric-looking object columns
        df = clean_punctuation_in_numeric(df, df.columns)
        st.success('Automatic cleaning attempted (stripping whitespace, removing $ , % from object cols and coercing).')
        st.experimental_rerun()

    st.write('Preview of data types after any cleaning:')
    st.dataframe(pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.values}))

    # allow manual fixes: choose a column and force type
    st.subheader('Manual type casting')
    col_to_cast = st.selectbox('Select column to cast (optional)', options=['-- none --'] + list(df.columns))
    if col_to_cast and col_to_cast != '-- none --':
        to_type = st.selectbox('Cast to type', options=['int', 'float', 'str', 'category', 'datetime'])
        if st.button('Apply cast'):
            try:
                if to_type == 'datetime':
                    df[col_to_cast] = pd.to_datetime(df[col_to_cast], errors='coerce')
                elif to_type == 'category':
                    df[col_to_cast] = df[col_to_cast].astype('category')
                else:
                    df[col_to_cast] = df[col_to_cast].astype(to_type)
                st.success(f'Column {col_to_cast} cast to {to_type}.')
            except Exception as e:
                st.error(f'Failed to cast: {e}')

    # SECTION 6: Categorical column analysis ---------------------------------
    st.header('6) Categorical column analysis')
    cat_choice = st.selectbox('Choose a categorical column to analyze', options=['-- none --'] + cat_cols)
    if cat_choice and cat_choice != '-- none --':
        st.subheader('A) Frequency table  B) Relative frequency')
        freq = df[cat_choice].value_counts(dropna=False).rename_axis(cat_choice).reset_index(name='counts')
        freq['relative'] = (freq['counts'] / len(df)).round(4)
        st.dataframe(freq)

        st.subheader('C) Bar chart  D) Pie chart')
        fig_bar = px.bar(freq, x=cat_choice, y='counts', title=f'Bar chart for {cat_choice}')
        fig_pie = px.pie(freq, names=cat_choice, values='counts', title=f'Pie chart for {cat_choice}')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('**Understanding (write-up):**')
        st.write('Check the frequency table for dominant categories, rare levels, and potential typos (e.g., same category spelled differently). If one level dominates, consider grouping rare levels into `Other` for modeling. Also check if missing values are meaningful or simply absent.')

    # SECTION 7: Numerical column analysis ----------------------------------
    st.header('7) Numerical column analysis')
    num_choice = st.selectbox('Choose a numerical column to analyze', options=['-- none --'] + num_cols)
    if num_choice and num_choice != '-- none --':
        st.subheader('A) Statistical summary')
        st.dataframe(statistical_summary(df, [num_choice]))

        st.subheader('B) Histogram & distribution')
        bins = st.slider('Bins for histogram', 5, 200, 30)
        fig_hist = plot_histogram(df, num_choice, bins=bins)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('**Understanding (write-up):**')
        st.write('Look at mean vs median to judge skewness. Heavy tails and long right tails indicate positive skew; consider log-transform. Multimodal distributions may suggest mixed populations or data-entry problems.')

    # SECTION 8: Outlier analysis -------------------------------------------
    st.header('8) Outlier analysis')
    out_col = st.selectbox('Select numerical column for outlier check', options=['-- none --'] + num_cols, key='outcol')
    if out_col and out_col != '-- none --':
        st.subheader('A) Box plot')
        fig_box = plot_box(df, out_col)
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader('B) Treat the outliers')
        method = st.radio('Treatment method', options=['None', 'IQR capping', 'Z-score removal', 'Winsorize (quantiles)'])
        if method == 'IQR capping':
            factor = st.number_input('IQR factor', value=1.5)
            lower, upper = iqr_treatment(df[out_col].dropna(), factor=factor)
            st.write('Lower bound:', lower, 'Upper bound:', upper)
            df[out_col] = np.where(df[out_col] < lower, lower, df[out_col])
            df[out_col] = np.where(df[out_col] > upper, upper, df[out_col])
            st.success('Applied IQR capping.')
        elif method == 'Z-score removal':
            thr = st.number_input('Z-score threshold', value=3.0)
            # careful approach to avoid dropping the whole dataframe unexpectedly
            mask = False
            try:
                z = np.abs(stats.zscore(df[out_col].fillna(df[out_col].mean())))
                mask = z > thr
                removed = mask.sum()
                df = df.loc[~mask]
                st.success(f'Removed {removed} rows as outliers by z-score (dropped).')
            except Exception as e:
                st.error(f'Z-score removal failed: {e}')
        elif method == 'Winsorize (quantiles)':
            q = st.slider('Winsorize quantile (each tail)', 0.0, 0.25, 0.05)
            lower = df[out_col].quantile(q)
            upper = df[out_col].quantile(1 - q)
            df[out_col] = df[out_col].clip(lower, upper)
            st.success('Applied winsorization.')

    # SECTION 9: Bivariate & multivariate analysis ---------------------------
    st.header('9) Bi-variate and Multi-variate analysis')
    st.subheader('A) How one column impacts another')
    x_col = st.selectbox('X (feature)', options=['-- none --'] + df.columns.tolist(), key='xcol')
    y_col = st.selectbox('Y (target)', options=['-- none --'] + df.columns.tolist(), key='ycol')
    if x_col != '-- none --' and y_col != '-- none --' and x_col != y_col:
        if x_col in num_cols and y_col in num_cols:
            fig = px.scatter(df, x=x_col, y=y_col, trendline='ols', title=f'{y_col} vs {x_col}')
            st.plotly_chart(fig, use_container_width=True)
            st.write('Interpretation: slope of trendline indicates direction; spread indicates variance and possible heteroscedasticity.')
        elif x_col in cat_cols and y_col in num_cols:
            fig = px.box(df, x=x_col, y=y_col, title=f'{y_col} distribution by {x_col}')
            st.plotly_chart(fig, use_container_width=True)
        elif x_col in num_cols and y_col in cat_cols:
            fig = px.violin(df, y=x_col, x=y_col, box=True, title=f'{x_col} distribution by {y_col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            ct = pd.crosstab(df[x_col], df[y_col])
            st.dataframe(ct)
            st.write('Use a stacked bar/heatmap to view association.')

    st.subheader('B) Correlation  C) Heatmap')
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        st.dataframe(corr)
        fig = px.imshow(corr, text_auto=True, title='Correlation heatmap (numerical features)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Need at least 2 numerical columns for correlation matrix.')

    # SECTION 10: Encoding -----------------------------------------------
    st.header('10) Encoding — Convert categorical to numerical')
    st.markdown('A) Label Encoder  B) OneHotEncoder')
    enc_col = st.multiselect('Select categorical columns to encode', options=cat_cols)
    encoding_method = st.radio('Choose encoding method', options=['LabelEncoder', 'OneHotEncoder (drop-first)'])
    if st.button('Apply encoding'):
        if not enc_col:
            st.error('Choose one or more categorical columns to encode.')
        else:
            if encoding_method == 'LabelEncoder':
                le = LabelEncoder()
                for c in enc_col:
                    df[c] = df[c].astype(str).fillna('##MISSING##')
                    df[c] = le.fit_transform(df[c])
                st.success('Applied LabelEncoder to selected columns.')
            else:
                df = pd.get_dummies(df, columns=enc_col, drop_first=True)
                st.success('Applied OneHotEncoding (drop_first=True).')

    # SECTION 11: Scaling -------------------------------------------------
    st.header('11) Scaling')
    st.markdown('A) Standardization  B) Normalization (MinMax)')
    scale_cols = st.multiselect('Select numerical columns to scale', options=num_cols)
    scaler_choice = st.radio('Scaler', options=['None', 'StandardScaler', 'MinMaxScaler'])
    if st.button('Apply scaling'):
        if scaler_choice == 'StandardScaler' and scale_cols:
            ss = StandardScaler()
            df[scale_cols] = ss.fit_transform(df[scale_cols])
            st.success('Applied StandardScaler.')
        elif scaler_choice == 'MinMaxScaler' and scale_cols:
            mm = MinMaxScaler()
            df[scale_cols] = mm.fit_transform(df[scale_cols])
            st.success('Applied MinMaxScaler.')
        else:
            st.info('No scaler applied or no columns selected.')

    # SECTION 12: Transformation -----------------------------------------
    st.header('12) Transformations')
    trans_col = st.selectbox('Choose a numerical column to transform', options=['-- none --'] + num_cols, key='trans')
    if trans_col and trans_col != '-- none --':
        trans_method = st.selectbox('Transformation', options=['log1p', 'sqrt', 'box-cox (positive only)', 'yeo-johnson (sklearn)'])
        if st.button('Apply transformation'):
            if trans_method == 'log1p':
                df[trans_col] = np.log1p(df[trans_col])
                st.success('Applied log1p.')
            elif trans_method == 'sqrt':
                df[trans_col] = np.sqrt(df[trans_col].clip(lower=0))
                st.success('Applied sqrt (negative values clipped).')
            elif trans_method == 'box-cox':
                from scipy.stats import boxcox
                series = df[trans_col].dropna()
                if (series <= 0).any():
                    st.error('Box-Cox requires positive values only.')
                else:
                    df[trans_col], _ = boxcox(series)
                    st.success('Applied boxcox.')
            elif trans_method == 'yeo-johnson (sklearn)':
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method='yeo-johnson')
                mask = df[trans_col].notna()
                df.loc[mask, trans_col] = pt.fit_transform(df.loc[mask, [trans_col]])
                st.success('Applied Yeo-Johnson transformation.')

    # SECTION 13: Feature selection --------------------------------------
    st.header('13) Selecting important features for ML model')
    st.markdown('This section gives a few quick unsupervised suggestions: variance-threshold, correlation filter, and simple tree-based importance (if target provided).')
    target_col = st.selectbox('(Optional) Choose a target column for supervised feature importance', options=['-- none --'] + df.columns.tolist(), key='target')
    fs_action = st.selectbox('Choose method', options=['Variance filter (low variance)', 'Correlation filter (highly correlated)', 'Tree-based importance (requires target)'])
    if st.button('Run feature-selection'):
        if fs_action.startswith('Variance'):
            from sklearn.feature_selection import VarianceThreshold
            vt = VarianceThreshold(threshold=0.0)
            numeric_for_fs = df.select_dtypes(include=[np.number]).fillna(0)
            vt.fit(numeric_for_fs)
            keep = numeric_for_fs.columns[vt.variances_ > 0]
            st.write('Kept columns (variance > 0):', list(keep))
        elif fs_action.startswith('Correlation'):
            num = df.select_dtypes(include=[np.number])
            corr = num.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
            st.write('Columns with high correlation (>0.9) to drop or inspect:', high_corr)
        else:
            if target_col == '-- none --':
                st.error('Provide a target column for tree-based importance.')
            else:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                X = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all').fillna(0)
                y = df[target_col]
                if y.dtype == 'object' or y.nunique() < 20:
                    model = RandomForestClassifier(n_estimators=50, random_state=0)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=0)
                # align shapes
                X = X.loc[y.dropna().index]
                y = y.dropna()
                model.fit(X, y)
                imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.dataframe(imp.head(30).rename('importance'))

    # SECTION 14: PCA & SVD -----------------------------------------------
    st.header('14) PCA & SVD')
    st.markdown('Compute PCA (for numerical features) and Truncated SVD for sparse/dummy-encoded features.')
    pca_cols = st.multiselect('Select columns for PCA (numerical recommended)', options=num_cols)
    if st.button('Run PCA'):
        if len(pca_cols) < 2:
            st.error('Choose at least 2 numeric columns for PCA.')
        else:
            pca = PCA(n_components=min(len(pca_cols), 10))
            X = df[pca_cols].dropna()
            comp = pca.fit_transform(X)
            var = pca.explained_variance_ratio_
            st.write('Explained variance ratio (components):')
            st.write(var.round(4))
            fig = px.bar(x=[f'PC{i+1}' for i in range(len(var))], y=var, title='PCA explained variance ratio')
            st.plotly_chart(fig, use_container_width=True)
            # 2D scatter if at least 2 components
            if comp.shape[1] >= 2:
                fig2 = px.scatter(x=comp[:, 0], y=comp[:, 1], title='PCA projection (PC1 vs PC2)')
                st.plotly_chart(fig2, use_container_width=True)

    svd_cols = st.multiselect('Select columns for Truncated SVD (sparse/dummy features)', options=df.columns.tolist())
    if st.button('Run Truncated SVD'):
        if len(svd_cols) < 2:
            st.error('Choose at least 2 columns for SVD.')
        else:
            X = pd.get_dummies(df[svd_cols].fillna('##MISSING##'))
            svd = TruncatedSVD(n_components=min(10, X.shape[1]-1))
            comp = svd.fit_transform(X)
            st.write('Explained variance ratio (SVD components):')
            st.write(svd.explained_variance_ratio_.round(4))

    # Export cleaned dataset
    st.header('Export / Save')
    st.markdown('Download the transformed / cleaned dataset')
    fmt = st.selectbox('Choose download format', options=['csv', 'xlsx'])
    if st.button('Create download link'):
        fn = f'cleaned.{fmt}'
        href = download_link(df, filename=fn)
        st.markdown(f"[Download cleaned dataset]({href})")

    st.markdown('---')
    st.info('This app attempts a broad, practical EDA pipeline. For production ML pipelines, convert interactive steps into reproducible scripts and tests.')

else:
    st.title('Full EDA pipeline')
    st.write('Upload a .csv / .xlsx / .xls file from the left sidebar to begin.')
