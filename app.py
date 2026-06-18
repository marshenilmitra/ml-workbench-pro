# ============================================================
# ML Workbench – Final Polished Edition (Overfitting Detection)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle, io, warnings, inspect, time, zipfile
warnings.filterwarnings('ignore')

# Global matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 100,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, accuracy_score, confusion_matrix,
    classification_report, r2_score
)
from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_wine, load_iris,
    fetch_california_housing, make_blobs, make_regression, make_classification,
    make_moons, make_circles
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------- Theme‑friendly CSS ----------
st.markdown(f"""
<style>
    .stMetric {{
        background: var(--st-color-background-secondary);
        padding: 12px;
        border-radius: 12px;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .badge-green {{ background: #d4edda; color: #155724; }}
    .badge-red {{ background: #f8d7da; color: #721c24; }}
    .badge-blue {{ background: #d1ecf1; color: #0c5460; }}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def filter_params(model_class, params):
    valid = set(inspect.signature(model_class).parameters.keys())
    valid -= {'self', 'kwargs', 'args'}
    return {k: v for k, v in params.items() if k in valid}

def export_to_excel(sheet_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, data in sheet_dict.items():
            if data.empty:
                data = pd.DataFrame({'Note': [f'No data for {sheet_name} – expected for this method.']})
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

def create_zip_download(figs_dict, excel_bytes, model_bytes, model_name):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('report.xlsx', excel_bytes)
        zf.writestr(f'{model_name}.pkl', model_bytes)
        for fig_name, fig in figs_dict.items():
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'figures/{fig_name}.png', img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ---------- Dataset metadata ----------
DATASETS = {
    "tips": {"source": "seaborn", "task": "regression", "target": "tip", "icon": "💵"},
    "titanic": {"source": "seaborn", "task": "classification", "target": "survived", "icon": "🚢"},
    "iris": {"source": "seaborn", "task": "classification", "target": "species", "icon": "🌸"},
    "diamonds": {"source": "seaborn", "task": "regression", "target": "price", "icon": "💎"},
    "mpg": {"source": "seaborn", "task": "regression", "target": "mpg", "icon": "🚗"},
    "penguins": {"source": "seaborn", "task": "classification", "target": "species", "icon": "🐧"},
    "taxis": {"source": "seaborn", "task": "regression", "target": "total", "icon": "🚕"},
    "car_crashes": {"source": "seaborn", "task": "regression", "target": "total", "icon": "💥"},
    "diabetes": {"source": "sklearn", "task": "regression", "target": "target", "icon": "🩺"},
    "breast_cancer": {"source": "sklearn", "task": "classification", "target": "target", "icon": "🎗️"},
    "wine": {"source": "sklearn", "task": "classification", "target": "target", "icon": "🍷"},
    "california_housing": {"source": "sklearn", "task": "regression", "target": "MedHouseVal", "icon": "🏠"},
    "blobs": {"source": "synthetic", "task": "clustering", "target": None, "icon": "🔵"},
    "moons": {"source": "synthetic", "task": "clustering", "target": None, "icon": "🌙"},
    "circles": {"source": "synthetic", "task": "clustering", "target": None, "icon": "⭕"},
    "iris_unsupervised": {"source": "real", "task": "clustering", "target": None, "icon": "🌺"},
    "synthetic_regression": {"source": "synthetic", "task": "regression", "target": "target", "icon": "🧪"},
    "synthetic_classification": {"source": "synthetic", "task": "classification", "target": "target", "icon": "🧬"},
    "flights": {"source": "seaborn", "task": "time_series", "target": "passengers", "icon": "✈️"},
    "synthetic_trend": {"source": "synthetic", "task": "time_series", "target": "value", "icon": "📈"},
    "synthetic_seasonal": {"source": "synthetic", "task": "time_series", "target": "value", "icon": "📊"},
}

def load_dataset(name):
    info = DATASETS[name]
    if name == "tips":
        df = sns.load_dataset('tips')
    elif name == "titanic":
        df = sns.load_dataset('titanic')[['survived','pclass','sex','age','sibsp','parch','fare','embarked']].dropna()
    elif name == "iris":
        df = sns.load_dataset('iris')
    elif name == "diamonds":
        df = sns.load_dataset('diamonds').dropna()
    elif name == "mpg":
        df = sns.load_dataset('mpg').dropna()
    elif name == "penguins":
        df = sns.load_dataset('penguins').dropna()
    elif name == "taxis":
        df = sns.load_dataset('taxis').dropna()
    elif name == "car_crashes":
        df = sns.load_dataset('car_crashes').dropna()
    elif name == "flights":
        df = sns.load_dataset('flights')
        df = df.pivot(index='month', columns='year', values='passengers').stack().reset_index()
        df.columns = ['month','year','passengers']
    elif name == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif name == "breast_cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif name == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif name == "california_housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['MedHouseVal'] = data.target
    elif name == "blobs":
        X, y = make_blobs(n_samples=300, centers=3, random_state=42)
        df = pd.DataFrame(X, columns=['x1','x2'])
        df['true_label'] = y
    elif name == "moons":
        X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=['x1','x2'])
        df['true_label'] = y
    elif name == "circles":
        X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
        df = pd.DataFrame(X, columns=['x1','x2'])
        df['true_label'] = y
    elif name == "iris_unsupervised":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    elif name == "synthetic_regression":
        X, y = make_regression(n_samples=200, n_features=5, noise=0.2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
    elif name == "synthetic_classification":
        X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
    elif name == "synthetic_trend":
        np.random.seed(42)
        t = np.arange(120)
        trend = 0.5 * t + 10
        noise = np.random.normal(scale=5, size=120)
        series = trend + noise
        df = pd.DataFrame({'date': pd.date_range(start='2000-01-01', periods=120, freq='ME'),
                           'value': series})
    elif name == "synthetic_seasonal":
        np.random.seed(42)
        t = np.arange(120)
        seasonal = 10 * np.sin(2 * np.pi * t / 12) + 50
        noise = np.random.normal(scale=3, size=120)
        series = seasonal + noise
        df = pd.DataFrame({'date': pd.date_range(start='2000-01-01', periods=120, freq='ME'),
                           'value': series})
    else:
        return None, None, None
    return df, info['task'], info['target']

# ---------- Model configs ----------
REGRESSION_MODELS = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": (0.001, 10.0, 1.0, 0.01)},
    "Lasso Regression": {"alpha": (0.001, 10.0, 1.0, 0.01)},
    "ElasticNet": {"alpha": (0.001, 10.0, 1.0, 0.01), "l1_ratio": (0.0, 1.0, 0.5, 0.05)},
    "Decision Tree Regressor": {"max_depth": (2, 30, 5, 1)},
    "Random Forest Regressor": {"n_estimators": (10, 200, 100, 10), "max_depth": (2, 30, 10, 1)},
    "Gradient Boosting Regressor": {"n_estimators": (10, 200, 100, 10), "learning_rate": (0.01, 1.0, 0.1, 0.01)},
    "SVR": {"C": (0.1, 100.0, 1.0, 0.1), "epsilon": (0.01, 1.0, 0.1, 0.01)},
    "K-Neighbors Regressor": {"n_neighbors": (1, 20, 5, 1)}
}
CLASSIFICATION_MODELS = {
    "Logistic Regression": {"C": (0.01, 10.0, 1.0, 0.01)},
    "K-Neighbors Classifier": {"n_neighbors": (1, 20, 5, 1)},
    "Decision Tree Classifier": {"max_depth": (2, 30, 5, 1)},
    "Random Forest Classifier": {"n_estimators": (10, 200, 100, 10), "max_depth": (2, 30, 10, 1)},
    "SVC": {"C": (0.1, 100.0, 1.0, 0.1), "kernel": ["rbf","linear","poly"]},
    "Gaussian Naive Bayes": {},
    "Gradient Boosting Classifier": {"n_estimators": (10, 200, 100, 10), "learning_rate": (0.01, 1.0, 0.1, 0.01)}
}

def build_model(model_name, task_type, params):
    if task_type == 'regression':
        mapping = {
            "Linear Regression": LinearRegression,
            "Ridge Regression": Ridge,
            "Lasso Regression": Lasso,
            "ElasticNet": ElasticNet,
            "Decision Tree Regressor": DecisionTreeRegressor,
            "Random Forest Regressor": RandomForestRegressor,
            "Gradient Boosting Regressor": GradientBoostingRegressor,
            "SVR": SVR,
            "K-Neighbors Regressor": KNeighborsRegressor
        }
    else:
        mapping = {
            "Logistic Regression": LogisticRegression,
            "K-Neighbors Classifier": KNeighborsClassifier,
            "Decision Tree Classifier": DecisionTreeClassifier,
            "Random Forest Classifier": RandomForestClassifier,
            "SVC": SVC,
            "Gaussian Naive Bayes": GaussianNB,
            "Gradient Boosting Classifier": GradientBoostingClassifier
        }
    model_class = mapping[model_name]
    clean_params = filter_params(model_class, params)
    if 'random_state' in inspect.signature(model_class).parameters:
        clean_params.setdefault('random_state', 42)
    return model_class(**clean_params)

# ---------- Data Quality Score ----------
def data_quality_score(df):
    score = 100
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100 if df.shape[0] > 0 else 0
    score -= missing_pct * 2
    duplicate_pct = df.duplicated().sum() / df.shape[0] * 100 if df.shape[0] > 0 else 0
    score -= duplicate_pct * 1.5
    if df.select_dtypes(include=np.number).shape[1] > 0:
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count += ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_pct = outlier_count / (df.shape[0] * len(numeric_cols)) * 100 if df.shape[0] > 0 else 0
        score -= outlier_pct * 0.5
    return max(0, min(100, score))

# ---------- Data Profiling ----------
def show_data_profile(df):
    dq_score = data_quality_score(df)
    if dq_score >= 90:
        st.success(f"📊 Data Quality Score: **{dq_score:.0f}/100** – Excellent")
    elif dq_score >= 70:
        st.warning(f"📊 Data Quality Score: **{dq_score:.0f}/100** – Good")
    else:
        st.error(f"📊 Data Quality Score: **{dq_score:.0f}/100** – Needs attention")

    with st.expander("🔍 Detailed Data Profile", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())
        st.caption("ℹ️ Datasets are pre‑cleaned (`.dropna()`) for reliable training. Missing values are 0 because cleaning has already been applied.")
        st.write("**Data Types:**")
        st.write(df.dtypes.astype(str))
        st.write("**Descriptive Statistics**")
        st.dataframe(df.describe(include='all'))

        num_df = df.select_dtypes(include=np.number)
        n_num = num_df.shape[1]
        if n_num > 1:
            st.write("**Correlation Heatmap**")
            show_annotations = st.checkbox("Show correlation values", value=n_num <= 10, key='corr_annot')
            fig, ax = plt.subplots(figsize=(max(8, n_num*0.8), max(6, n_num*0.65)))
            corr = num_df.corr()
            sns.heatmap(corr, annot=show_annotations, fmt=".2f", cmap='coolwarm',
                        linewidths=0.5, square=True,
                        annot_kws={'size': max(7, 12 - n_num*0.5)} if show_annotations else None,
                        ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title("Feature Correlation Matrix", fontsize=14, weight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# ---------- Auto Insights ----------
def regression_insights(y_test, y_pred, r2, mse, feature_names, importance, train_time, train_score, test_score):
    lines = [f"⏱️ Training completed in **{train_time:.2f} seconds**."]
    if r2 > 0.8:
        lines.append(f"✅ Excellent fit: R² = {r2:.2%} – model explains most variance.")
    elif r2 > 0.5:
        lines.append(f"⚠️ Moderate fit: R² = {r2:.2%} – some patterns still unexplained.")
    else:
        lines.append(f"🔻 Weak fit: R² = {r2:.2%} – model not capturing the trend well.")
    lines.append(f"📏 Average prediction error (RMSE) is **{np.sqrt(mse):.2f}** units.")
    errors = np.abs(y_test - y_pred)
    worst_idx = np.argmax(errors)
    lines.append(f"🔍 Largest single error: {errors.iloc[worst_idx]:.2f} (Actual: {y_test.iloc[worst_idx]:.2f}, Pred: {y_pred[worst_idx]:.2f})")
    if importance is not None and np.any(importance):
        top_idx = np.argsort(np.abs(importance))[-2:]
        top_features = feature_names[top_idx][::-1]
        lines.append(f"📌 Top two most influential features: **{top_features[0]}**, **{top_features[1]}**.")
    else:
        lines.append("📌 Feature importance not available for this model type.")

    # Overfitting detection
    if train_score is not None and test_score is not None:
        gap = abs(train_score - test_score)
        if gap > 0.1:
            lines.append(f"⚠️ **Possible overfitting:** Training R² = {train_score:.2%}, Test R² = {test_score:.2%} — a large gap may indicate the model memorised training data.")
    return lines

def classification_insights(y_test, y_pred, acc, cm, feature_names, importance, train_time, train_acc, test_acc):
    lines = [f"⏱️ Training completed in **{train_time:.2f} seconds**."]
    lines.append(f"📊 Test Accuracy: **{test_acc:.2%}**.")
    if acc > 0.9:
        lines.append("✅ High accuracy – model performs very well.")
    elif acc > 0.7:
        lines.append("⚠️ Moderate accuracy – consider feature engineering.")
    else:
        lines.append("🔻 Low accuracy – the model may be underfitting or the problem is hard.")
    misclass = cm.sum(axis=1) - np.diag(cm)
    if np.sum(misclass) > 0:
        worst_class = np.argmax(misclass)
        lines.append(f"🔁 Class **{worst_class}** is most often misclassified ({misclass[worst_class]} times).")
    if importance is not None and np.any(importance):
        top_idx = np.argsort(np.abs(importance))[-2:]
        top_features = feature_names[top_idx][::-1]
        lines.append(f"📌 Top two most important features: **{top_features[0]}**, **{top_features[1]}**.")
    else:
        lines.append("📌 Feature importance not available for this model type.")

    # Overfitting detection
    if train_acc is not None and test_acc is not None:
        gap = train_acc - test_acc
        if gap > 0.1:
            lines.append(f"⚠️ **Possible overfitting:** Training accuracy = {train_acc:.2%}, Test accuracy = {test_acc:.2%} — a large gap may indicate the model memorised training data.")
    return lines

def clustering_insights(labels, centers_df):
    lines = []
    unique, counts = np.unique(labels, return_counts=True)
    lines.append(f"📊 Found **{len(unique)}** clusters with sizes: {dict(zip(unique, counts))}.")
    dominant = unique[np.argmax(counts)]
    lines.append(f"📌 Cluster **{dominant}** is the largest ({(max(counts)/sum(counts)):.0%} of points).")
    if centers_df is not None:
        lines.append("📈 Cluster centers (original scale) show defining characteristics of each group.")
    return lines

def pca_insights(explained_variance, loadings):
    lines = []
    cum_var = np.cumsum(explained_variance)
    lines.append(f"📊 First 2 components explain **{cum_var[1]:.1%}** of total variance.")
    if cum_var[1] > 0.8:
        lines.append("✅ Excellent – 2 components capture most of the data's structure.")
    elif cum_var[1] > 0.5:
        lines.append("⚠️ Moderate – consider using more components for better representation.")
    else:
        lines.append("🔻 Low – the data may not be well‑suited for 2D projection.")
    top_features_pc1 = loadings.iloc[:, 0].abs().sort_values(ascending=False).head(2).index.tolist()
    lines.append(f"📌 Top features for PC1: **{top_features_pc1[0]}**, **{top_features_pc1[1]}**.")
    return lines

def tsne_insights(perplexity):
    lines = [
        f"🔧 t‑SNE run with perplexity = **{perplexity}**.",
        "💡 t‑SNE is excellent for visualizing high‑dimensional data in 2D.",
        "⚠️ Note: t‑SNE distances between clusters are not directly interpretable as in PCA."
    ]
    return lines

def time_series_insights(series, fitted, forecast, model_name):
    lines = [f"⏱️ {model_name} forecast generated successfully."]
    if series.index.is_monotonic_increasing:
        first_val = series.iloc[0]
        last_val = series.iloc[-1]
        lines.append("📈 Overall **upward trend** in the historical data." if last_val > first_val else "📉 Overall **downward trend**.")
    if isinstance(series.index, pd.DatetimeIndex) and len(series) > 12:
        lines.append("🔁 Data likely contains **seasonal patterns** (monthly/yearly).")
    lines.append(f"🔮 Forecast for next {len(forecast)} periods generated.")
    if fitted is not None and hasattr(fitted, 'aic'):
        lines.append(f"🧾 Model AIC: {fitted.aic:.2f} – lower is better.")
    return lines

# ---------- Compatibility check ----------
def check_compatibility(df, target_col, task_type, model_name):
    if df is None or df.empty:
        return False, "Dataset is empty."
    if target_col and target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset."
    if task_type in ['regression', 'classification']:
        if len(df) < 10:
            return False, "Dataset has fewer than 10 rows – not enough for training."
        if task_type == 'classification' and df[target_col].nunique() < 2:
            return False, "Target variable has only one unique class – cannot train a classifier."
        if "K-Neighbors" in model_name and len(df) < 30:
            return True, "⚠️ Dataset is small for K‑Neighbors – results may be unstable."
    return True, ""

# ---------- Leaderboard ----------
def update_leaderboard(model_name, cv_mean, cv_std):
    if 'leaderboard' not in st.session_state:
        st.session_state.leaderboard = []
    st.session_state.leaderboard.append({
        'Model': model_name,
        'Dataset': dataset_display,
        'CV Score': f"{cv_mean:.4f}",
        'Std': f"{cv_std:.4f}"
    })

def render_sidebar_leaderboard():
    if 'leaderboard' in st.session_state and len(st.session_state.leaderboard) > 0:
        lb_df = pd.DataFrame(st.session_state.leaderboard)
        lb_df = lb_df.sort_values('CV Score', ascending=False).head(3)
        st.sidebar.markdown("**🏆 Leaderboard (Top 3)**")
        st.sidebar.dataframe(lb_df, use_container_width=True, hide_index=True)
        if st.sidebar.button("Clear Leaderboard"):
            st.session_state.leaderboard = []
            st.rerun()
    else:
        st.sidebar.info("No models trained yet. Train a supervised model to populate the leaderboard.")

# ---------- Supervised App ----------
def supervised_app(df, target_col, task_type):
    icon = DATASETS[dataset_name]['icon']
    st.header(f"{icon} {task_type.capitalize()} – {dataset_display}")
    st.markdown(f'<span class="badge badge-green">✅ Compatible: {task_type.capitalize()} dataset</span>', unsafe_allow_html=True)

    show_data_profile(df)

    df_enc = df.copy()
    non_num_cols = df_enc.select_dtypes(exclude=[np.number]).columns
    for col in non_num_cols:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    datetime_cols = df_enc.select_dtypes(include=['datetime64','timedelta64']).columns
    for col in datetime_cols:
        df_enc[col] = df_enc[col].astype('int64') // 10**9
        st.caption(f"🕒 Converted datetime column `{col}` to numeric timestamp.")
    df_enc.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df_enc.columns:
        if df_enc[col].isnull().any():
            df_enc[col] = df_enc[col].fillna(df_enc[col].median())

    X = df_enc.drop(target_col, axis=1)
    y = df_enc[target_col]

    st.sidebar.subheader("⚙️ Model & Hyperparameters")
    if task_type == 'regression':
        model_name = st.sidebar.selectbox("Choose model", list(REGRESSION_MODELS.keys()))
    else:
        model_name = st.sidebar.selectbox("Choose model", list(CLASSIFICATION_MODELS.keys()))

    config = REGRESSION_MODELS[model_name] if task_type == 'regression' else CLASSIFICATION_MODELS[model_name]
    use_defaults = st.sidebar.checkbox("Use default hyperparameters", True)
    param_dict = {}
    if not use_defaults:
        with st.sidebar.expander("Tune hyperparameters", expanded=True):
            for param, val in config.items():
                if isinstance(val, tuple):
                    param_dict[param] = st.slider(param, *val)
                else:
                    param_dict[param] = st.selectbox(param, val)
    else:
        for param, val in config.items():
            param_dict[param] = val[2] if isinstance(val, tuple) else val[0]

    test_size = st.sidebar.slider("Test size (%)", 10, 40, 20) / 100
    random_state = st.sidebar.number_input("Random state", 0, 100, 42)

    ok, msg = check_compatibility(df, target_col, task_type, model_name)
    if not ok:
        st.error(f"❌ Incompatible model/dataset: {msg}")
        st.stop()
    if msg.startswith("⚠️"):
        st.warning(msg)

    model = build_model(model_name, task_type, param_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    start_time = time.time()
    try:
        with st.spinner(f"Training **{model_name}** on {dataset_display}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        train_time = time.time() - start_time
    except Exception as e:
        st.markdown(f'<span class="badge badge-red">❌ Training Failed</span>', unsafe_allow_html=True)
        st.error(f"❌ Training failed: {str(e)}")
        st.info("Try a different model, adjust hyperparameters, or check your data.")
        st.stop()

    st.markdown(f'<span class="badge badge-blue">⏱️ Trained in {train_time:.2f} seconds</span>', unsafe_allow_html=True)

    # Compute training score for overfitting detection
    train_score = None
    try:
        if task_type == 'regression':
            train_score = model.score(X_train, y_train)
        else:
            train_score = model.score(X_train, y_train)   # accuracy on training set
    except:
        pass

    # Cross‑validation
    st.markdown("### 🔁 Cross‑Validation (5‑fold)")
    cv_mean, cv_std = 0, 0
    cv_success = False
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, error_score='raise')
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        cv_success = True
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean CV score", f"{cv_mean:.4f}")
        c2.metric("Std deviation", f"{cv_std:.4f}")
        c3.metric("Median CV score", f"{np.median(cv_scores):.4f}")
        fig, ax = plt.subplots(figsize=(7, 5))
        bp = ax.boxplot(cv_scores, patch_artist=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_facecolor('#4CAF50')
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"Fold {i+1}" for i in range(5)], fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Per‑Fold CV Score Distribution", fontsize=13, weight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        update_leaderboard(model_name, cv_mean, cv_std)
    except Exception as e:
        error_msg = str(e)
        if "too small" in error_msg.lower() or "less than" in error_msg.lower():
            st.info("Dataset too small for 5‑fold CV – showing single train/test split only.")
        else:
            st.warning(f"⚠️ Cross‑validation skipped: {error_msg}")
            st.info("Training results are still valid. Try a different model or preprocess the data further.")

    figs_dict = {}
    if task_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mse:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("R²", f"{r2:.4f}")
        eval_df = pd.DataFrame({'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]})

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel("Actual Values", fontsize=11)
        ax.set_ylabel("Predicted Values", fontsize=11)
        ax.set_title("Actual vs Predicted", fontsize=13, weight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        figs_dict['actual_vs_predicted'] = fig

        if hasattr(model, 'coef_'):
            imp = model.coef_
        elif hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        else:
            imp = np.zeros(X.shape[1])
        coeff_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp})
        if np.any(imp):
            coeff_df['AbsImportance'] = np.abs(coeff_df['Importance'])
            coeff_df = coeff_df.sort_values('AbsImportance', ascending=True)
            n_features = len(coeff_df)
            fig_height = max(4, n_features * 0.35)
            fig, ax = plt.subplots(figsize=(8, fig_height))
            ax.barh(coeff_df['Feature'], coeff_df['Importance'],
                   color=['#2E86AB' if v >= 0 else '#D64045' for v in coeff_df['Importance']],
                   edgecolor='white', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("Importance / Coefficient Value", fontsize=11)
            ax.set_title("Feature Importances / Coefficients", fontsize=13, weight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            figs_dict['feature_importance'] = fig

        st.markdown("### 💡 Model Insights")
        test_score = r2   # for regression, test_score is R²
        for line in regression_insights(y_test, y_pred, r2, mse, X.columns, imp, train_time, train_score, test_score):
            st.markdown(f"- {line}")

    else:  # classification
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        n_classes = cm.shape[0]
        fig_size = max(5, n_classes * 1.2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    linewidths=1, linecolor='white',
                    annot_kws={'size': 12 if n_classes <= 5 else 10},
                    ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, weight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        figs_dict['confusion_matrix'] = fig
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
        eval_df = pd.DataFrame(cm)

        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            coeff_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp})
            coeff_df = coeff_df.sort_values('Importance', ascending=True)
            n_features = len(coeff_df)
            fig_height = max(4, n_features * 0.35)
            fig, ax = plt.subplots(figsize=(8, fig_height))
            ax.barh(coeff_df['Feature'], coeff_df['Importance'],
                   color='#2E86AB', edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Importance", fontsize=11)
            ax.set_title("Feature Importances", fontsize=13, weight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            figs_dict['feature_importance'] = fig
        elif hasattr(model, 'coef_'):
            imp = model.coef_[0]
            coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': imp})
        else:
            imp = np.ones(X.shape[1])
            coeff_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp})

        st.markdown("### 💡 Model Insights")
        test_acc = acc
        for line in classification_insights(y_test, y_pred, acc, cm, X.columns, imp, train_time, train_score, test_acc):
            st.markdown(f"- {line}")

    # Downloads
    model_bytes = pickle.dumps(model)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📥 Download Model", model_bytes, file_name=f"{model_name}.pkl")
    with col2:
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        test_df['Predicted'] = y_pred
        sheets = {
            'Raw Data': df,
            'Training Set': train_df,
            'Test Set': test_df,
            'Model Coefficients': coeff_df,
            'Model Evaluation': eval_df
        }
        xlsx = export_to_excel(sheets)
        st.download_button("📥 Download Excel", xlsx, file_name=f"{dataset_name}_{model_name}_report.xlsx")
    with col3:
        zip_bytes = create_zip_download(figs_dict, xlsx, model_bytes, model_name)
        st.download_button("📦 Download All (ZIP)", zip_bytes, file_name=f"{dataset_name}_{model_name}_all.zip")

# ---------- Unsupervised App ----------
def unsupervised_app(df):
    icon = DATASETS[dataset_name]['icon']
    st.header(f"{icon} Unsupervised – {dataset_display}")
    st.markdown(f'<span class="badge badge-green">✅ Compatible: Unsupervised dataset</span>', unsafe_allow_html=True)

    show_data_profile(df)

    X = df.select_dtypes(include=np.number)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    method = st.sidebar.selectbox("Unsupervised technique", [
        "K-Means", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture", "PCA", "t-SNE"
    ])
    st.sidebar.subheader("⚙️ Method Settings")

    if method in ["K-Means", "Agglomerative Clustering", "Gaussian Mixture"]:
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

    centers_df = None
    labels = None
    figs_dict = {}

    try:
        with st.spinner(f"Running {method}..."):
            if method == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                centers_df = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=X.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sc = ax.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis',
                               alpha=0.7, edgecolors='k', linewidth=0.3, s=60)
                ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],
                          marker='X', s=250, c='red', edgecolors='white', linewidth=1.5, label='Centroids')
                ax.set_xlabel(f"Feature 1 (scaled)", fontsize=11)
                ax.set_ylabel(f"Feature 2 (scaled)", fontsize=11)
                ax.set_title(f"K‑Means Clustering (k={n_clusters})", fontsize=13, weight='bold')
                ax.legend(fontsize=9)
                plt.colorbar(sc, ax=ax, shrink=0.8)
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['kmeans_clusters'] = fig
                st.subheader("Cluster Centers (original scale)")
                st.dataframe(centers_df)
                sheets = {
                    'Raw Data': df, 'Training Set': pd.DataFrame(), 'Test Set': pd.DataFrame(),
                    'Model Coefficients': centers_df,
                    'Model Evaluation': pd.DataFrame({'Inertia': [model.inertia_], 'Clusters': [n_clusters]})
                }
            elif method == "Agglomerative Clustering":
                linkage = st.sidebar.selectbox("Linkage", ["ward","complete","average","single"])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                labels = model.fit_predict(X_scaled)
                df['Cluster'] = labels
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis',
                          alpha=0.7, edgecolors='k', linewidth=0.3, s=60)
                ax.set_title(f"Agglomerative Clustering (k={n_clusters}, {linkage})", fontsize=13, weight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['agglomerative_clusters'] = fig
                sheets = {
                    'Raw Data': df.drop(columns=['Cluster']), 'Training Set': pd.DataFrame(), 'Test Set': pd.DataFrame(),
                    'Model Coefficients': pd.DataFrame(),
                    'Model Evaluation': pd.DataFrame({'Clusters': [n_clusters], 'Linkage': [linkage]})
                }
            elif method == "DBSCAN":
                eps = st.sidebar.slider("eps", 0.1, 5.0, 0.5, 0.1)
                min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                df['Cluster'] = labels
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis',
                          alpha=0.7, edgecolors='k', linewidth=0.3, s=60)
                ax.set_title(f"DBSCAN (eps={eps}, min_samples={min_samples}) → {n_clusters_found} clusters", fontsize=13, weight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['dbscan_clusters'] = fig
                sheets = {
                    'Raw Data': df.drop(columns=['Cluster']), 'Training Set': pd.DataFrame(), 'Test Set': pd.DataFrame(),
                    'Model Coefficients': pd.DataFrame(),
                    'Model Evaluation': pd.DataFrame({'eps': [eps], 'min_samples': [min_samples], 'Clusters': [n_clusters_found]})
                }
            elif method == "Gaussian Mixture":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = model.fit_predict(X_scaled)
                df['Cluster'] = labels
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis',
                          alpha=0.7, edgecolors='k', linewidth=0.3, s=60)
                ax.set_title(f"Gaussian Mixture Model (k={n_clusters})", fontsize=13, weight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['gmm_clusters'] = fig
                sheets = {
                    'Raw Data': df.drop(columns=['Cluster']), 'Training Set': pd.DataFrame(), 'Test Set': pd.DataFrame(),
                    'Model Coefficients': pd.DataFrame(),
                    'Model Evaluation': pd.DataFrame({'Clusters': [n_clusters], 'Covariance': ['full']})
                }
            elif method == "PCA":
                n_comp = st.sidebar.slider("n_components", 2, min(X.shape[1], 10), 2)
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X_scaled)
                loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_comp)], index=X.columns)
                explained = pca.explained_variance_ratio_
                st.bar_chart(explained)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7, edgecolors='k', linewidth=0.3, s=60)
                ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)", fontsize=11)
                ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)", fontsize=11)
                ax.set_title("PCA – First Two Principal Components", fontsize=13, weight='bold')
                ax.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['pca_projection'] = fig
                sheets = {
                    'Raw Data': df, 'Training Set': pd.DataFrame(), 'Test Set': pd.DataFrame(),
                    'Model Coefficients': loadings,
                    'Model Evaluation': pd.DataFrame({'Component': [f'PC{i+1}' for i in range(n_comp)], 'Variance': explained})
                }
                labels = None
                st.markdown("### 💡 PCA Insights")
                for line in pca_insights(explained, loadings):
                    st.markdown(f"- {line}")
            elif method == "t-SNE":
                perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                X_tsne = tsne.fit_transform(X_scaled)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X_tsne[:,0], X_tsne[:,1], alpha=0.7, edgecolors='k', linewidth=0.3, s=60)
                ax.set_xlabel("t‑SNE Component 1", fontsize=11)
                ax.set_ylabel("t‑SNE Component 2", fontsize=11)
                ax.set_title(f"t‑SNE Visualization (perplexity={perplexity})", fontsize=13, weight='bold')
                ax.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['tsne_projection'] = fig
                sheets = {
                    'Raw Data': df, 'Training Set': pd.DataFrame(), 'Test Set': pd.DataFrame(),
                    'Model Coefficients': pd.DataFrame({'Perplexity': [perplexity]}), 'Model Evaluation': pd.DataFrame()
                }
                labels = None
                st.markdown("### 💡 t‑SNE Insights")
                for line in tsne_insights(perplexity):
                    st.markdown(f"- {line}")
    except Exception as e:
        st.markdown(f'<span class="badge badge-red">❌ Method Failed</span>', unsafe_allow_html=True)
        st.error(f"❌ Unsupervised method failed: {str(e)}")
        st.stop()

    st.info("ℹ️ Cross‑validation and Leaderboard are not applicable for Unsupervised Learning. Model evaluation is done via internal metrics (inertia, etc.).")
    if labels is not None:
        st.markdown("### 💡 Clustering Insights")
        for line in clustering_insights(labels, centers_df):
            st.markdown(f"- {line}")

    xlsx = export_to_excel(sheets)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Excel Report", xlsx, file_name=f"unsupervised_{method}_report.xlsx")
    with col2:
        zip_bytes = create_zip_download(figs_dict, xlsx, pickle.dumps('no_model'), method)
        st.download_button("📦 Download All (ZIP)", zip_bytes, file_name=f"unsupervised_{method}_all.zip")

# ---------- Time Series App ----------
def time_series_app(df, target_col):
    icon = DATASETS[dataset_name]['icon']
    st.header(f"{icon} Time Series Forecasting – {dataset_display}")
    st.markdown(f'<span class="badge badge-green">✅ Compatible: Time Series dataset</span>', unsafe_allow_html=True)

    # Prepare numeric time series
    if 'date' in df.columns and 'value' in df.columns:
        ts = df.set_index('date')[target_col]
    elif 'month' in df.columns and 'year' in df.columns:
        df['date'] = pd.to_datetime(
            df['year'].astype(str).str.cat(df['month'].astype(str), sep='-'),
            format='%Y-%b', errors='coerce'
        )
        ts = df.dropna(subset=['date']).sort_values('date').set_index('date')[target_col]
    else:
        ts = df[target_col]

    # Simple time‑series summary
    st.markdown("### 📋 Series Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Length", len(ts))
    if isinstance(ts.index, pd.DatetimeIndex):
        c2.metric("Start", ts.index[0].strftime('%Y-%m-%d'))
        c3.metric("End", ts.index[-1].strftime('%Y-%m-%d'))
    c4.metric("Mean", f"{ts.mean():.2f}")
    st.caption("Use the controls in the sidebar to configure and run a forecast.")

    st.line_chart(ts)

    st.sidebar.subheader("⚙️ Time Series Model")
    ts_model = st.sidebar.selectbox("Choose model", [
        "ARIMA",
        "SARIMA",
        "Holt‑Winters",
        "Holt’s Linear Trend",
        "Simple Moving Average"
    ])

    forecast_steps = st.sidebar.slider("Forecast steps", 1, 24, 12)

    if ts_model == "ARIMA":
        st.sidebar.markdown("**ARIMA Order**")
        p = st.sidebar.slider("p (AR)", 0, 5, 1)
        d = st.sidebar.slider("d (I)", 0, 2, 1)
        q = st.sidebar.slider("q (MA)", 0, 5, 1)
    elif ts_model == "SARIMA":
        st.sidebar.markdown("**Non‑seasonal Order**")
        p = st.sidebar.slider("p (AR)", 0, 5, 1)
        d = st.sidebar.slider("d (I)", 0, 2, 1)
        q = st.sidebar.slider("q (MA)", 0, 5, 1)
        st.sidebar.markdown("**Seasonal Order**")
        P = st.sidebar.slider("P (Seasonal AR)", 0, 3, 1)
        D = st.sidebar.slider("D (Seasonal I)", 0, 2, 1)
        Q = st.sidebar.slider("Q (Seasonal MA)", 0, 3, 1)
        s = st.sidebar.slider("Seasonal period (s)", 2, 12, 12)
    elif ts_model == "Holt‑Winters":
        st.sidebar.markdown("**Holt‑Winters Settings**")
        trend = st.sidebar.selectbox("Trend", ["add", "mul"], index=0)
        seasonal = st.sidebar.selectbox("Seasonal", ["add", "mul"], index=0)
        seasonal_periods = st.sidebar.slider("Seasonal periods", 2, 12, 12)
    elif ts_model == "Holt’s Linear Trend":
        trend = "add"
        seasonal = None
        seasonal_periods = None
    elif ts_model == "Simple Moving Average":
        window = st.sidebar.slider("Window size", 2, 12, 3)

    if st.sidebar.button("Run Forecast"):
        figs_dict = {}
        try:
            with st.spinner(f"Fitting {ts_model}..."):
                if ts_model == "ARIMA":
                    model = ARIMA(ts, order=(p, d, q))
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=forecast_steps)
                    model_name_used = f"ARIMA({p},{d},{q})"
                elif ts_model == "SARIMA":
                    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
                    fitted = model.fit(disp=False)
                    forecast = fitted.forecast(steps=forecast_steps)
                    model_name_used = f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
                elif ts_model == "Holt‑Winters":
                    model = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal,
                                                 seasonal_periods=seasonal_periods)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=forecast_steps)
                    model_name_used = f"Holt‑Winters ({trend}, {seasonal})"
                elif ts_model == "Holt’s Linear Trend":
                    model = ExponentialSmoothing(ts, trend='add', seasonal=None)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=forecast_steps)
                    model_name_used = "Holt’s Linear Trend"
                elif ts_model == "Simple Moving Average":
                    roll = ts.rolling(window=window).mean()
                    forecast = np.full(forecast_steps, roll.iloc[-1])
                    fitted = None
                    model_name_used = f"SMA({window})"

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ts.index, ts.values, label='Historical', linewidth=2)
                if isinstance(ts.index, pd.DatetimeIndex):
                    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps+1,
                                                   freq=ts.index.freq or 'MS')[1:]
                else:
                    forecast_index = np.arange(len(ts), len(ts)+forecast_steps)
                ax.plot(forecast_index, forecast, label='Forecast', linestyle='--', linewidth=2, color='red')
                ax.set_xlabel("Date", fontsize=11)
                ax.set_ylabel("Value", fontsize=11)
                ax.set_title(f"{model_name_used} Forecast", fontsize=13, weight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                figs_dict['forecast'] = fig

                forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
                param_df = pd.DataFrame({'Parameter': ['Model'], 'Value': [model_name_used]})
                sheets = {
                    'Raw Data': ts.reset_index(),
                    'Training Set': ts.reset_index(),
                    'Test Set': pd.DataFrame(),
                    'Model Coefficients': param_df,
                    'Model Evaluation': pd.DataFrame({'Note': ['N/A for this model']})
                }
                xlsx = export_to_excel(sheets)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 Download Excel Report", xlsx, file_name=f"{dataset_name}_{ts_model}_report.xlsx")
                with col2:
                    zip_bytes = create_zip_download(figs_dict, xlsx, pickle.dumps('model'), ts_model)
                    st.download_button("📦 Download All (ZIP)", zip_bytes, file_name=f"{dataset_name}_{ts_model}_all.zip")

                st.markdown("### 💡 Time Series Insights")
                if fitted is not None and hasattr(fitted, 'aic'):
                    lines = time_series_insights(ts, fitted, forecast, model_name_used)
                else:
                    lines = [f"⏱️ {model_name_used} forecast generated successfully.",
                             f"🔮 Forecast for next {len(forecast)} periods."]
                for line in lines:
                    st.markdown(f"- {line}")
        except Exception as e:
            st.markdown(f'<span class="badge badge-red">❌ {ts_model} Failed</span>', unsafe_allow_html=True)
            st.error(f"❌ {ts_model} failed: {str(e)}")

    st.info("ℹ️ Cross‑validation and Leaderboard are not applicable for Time Series Forecasting. Model evaluation is done via AIC/BIC and residual analysis.")

# ---------- Main App ----------
def main():
    st.set_page_config(page_title="ML Workbench – Premium", page_icon="🧠", layout="wide")

    st.sidebar.title("🧠 ML Workbench")
    st.sidebar.markdown("**Premium Model Trainer**")
    st.sidebar.markdown("---")

    technique = st.sidebar.radio(
        "🎯 Choose Technique Type",
        ["📈 Supervised Learning", "🧩 Unsupervised Learning", "⏳ Time Series Forecasting"]
    )

    available = []
    for name, info in DATASETS.items():
        if technique == "📈 Supervised Learning" and info['task'] in ['regression','classification']:
            available.append(name)
        elif technique == "🧩 Unsupervised Learning" and info['task'] == 'clustering':
            available.append(name)
        elif technique == "⏳ Time Series Forecasting" and info['task'] == 'time_series':
            available.append(name)

    dataset_options = [f"{DATASETS[ds]['icon']} {ds} ({DATASETS[ds]['source']} - {DATASETS[ds]['task']})" for ds in available]
    chosen_label = st.sidebar.selectbox("📦 Select Dataset", dataset_options)
    global dataset_name, dataset_display
    dataset_name = chosen_label.split(" ")[1]
    dataset_display = chosen_label

    df, task, target = load_dataset(dataset_name)
    if df is None:
        st.error("Dataset could not be loaded.")
        return

    if technique == "📈 Supervised Learning":
        supervised_app(df, target, task)
        st.sidebar.markdown("---")
        render_sidebar_leaderboard()
    elif technique == "🧩 Unsupervised Learning":
        unsupervised_app(df)
        st.sidebar.markdown("---")
        st.sidebar.info("ℹ️ Leaderboard is only available for Supervised Learning.")
    else:
        time_series_app(df, target)
        st.sidebar.markdown("---")
        st.sidebar.info("ℹ️ Leaderboard is only available for Supervised Learning.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Streamlit & scikit-learn")
    st.sidebar.markdown("[📂 View on GitHub](https://github.com)")

if __name__ == "__main__":
    main()