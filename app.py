import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import warnings, io, joblib, os
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flue Gas Flow Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}

.stApp { background: #080d14; }

.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 50%, #061020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #0080ff, #00d4ff, transparent);
}
.main-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f4fd;
    margin: 0;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.main-header p {
    color: #4a9eda;
    font-size: 0.95rem;
    margin: 0.4rem 0 0;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
}

.metric-panel {
    background: linear-gradient(135deg, #0d1b2a, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-panel::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20%; right: 20%; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff44, transparent);
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.7rem;
    font-weight: 400;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.72rem;
    color: #4a7090;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}
.metric-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #2a6090;
    margin-top: 2px;
}

.status-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
}
.status-train { background: #002244; color: #0080ff; border: 1px solid #0044aa; }
.status-test  { background: #001a22; color: #00d4ff; border: 1px solid #006688; }

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #7ec8e3;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-left: 3px solid #00d4ff;
    padding-left: 10px;
    margin: 1.5rem 0 1rem;
}

.pred-result {
    background: linear-gradient(135deg, #001a2e, #00253d);
    border: 1px solid #004d7a;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-result::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.pred-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 3px;
    line-height: 1;
}
.pred-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    color: #4a9eda;
    margin-top: 6px;
}

div[data-testid="stSidebar"] {
    background: #060c14;
    border-right: 1px solid #1e3a5f;
}
div[data-testid="stSidebar"] .stMarkdown h2,
div[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Rajdhani', sans-serif;
    color: #4a9eda;
    letter-spacing: 1.5px;
}

.stButton > button {
    background: linear-gradient(135deg, #003366, #004d99);
    color: #b0d4f1;
    border: 1px solid #0066cc;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #004499, #0066cc);
    border-color: #00aaff;
    color: #ffffff;
}

.stSlider > div > div { background: #1e3a5f !important; }
.stNumberInput input, .stTextInput input {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    color: #b0d4f1 !important;
    border-radius: 6px !important;
}
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Utility functions ──────────────────────────────────────────────────────────

def styled_fig():
    """Return a dark-themed matplotlib figure."""
    fig = plt.figure(facecolor="#0a1628")
    return fig


def apply_dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="#7ec8e3", labelsize=9)
    ax.xaxis.label.set_color("#7ec8e3")
    ax.yaxis.label.set_color("#7ec8e3")
    ax.title.set_color("#b0d4f1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1a2e45", linewidth=0.5, linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if title:  ax.set_title(title, fontsize=11, pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)


@st.cache_data
def load_data(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return None


def train_mlp(X_train, y_train, hidden_layers, activation, lr, max_iter, alpha):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    y_tr = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        learning_rate_init=lr,
        max_iter=max_iter,
        alpha=alpha,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(X_tr, y_tr)
    return model, scaler_X, scaler_y


def evaluate(model, scaler_X, scaler_y, X, y):
    Xs = scaler_X.transform(X)
    ys_pred = model.predict(Xs)
    y_pred = scaler_y.inverse_transform(ys_pred.reshape(-1, 1)).ravel()
    y_true = y.values
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return y_pred, r2, rmse, mae, mape


# ── Plot functions ──────────────────────────────────────────────────────────────

def make_parity_plot(y_train_true, y_train_pred, y_test_true, y_test_pred):
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#0a1628")
    apply_dark_ax(ax, title="Parity Plot  ·  Actual vs Predicted",
                  xlabel="Actual Flow Rate", ylabel="Predicted Flow Rate")

    all_vals = np.concatenate([y_train_true, y_test_true, y_train_pred, y_test_pred])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            color="#00d4ff", linewidth=1.5, linestyle="--", label="Perfect fit", zorder=2)

    ax.scatter(y_train_true, y_train_pred, c="#0066ff", alpha=0.65, s=35,
               edgecolors="#0044aa", linewidths=0.5, label="Train", zorder=3)
    ax.scatter(y_test_true,  y_test_pred,  c="#00d4ff", alpha=0.80, s=45,
               edgecolors="#007799", linewidths=0.5, label="Test",  zorder=4)

    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#b0d4f1", fontsize=9)
    fig.tight_layout()
    return fig


def make_validation_plot(model, y_train_true, y_train_pred, y_test_true, y_test_pred):
    fig = plt.figure(figsize=(13, 5), facecolor="#0a1628")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

    # ── Left: Loss curve ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    apply_dark_ax(ax1, title="Training Loss Curve", xlabel="Iteration", ylabel="Loss (MSE)")
    loss = model.loss_curve_
    iters = np.arange(1, len(loss) + 1)
    ax1.plot(iters, loss, color="#0066ff", linewidth=1.8, label="Train loss")
    if hasattr(model, "validation_scores_") and model.validation_scores_ is not None:
        val_loss = [-s for s in model.validation_scores_]
        if len(val_loss) == len(loss):
            ax1.plot(iters, val_loss, color="#00d4ff", linewidth=1.4,
                     linestyle="--", label="Val loss")
    ax1.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#b0d4f1", fontsize=9)

    # ── Right: Residuals ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    apply_dark_ax(ax2, title="Residuals  (Actual − Predicted)",
                  xlabel="Sample Index", ylabel="Residual")
    train_res = y_train_true - y_train_pred
    test_res  = y_test_true  - y_test_pred
    ax2.scatter(range(len(train_res)), train_res, c="#0066ff", alpha=0.6, s=22,
                edgecolors="none", label="Train")
    ax2.scatter(range(len(train_res), len(train_res) + len(test_res)), test_res,
                c="#00d4ff", alpha=0.75, s=28, edgecolors="none", label="Test")
    ax2.axhline(0, color="#00d4ff", linewidth=1, linestyle="--", alpha=0.6)
    ax2.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#b0d4f1", fontsize=9)

    fig.tight_layout()
    return fig


def make_train_test_series(y_train_true, y_train_pred, y_test_true, y_test_pred):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="#0a1628")
    for ax, y_true, y_pred, split in zip(axes,
            [y_train_true, y_test_true],
            [y_train_pred, y_test_pred],
            ["Train", "Test"]):
        apply_dark_ax(ax, title=f"{split} Set  ·  Actual vs Predicted",
                      xlabel="Sample Index", ylabel="Flow Rate")
        idx = np.arange(len(y_true))
        ax.plot(idx, y_true, color="#0066ff", linewidth=1.6, label="Actual")
        ax.plot(idx, y_pred, color="#00d4ff", linewidth=1.4,
                linestyle="--", alpha=0.85, label="Predicted")
        ax.fill_between(idx, y_true, y_pred, alpha=0.12, color="#00aaff")
        ax.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#b0d4f1", fontsize=9)
    fig.tight_layout()
    return fig


# ── Session state ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model     = None
    st.session_state.scaler_X  = None
    st.session_state.scaler_y  = None
    st.session_state.col_names = None
    st.session_state.trained   = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ Flue Gas MLP")
    st.markdown("---")
    uploaded = st.file_uploader("Upload dataset (CSV / Excel)",
                                type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.markdown("### ⚙️ Model Hyperparameters")

    n_layers = st.slider("Hidden layers", 1, 5, 3)
    neurons  = st.slider("Neurons per layer", 8, 256, 64)
    hidden_layers = tuple([neurons] * n_layers)

    activation = st.selectbox("Activation function",
                               ["relu", "tanh", "logistic"], index=0)
    lr         = st.select_slider("Learning rate",
                                   options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                   value=1e-3,
                                   format_func=lambda x: f"{x:.4f}")
    max_iter   = st.slider("Max iterations", 100, 2000, 500, step=100)
    alpha      = st.select_slider("L2 regularisation (α)",
                                   options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                                   value=1e-4,
                                   format_func=lambda x: f"{x:.5f}")

    st.markdown("---")
    train_btn = st.button("🚀  Train Model", use_container_width=True)
    st.markdown("---")
    st.caption("v1.0 · MLP Regressor · scikit-learn")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚗️ Flue Gas Flow Rate Predictor</h1>
  <p>MLP Neural Network · 5 inputs → 1 output · 70/30 Train-Test Split</p>
</div>
""", unsafe_allow_html=True)


# ── Main logic ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar. The first 5 columns must be input parameters and the 6th column must be the output (total flow rate).")
    st.stop()

df = load_data(uploaded)
if df is None or df.shape[1] < 6:
    st.error("Dataset must have at least 6 columns (5 inputs + 1 output).")
    st.stop()

# Use first 6 columns
df = df.iloc[:, :6].dropna()
input_cols  = df.columns[:5].tolist()
output_col  = df.columns[5]

# ── Data preview ───────────────────────────────────────────────────────────────
with st.expander("📂 Dataset Preview", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Train Samples", f"{int(len(df)*0.7):,}  (70%)")
    c3.metric("Test Samples",  f"{int(len(df)*0.3):,}  (30%)")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Input features: {', '.join(input_cols)} | Target: **{output_col}**")


# ── Train ──────────────────────────────────────────────────────────────────────
if train_btn:
    X = df[input_cols]
    y = df[output_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    with st.spinner("⚡ Training MLP…"):
        model, scaler_X, scaler_y = train_mlp(
            X_train, y_train, hidden_layers, activation, lr, max_iter, alpha
        )

    y_train_pred, r2_tr, rmse_tr, mae_tr, mape_tr = evaluate(
        model, scaler_X, scaler_y, X_train, y_train)
    y_test_pred, r2_te, rmse_te, mae_te, mape_te = evaluate(
        model, scaler_X, scaler_y, X_test, y_test)

    # Persist in session state
    st.session_state.model      = model
    st.session_state.scaler_X   = scaler_X
    st.session_state.scaler_y   = scaler_y
    st.session_state.col_names  = input_cols
    st.session_state.trained    = True
    st.session_state.results    = dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_train_pred=y_train_pred, y_test_pred=y_test_pred,
        r2_tr=r2_tr, rmse_tr=rmse_tr, mae_tr=mae_tr, mape_tr=mape_tr,
        r2_te=r2_te, rmse_te=rmse_te, mae_te=mae_te, mape_te=mape_te,
    )
    st.success("✅ Model trained successfully!")


# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.trained:
    r = st.session_state.results
    model     = st.session_state.model
    scaler_X  = st.session_state.scaler_X
    scaler_y  = st.session_state.scaler_y

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Performance Metrics</div>',
                unsafe_allow_html=True)

    col_labels = ["R² Score", "RMSE", "MAE", "MAPE (%)"]
    train_vals = [f"{r['r2_tr']:.4f}", f"{r['rmse_tr']:.4f}",
                  f"{r['mae_tr']:.4f}",  f"{r['mape_tr']:.2f}"]
    test_vals  = [f"{r['r2_te']:.4f}", f"{r['rmse_te']:.4f}",
                  f"{r['mae_te']:.4f}",  f"{r['mape_te']:.2f}"]

    cols = st.columns(4)
    for col, lbl, tv, tsv in zip(cols, col_labels, train_vals, test_vals):
        col.markdown(f"""
        <div class="metric-panel">
          <div class="metric-value">{tsv}</div>
          <div class="metric-label">{lbl} · Test</div>
          <div class="metric-sub">Train: {tv}</div>
        </div>""", unsafe_allow_html=True)

    # ── Architecture summary ──────────────────────────────────────────────────
    arch_str = " → ".join(
        [f"Input({len(input_cols)})"] +
        [f"Dense({n})" for n in hidden_layers] +
        ["Output(1)"]
    )
    st.caption(f"🏗  Architecture: **{arch_str}**  |  "
               f"Activation: **{activation}**  |  "
               f"Iterations run: **{model.n_iter_}**  |  "
               f"Final loss: **{model.loss_:.6f}**")

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Parity Plot</div>', unsafe_allow_html=True)
    fig_parity = make_parity_plot(
        r["y_train"].values, r["y_train_pred"],
        r["y_test"].values,  r["y_test_pred"]
    )
    st.pyplot(fig_parity, use_container_width=True)

    st.markdown('<div class="section-title">Validation Plot</div>', unsafe_allow_html=True)
    fig_val = make_validation_plot(
        model,
        r["y_train"].values, r["y_train_pred"],
        r["y_test"].values,  r["y_test_pred"]
    )
    st.pyplot(fig_val, use_container_width=True)

    st.markdown('<div class="section-title">Train / Test Series Comparison</div>',
                unsafe_allow_html=True)
    fig_series = make_train_test_series(
        r["y_train"].values, r["y_train_pred"],
        r["y_test"].values,  r["y_test_pred"]
    )
    st.pyplot(fig_series, use_container_width=True)

    # ── Download predictions ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Download Predictions</div>', unsafe_allow_html=True)
    pred_df = pd.concat([
        r["X_train"].assign(Split="Train",
                             Actual=r["y_train"].values,
                             Predicted=r["y_train_pred"]),
        r["X_test"].assign(Split="Test",
                            Actual=r["y_test"].values,
                            Predicted=r["y_test_pred"])
    ], ignore_index=True)
    pred_df["Residual"] = pred_df["Actual"] - pred_df["Predicted"]
    st.dataframe(pred_df.head(20), use_container_width=True)
    csv_bytes = pred_df.to_csv(index=False).encode()
    st.download_button("⬇️  Download All Predictions (CSV)", csv_bytes,
                       file_name="predictions.csv", mime="text/csv")


# ── Prediction panel ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Single-Point Prediction</div>', unsafe_allow_html=True)

if not st.session_state.trained:
    st.info("Train the model first using the sidebar controls.")
else:
    cols_input = st.session_state.col_names
    st.markdown("Enter values for each input parameter:")
    pcols = st.columns(5)
    user_inputs = []
    for i, (col, pc) in enumerate(zip(cols_input, pcols)):
        val = pc.number_input(col, value=0.0, format="%.4f", key=f"inp_{i}")
        user_inputs.append(val)

    if st.button("⚡  Predict Flow Rate"):
        X_new = np.array(user_inputs).reshape(1, -1)
        Xs    = st.session_state.scaler_X.transform(X_new)
        ys    = st.session_state.model.predict(Xs)
        y_out = st.session_state.scaler_y.inverse_transform(
                    ys.reshape(-1, 1)).ravel()[0]

        st.markdown(f"""
        <div class="pred-result">
          <div class="pred-value">{y_out:,.4f}</div>
          <div class="pred-unit">Total Flue Gas Flow Rate</div>
        </div>""", unsafe_allow_html=True)
