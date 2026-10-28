"""Aplikasi Streamlit: Prediksi Harga Rumah (Gradient Boosting Regressor)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_PATH = Path(__file__).parent / "artifacts.joblib"

# Fitur yang diisi pengguna. Fitur lain diisi nilai rata-rata / modus data latih.
NUMERIC_INPUTS = [
    ("OverallQual", "Kualitas material & finishing (1-10)"),
    ("OverallCond", "Kondisi rumah secara umum (1-10)"),
    ("GrLivArea", "Luas area tinggal di atas tanah (sq ft)"),
    ("TotalBsmtSF", "Luas basement (sq ft)"),
    ("LotArea", "Luas tanah (sq ft)"),
    ("YearBuilt", "Tahun dibangun"),
    ("YearRemodAdd", "Tahun renovasi terakhir"),
    ("FullBath", "Jumlah kamar mandi lengkap"),
    ("TotRmsAbvGrd", "Jumlah ruangan di atas tanah"),
    ("GarageCars", "Kapasitas garasi (jumlah mobil)"),
    ("Fireplaces", "Jumlah perapian"),
]

CATEGORICAL_INPUTS = [
    ("Neighborhood", "Lingkungan / kawasan"),
    ("KitchenQual", "Kualitas dapur"),
    ("ExterQual", "Kualitas eksterior"),
]

QUALITY_LABELS = {
    "Ex": "Ex - Excellent",
    "Gd": "Gd - Good",
    "TA": "TA - Typical / Average",
    "Fa": "Fa - Fair",
    "Po": "Po - Poor",
}

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠")


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACTS_PATH)


def numeric_bounds(A, col):
    """Batas input diambil dari rentang data latih supaya tidak ekstrapolasi."""
    lo, hi = A["num_range"][col]
    # dibulatkan dulu: sisa desimal kecil (mis. -1e-15) bisa membuat floor() jadi -1
    lo, hi = int(np.floor(round(lo, 4))), int(np.ceil(round(hi, 4)))
    if hi <= lo:
        hi = lo + 1
    default = int(np.clip(round(A["num_mean"][col]), lo, hi))
    return lo, hi, default


def predict_price(A, user_num, user_cat):
    """Susun 1 baris fitur seperti saat training, prediksi, lalu kembalikan ke USD."""
    row = {}
    for col in A["feature_columns"]:
        if col in A["cat_cols"]:
            if col in user_cat:
                row[col] = A["classes"][col].index(user_cat[col])
            else:
                row[col] = A["cat_default"][col]
        else:
            raw = user_num.get(col, A["num_mean"][col])
            row[col] = (raw - A["num_mean"][col]) / A["num_scale"][col]

    X = pd.DataFrame([row], columns=A["feature_columns"])
    y_std = float(A["model"].predict(X)[0])
    price = y_std * A["num_scale"]["SalePrice"] + A["num_mean"]["SalePrice"]
    return max(price, 0.0)


A = load_artifacts()

st.title("🏠 Prediksi Harga Rumah")
st.caption(
    "Model: Gradient Boosting Regressor · Dataset: House Prices (Ames, Iowa) · "
    "Harga dalam USD"
)

tab_pred, tab_eval = st.tabs(["Prediksi", "Evaluasi Model"])

with tab_pred:
    with st.form("form_rumah"):
        left, right = st.columns(2)
        user_num = {}
        for i, (col, label) in enumerate(NUMERIC_INPUTS):
            lo, hi, default = numeric_bounds(A, col)
            target = left if i % 2 == 0 else right
            user_num[col] = target.number_input(
                label, min_value=lo, max_value=hi, value=default, step=1
            )

        user_cat = {}
        for col, label in CATEGORICAL_INPUTS:
            user_cat[col] = st.selectbox(
                label,
                A["classes"][col],
                index=int(A["cat_default"][col]),
                format_func=lambda v: QUALITY_LABELS.get(v, v),
            )

        submitted = st.form_submit_button("Prediksi harga", type="primary")

    if submitted:
        price = predict_price(A, user_num, user_cat)
        st.metric("Perkiraan harga rumah", f"${price:,.0f}")
        st.caption(
            f"Rata-rata selisih prediksi model terhadap harga asli pada data uji (MAE) "
            f"sekitar ${A['mae_usd']:,.0f}. Baca hasil ini sebagai perkiraan kasar, "
            "bukan harga pasti."
        )

    st.info(
        "Model memakai 76 fitur, sedangkan form ini hanya meminta 14. Fitur lainnya "
        "diisi nilai rata-rata (numerik) atau nilai tersering (kategori) dari data latih, "
        "jadi mengubah satu input saja biasanya berdampak kecil pada harga. "
        "Ubah beberapa input sekaligus untuk melihat perbedaan yang jelas."
    )

with tab_eval:
    st.subheader("Hasil evaluasi pada data uji")
    results = pd.DataFrame(A["results"]).set_index("Model")
    st.table(results.round(4))

    best = results["R2"].idxmax()
    st.success(f"Model dengan R² tertinggi: **{best}**")
    st.metric("MAE model terbaik dalam USD", f"${A['mae_usd']:,.0f}")

    st.markdown(
        """
**Cara membaca metrik**
- **MAE**: rata-rata selisih absolut antara prediksi dan nilai asli. Makin kecil makin baik.
- **MSE**: rata-rata kuadrat selisih, sehingga error besar lebih dihukum. Makin kecil makin baik.
- **R²**: proporsi variasi harga yang bisa dijelaskan model. Maksimal 1; nilai negatif
  berarti lebih buruk daripada sekadar menebak rata-rata.

Target `SalePrice` distandarisasi saat training, jadi MAE dan MSE pada tabel berada
dalam satuan standar deviasi, bukan USD.
"""
    )
