# Prediksi Harga Rumah

Prediksi harga rumah (`SalePrice`) pada dataset House Prices (Ames, Iowa) menggunakan tiga model regresi:
Least Angle Regression (LARS), Linear Regression, dan Gradient Boosting Regressor.
Model terbaik dipakai pada aplikasi web Streamlit.

## Hasil Evaluasi

Dievaluasi pada data uji (112 baris dari 557 baris setelah pembersihan data, pembagian 80:20).

| Model             |    MAE |    MSE |      R² |
|:------------------|-------:|-------:|--------:|
| LARS              | 0.7201 | 0.8176 | -0.0185 |
| Linear Regression | 0.2374 | 0.0958 |  0.8807 |
| Gradient Boosting | 0.2174 | 0.0891 |  0.8891 |

Catatan: `SalePrice` distandarisasi sebelum training, sehingga MAE dan MSE berada dalam satuan standar deviasi.

## Pembahasan

- **Gradient Boosting** menjadi model terbaik pada ketiga metrik: MAE dan MSE paling kecil, R² paling tinggi (0.8891).
- **Linear Regression** hanya sedikit di bawah Gradient Boosting (R² 0.8807), jadi hubungan antara fitur dan harga cukup baik ditangkap model linear.
- **LARS** paling buruk dengan R² negatif (-0.0185), artinya tidak lebih baik daripada menebak harga rata-rata. Hal ini wajar karena konfigurasinya `n_nonzero_coefs=1` membatasi model hanya memakai satu fitur.
- Hasil berasal dari satu kali pembagian data (`random_state=1`) dengan data uji yang kecil, sehingga angkanya bisa berubah jika pembagian data berbeda.

## Struktur

- `LAB_2_Houseprice.ipynb`: notebook pembersihan data, EDA, pelatihan, dan evaluasi
- `app.py`: aplikasi Streamlit
- `artifacts.joblib`: model dan parameter preprocessing yang dipakai aplikasi
- `requirements.txt`: daftar dependensi

## Menjalankan Aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi juga bisa di-deploy gratis lewat [Streamlit Community Cloud](https://share.streamlit.io) dengan memilih repo ini dan file utama `app.py`.

## Referensi

1. scikit-learn developers. *sklearn.metrics.mean_squared_error*. https://scikit-learn.org/1.3/modules/generated/sklearn.metrics.mean_squared_error.html
2. scikit-learn developers. *sklearn.metrics.r2_score*. https://scikit-learn.org/1.2/modules/generated/sklearn.metrics.r2_score.html
3. Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation. *PeerJ Computer Science*. https://www.peerj.com/articles/cs-623
4. Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. *Climate Research*, 30(1), 79-82. https://int-res.com/abstracts/cr/v30/n1/p79-82
5. Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7, 1247-1250. https://doi.org/10.5194/gmd-7-1247-2014
