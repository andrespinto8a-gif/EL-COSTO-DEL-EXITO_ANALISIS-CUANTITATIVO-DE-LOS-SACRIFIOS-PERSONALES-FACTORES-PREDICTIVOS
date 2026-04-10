"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 01 — ESTADÍSTICAS DESCRIPTIVAS COMPLETAS
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Estadísticas descriptivas completas de todas las variables numéricas
    del dataset: media, mediana, desviación estándar, percentiles,
    sesgo, curtosis, coeficiente de variación e intervalos de confianza.

Output:
    - results/01_descriptive_stats_full.csv
    - results/01_descriptive_stats_by_sector.csv
    - results/01_descriptive_stats_by_wealth.csv
    - results/01_frecuencias_categoricas.csv
    - Prints resumen en consola
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)

# ── CARGA DE DATOS ───────────────────────────────────────────────────────────
df = pd.read_csv("../data/dataset_5000.csv")
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas\n")

numeric_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# ── 1. ESTADÍSTICAS DESCRIPTIVAS GLOBALES ───────────────────────────────────
print("=" * 70)
print("1. ESTADÍSTICAS DESCRIPTIVAS GLOBALES")
print("=" * 70)

rows = []
for col in numeric_cols:
    x = df[col].dropna()
    n = len(x)
    mean = x.mean()
    std  = x.std()
    sem  = stats.sem(x)
    ci95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
    skew = x.skew()
    kurt = x.kurtosis()
    cv   = (std / mean * 100) if mean != 0 else np.nan
    rows.append({
        "variable":    col,
        "n":           n,
        "media":       round(mean, 4),
        "mediana":     round(x.median(), 4),
        "moda":        round(float(x.mode().iloc[0]), 4) if not x.mode().empty else np.nan,
        "desv_std":    round(std, 4),
        "varianza":    round(x.var(), 4),
        "cv_pct":      round(cv, 2),
        "min":         round(x.min(), 4),
        "p5":          round(x.quantile(0.05), 4),
        "p10":         round(x.quantile(0.10), 4),
        "p25":         round(x.quantile(0.25), 4),
        "p50":         round(x.quantile(0.50), 4),
        "p75":         round(x.quantile(0.75), 4),
        "p90":         round(x.quantile(0.90), 4),
        "p95":         round(x.quantile(0.95), 4),
        "max":         round(x.max(), 4),
        "iqr":         round(x.quantile(0.75) - x.quantile(0.25), 4),
        "sesgo":       round(skew, 4),
        "curtosis":    round(kurt, 4),
        "ci95_lower":  round(ci95[0], 4),
        "ci95_upper":  round(ci95[1], 4),
        "sem":         round(sem, 6),
        "normalidad_sw_p": round(stats.shapiro(x[:5000])[1], 6) if n >= 3 else np.nan,
    })

desc_df = pd.DataFrame(rows)
desc_df.to_csv("results/01_descriptive_stats_full.csv", index=False)
print(desc_df[["variable","n","media","mediana","desv_std","cv_pct","sesgo","curtosis"]].to_string(index=False))

# ── 2. ESTADÍSTICAS POR SECTOR ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. ESTADÍSTICAS CLAVE POR SECTOR")
print("=" * 70)

key_vars = ['patrimonio_m_usd','roi_anualizado_pct','horas_sem_pico',
            'sacrificio_total','score_exito','bienestar_psicologico',
            'score_disciplina','divorcios_num','anos_primer_millon']

sector_stats = df.groupby('sector')[key_vars].agg(['mean','median','std','min','max']).round(2)
sector_stats.to_csv("results/01_descriptive_stats_by_sector.csv")
print(df.groupby('sector')[key_vars].mean().round(2).to_string())

# ── 3. ESTADÍSTICAS POR NIVEL DE RIQUEZA ────────────────────────────────────
print("\n" + "=" * 70)
print("3. ESTADÍSTICAS POR NIVEL DE RIQUEZA")
print("=" * 70)

df['rango_riqueza'] = pd.cut(df['patrimonio_m_usd'],
    bins=[0, 5, 20, 100, 980],
    labels=['$1M–$5M', '$5M–$20M', '$20M–$100M', '$100M+'])

sac_vars = ['sacrificio_familia','sacrificio_salud','sacrificio_amor_pareja',
            'sacrificio_amigos','sacrificio_ocio_hobbies','sacrificio_sueno','sacrificio_total']
wealth_stats = df.groupby('rango_riqueza', observed=True)[sac_vars + ['score_exito','bienestar_psicologico','roi_anualizado_pct']].mean().round(2)
wealth_stats.to_csv("results/01_descriptive_stats_by_wealth.csv")
print(wealth_stats.to_string())

# ── 4. FRECUENCIAS CATEGÓRICAS ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. DISTRIBUCIONES CATEGÓRICAS")
print("=" * 70)

freq_rows = []
for col in cat_cols:
    vc = df[col].value_counts()
    for val, cnt in vc.items():
        freq_rows.append({"variable": col, "categoria": val,
                          "frecuencia": cnt, "porcentaje": round(cnt/len(df)*100, 2)})
        print(f"  {col} = {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

pd.DataFrame(freq_rows).to_csv("results/01_frecuencias_categoricas.csv", index=False)

# ── 5. OUTLIERS ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. DETECCIÓN DE OUTLIERS (método IQR)")
print("=" * 70)

outlier_rows = []
for col in numeric_cols:
    x = df[col]
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((x < lo) | (x > hi)).sum()
    pct_out = n_out / len(x) * 100
    outlier_rows.append({"variable": col, "q1": round(q1,3), "q3": round(q3,3),
                          "iqr": round(iqr,3), "fence_lo": round(lo,3),
                          "fence_hi": round(hi,3), "n_outliers": n_out,
                          "pct_outliers": round(pct_out,2)})
    if n_out > 0:
        print(f"  {col}: {n_out} outliers ({pct_out:.1f}%)")

pd.DataFrame(outlier_rows).to_csv("results/01_outliers.csv", index=False)

print("\n✅ Análisis 01 completado. Resultados guardados en results/")
