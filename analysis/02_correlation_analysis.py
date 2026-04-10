"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 02 — MATRIZ DE CORRELACIÓN Y ANÁLISIS DE ASOCIACIÓN
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Calcula correlaciones de Pearson, Spearman y Kendall entre todas las
    variables numéricas. Incluye p-values, intervalos de confianza del 95%,
    clasificación de fuerza e interpretación textual.

Output:
    - results/02_pearson_matrix.csv
    - results/02_spearman_matrix.csv
    - results/02_correlation_ranking.csv      ← ranking de correlaciones más fuertes
    - results/02_pvalues_matrix.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("../data/dataset_5000.csv")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
num_df = df[numeric_cols]

print(f"Variables numéricas: {len(numeric_cols)}")
print(f"Pares posibles: {len(numeric_cols)*(len(numeric_cols)-1)//2}")

# ── 1. MATRICES DE CORRELACIÓN ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. MATRICES DE CORRELACIÓN")
print("=" * 70)

pearson_r  = num_df.corr(method='pearson').round(4)
spearman_r = num_df.corr(method='spearman').round(4)

pearson_r.to_csv("results/02_pearson_matrix.csv")
spearman_r.to_csv("results/02_spearman_matrix.csv")
print("✓ Matriz de Pearson guardada")
print("✓ Matriz de Spearman guardada")

# ── 2. P-VALUES Y TESTS DE SIGNIFICANCIA ────────────────────────────────────
print("\n" + "=" * 70)
print("2. P-VALUES DE PEARSON (primeras 10 columnas × 10)")
print("=" * 70)

n_vars = len(numeric_cols)
pval_matrix = pd.DataFrame(np.ones((n_vars, n_vars)), index=numeric_cols, columns=numeric_cols)

for i, c1 in enumerate(numeric_cols):
    for j, c2 in enumerate(numeric_cols):
        if i != j:
            r, p = stats.pearsonr(num_df[c1].dropna(), num_df[c2].dropna())
            pval_matrix.loc[c1, c2] = round(p, 6)

pval_matrix.to_csv("results/02_pvalues_matrix.csv")
print("✓ Matriz de p-values guardada")

# ── 3. RANKING DE CORRELACIONES ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. RANKING COMPLETO DE CORRELACIONES (pares únicos)")
print("=" * 70)

def classify_corr(r):
    ar = abs(r)
    if ar >= 0.80: return "Muy fuerte"
    if ar >= 0.60: return "Fuerte"
    if ar >= 0.40: return "Moderada"
    if ar >= 0.20: return "Débil"
    return "Muy débil / ruido"

rows = []
for i, c1 in enumerate(numeric_cols):
    for j, c2 in enumerate(numeric_cols):
        if j > i:
            x1, x2 = num_df[c1].dropna(), num_df[c2].dropna()
            idx = x1.index.intersection(x2.index)
            x1_, x2_ = x1[idx], x2[idx]
            r_p, p_p   = stats.pearsonr(x1_, x2_)
            r_s, p_s   = stats.spearmanr(x1_, x2_)
            r2 = r_p ** 2
            # IC 95% via Fisher z
            n = len(idx)
            z = np.arctanh(r_p)
            se = 1 / np.sqrt(n - 3)
            ci_lo = np.tanh(z - 1.96 * se)
            ci_hi = np.tanh(z + 1.96 * se)
            rows.append({
                "var_a":             c1,
                "var_b":             c2,
                "pearson_r":         round(r_p, 4),
                "pearson_r2":        round(r2, 4),
                "pearson_p":         round(p_p, 6),
                "significativo_95":  p_p < 0.05,
                "ci95_lo":           round(ci_lo, 4),
                "ci95_hi":           round(ci_hi, 4),
                "spearman_r":        round(r_s, 4),
                "spearman_p":        round(p_s, 6),
                "fuerza":            classify_corr(r_p),
                "direccion":         "Positiva" if r_p > 0 else "Negativa",
                "abs_r":             round(abs(r_p), 4),
            })

corr_rank = pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
corr_rank.to_csv("results/02_correlation_ranking.csv", index=False)

print("\nTOP 30 CORRELACIONES MÁS FUERTES:")
print(corr_rank[["var_a","var_b","pearson_r","pearson_r2","pearson_p","fuerza"]].head(30).to_string(index=True))

print("\n10 CORRELACIONES MÁS DÉBILES (cercanas a 0):")
print(corr_rank[["var_a","var_b","pearson_r","fuerza"]].tail(10).to_string(index=True))

print(f"\n✅ Análisis 02 completado — {len(rows)} pares de correlación calculados.")

# ── 4. CORRELACIONES CON SCORE_EXITO ─────────────────────────────────────────
print("\n" + "=" * 70)
print("4. CORRELACIONES CON SCORE_EXITO (variable objetivo)")
print("=" * 70)

target_corr = corr_rank[(corr_rank['var_a']=='score_exito') | (corr_rank['var_b']=='score_exito')].copy()
target_corr = target_corr.sort_values('abs_r', ascending=False)
print(target_corr[["var_a","var_b","pearson_r","fuerza","direccion","pearson_p"]].to_string(index=False))

# ── 5. CORRELACIONES CON SACRIFICIO_TOTAL ────────────────────────────────────
print("\n" + "=" * 70)
print("5. CORRELACIONES CON SACRIFICIO_TOTAL")
print("=" * 70)

sac_corr = corr_rank[(corr_rank['var_a']=='sacrificio_total') | (corr_rank['var_b']=='sacrificio_total')].copy()
sac_corr = sac_corr.sort_values('abs_r', ascending=False)
print(sac_corr[["var_a","var_b","pearson_r","fuerza","direccion"]].to_string(index=False))
