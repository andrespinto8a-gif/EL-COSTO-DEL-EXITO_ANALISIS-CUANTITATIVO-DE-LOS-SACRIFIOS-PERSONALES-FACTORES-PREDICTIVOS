"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 05 — MODELOS DE REGRESIÓN
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Modelos de regresión lineal múltiple para predecir:
    (A) score_exito          — éxito empresarial global
    (B) roi_anualizado_pct   — retorno financiero
    (C) bienestar_psicologico — bienestar (costo oculto)
    (D) anos_primer_millon    — velocidad hasta el éxito

    Incluye: coeficientes, errores estándar, t-stats, p-values,
    VIF para multicolinealidad, diagnósticos de residuos, y
    comparación de modelos por AIC/BIC.

Output:
    - results/05_regresion_score_exito.csv
    - results/05_regresion_roi.csv
    - results/05_regresion_bienestar.csv
    - results/05_regresion_tiempo.csv
    - results/05_model_comparison.csv
    - results/05_vif.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("../data/dataset_5000.csv")

# ── FUNCIONES AUXILIARES ──────────────────────────────────────────────────────

def ols_manual(X_df, y_series, model_name="Modelo"):
    """Regresión OLS manual con estadísticas completas."""
    X_raw = X_df.copy()
    X_raw.insert(0, 'intercepto', 1.0)
    X = X_raw.values.astype(float)
    y = y_series.values.astype(float)
    n, k = X.shape

    # Coeficientes
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    resid = y - y_hat

    # Métricas
    ss_res = (resid**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)
    mse = ss_res / (n - k)
    rmse = np.sqrt(mse)
    mae = np.abs(resid).mean()

    # F-statistic
    f_stat = ((ss_tot - ss_res) / (k - 1)) / (ss_res / (n - k))
    f_p = 1 - stats.f.cdf(f_stat, k - 1, n - k)

    # Coef. std errors
    se_beta = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    ci_lo = beta - 1.96 * se_beta
    ci_hi = beta + 1.96 * se_beta

    # AIC / BIC
    log_lik = -n/2 * np.log(2 * np.pi * mse) - ss_res / (2 * mse)
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + k * np.log(n)

    coef_df = pd.DataFrame({
        "variable":   list(X_raw.columns),
        "coeficiente": beta.round(6),
        "se":         se_beta.round(6),
        "t_stat":     t_stats.round(4),
        "p_value":    p_vals.round(6),
        "sig":        ["***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")) for p in p_vals],
        "ci95_lo":    ci_lo.round(6),
        "ci95_hi":    ci_hi.round(6),
    })

    print(f"\n{'─'*65}")
    print(f"  MODELO: {model_name}")
    print(f"  n={n} | R²={r2:.4f} | R²-adj={r2_adj:.4f} | RMSE={rmse:.4f}")
    print(f"  F={f_stat:.2f} (p={f_p:.6f}) | AIC={aic:.1f} | BIC={bic:.1f}")
    print(f"{'─'*65}")
    print(coef_df.to_string(index=False))

    return coef_df, {"model": model_name, "n": n, "k": k, "r2": round(r2,4),
                     "r2_adj": round(r2_adj,4), "rmse": round(rmse,4),
                     "mae": round(mae,4), "f_stat": round(f_stat,4),
                     "f_pval": round(f_p,6), "aic": round(aic,2),
                     "bic": round(bic,2), "log_lik": round(log_lik,4)}

def compute_vif(X_df):
    """Calcula el VIF para detectar multicolinealidad."""
    vif_rows = []
    cols = X_df.columns.tolist()
    X = X_df.values.astype(float)
    for i, col in enumerate(cols):
        y_v = X[:, i]
        X_v = np.delete(X, i, axis=1)
        X_v = np.column_stack([np.ones(len(y_v)), X_v])
        XtX_inv = np.linalg.pinv(X_v.T @ X_v)
        beta = XtX_inv @ X_v.T @ y_v
        y_hat = X_v @ beta
        ss_res = ((y_v - y_hat)**2).sum()
        ss_tot = ((y_v - y_v.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_rows.append({"variable": col, "VIF": round(vif, 3),
                         "tolerancia": round(1/vif if vif > 0 else 0, 4),
                         "alerta": "⚠ ALTA" if vif > 10 else ("MODERADA" if vif > 5 else "OK")})
    return pd.DataFrame(vif_rows)

# ── PREPARAR VARIABLES ────────────────────────────────────────────────────────
features_main = ['score_disciplina','score_conocimiento','horas_sem_pico',
                  'sacrificio_total','anos_trayectoria_total','capital_inicial_usd',
                  'score_red_contactos','score_vision','score_resiliencia']

features_ext = features_main + ['edad_inicio','divorcios_num','sueno_horas_noche','ejercicio_dias_sem']

Xm = df[features_main].copy()
Xe = df[features_ext].copy()

model_results = []

# ── MODELO A: SCORE ÉXITO ────────────────────────────────────────────────────
print("=" * 65)
print("MODELO A — Variable Dependiente: SCORE_EXITO (0–100)")
print("=" * 65)

y_exito = df['score_exito']

coef_a1, meta_a1 = ols_manual(Xm, y_exito, "A1: Éxito ~ features principales")
coef_a2, meta_a2 = ols_manual(Xe, y_exito, "A2: Éxito ~ features extendidas")
model_results.extend([meta_a1, meta_a2])
coef_a2.to_csv("results/05_regresion_score_exito.csv", index=False)

# ── MODELO B: ROI ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODELO B — Variable Dependiente: ROI_ANUALIZADO_PCT")
print("=" * 65)

y_roi = df['roi_anualizado_pct']

coef_b1, meta_b1 = ols_manual(Xm, y_roi, "B1: ROI ~ features principales")
model_results.append(meta_b1)
coef_b1.to_csv("results/05_regresion_roi.csv", index=False)

# ── MODELO C: BIENESTAR ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODELO C — Variable Dependiente: BIENESTAR_PSICOLOGICO")
print("=" * 65)

y_bien = df['bienestar_psicologico']
feat_bien = ['sacrificio_total','sacrificio_salud','sacrificio_amor_pareja',
             'sacrificio_familia','horas_sem_pico','divorcios_num',
             'sueno_horas_noche','ejercicio_dias_sem','score_exito']

coef_c1, meta_c1 = ols_manual(df[feat_bien], y_bien, "C1: Bienestar ~ sacrificios + variables control")
model_results.append(meta_c1)
coef_c1.to_csv("results/05_regresion_bienestar.csv", index=False)

# ── MODELO D: TIEMPO HASTA PRIMER MILLÓN ──────────────────────────────────────
print("\n" + "=" * 65)
print("MODELO D — Variable Dependiente: AÑOS HASTA PRIMER MILLÓN")
print("=" * 65)

y_tiempo = df['anos_primer_millon']
feat_tiempo = ['score_disciplina','score_conocimiento','score_red_contactos',
               'edad_inicio','capital_inicial_usd','horas_sem_pico','score_vision']

coef_d1, meta_d1 = ols_manual(df[feat_tiempo], y_tiempo, "D1: Tiempo ~ factores de velocidad")
model_results.append(meta_d1)
coef_d1.to_csv("results/05_regresion_tiempo.csv", index=False)

# ── COMPARACIÓN DE MODELOS ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPARACIÓN DE TODOS LOS MODELOS")
print("=" * 65)

comp_df = pd.DataFrame(model_results)
comp_df.to_csv("results/05_model_comparison.csv", index=False)
print(comp_df[['model','n','r2','r2_adj','rmse','aic','bic']].to_string(index=False))

# ── VIF — MULTICOLINEALIDAD ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FACTOR DE INFLACIÓN DE VARIANZA (VIF)")
print("=" * 65)

vif_df = compute_vif(Xm)
vif_df.to_csv("results/05_vif.csv", index=False)
print(vif_df.to_string(index=False))

# ── BETAS ESTANDARIZADAS (importancia relativa) ────────────────────────────────
print("\n" + "=" * 65)
print("BETAS ESTANDARIZADAS — IMPORTANCIA RELATIVA DE PREDICTORES")
print("=" * 65)

Xm_std = (Xm - Xm.mean()) / Xm.std()
y_std  = (y_exito - y_exito.mean()) / y_exito.std()
_, meta_std = ols_manual(Xm_std, y_std, "Modelo Estandarizado (Betas)")

print("\n✅ Análisis 05 completado.")
