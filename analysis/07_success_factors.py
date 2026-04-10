"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 07 — FACTORES DE ÉXITO Y ATRIBUCIÓN SUBJETIVA
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Analiza qué factores predicen el éxito y cómo los individuos
    atribuyen subjetivamente su propio éxito. Incluye:
    - Pesos de atribución subjetiva por factor
    - Diferencia entre atribución declarada vs correlación real
    - Análisis por sector, nivel de riqueza y arquetipos
    - Ranking de predictores por beta estandarizada

Output:
    - results/07_atribucion_global.csv
    - results/07_atribucion_por_sector.csv
    - results/07_atribucion_por_riqueza.csv
    - results/07_prediccion_vs_atribucion.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)
df = pd.read_csv("../data/dataset_5000.csv")

ATTR_COLS = {
    'attr_disciplina_pct':     'Disciplina / Constancia',
    'attr_oportunidad_pct':    'Timing / Oportunidad',
    'attr_conocimiento_pct':   'Conocimiento técnico',
    'attr_red_pct':            'Red de contactos',
    'attr_trabajo_duro_pct':   'Trabajo duro (horas)',
    'attr_vision_pct':         'Visión / Estrategia',
    'attr_suerte_pct':         'Suerte / Azar',
    'attr_capital_inicial_pct':'Capital inicial',
}

REAL_PREDICTORS = {
    'score_disciplina':     'Disciplina / Constancia',
    'score_conocimiento':   'Conocimiento técnico',
    'horas_sem_pico':       'Trabajo duro (horas)',
    'score_red_contactos':  'Red de contactos',
    'score_vision':         'Visión / Estrategia',
    'score_resiliencia':    'Resiliencia',
    'capital_inicial_usd':  'Capital inicial',
    'attr_suerte_pct':      'Suerte / Azar',
}

# ── 1. ATRIBUCIÓN GLOBAL ──────────────────────────────────────────────────────
print("=" * 70)
print("1. ATRIBUCIÓN SUBJETIVA GLOBAL — ¿A qué atribuyen su éxito?")
print("=" * 70)

attr_rows = []
for col, label in ATTR_COLS.items():
    x = df[col]
    r_exito, p_exito = stats.pearsonr(x, df['score_exito'])
    r_roi, p_roi     = stats.pearsonr(x, df['roi_anualizado_pct'])
    attr_rows.append({
        "factor":                 label,
        "variable":               col,
        "media_pct_declarado":    round(x.mean(), 2),
        "mediana_pct_declarado":  round(x.median(), 2),
        "desv_std":               round(x.std(), 2),
        "min":                    round(x.min(), 2),
        "max":                    round(x.max(), 2),
        "r_con_score_exito":      round(r_exito, 4),
        "p_con_score_exito":      round(p_exito, 6),
        "r_con_roi":              round(r_roi, 4),
        "p_con_roi":              round(p_roi, 6),
    })

attr_df = pd.DataFrame(attr_rows).sort_values("media_pct_declarado", ascending=False)
attr_df.to_csv("results/07_atribucion_global.csv", index=False)
print(attr_df[['factor','media_pct_declarado','r_con_score_exito','r_con_roi']].to_string(index=False))

# ── 2. PREDICCIÓN REAL vs ATRIBUCIÓN DECLARADA ───────────────────────────────
print("\n" + "=" * 70)
print("2. PREDICCIÓN REAL (correlación con éxito) vs ATRIBUCIÓN DECLARADA")
print("=" * 70)

compare_rows = []
for col, label in REAL_PREDICTORS.items():
    x = df[col]
    r_exito, p_exito = stats.pearsonr(x, df['score_exito'])
    r_roi, _         = stats.pearsonr(x, df['roi_anualizado_pct'])

    # Buscar peso declarado equivalente
    attr_col = None
    for ac, al in ATTR_COLS.items():
        if al == label:
            attr_col = ac
            break
    pct_decl = df[attr_col].mean() if attr_col else np.nan

    compare_rows.append({
        "factor":            label,
        "r_con_exito":       round(r_exito, 4),
        "r2_con_exito":      round(r_exito**2, 4),
        "r_con_roi":         round(r_roi, 4),
        "pct_atribuido_decl":round(pct_decl, 2) if not np.isnan(pct_decl) else "—",
        "gap":               round(abs(r_exito) * 100 - pct_decl, 2) if not np.isnan(pct_decl) else "—",
        "subvalorado":       abs(r_exito) * 100 > float(pct_decl) if not np.isnan(pct_decl) else "—",
    })

compare_df = pd.DataFrame(compare_rows).sort_values("r_con_exito", ascending=False)
compare_df.to_csv("results/07_prediccion_vs_atribucion.csv", index=False)
print(compare_df.to_string(index=False))
print("\nNOTA: Gap (+) = factor más importante de lo que la gente declara")

# ── 3. ATRIBUCIÓN POR SECTOR ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. ATRIBUCIÓN DECLARADA POR SECTOR")
print("=" * 70)

attr_sector = df.groupby('sector')[list(ATTR_COLS.keys())].mean().round(2)
attr_sector.columns = [ATTR_COLS[c] for c in attr_sector.columns]
attr_sector.to_csv("results/07_atribucion_por_sector.csv")
print(attr_sector.to_string())

# ── 4. ATRIBUCIÓN POR NIVEL DE RIQUEZA ───────────────────────────────────────
print("\n" + "=" * 70)
print("4. ATRIBUCIÓN DECLARADA POR NIVEL DE RIQUEZA")
print("=" * 70)

df['rango_riqueza'] = pd.cut(df['patrimonio_m_usd'],
    bins=[0, 5, 20, 100, 980],
    labels=['$1M–$5M', '$5M–$20M', '$20M–$100M', '$100M+'])

attr_wealth = df.groupby('rango_riqueza', observed=True)[list(ATTR_COLS.keys())].mean().round(2)
attr_wealth.columns = [ATTR_COLS[c] for c in attr_wealth.columns]
attr_wealth.to_csv("results/07_atribucion_por_riqueza.csv")
print(attr_wealth.to_string())

# ── 5. ANÁLISIS DEL ROL DE LA SUERTE ─────────────────────────────────────────
print("\n" + "=" * 70)
print("5. ANÁLISIS DEL ROL DE LA SUERTE")
print("=" * 70)

suerte = df['attr_suerte_pct']
print(f"  % suerte declarado — media: {suerte.mean():.2f}%, mediana: {suerte.median():.2f}%")
print(f"  Rango: {suerte.min():.2f}% – {suerte.max():.2f}%")
print(f"  % que atribuye >10% a la suerte: {(suerte > 10).mean()*100:.1f}%")
print(f"  Correlación suerte ↔ score éxito: {stats.pearsonr(suerte, df['score_exito'])[0]:.4f}")
print(f"  Correlación suerte ↔ ROI: {stats.pearsonr(suerte, df['roi_anualizado_pct'])[0]:.4f}")

# Suerte por nivel educativo
edu_suerte = df.groupby('nivel_educacion')['attr_suerte_pct'].mean().round(2)
print(f"\n  Suerte declarada por nivel educativo:")
print(edu_suerte.to_string())

# ── 6. DISCIPLINA: EL FACTOR DOMINANTE ───────────────────────────────────────
print("\n" + "=" * 70)
print("6. DISCIPLINA — ANÁLISIS PROFUNDO DEL FACTOR #1")
print("=" * 70)

disc = df['score_disciplina']
print(f"  Score disciplina — media: {disc.mean():.2f} / 10")
print(f"  % con disciplina ≥8: {(disc >= 8).mean()*100:.1f}%")
print(f"  r con score_exito: {stats.pearsonr(disc, df['score_exito'])[0]:.4f}")
print(f"  r con ROI: {stats.pearsonr(disc, df['roi_anualizado_pct'])[0]:.4f}")
print(f"  r con años_primer_millon (neg=más rápido): {stats.pearsonr(disc, df['anos_primer_millon'])[0]:.4f}")
print(f"  r con bienestar: {stats.pearsonr(disc, df['bienestar_psicologico'])[0]:.4f}")

# Disciplina por cuartil → éxito promedio
df['cuartil_disciplina'] = pd.qcut(disc, q=4, labels=['Q1 Baja','Q2 Med-Baja','Q3 Med-Alta','Q4 Alta'])
disc_exito = df.groupby('cuartil_disciplina', observed=True)[['score_exito','roi_anualizado_pct','anos_primer_millon','bienestar_psicologico']].mean().round(2)
print(f"\n  Éxito y ROI por cuartil de disciplina:")
print(disc_exito.to_string())

print("\n✅ Análisis 07 completado.")
