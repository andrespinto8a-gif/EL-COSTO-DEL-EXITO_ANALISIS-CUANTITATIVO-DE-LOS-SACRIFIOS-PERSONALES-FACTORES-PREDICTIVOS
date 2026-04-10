"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 03 — ANÁLISIS PROFUNDO DE SACRIFICIOS
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Análisis exhaustivo de los 6 tipos de sacrificio:
    - Ranking de sacrificios por prevalencia e intensidad
    - Sacrificio dominante por individuo
    - Sacrificio por sector, nivel de riqueza y nivel educativo
    - Tests estadísticos de diferencias entre grupos (ANOVA, Kruskal-Wallis)
    - Índice de sacrificio compuesto y sus componentes

Output:
    - results/03_sacrificio_ranking.csv
    - results/03_sacrificio_por_sector.csv
    - results/03_sacrificio_por_riqueza.csv
    - results/03_sacrificio_dominante.csv
    - results/03_anova_resultados.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("../data/dataset_5000.csv")

SAC_COLS = {
    'sacrificio_familia':        'Familia',
    'sacrificio_salud':          'Salud física',
    'sacrificio_amor_pareja':    'Amor / Pareja',
    'sacrificio_amigos':         'Amigos',
    'sacrificio_ocio_hobbies':   'Ocio / Hobbies',
    'sacrificio_sueno':          'Sueño',
}
SAC_LIST = list(SAC_COLS.keys())

# ── 1. RANKING DE SACRIFICIOS ─────────────────────────────────────────────────
print("=" * 70)
print("1. RANKING DE SACRIFICIOS — PREVALENCIA E INTENSIDAD")
print("=" * 70)

THRESHOLD_ALTO  = 7.0
THRESHOLD_MED   = 4.0

ranking_rows = []
for col, label in SAC_COLS.items():
    x = df[col]
    pct_alto = (x >= THRESHOLD_ALTO).mean() * 100
    pct_med  = ((x >= THRESHOLD_MED) & (x < THRESHOLD_ALTO)).mean() * 100
    pct_bajo = (x < THRESHOLD_MED).mean() * 100
    ranking_rows.append({
        "sacrificio":           label,
        "variable":             col,
        "media":                round(x.mean(), 3),
        "mediana":              round(x.median(), 3),
        "desv_std":             round(x.std(), 3),
        "pct_alto_gte7":        round(pct_alto, 1),
        "pct_medio_4a7":        round(pct_med, 1),
        "pct_bajo_lt4":         round(pct_bajo, 1),
        "p90":                  round(x.quantile(0.90), 2),
        "n_exacto_10":          int((x == 10).sum()),
        "n_exacto_0":           int((x == 0).sum()),
    })

rank_df = pd.DataFrame(ranking_rows).sort_values("pct_alto_gte7", ascending=False)
rank_df.to_csv("results/03_sacrificio_ranking.csv", index=False)
print(rank_df[["sacrificio","media","pct_alto_gte7","pct_medio_4a7","pct_bajo_lt4"]].to_string(index=False))

# ── 2. SACRIFICIO DOMINANTE POR INDIVIDUO ────────────────────────────────────
print("\n" + "=" * 70)
print("2. SACRIFICIO DOMINANTE POR INDIVIDUO")
print("=" * 70)

df['sacrificio_dominante'] = df[SAC_LIST].idxmax(axis=1).map(SAC_COLS)
dom_dist = df['sacrificio_dominante'].value_counts().reset_index()
dom_dist.columns = ['sacrificio','n']
dom_dist['pct'] = (dom_dist['n'] / len(df) * 100).round(2)
dom_dist.to_csv("results/03_sacrificio_dominante.csv", index=False)
print(dom_dist.to_string(index=False))

# ── 3. SACRIFICIO POR SECTOR ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. SACRIFICIO PROMEDIO POR SECTOR")
print("=" * 70)

sector_sac = df.groupby('sector')[SAC_LIST + ['sacrificio_total']].mean().round(3)
sector_sac.to_csv("results/03_sacrificio_por_sector.csv")
print(sector_sac.to_string())

# ── 4. SACRIFICIO POR NIVEL DE RIQUEZA ───────────────────────────────────────
print("\n" + "=" * 70)
print("4. SACRIFICIO POR NIVEL DE RIQUEZA")
print("=" * 70)

df['rango_riqueza'] = pd.cut(df['patrimonio_m_usd'],
    bins=[0, 5, 20, 100, 980],
    labels=['$1M–$5M', '$5M–$20M', '$20M–$100M', '$100M+'])

wealth_sac = df.groupby('rango_riqueza', observed=True)[SAC_LIST + ['sacrificio_total']].mean().round(3)
wealth_sac.to_csv("results/03_sacrificio_por_riqueza.csv")
print(wealth_sac.to_string())

# ── 5. SACRIFICIO POR NIVEL EDUCATIVO ─────────────────────────────────────────
print("\n" + "=" * 70)
print("5. SACRIFICIO POR NIVEL EDUCATIVO")
print("=" * 70)

edu_order = ['Bachillerato','Técnico','Universitario','Posgrado','Maestría','Doctorado']
edu_sac = df.groupby('nivel_educacion')[SAC_LIST + ['sacrificio_total','score_exito']].mean().round(3)
edu_sac = edu_sac.reindex([e for e in edu_order if e in edu_sac.index])
print(edu_sac.to_string())

# ── 6. ANOVA — ¿DIFIEREN LOS SACRIFICIOS ENTRE SECTORES? ────────────────────
print("\n" + "=" * 70)
print("6. ANOVA — DIFERENCIAS DE SACRIFICIO ENTRE SECTORES")
print("=" * 70)

anova_rows = []
for col, label in SAC_COLS.items():
    grupos = [g[col].values for _, g in df.groupby('sector')]
    f_stat, p_val = stats.f_oneway(*grupos)
    kw_stat, kw_p = stats.kruskal(*grupos)
    ss_within = sum(((g - g.mean())**2).sum() for g in grupos)
    ss_total = ((df[col] - df[col].mean())**2).sum()
    eta2 = 1 - ss_within / ss_total if ss_total > 0 else 0
    anova_rows.append({
        "sacrificio":   label,
        "variable":     col,
        "F_statistic":  round(f_stat, 4),
        "p_value_anova":round(p_val, 6),
        "sig_anova":    p_val < 0.05,
        "kruskal_H":    round(kw_stat, 4),
        "p_kruskal":    round(kw_p, 6),
        "sig_kruskal":  kw_p < 0.05,
        "eta_squared":  round(eta2, 4),
        "efecto":       "Grande" if eta2 > 0.14 else ("Mediano" if eta2 > 0.06 else "Pequeño"),
    })

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv("results/03_anova_resultados.csv", index=False)
print(anova_df.to_string(index=False))

# ── 7. DISTRIBUCIÓN TEMPORAL DEL SACRIFICIO ───────────────────────────────────
print("\n" + "=" * 70)
print("7. SACRIFICIO POR TIEMPO HASTA PRIMER MILLÓN")
print("=" * 70)

df['rango_anos'] = pd.cut(df['anos_primer_millon'],
    bins=[0,3,6,10,15,35], labels=['1–3 años','4–6 años','7–10 años','11–15 años','16+ años'])

tiempo_sac = df.groupby('rango_anos', observed=True)[['sacrificio_total','score_exito','bienestar_psicologico']].mean().round(3)
print(tiempo_sac.to_string())

# ── 8. CORRELACIÓN ENTRE TIPOS DE SACRIFICIO ─────────────────────────────────
print("\n" + "=" * 70)
print("8. CORRELACIÓN ENTRE TIPOS DE SACRIFICIO")
print("=" * 70)

sac_corr = df[SAC_LIST].corr(method='pearson').round(4)
print(sac_corr.to_string())
sac_corr.to_csv("results/03_sacrificio_intercorrelacion.csv")

print("\n✅ Análisis 03 completado.")
