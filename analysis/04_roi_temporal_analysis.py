"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 04 — ROI, TIEMPO Y ANÁLISIS FINANCIERO
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Análisis del retorno sobre inversión (ROI CAGR), tiempo hasta primer
    millón, curvas de crecimiento patrimonial por percentiles y análisis
    financiero comparado por sector y perfil del emprendedor.

Fórmula ROI:
    ROI_anualizado = (Patrimonio_final / Capital_inicial)^(1/años) − 1

Output:
    - results/04_roi_por_sector.csv
    - results/04_roi_por_educacion.csv
    - results/04_tiempo_primer_millon.csv
    - results/04_curva_patrimonial_percentiles.csv
    - results/04_roi_distribucion.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("../data/dataset_5000.csv")

# ── 1. DISTRIBUCIÓN GENERAL DEL ROI ─────────────────────────────────────────
print("=" * 70)
print("1. DISTRIBUCIÓN GENERAL DEL ROI ANUALIZADO")
print("=" * 70)

roi = df['roi_anualizado_pct']
bins = [0, 10, 20, 30, 45, 60, 80, 100, 210]
labels = ['<10%','10-20%','20-30%','30-45%','45-60%','60-80%','80-100%','100%+']
df['roi_rango'] = pd.cut(roi, bins=bins, labels=labels, right=False)
roi_dist = df['roi_rango'].value_counts().sort_index().reset_index()
roi_dist.columns = ['rango_roi','n']
roi_dist['pct'] = (roi_dist['n'] / len(df) * 100).round(2)
roi_dist['roi_medio_rango'] = [5,15,25,37.5,52.5,70,90,150]
roi_dist.to_csv("results/04_roi_distribucion.csv", index=False)

print(f"  Media:       {roi.mean():.2f}%")
print(f"  Mediana:     {roi.median():.2f}%")
print(f"  Desv. Std:   {roi.std():.2f}%")
print(f"  Mínimo:      {roi.min():.2f}%")
print(f"  Máximo:      {roi.max():.2f}%")
print(f"  P25:         {roi.quantile(0.25):.2f}%")
print(f"  P75:         {roi.quantile(0.75):.2f}%")
print(f"  % con ROI>50%: {(roi > 50).mean()*100:.1f}%")
print(f"  % con ROI>100%:{(roi > 100).mean()*100:.1f}%")
print(f"\n{roi_dist.to_string(index=False)}")

# ── 2. ROI POR SECTOR ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. ROI POR SECTOR")
print("=" * 70)

roi_sector = df.groupby('sector').agg(
    n=('roi_anualizado_pct', 'count'),
    roi_media=('roi_anualizado_pct', 'mean'),
    roi_mediana=('roi_anualizado_pct', 'median'),
    roi_std=('roi_anualizado_pct', 'std'),
    roi_p25=('roi_anualizado_pct', lambda x: x.quantile(0.25)),
    roi_p75=('roi_anualizado_pct', lambda x: x.quantile(0.75)),
    roi_max=('roi_anualizado_pct', 'max'),
    patrimonio_medio=('patrimonio_m_usd', 'mean'),
    patrimonio_mediano=('patrimonio_m_usd', 'median'),
    capital_inicial_medio=('capital_inicial_usd', 'mean'),
    anos_millon_medio=('anos_primer_millon', 'mean'),
).round(2).reset_index()
roi_sector = roi_sector.sort_values('roi_media', ascending=False)
roi_sector.to_csv("results/04_roi_por_sector.csv", index=False)
print(roi_sector[['sector','n','roi_media','roi_mediana','roi_std','patrimonio_medio']].to_string(index=False))

# ── 3. ROI POR NIVEL EDUCATIVO ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. ROI POR NIVEL EDUCATIVO")
print("=" * 70)

edu_order = ['Bachillerato','Técnico','Universitario','Posgrado','Maestría','Doctorado']
roi_edu = df.groupby('nivel_educacion').agg(
    n=('roi_anualizado_pct','count'),
    roi_media=('roi_anualizado_pct','mean'),
    roi_mediana=('roi_anualizado_pct','median'),
    score_exito_medio=('score_exito','mean'),
    patrimonio_medio=('patrimonio_m_usd','mean'),
    anos_millon_medio=('anos_primer_millon','mean'),
).round(2).reindex([e for e in edu_order if e in df['nivel_educacion'].unique()])
roi_edu.to_csv("results/04_roi_por_educacion.csv")
print(roi_edu.to_string())

# ── 4. TIEMPO HASTA PRIMER MILLÓN ───────────────────────────────────────────
print("\n" + "=" * 70)
print("4. TIEMPO HASTA PRIMER MILLÓN — DISTRIBUCIÓN COMPLETA")
print("=" * 70)

am = df['anos_primer_millon']
print(f"  Media:    {am.mean():.1f} años")
print(f"  Mediana:  {am.median():.1f} años")
print(f"  Moda:     {int(am.mode().iloc[0])} años")
print(f"  Desv:     {am.std():.1f} años")
print(f"  % en ≤5 años:  {(am <= 5).mean()*100:.1f}%")
print(f"  % en ≤10 años: {(am <= 10).mean()*100:.1f}%")
print(f"  % en >20 años: {(am > 20).mean()*100:.1f}%")

tiempo_dist = am.value_counts().sort_index().reset_index()
tiempo_dist.columns = ['anos', 'n']
tiempo_dist['pct'] = (tiempo_dist['n'] / len(df) * 100).round(2)
tiempo_dist['pct_acumulado'] = tiempo_dist['pct'].cumsum().round(2)
tiempo_dist.to_csv("results/04_tiempo_primer_millon.csv", index=False)
print(f"\n{tiempo_dist.to_string(index=False)}")

# ── 5. TIEMPO POR SECTOR Y PERFIL ────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. TIEMPO HASTA PRIMER MILLÓN — POR SECTOR")
print("=" * 70)

tiempo_sector = df.groupby('sector').agg(
    n=('anos_primer_millon','count'),
    media=('anos_primer_millon','mean'),
    mediana=('anos_primer_millon','median'),
    std=('anos_primer_millon','std'),
    pct_lt5=('anos_primer_millon', lambda x: (x<=5).mean()*100),
    pct_lt10=('anos_primer_millon', lambda x: (x<=10).mean()*100),
    roi_medio=('roi_anualizado_pct','mean'),
).round(2).sort_values('media')
print(tiempo_sector.to_string())

# ── 6. CURVA DE CRECIMIENTO PATRIMONIAL ──────────────────────────────────────
print("\n" + "=" * 70)
print("6. CURVA DE CRECIMIENTO PATRIMONIAL SIMULADA POR PERCENTILES")
print("=" * 70)

np.random.seed(42)
years = list(range(0, 21))
percentiles = {'p10': 0.10, 'p25': 0.25, 'p50': 0.50, 'p75': 0.75, 'p90': 0.90}

roi_by_p = {k: df['roi_anualizado_pct'].quantile(v)/100 for k, v in percentiles.items()}
cap_by_p = {k: df['capital_inicial_usd'].quantile(v) for k, v in percentiles.items()}

curve_rows = []
for yr in years:
    row = {'ano': yr}
    for pk, rv in roi_by_p.items():
        capital = cap_by_p[pk]
        pat = capital * (1 + rv)**yr / 1e6
        row[f'patrimonio_{pk}_m_usd'] = round(pat, 4)
    curve_rows.append(row)

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv("results/04_curva_patrimonial_percentiles.csv", index=False)
print(curve_df.to_string(index=False))

# ── 7. ANÁLISIS DE COSTO-BENEFICIO: ROI vs SACRIFICIO ──────────────────────
print("\n" + "=" * 70)
print("7. ANÁLISIS COSTO-BENEFICIO: ROI vs SACRIFICIO TOTAL")
print("=" * 70)

df['cuartil_roi'] = pd.qcut(df['roi_anualizado_pct'], q=4, labels=['Q1 Bajo','Q2 Med-Bajo','Q3 Med-Alto','Q4 Alto'])
roi_vs_sac = df.groupby('cuartil_roi', observed=True).agg(
    n=('roi_anualizado_pct','count'),
    roi_medio=('roi_anualizado_pct','mean'),
    sac_total=('sacrificio_total','mean'),
    sac_familia=('sacrificio_familia','mean'),
    sac_salud=('sacrificio_salud','mean'),
    bienestar=('bienestar_psicologico','mean'),
    divorcios=('divorcios_num','mean'),
    patrimonio=('patrimonio_m_usd','mean'),
).round(2)
print(roi_vs_sac.to_string())
roi_vs_sac.to_csv("results/04_roi_vs_sacrificio.csv")

# ── 8. RENTABILIDAD POR TIPO DE NEGOCIO ──────────────────────────────────────
print("\n" + "=" * 70)
print("8. ROI Y PATRIMONIO POR TIPO DE NEGOCIO")
print("=" * 70)

roi_tipo = df.groupby('tipo_negocio').agg(
    n=('roi_anualizado_pct','count'),
    roi_media=('roi_anualizado_pct','mean'),
    roi_mediana=('roi_anualizado_pct','median'),
    patrimonio_medio=('patrimonio_m_usd','mean'),
    score_exito_medio=('score_exito','mean'),
    anos_primer_millon=('anos_primer_millon','mean'),
).round(2).sort_values('roi_media', ascending=False)
print(roi_tipo.to_string())
roi_tipo.to_csv("results/04_roi_por_tipo_negocio.csv")

print("\n✅ Análisis 04 completado.")
