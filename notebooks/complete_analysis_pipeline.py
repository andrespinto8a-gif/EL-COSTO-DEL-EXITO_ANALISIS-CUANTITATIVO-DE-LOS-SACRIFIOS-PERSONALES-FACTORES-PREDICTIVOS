# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK COMPLETO — EL COSTO DEL ÉXITO
# Pipeline unificado: carga → limpieza → análisis → resultados
# El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
# ══════════════════════════════════════════════════════════════════════════════
# Para ejecutar: python complete_analysis_pipeline.py
# Genera TODOS los resultados en la carpeta results/
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("results", exist_ok=True)

print("╔══════════════════════════════════════════════════════════════════════╗")
print("║        EL COSTO DEL ÉXITO — PIPELINE DE ANÁLISIS COMPLETO          ║")
print("║              n = 5,000 | 41 Variables | 2024                        ║")
print("╚══════════════════════════════════════════════════════════════════════╝\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0: CARGA Y VALIDACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("SECCIÓN 0: CARGA Y VALIDACIÓN")
print("─" * 70)

df = pd.read_csv("../data/dataset_5000.csv")
print(f"✓ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"✓ Valores nulos: {df.isnull().sum().sum()}")
print(f"✓ Tipos: {df.dtypes.value_counts().to_dict()}")
print(f"✓ Columnas numéricas: {len(df.select_dtypes(include='number').columns)}")
print(f"✓ Columnas categóricas: {len(df.select_dtypes(include='object').columns)}")

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

SAC_COLS = ['sacrificio_familia','sacrificio_salud','sacrificio_amor_pareja',
            'sacrificio_amigos','sacrificio_ocio_hobbies','sacrificio_sueno']

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: KPIs GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 1: KPIs GLOBALES DEL ESTUDIO")
print("─" * 70)

kpis = {
    "Patrimonio mediano (M USD)":         df['patrimonio_m_usd'].median(),
    "Patrimonio medio (M USD)":           df['patrimonio_m_usd'].mean(),
    "ROI anualizado promedio (%)":        df['roi_anualizado_pct'].mean(),
    "ROI anualizado mediano (%)":         df['roi_anualizado_pct'].median(),
    "Horas/sem pico promedio":            df['horas_sem_pico'].mean(),
    "Años hasta primer millón (media)":   df['anos_primer_millon'].mean(),
    "Años hasta primer millón (mediana)": df['anos_primer_millon'].median(),
    "Score disciplina promedio (0-10)":   df['score_disciplina'].mean(),
    "Score conocimiento promedio (0-50)": df['score_conocimiento'].mean(),
    "Sacrificio total promedio (0-10)":   df['sacrificio_total'].mean(),
    "Sacrificio familia promedio":        df['sacrificio_familia'].mean(),
    "Sacrificio salud promedio":          df['sacrificio_salud'].mean(),
    "Sacrificio amor/pareja promedio":    df['sacrificio_amor_pareja'].mean(),
    "% con sacrificio total ≥7":         (df['sacrificio_total'] >= 7).mean()*100,
    "% tiempo libre ≥7 (ocio)":          (df['sacrificio_ocio_hobbies'] >= 7).mean()*100,
    "Score éxito promedio (0-100)":       df['score_exito'].mean(),
    "Bienestar psicológico promedio":     df['bienestar_psicologico'].mean(),
    "Satisfacción de vida promedio":      df['satisfaccion_vida'].mean(),
    "Divorcios promedio":                 df['divorcios_num'].mean(),
    "% con ≥1 divorcio":                 (df['divorcios_num'] >= 1).mean()*100,
    "Sueño promedio (hrs/noche)":        df['sueno_horas_noche'].mean(),
    "% con sueño <6h":                   (df['sueno_horas_noche'] < 6).mean()*100,
    "Empleos generados (media)":          df['empleos_generados'].mean(),
    "Capital inicial promedio (USD)":     df['capital_inicial_usd'].mean(),
    "Trayectoria total promedio (años)":  df['anos_trayectoria_total'].mean(),
    "Attr. disciplina declarada (%)":     df['attr_disciplina_pct'].mean(),
    "Attr. suerte declarada (%)":         df['attr_suerte_pct'].mean(),
}

kpi_df = pd.DataFrame(list(kpis.items()), columns=['indicador','valor'])
kpi_df['valor'] = kpi_df['valor'].round(3)
kpi_df.to_csv("results/00_kpis_globales.csv", index=False)
for k, v in kpis.items():
    print(f"  {k:<45}: {v:>10.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: MATRIZ DE CORRELACIÓN COMPLETA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 2: CORRELACIONES DE PEARSON — TOP 40 PARES")
print("─" * 70)

num_df = df[num_cols]
corr_rows = []
for i, c1 in enumerate(num_cols):
    for j, c2 in enumerate(num_cols):
        if j > i:
            r, p = stats.pearsonr(num_df[c1], num_df[c2])
            n = len(num_df)
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            corr_rows.append({
                "var_a": c1, "var_b": c2,
                "r": round(r, 4), "r2": round(r**2, 4),
                "p": round(p, 6), "sig": p < 0.05,
                "ci95_lo": round(np.tanh(z - 1.96*se), 4),
                "ci95_hi": round(np.tanh(z + 1.96*se), 4),
                "fuerza": "Muy fuerte" if abs(r)>=0.80 else
                           ("Fuerte" if abs(r)>=0.60 else
                           ("Moderada" if abs(r)>=0.40 else
                           ("Débil" if abs(r)>=0.20 else "Muy débil"))),
                "dir": "+" if r > 0 else "−",
            })

corr_df = pd.DataFrame(corr_rows).sort_values("r", key=abs, ascending=False)
corr_df.to_csv("results/02_all_correlations.csv", index=False)
top40 = corr_df.head(40)
print(top40[["var_a","var_b","r","r2","p","fuerza"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: ANÁLISIS DE SACRIFICIOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 3: ANÁLISIS DE SACRIFICIOS")
print("─" * 70)

sac_labels = {'sacrificio_familia':'Familia','sacrificio_salud':'Salud física',
              'sacrificio_amor_pareja':'Amor/Pareja','sacrificio_amigos':'Amigos',
              'sacrificio_ocio_hobbies':'Ocio/Hobbies','sacrificio_sueno':'Sueño'}

sac_summary = []
for col, label in sac_labels.items():
    x = df[col]
    sac_summary.append({
        "sacrificio": label,
        "media": round(x.mean(),3),
        "mediana": round(x.median(),3),
        "std": round(x.std(),3),
        "pct_gte7": round((x>=7).mean()*100,1),
        "pct_gte8": round((x>=8).mean()*100,1),
        "pct_lt4":  round((x<4).mean()*100,1),
        "r_exito":  round(stats.pearsonr(x, df['score_exito'])[0],4),
        "r_roi":    round(stats.pearsonr(x, df['roi_anualizado_pct'])[0],4),
        "r_bien":   round(stats.pearsonr(x, df['bienestar_psicologico'])[0],4),
        "r_horas":  round(stats.pearsonr(x, df['horas_sem_pico'])[0],4),
    })

sac_df = pd.DataFrame(sac_summary).sort_values("pct_gte7", ascending=False)
sac_df.to_csv("results/03_sacrificio_completo.csv", index=False)
print(sac_df.to_string(index=False))

# Por sector
df['rango_riqueza'] = pd.cut(df['patrimonio_m_usd'], bins=[0,5,20,100,980],
    labels=['$1M–$5M','$5M–$20M','$20M–$100M','$100M+'])

sector_sac = df.groupby('sector')[SAC_COLS + ['sacrificio_total']].mean().round(3)
wealth_sac  = df.groupby('rango_riqueza', observed=True)[SAC_COLS + ['sacrificio_total']].mean().round(3)
sector_sac.to_csv("results/03_sac_por_sector.csv")
wealth_sac.to_csv("results/03_sac_por_riqueza.csv")

# ANOVA — diferencias entre sectores
print(f"\n  ANOVA sacrificio_total entre sectores:")
grupos = [g['sacrificio_total'].values for _, g in df.groupby('sector')]
f, p = stats.f_oneway(*grupos)
print(f"  F={f:.2f}, p={p:.6f} {'*** SIGNIFICATIVO' if p<0.001 else ''}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: ROI Y ANÁLISIS TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 4: ROI Y ANÁLISIS TEMPORAL")
print("─" * 70)

roi_sector = df.groupby('sector').agg(
    n=('roi_anualizado_pct','count'),
    roi_media=('roi_anualizado_pct','mean'),
    roi_mediana=('roi_anualizado_pct','median'),
    patrimonio_medio=('patrimonio_m_usd','mean'),
    anos_millon_medio=('anos_primer_millon','mean'),
).round(2).sort_values('roi_media', ascending=False)
roi_sector.to_csv("results/04_roi_por_sector_pipeline.csv")
print("  ROI por sector:")
print(roi_sector.to_string())

# Distribución años hasta primer millón
am_bins = pd.cut(df['anos_primer_millon'], bins=[0,3,5,7,10,15,20,35],
    labels=['1-3','4-5','6-7','8-10','11-15','16-20','21+'])
am_dist = am_bins.value_counts().sort_index().reset_index()
am_dist.columns = ['rango_anos','n']
am_dist['pct'] = (am_dist['n']/len(df)*100).round(1)
am_dist['pct_acum'] = am_dist['pct'].cumsum().round(1)
am_dist.to_csv("results/04_distribucion_anos_millon_pipeline.csv", index=False)
print(f"\n  Distribución años hasta primer millón:")
print(am_dist.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: MODELOS DE REGRESIÓN OLS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 5: REGRESIÓN OLS — score_exito ~ predictores")
print("─" * 70)

PRED = ['score_disciplina','score_conocimiento','horas_sem_pico',
        'sacrificio_total','anos_trayectoria_total','score_vision','score_resiliencia']

Xv = np.column_stack([np.ones(len(df))] + [df[c].values for c in PRED])
yv = df['score_exito'].values
b = np.linalg.pinv(Xv.T @ Xv) @ Xv.T @ yv
yh = Xv @ b
resid = yv - yh
n, k = Xv.shape
mse = (resid**2).sum() / (n-k)
ss_tot = ((yv - yv.mean())**2).sum()
r2 = 1 - (resid**2).sum()/ss_tot
r2_adj = 1 - (1-r2)*(n-1)/(n-k)
se_b = np.sqrt(np.diag(np.linalg.pinv(Xv.T @ Xv) * mse))
t_s = b / se_b
p_s = 2*(1-stats.t.cdf(np.abs(t_s), df=n-k))

reg_df = pd.DataFrame({
    "variable": ["intercepto"] + PRED,
    "coef": b.round(6), "se": se_b.round(6),
    "t": t_s.round(4), "p": p_s.round(6),
    "sig": ["***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "")) for p in p_s],
    "ci95_lo": (b - 1.96*se_b).round(6), "ci95_hi": (b + 1.96*se_b).round(6),
})
reg_df.to_csv("results/05_regresion_principal.csv", index=False)
print(f"  R² = {r2:.4f} | R²-adj = {r2_adj:.4f} | n = {n}")
print(reg_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: BIENESTAR Y PARADOJA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 6: BIENESTAR Y LA PARADOJA ÉXITO-FELICIDAD")
print("─" * 70)

r_bien, _ = stats.pearsonr(df['score_exito'], df['bienestar_psicologico'])
r_sat,  _ = stats.pearsonr(df['score_exito'], df['satisfaccion_vida'])
r_sue,  _ = stats.pearsonr(df['score_exito'], df['sueno_horas_noche'])
print(f"  r(éxito ↔ bienestar):       {r_bien:+.4f}")
print(f"  r(éxito ↔ satisfacción):    {r_sat:+.4f}")
print(f"  r(éxito ↔ sueño):           {r_sue:+.4f}")
print(f"  r(sacrificio ↔ bienestar):  {stats.pearsonr(df['sacrificio_total'], df['bienestar_psicologico'])[0]:+.4f}")
print(f"  r(horas ↔ divorcios):       {stats.pearsonr(df['horas_sem_pico'], df['divorcios_num'])[0]:+.4f}")

# Tabla paradoja por quintil de éxito
df['q_exito'] = pd.qcut(df['score_exito'], q=5, labels=['Q1','Q2','Q3','Q4','Q5'])
paradox = df.groupby('q_exito', observed=True)[['score_exito','bienestar_psicologico',
    'satisfaccion_vida','sacrificio_total','divorcios_num','sueno_horas_noche']].mean().round(2)
paradox.to_csv("results/08_paradoja_pipeline.csv")
print(f"\n  Bienestar por quintil de éxito:")
print(paradox.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: ATRIBUCIÓN SUBJETIVA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 7: ATRIBUCIÓN SUBJETIVA DEL ÉXITO")
print("─" * 70)

ATTR = ['attr_disciplina_pct','attr_oportunidad_pct','attr_conocimiento_pct',
        'attr_red_pct','attr_trabajo_duro_pct','attr_vision_pct',
        'attr_suerte_pct','attr_capital_inicial_pct']
ATTR_LABELS = ['Disciplina','Oportunidad','Conocimiento','Red contactos',
               'Trabajo duro','Visión','Suerte','Capital inicial']

attr_means = df[ATTR].mean().round(2)
print("  Factor                   | % declarado | r con éxito")
print("  " + "─" * 55)
for col, label in zip(ATTR, ATTR_LABELS):
    r_e, _ = stats.pearsonr(df[col], df['score_exito'])
    print(f"  {label:<25}| {df[col].mean():>10.2f}% | {r_e:>+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: RESUMEN EJECUTIVO JSON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN 8: GENERANDO RESUMEN EJECUTIVO JSON")
print("─" * 70)

top_corr = corr_df.head(10)[['var_a','var_b','r','fuerza']].to_dict('records')

executive_summary = {
    "estudio": "El Costo del Éxito",
    "n": 5000,
    "variables": 41,
    "fecha": "2024",
    "kpis": {k: round(v,3) for k,v in kpis.items()},
    "top_10_correlaciones": top_corr,
    "sacrificio_ranking": sac_df[['sacrificio','pct_gte7']].to_dict('records'),
    "roi_por_sector": roi_sector[['roi_media']].to_dict(),
    "regresion_r2": round(r2, 4),
    "regresion_r2_adj": round(r2_adj, 4),
    "paradoja_exito_bienestar_r": round(r_bien, 4),
    "atribucion_top_factor": ATTR_LABELS[attr_means.values.argmax()],
    "atribucion_top_pct": round(attr_means.max(), 2),
    "nota_metodologica": "Dataset sintético calibrado con distribuciones de literatura académica. Referencias: Hurst & Pugsley (2011), Moskowitz & Vissing-Jørgensen (2002), Cagetti & De Nardi (2006)."
}

with open("results/00_executive_summary.json", "w", encoding="utf-8") as f:
    json.dump(executive_summary, f, indent=2, ensure_ascii=False, default=str)

print("  ✓ executive_summary.json guardado")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  ✅ PIPELINE COMPLETO EJECUTADO")
print(f"  Archivos generados en results/")
import glob
files = glob.glob("results/*.csv") + glob.glob("results/*.json")
for f in sorted(files):
    size = os.path.getsize(f)
    print(f"    {f:<45} {size:>8,} bytes")
print("═" * 70)
