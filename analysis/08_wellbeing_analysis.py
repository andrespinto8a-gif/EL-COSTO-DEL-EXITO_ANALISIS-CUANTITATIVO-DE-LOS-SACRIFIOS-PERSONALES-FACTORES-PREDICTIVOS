"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 08 — BIENESTAR, SALUD Y COSTO PERSONAL
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Análisis del impacto del éxito empresarial en el bienestar físico
    y psicológico. Incluye análisis de divorcios, sueño, ejercicio,
    satisfacción de vida y la paradoja éxito-felicidad.

Output:
    - results/08_bienestar_por_exito.csv
    - results/08_divorcios_analisis.csv
    - results/08_salud_comparacion.csv
    - results/08_paradoja_exito_felicidad.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)
df = pd.read_csv("../data/dataset_5000.csv")

# ── 1. BIENESTAR POR NIVEL DE ÉXITO ──────────────────────────────────────────
print("=" * 70)
print("1. BIENESTAR POR NIVEL DE ÉXITO")
print("=" * 70)

df['quintil_exito'] = pd.qcut(df['score_exito'], q=5,
    labels=['Q1 Más bajo','Q2 Bajo','Q3 Medio','Q4 Alto','Q5 Más alto'])

bien_exito = df.groupby('quintil_exito', observed=True).agg(
    n=('bienestar_psicologico','count'),
    exito_medio=('score_exito','mean'),
    bienestar_medio=('bienestar_psicologico','mean'),
    satisfaccion_vida=('satisfaccion_vida','mean'),
    sueno_horas=('sueno_horas_noche','mean'),
    ejercicio_dias=('ejercicio_dias_sem','mean'),
    divorcios=('divorcios_num','mean'),
    horas_pico=('horas_sem_pico','mean'),
    sacrificio=('sacrificio_total','mean'),
    patrimonio=('patrimonio_m_usd','mean'),
).round(2)

bien_exito.to_csv("results/08_bienestar_por_exito.csv")
print(bien_exito.to_string())

r_bien_exito, p = stats.pearsonr(df['score_exito'], df['bienestar_psicologico'])
print(f"\n  r(éxito, bienestar) = {r_bien_exito:.4f} (p={p:.6f})")
print(f"  → La paradoja éxito-bienestar es {'REAL' if r_bien_exito < 0 else 'no confirmada'}")

# ── 2. ANÁLISIS DE DIVORCIOS ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. ANÁLISIS DE DIVORCIOS Y RUPTURAS")
print("=" * 70)

div = df['divorcios_num']
print(f"  Media divorcios:    {div.mean():.2f}")
print(f"  % con ≥1 divorcio:  {(div >= 1).mean()*100:.1f}%")
print(f"  % con ≥2 divorcios: {(div >= 2).mean()*100:.1f}%")
print(f"  % con 0 divorcios:  {(div == 0).mean()*100:.1f}%")
print(f"  r(divorcios, horas): {stats.pearsonr(div, df['horas_sem_pico'])[0]:.4f}")
print(f"  r(divorcios, patrimonio): {stats.pearsonr(div, df['patrimonio_m_usd'])[0]:.4f}")
print(f"  r(divorcios, sacrificio_amor): {stats.pearsonr(div, df['sacrificio_amor_pareja'])[0]:.4f}")

div_dist = div.value_counts().sort_index().reset_index()
div_dist.columns = ['divorcios', 'n']
div_dist['pct'] = (div_dist['n'] / len(df) * 100).round(2)
div_dist.to_csv("results/08_divorcios_analisis.csv", index=False)
print(f"\n  Distribución de divorcios:")
print(div_dist.to_string(index=False))

# Por sector
div_sector = df.groupby('sector')['divorcios_num'].agg(['mean','median']).round(2).sort_values('mean', ascending=False)
print(f"\n  Divorcios por sector:")
print(div_sector.to_string())

# Por patrimonio
df['rango_riqueza'] = pd.cut(df['patrimonio_m_usd'], bins=[0,5,20,100,980],
    labels=['$1M–$5M','$5M–$20M','$20M–$100M','$100M+'])
div_wealth = df.groupby('rango_riqueza', observed=True)['divorcios_num'].agg(['mean','median']).round(2)
print(f"\n  Divorcios por nivel de riqueza:")
print(div_wealth.to_string())

# ── 3. ANÁLISIS DE SUEÑO ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. ANÁLISIS DE SUEÑO")
print("=" * 70)

sl = df['sueno_horas_noche']
print(f"  Media:    {sl.mean():.2f} hrs/noche")
print(f"  Mediana:  {sl.median():.2f} hrs/noche")
print(f"  % <6h:    {(sl < 6).mean()*100:.1f}%")
print(f"  % <5h:    {(sl < 5).mean()*100:.1f}%")
print(f"  % ≥7h:    {(sl >= 7).mean()*100:.1f}%")
print(f"  r(sueño, score_exito): {stats.pearsonr(sl, df['score_exito'])[0]:.4f}")
print(f"  r(sueño, bienestar):   {stats.pearsonr(sl, df['bienestar_psicologico'])[0]:.4f}")
print(f"  r(sueño, horas_pico):  {stats.pearsonr(sl, df['horas_sem_pico'])[0]:.4f}")

# ── 4. PARADOJA ÉXITO-FELICIDAD ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. LA PARADOJA ÉXITO-FELICIDAD — Análisis completo")
print("=" * 70)

paradox_vars = ['score_exito','patrimonio_m_usd','roi_anualizado_pct']
wellbeing_vars = ['bienestar_psicologico','satisfaccion_vida','sueno_horas_noche',
                  'ejercicio_dias_sem']

paradox_rows = []
for pv in paradox_vars:
    for wv in wellbeing_vars:
        r, p = stats.pearsonr(df[pv], df[wv])
        paradox_rows.append({
            "variable_exito":    pv,
            "variable_bienestar":wv,
            "pearson_r":         round(r, 4),
            "r2":                round(r**2, 4),
            "p_value":           round(p, 6),
            "direccion":         "Positiva" if r > 0 else "Negativa",
            "interpretacion":    "Mayor éxito = mejor bienestar" if r > 0 else "Mayor éxito = PEOR bienestar",
        })

paradox_df = pd.DataFrame(paradox_rows)
paradox_df.to_csv("results/08_paradoja_exito_felicidad.csv", index=False)
print(paradox_df.to_string(index=False))

# ── 5. COMPARACIÓN POBLACIÓN GENERAL vs EXITOSOS ──────────────────────────────
print("\n" + "=" * 70)
print("5. COMPARACIÓN: EXITOSOS vs POBLACIÓN GENERAL (benchmarks literatura)")
print("=" * 70)

benchmarks = {
    "Bienestar psicológico (0-100)":   (df['bienestar_psicologico'].mean(), 71.8),
    "Sueño promedio (hrs/noche)":      (df['sueno_horas_noche'].mean(), 7.1),
    "Ejercicio (días/semana)":         (df['ejercicio_dias_sem'].mean(), 3.5),
    "Satisfacción de vida (0-100)":    (df['satisfaccion_vida'].mean(), 68.0),
    "Divorcios promedio":              (df['divorcios_num'].mean(), 0.8),
}

print(f"  {'Dimensión':<40} {'Exitosos':>10} {'Pob. General':>14} {'Diferencia':>12} {'Gap %':>8}")
print("  " + "─" * 88)
comp_rows = []
for dim, (val_ex, val_gen) in benchmarks.items():
    diff = val_ex - val_gen
    gap_pct = (diff / val_gen * 100)
    direction = "▲ MEJOR" if diff > 0 else "▼ PEOR"
    print(f"  {dim:<40} {val_ex:>10.2f} {val_gen:>14.2f} {diff:>+12.2f} {gap_pct:>7.1f}% {direction}")
    comp_rows.append({"dimension": dim, "exitosos": round(val_ex,2),
                      "poblacion_general": val_gen, "diferencia": round(diff,2),
                      "gap_pct": round(gap_pct,1)})

pd.DataFrame(comp_rows).to_csv("results/08_salud_comparacion.csv", index=False)

# ── 6. HORAS TRABAJADAS: PUNTO DE INFLEXIÓN ───────────────────────────────────
print("\n" + "=" * 70)
print("6. HORAS TRABAJADAS — PUNTO DE INFLEXIÓN BIENESTAR vs ÉXITO")
print("=" * 70)

df['rango_horas'] = pd.cut(df['horas_sem_pico'],
    bins=[39,50,60,70,80,90,100,120],
    labels=['40-50','51-60','61-70','71-80','81-90','91-100','101+'])

horas_analysis = df.groupby('rango_horas', observed=True).agg(
    n=('horas_sem_pico','count'),
    horas_media=('horas_sem_pico','mean'),
    exito=('score_exito','mean'),
    roi=('roi_anualizado_pct','mean'),
    bienestar=('bienestar_psicologico','mean'),
    sacrificio=('sacrificio_total','mean'),
    divorcios=('divorcios_num','mean'),
    sueno=('sueno_horas_noche','mean'),
).round(2)

print(horas_analysis.to_string())
horas_analysis.to_csv("results/08_horas_inflexion.csv")

print("\n✅ Análisis 08 completado.")
