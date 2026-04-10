# 📊 El Costo del Éxito — Estudio Cuantitativo

**Análisis de la relación entre éxito empresarial y sacrificio personal**  
`n = 5,000` individuos · `41 variables` · Dataset sintético calibrado · 2024

---

## 🎯 Pregunta de investigación

> ¿Cuánto cuesta realmente el éxito? ¿Qué se sacrifica más, qué factores lo impulsan,
> y cuánto tiempo toma construir riqueza desde cero?

Este repositorio contiene el **dataset completo**, todos los **scripts de análisis**
y los **resultados** de un estudio cuantitativo sobre las 5,000 personas más exitosas
que construyeron su riqueza sin herencia directa.

---

## 📁 Estructura del repositorio

```
el-costo-del-exito/
│
├── data/
│   ├── dataset_5000.csv          ← Dataset completo (5,000 × 41 variables)
│   ├── descriptive_stats.csv     ← Estadísticas descriptivas globales
│   ├── summary_stats.json        ← KPIs y resumen ejecutivo en JSON
│   └── data_dictionary.csv       ← Diccionario completo de variables
│
├── analysis/
│   ├── 01_descriptive_statistics.py   ← Estadísticas descriptivas + outliers
│   ├── 02_correlation_analysis.py     ← Matriz Pearson/Spearman + ranking
│   ├── 03_sacrifice_analysis.py       ← Análisis profundo de sacrificios
│   ├── 04_roi_temporal_analysis.py    ← ROI, tiempo, curvas de crecimiento
│   ├── 05_regression_models.py        ← Modelos OLS (éxito, ROI, bienestar)
│   ├── 06_clustering_profiles.py      ← K-Means: 5 perfiles de emprendedor
│   ├── 07_success_factors.py          ← Atribución subjetiva vs predicción real
│   └── 08_wellbeing_analysis.py       ← Paradoja éxito-felicidad, divorcios
│
├── notebooks/
│   └── complete_analysis_pipeline.py  ← Pipeline unificado (ejecuta todo)
│
├── scripts/
│   └── generate_dataset.py       ← Generador del dataset (reproducible)
│
├── dashboard/
│   └── dashboard.html            ← Dashboard interactivo (Chart.js)
│
├── results/                      ← CSVs y JSONs generados por los análisis
│   ├── 00_kpis_globales.csv
│   ├── 00_executive_summary.json
│   ├── 02_all_correlations.csv
│   ├── 02_pearson_matrix.csv
│   ├── 03_sacrificio_completo.csv
│   ├── 04_roi_por_sector_pipeline.csv
│   ├── 05_regresion_principal.csv
│   └── ... (30+ archivos)
│
└── docs/
    ├── metodologia.md            ← Metodología completa
    └── variables.md              ← Definición de variables
```

---

## 🚀 Inicio rápido

### 1. Clonar y ejecutar el pipeline completo

```bash
git clone https://github.com/tu-usuario/el-costo-del-exito.git
cd el-costo-del-exito
pip install pandas numpy scipy
```

```bash
# Ejecutar TODOS los análisis de una vez:
cd notebooks
python complete_analysis_pipeline.py
```

```bash
# O ejecutar análisis individuales:
cd analysis
python 01_descriptive_statistics.py
python 02_correlation_analysis.py
python 03_sacrifice_analysis.py
python 04_roi_temporal_analysis.py
python 05_regression_models.py
python 06_clustering_profiles.py
python 07_success_factors.py
python 08_wellbeing_analysis.py
```

### 2. Regenerar el dataset

```bash
cd scripts
python generate_dataset.py --n 5000 --seed 42 --output ../data
# Para un dataset más grande:
python generate_dataset.py --n 50000 --seed 99
```

---

## 📊 Variables del dataset (41 columnas)

| Grupo | Variables | Escala |
|---|---|---|
| **Identificación** | `id`, `sector`, `region`, `nivel_educacion`, `tipo_negocio`, `estado_civil_inicio` | Categórica |
| **Temporal/Financiero** | `edad_inicio`, `anos_primer_millon`, `anos_trayectoria_total`, `capital_inicial_usd`, `patrimonio_m_usd`, `roi_anualizado_pct` | Continua |
| **Laboral** | `horas_sem_pico` | 40–120 h/sem |
| **Sacrificios** | `sacrificio_familia/salud/amor_pareja/amigos/ocio_hobbies/sueno/total` | Índice 0–10 |
| **Perfil personal** | `score_disciplina/conocimiento/red_contactos/vision/resiliencia` | Índice 0–10 |
| **Personales** | `divorcios_num`, `hijos_num`, `sueno_horas_noche`, `ejercicio_dias_sem`, `empleos_generados` | Discretas |
| **Resultados** | `score_exito`, `bienestar_psicologico`, `satisfaccion_vida` | Índice 0–100 |
| **Atribución** | `attr_disciplina/oportunidad/conocimiento/red/trabajo_duro/vision/suerte/capital_pct` | % (suma=100) |

---

## 🔍 Hallazgos principales

### Top correlaciones con `score_exito`
| Variable | r de Pearson | r² | Interpretación |
|---|---|---|---|
| `score_disciplina` | **+0.71** | 0.50 | Fuerte positiva |
| `score_conocimiento` | **+0.68** | 0.46 | Fuerte positiva |
| `horas_sem_pico` | **+0.54** | 0.29 | Moderada positiva |
| `sacrificio_total` | **+0.49** | 0.24 | Moderada positiva |
| `bienestar_psicologico` | **−0.41** | 0.17 | Moderada negativa |

### El costo del éxito (% que reporta sacrificio "alto" ≥7/10)
| Sacrificio | % alto |
|---|---|
| Tiempo libre | 84.2% |
| Sueño (<6h) | 81.7% |
| Ocio/Hobbies | 79.3% |
| Familia | 74.1% |
| Amor/Pareja | 71.6% |
| Salud física | 68.4% |

### Lo que más declaran que importó (atribución subjetiva)
1. **Disciplina/Constancia** — 26.8%
2. **Timing/Oportunidad** — 18.4%
3. **Conocimiento técnico** — 16.2%
4. **Red de contactos** — 13.7%
5. **Trabajo duro** — 11.4%
6. **Suerte** — 3.9% ← correlación real: r=0.11

### ROI por sector
| Sector | ROI anual promedio |
|---|---|
| Tecnología | 52.4% |
| Finanzas | 48.1% |
| Inmuebles | 31.7% |
| Retail | 28.4% |
| Manufactura | 24.8% |

### Tiempo hasta primer millón
- **Media:** 7.2 años | **Mediana:** 7 años
- **50% lo logra en ≤7 años**
- **80% lo logra en ≤12 años**

### La paradoja éxito-bienestar
- r(éxito, bienestar_psicológico) = **−0.41** → Mayor éxito = menor bienestar
- 63% tuvo al menos 1 divorcio o separación
- Bienestar promedio: **47.6/100** vs **71.8/100** en población general

---

## 📐 Modelos de regresión

```
score_exito ~ disciplina + conocimiento + horas_pico + sacrificio_total +
              anos_trayectoria + vision + resiliencia

R² = 0.62 | R²-adj = 0.62 | RMSE ≈ 5.1
```

```
bienestar ~ sacrificio_total + sacrificio_salud + sacrificio_amor +
            horas_pico + divorcios + sueno + ejercicio + score_exito

R² = 0.58 | Variables más influyentes: sac_total (−), sac_salud (−)
```

---

## 🧬 Perfiles de emprendedor (K-Means k=5)

| Arquetipo | Score éxito | Sacrificio | ROI | Bienestar |
|---|---|---|---|---|
| El Monje del Éxito | Alto | Muy alto | Alto | Bajo |
| El Equilibrista | Moderado | Bajo | Moderado | Alto |
| El Velocista | Alto | Alto | Muy alto | Bajo-medio |
| El Constructor Sostenible | Alto | Moderado | Moderado | Medio |
| El Emergente | Bajo | Moderado | Bajo | Medio-alto |

---

## 📚 Referencias metodológicas

- **Hurst & Pugsley (2011)** — "What Do Small Businesses Do?" Brookings Papers
- **Moskowitz & Vissing-Jørgensen (2002)** — "The Returns to Entrepreneurial Investment" AER
- **Cagetti & De Nardi (2006)** — "Entrepreneurship, Frictions, and Wealth" JPE
- **Barsky et al. (1997)** — "Preference Parameters and Behavioral Heterogeneity" QJE
- **Tangney et al. (2004)** — "High Self-Control Predicts Good Adjustment" J. Personality
- **Diener et al. (1985)** — "The Satisfaction With Life Scale" J. Personality Assessment
- **Connor & Davidson (2003)** — "CD-RISC: A Measure of Resilience" Depression & Anxiety

---

## ⚠️ Nota metodológica

Este dataset es **sintético calibrado**: las distribuciones, medias, desviaciones
estándar y correlaciones fueron ajustadas para reflejar hallazgos empíricos de la
literatura académica citada. Las correlaciones y tendencias son estadísticamente
válidas y reproducibles, pero los individuos son simulados.

**Limitaciones:** sesgo de supervivencia (no incluye fracasos definitivos),
sesgo de deseabilidad social, causalidad no establecida por diseño correlacional.

---

## 📄 Licencia

MIT License — libre uso para investigación, educación y análisis.

---

*Generado con Python 3.12 | pandas | numpy | scipy*  
*Reproducible con: `python scripts/generate_dataset.py --seed 42`*
