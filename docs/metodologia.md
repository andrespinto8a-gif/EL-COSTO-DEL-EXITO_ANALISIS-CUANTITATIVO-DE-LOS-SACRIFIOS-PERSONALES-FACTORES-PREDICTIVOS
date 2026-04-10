# 📐 Metodología — El Costo del Éxito

## 1. Naturaleza del dataset

Dataset **sintético calibrado**. Las distribuciones, correlaciones y parámetros estadísticos
fueron ajustados para reflejar fielmente los hallazgos de la literatura académica sobre
emprendimiento y éxito empresarial.

## 2. Diseño muestral

| Parámetro | Valor |
|---|---|
| n total | 5,000 individuos |
| Criterio inclusión | Patrimonio ≥$1M USD autogenerado (sin herencia) |
| Estratificación | Sector (8), región (12), nivel de riqueza (4) |
| Corte de datos | 31/12/2024 |
| Períodos cubiertos | Trayectorias 1984–2023 |

## 3. Definición de variables compuestas

### Score de éxito (0–100)
```
score_exito = Patrimonio(40%) + ROI(30%) + Empleos(20%) + Visión(10%)
```
- Patrimonio: normalizado logarítmicamente sobre rango $1M–$980M
- ROI: normalizado sobre rango 0–210%
- Empleos: normalizado logarítmicamente sobre rango 1–5,000
- Visión: score_vision / 10

### ROI Anualizado (CAGR)
```
ROI = (Patrimonio_final_USD / Capital_inicial_USD)^(1/años) − 1
```

### Índices de sacrificio (0–10)

Cada índice es un compuesto de sub-ítems Likert × frecuencia × duración:

| Variable | Instrumento base |
|---|---|
| `sacrificio_familia` | Family Assessment Device (FAD) adaptado |
| `sacrificio_salud` | SF-36 Health Survey + biomarcadores |
| `sacrificio_amor_pareja` | Dyadic Adjustment Scale (DAS) |
| `sacrificio_amigos` | Social Network Analysis (Burt 2004) |
| `sacrificio_ocio_hobbies` | Time-use survey retrospectivo |
| `sacrificio_sueno` | Pittsburgh Sleep Quality Index (PSQI) |

### Sacrificio total
```
sacrificio_total = mean(familia, salud, amor, amigos, ocio, sueno)
```

### Bienestar psicológico (0–100)
```
bienestar = PHQ-9 + GAD-7 + SWLS (normalizados e invertidos)
```
Escalas: Patient Health Questionnaire-9, Generalized Anxiety Disorder-7,
Satisfaction With Life Scale (Diener 1985).

### Score disciplina (0–10)
Brief Self-Control Scale (BSCS) de 13 ítems (Tangney, Baumeister & Boone, 2004).

### Score conocimiento (0–50)
```
conocimiento = certificaciones + libros_ano × 0.5 + anos_con_mentor
```

## 4. Correlaciones calibradas (estructura de covarianza)

Las correlaciones objetivo fijadas en la generación del dataset:

| Par de variables | r objetivo | Fundamento |
|---|---|---|
| disciplina ↔ score_exito | +0.71 | Baumeister & Tierney (2011) |
| conocimiento ↔ score_exito | +0.68 | Heckman & Kautz (2012) |
| sac_salud ↔ horas_pico | +0.74 | Kivimäki et al. (2015) |
| sac_familia ↔ patrimonio $100M+ | +0.81 | Kasser & Ryan (1993) |
| bienestar ↔ sac_total | −0.62 | Kahneman & Deaton (2010) |
| bienestar ↔ sac_salud | −0.71 | Steptoe et al. (2015) |
| divorcios ↔ horas_pico | +0.59 | Johnson et al. (2012) |
| disciplina ↔ anos_primer_millon | −0.57 | Duckworth et al. (2007) |

## 5. Parámetros de distribución

| Variable | Distribución | μ | σ | Min | Max |
|---|---|---|---|---|---|
| `edad_inicio` | Normal | 26 | 6 | 17 | 55 |
| `anos_primer_millon` | Normal | 7.2 | 3.8 | 1 | 35 |
| `capital_inicial_usd` | Exponencial | λ=8,000 | — | 500 | 150,000 |
| `roi_anualizado_pct` | Normal (sector-base) | variable | 14 | 4 | 210 |
| `patrimonio_m_usd` | Derivada | — | — | 1 | 980 |
| `horas_sem_pico` | Normal | 72.3 | 13.5 | 40 | 120 |
| `score_disciplina` | Normal | 7.4 | 1.5 | 1 | 10 |
| `score_conocimiento` | Normal | 22 | 10 | 0 | 50 |
| `divorcios_num` | Poisson | λ=1.4 | — | 0 | 6 |
| `hijos_num` | Poisson | λ=1.8 | — | 0 | 7 |
| `sueno_horas_noche` | Normal | 5.2 | 0.9 | 3 | 9 |

## 6. Modelos estadísticos recomendados

### Regresión lineal múltiple
```python
Y = score_exito
X = [disciplina, conocimiento, horas_pico, sacrificio_total,
     anos_trayectoria, vision, resiliencia]
# R² esperado: 0.61–0.74
```

### Regresión logística
```python
Y = exito_alto (score_exito > 60)
X = [disciplina, conocimiento, red_contactos, sector_dummy]
```

### Clustering K-Means
```python
k = 5  # perfiles óptimos
vars = [score_exito, sacrificio_total, roi, disciplina,
        conocimiento, horas_pico, bienestar, anos_millon, patrimonio]
```

### Análisis de supervivencia (Kaplan-Meier)
```python
time = anos_primer_millon
event = 1  # todos llegaron al millón (sesgo supervivencia)
covariates = [disciplina, conocimiento, red, capital_ini]
```

### Análisis de mediación
```
disciplina → score_exito  (efecto total)
disciplina → horas_pico → score_exito  (mediación parcial)
disciplina → conocimiento → score_exito  (mediación parcial)
```

## 7. Significancia estadística

Con n=5,000 y α=0.05:
- Valor crítico |r| > **0.028** para significancia estadística
- Potencia estadística (1-β) > **0.99** para |r| ≥ 0.07
- Corrección de Bonferroni recomendada para comparaciones múltiples (α/k)

## 8. Limitaciones

1. **Sesgo de supervivencia**: excluye emprendedores que fracasaron permanentemente
2. **Sesgo de memoria retrospectiva**: variables autodeclaradas con horizonte largo
3. **Sesgo de deseabilidad social**: especialmente en sacrificios y atribución
4. **Causalidad**: las correlaciones no implican relaciones causales directas
5. **Generalización**: muestra de países latinoamericanos, no necesariamente universal
6. **Sincronía**: snapshot a 2024, no captura evolución temporal longitudinal

## 9. Referencias completas

- Barsky, R.B., Juster, F.T., Kimball, M.S., & Shapiro, M.D. (1997). Preference Parameters and Behavioral Heterogeneity. *QJE*, 112(2), 537–579.
- Baumeister, R.F., & Tierney, J. (2011). *Willpower: Rediscovering the Greatest Human Strength*. Penguin.
- Cagetti, M., & De Nardi, M. (2006). Entrepreneurship, Frictions, and Wealth. *JPE*, 114(5), 835–870.
- Connor, K.M., & Davidson, J.R. (2003). Development of a new resilience scale: the CD-RISC. *Depression and Anxiety*, 18(2), 76–82.
- Diener, E., Emmons, R.A., Larsen, R.J., & Griffin, S. (1985). The Satisfaction With Life Scale. *Journal of Personality Assessment*, 49(1), 71–75.
- Duckworth, A.L., Peterson, C., Matthews, M.D., & Kelly, D.R. (2007). Grit: Perseverance and Passion for Long-Term Goals. *JPSP*, 92(6), 1087.
- Hurst, E., & Pugsley, B.W. (2011). What Do Small Businesses Do? *Brookings Papers on Economic Activity*, 2011(2), 73–118.
- Kahneman, D., & Deaton, A. (2010). High income improves evaluation of life but not emotional well-being. *PNAS*, 107(38), 16489–16493.
- Moskowitz, T.J., & Vissing-Jørgensen, A. (2002). The Returns to Entrepreneurial Investment. *AER*, 92(4), 745–778.
- Tangney, J.P., Baumeister, R.F., & Boone, A.L. (2004). High self-control predicts good adjustment. *Journal of Personality*, 72(2), 271–324.
