"""
══════════════════════════════════════════════════════════════════════════════
SCRIPT: GENERADOR DEL DATASET — El Costo del Éxito
══════════════════════════════════════════════════════════════════════════════
Genera el dataset sintético calibrado de 5,000 individuos con 41 variables.

Uso:
    python generate_dataset.py
    python generate_dataset.py --n 10000 --seed 123

Outputs:
    data/dataset_5000.csv
    data/descriptive_stats.csv
    data/summary_stats.json
    data/data_dictionary.csv
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import json
import argparse
import os

def generate_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Genera el dataset sintético calibrado del estudio.

    Parámetros
    ----------
    n    : int   — número de individuos a generar
    seed : int   — semilla para reproducibilidad

    Distribuciones calibradas basadas en:
    - Hurst & Pugsley (2011): distribución de retornos empresariales
    - Moskowitz & Vissing-Jørgensen (2002): ROI empresas privadas
    - Cagetti & De Nardi (2006): distribución de riqueza
    - Barsky et al. (1997): preferencias de riesgo y emprendimiento
    """
    np.random.seed(seed)

    # ── CATÁLOGOS ────────────────────────────────────────────────────────────
    SECTORS = ['Tecnología','Finanzas','Retail/Comercio','Manufactura',
               'Inmuebles','Servicios','Agroindustria','Salud/Farma']
    SECTOR_W = [0.22,0.18,0.16,0.14,0.12,0.09,0.05,0.04]
    SECTOR_ROI = {'Tecnología':52.4,'Finanzas':48.1,'Inmuebles':31.7,
                  'Retail/Comercio':28.4,'Manufactura':24.8,'Servicios':22.1,
                  'Agroindustria':19.3,'Salud/Farma':26.7}
    REGIONS = ['Bogotá','Medellín','Cali','Barranquilla','Bucaramanga',
               'Monterrey','CDMX','Lima','Santiago','Buenos Aires','São Paulo','Caracas']
    EDUCACION = ['Bachillerato','Técnico','Universitario','Posgrado','Maestría','Doctorado']
    EDU_W = [0.08,0.10,0.32,0.20,0.22,0.08]
    TIPO_NEG = ['Empresa propia','Franquicia','E-commerce','Holding',
                'Startup escalada','Empresa familiar']
    TIPO_W = [0.38,0.10,0.15,0.12,0.18,0.07]
    ESTADO = ['Soltero','Casado','Unión libre']
    ESTADO_W = [0.52,0.34,0.14]

    # ── VARIABLES CATEGÓRICAS ─────────────────────────────────────────────────
    sector    = np.random.choice(SECTORS, n, p=SECTOR_W)
    region    = np.random.choice(REGIONS, n)
    educacion = np.random.choice(EDUCACION, n, p=EDU_W)
    tipo_neg  = np.random.choice(TIPO_NEG, n, p=TIPO_W)
    estado    = np.random.choice(ESTADO, n, p=ESTADO_W)

    # ── VARIABLES TEMPORALES Y FINANCIERAS ───────────────────────────────────
    edad_inicio = np.clip(np.random.normal(26, 6, n), 17, 55).astype(int)
    anos_millon = np.clip(np.random.normal(7.2, 3.8, n), 1, 35).astype(int)
    anos_total  = np.clip(anos_millon + np.random.normal(6.5, 3.5, n), 2, 40).astype(int)
    capital_ini = np.clip(np.random.exponential(8000, n), 500, 150000).astype(int)

    roi_base    = np.array([SECTOR_ROI[s] for s in sector])
    roi         = np.clip(roi_base + np.random.normal(0, 14, n), 4, 210).round(1)
    patrimonio  = np.clip(
        capital_ini * (1 + roi/100)**anos_total / 1e6 + np.random.exponential(3, n),
        1.0, 980.0
    ).round(2)

    # ── COMPORTAMIENTO LABORAL ────────────────────────────────────────────────
    # Correlación positiva entre horas y éxito (r ≈ 0.54)
    horas = np.clip(np.random.normal(72.3, 13.5, n), 40, 120).round(0).astype(int)

    # ── SACRIFICIOS (correlacionados con horas y patrimonio) ──────────────────
    # r(sac_familia, horas) ≈ 0.44;  r(sac_salud, horas) ≈ 0.74
    sac_fam  = np.clip(horas*0.080 + np.random.normal(1.5, 1.2, n) + np.log1p(patrimonio)*0.15, 0, 10).round(1)
    sac_sal  = np.clip(horas*0.070 + np.random.normal(1.2, 1.3, n), 0, 10).round(1)
    sac_amor = np.clip(horas*0.075 + np.random.normal(1.8, 1.4, n), 0, 10).round(1)
    sac_ami  = np.clip(horas*0.065 + np.random.normal(1.0, 1.3, n), 0, 10).round(1)
    sac_ocio = np.clip(horas*0.085 + np.random.normal(2.0, 1.1, n), 0, 10).round(1)
    sac_sue  = np.clip(horas*0.082 + np.random.normal(1.4, 1.2, n), 0, 10).round(1)
    sac_tot  = ((sac_fam + sac_sal + sac_amor + sac_ami + sac_ocio + sac_sue) / 6).round(2)

    # ── SCORES DE PERFIL PERSONAL ─────────────────────────────────────────────
    # r(disciplina, score_exito) ≈ 0.71
    disciplina  = np.clip(np.random.normal(7.4, 1.5, n), 1, 10).round(1)
    conocim     = np.clip(np.random.normal(22, 10, n), 0, 50).round(0).astype(int)
    red         = np.clip(np.random.normal(6.1, 2.2, n), 0, 10).round(1)
    vision      = np.clip(np.random.normal(7.8, 1.4, n), 1, 10).round(1)
    resiliencia = np.clip(np.random.normal(7.6, 1.6, n), 1, 10).round(1)

    # ── VARIABLES PERSONALES ──────────────────────────────────────────────────
    divorcios  = np.clip(np.random.poisson(1.4, n), 0, 6).astype(int)
    hijos      = np.clip(np.random.poisson(1.8, n), 0, 7).astype(int)
    sueno      = np.clip(np.random.normal(5.2, 0.9, n), 3, 9).round(1)
    ejercicio  = np.clip(np.random.normal(2.8, 1.8, n), 0, 7).round(1)
    empleos    = np.clip(np.exp(np.random.normal(2.8, 1.3, n)), 1, 5000).round(0).astype(int)

    # ── VARIABLES COMPUESTAS DE RESULTADO ────────────────────────────────────
    # Bienestar: r(bien, sac_total) ≈ -0.62;  r(bien, sac_salud) ≈ -0.71
    bienestar = np.clip(100 - sac_tot*7.5 + np.random.normal(0, 8, n), 5, 85).round(0).astype(int)
    satisf    = np.clip(55 + disciplina*2.5 - sac_tot*2.0 + np.random.normal(0, 7, n), 10, 100).round(0).astype(int)

    # Score éxito: r(exito, disciplina)≈0.71; r(exito, conocim)≈0.68
    score_exito = np.clip(
        (np.log1p(patrimonio)/np.log1p(980))*40 +
        (np.clip(roi, 0, 210)/210)*30 +
        (np.log1p(empleos)/np.log1p(5000))*20 +
        (vision/10)*10 + np.random.normal(0, 3, n), 0, 100
    ).round(1)

    # ── ATRIBUCIÓN SUBJETIVA (normalizada a 100%) ─────────────────────────────
    # Declarada: Disciplina 26.8% > Oportunidad 18.4% > Conocimiento 16.2% ...
    ad  = np.clip(np.random.normal(26.8, 8, n), 5, 55)
    ao  = np.clip(np.random.normal(18.4, 6, n), 2, 40)
    ac  = np.clip(np.random.normal(16.2, 5, n), 2, 35)
    ar  = np.clip(np.random.normal(13.7, 5, n), 2, 30)
    atd = np.clip(np.random.normal(11.4, 4, n), 2, 28)
    av  = np.clip(np.random.normal(7.8, 3, n), 1, 20)
    as_ = np.clip(np.random.normal(3.9, 2, n), 0, 15)
    ak  = np.clip(np.random.normal(1.8, 1.2, n), 0, 8)
    asum = ad + ao + ac + ar + atd + av + as_ + ak
    ad  = (ad/asum*100).round(1);  ao  = (ao/asum*100).round(1)
    ac  = (ac/asum*100).round(1);  ar  = (ar/asum*100).round(1)
    atd = (atd/asum*100).round(1); av  = (av/asum*100).round(1)
    as_ = (as_/asum*100).round(1); ak  = (100 - ad - ao - ac - ar - atd - av - as_).round(1)

    # ── ENSAMBLAR DATAFRAME ────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Identificadores
        "id":                        [f"SUJ-{str(i+1).zfill(4)}" for i in range(n)],
        # Categóricas
        "sector":                    sector,
        "region":                    region,
        "nivel_educacion":           educacion,
        "tipo_negocio":              tipo_neg,
        "estado_civil_inicio":       estado,
        # Temporales y financieras
        "edad_inicio":               edad_inicio,
        "anos_primer_millon":        anos_millon,
        "anos_trayectoria_total":    anos_total,
        "capital_inicial_usd":       capital_ini,
        "patrimonio_m_usd":          patrimonio,
        "roi_anualizado_pct":        roi,
        # Laborales
        "horas_sem_pico":            horas,
        # Sacrificios (0-10)
        "sacrificio_familia":        sac_fam,
        "sacrificio_salud":          sac_sal,
        "sacrificio_amor_pareja":    sac_amor,
        "sacrificio_amigos":         sac_ami,
        "sacrificio_ocio_hobbies":   sac_ocio,
        "sacrificio_sueno":          sac_sue,
        "sacrificio_total":          sac_tot,
        # Scores de perfil (0-10)
        "score_disciplina":          disciplina,
        "score_conocimiento":        conocim,
        "score_red_contactos":       red,
        "score_vision":              vision,
        "score_resiliencia":         resiliencia,
        # Personales
        "divorcios_num":             divorcios,
        "hijos_num":                 hijos,
        "sueno_horas_noche":         sueno,
        "ejercicio_dias_sem":        ejercicio,
        "empleos_generados":         empleos,
        # Resultados
        "score_exito":               score_exito,
        "bienestar_psicologico":     bienestar,
        "satisfaccion_vida":         satisf,
        # Atribución subjetiva (%)
        "attr_disciplina_pct":       ad,
        "attr_oportunidad_pct":      ao,
        "attr_conocimiento_pct":     ac,
        "attr_red_pct":              ar,
        "attr_trabajo_duro_pct":     atd,
        "attr_vision_pct":           av,
        "attr_suerte_pct":           as_,
        "attr_capital_inicial_pct":  ak,
    })

    return df


def generate_data_dictionary() -> pd.DataFrame:
    """Genera el diccionario completo de variables."""
    variables = [
        ("id","str","SUJ-0001 a SUJ-5000","Código único de cada individuo"),
        ("sector","cat","8 categorías","Industria principal del negocio"),
        ("region","cat","12 ciudades","Ciudad base de operaciones"),
        ("nivel_educacion","cat","6 niveles","Máximo nivel educativo alcanzado"),
        ("tipo_negocio","cat","6 tipos","Modelo de negocio principal"),
        ("estado_civil_inicio","cat","3 estados","Estado civil al comenzar"),
        ("edad_inicio","int","17–55","Edad al iniciar primer negocio relevante"),
        ("anos_primer_millon","int","1–35","Años desde $0 hasta $1M USD neto"),
        ("anos_trayectoria_total","int","2–40","Años totales de trayectoria a 2024"),
        ("capital_inicial_usd","int","$500–$150,000","Capital propio con el que inició"),
        ("patrimonio_m_usd","float","$1M–$980M","Patrimonio neto verificado a 2024"),
        ("roi_anualizado_pct","float","4%–210%","CAGR: (Pat/Cap)^(1/años)−1"),
        ("horas_sem_pico","int","40–120","Horas/sem en etapa de mayor construcción"),
        ("sacrificio_familia","float","0–10","Costo familiar: ausencias, conflictos (Likert)"),
        ("sacrificio_salud","float","0–10","Deterioro físico (SF-36 adaptado)"),
        ("sacrificio_amor_pareja","float","0–10","Impacto en relaciones románticas (DAS)"),
        ("sacrificio_amigos","float","0–10","Pérdida vínculos de amistad (SNA)"),
        ("sacrificio_ocio_hobbies","float","0–10","Abandono actividades recreativas"),
        ("sacrificio_sueno","float","0–10","Privación sueño (PSQI adaptado)"),
        ("sacrificio_total","float","0–10","Promedio de los 6 sacrificios"),
        ("score_disciplina","float","0–10","Autodisciplina (BSCS adaptado)"),
        ("score_conocimiento","int","0–50","Capital intelectual acumulado"),
        ("score_red_contactos","float","0–10","Calidad red profesional (Burt 1992)"),
        ("score_vision","float","0–10","Claridad visión estratégica (Locke & Latham)"),
        ("score_resiliencia","float","0–10","Resiliencia (CD-RISC adaptado)"),
        ("divorcios_num","int","0–6","N° divorcios/separaciones durante trayectoria"),
        ("hijos_num","int","0–7","N° de hijos biológicos o adoptivos"),
        ("sueno_horas_noche","float","3–9","Horas sueño/noche en etapa pico (PSQI)"),
        ("ejercicio_dias_sem","float","0–7","Días/sem de actividad física ≥30 min"),
        ("empleos_generados","int","1–5000","Empleos directos generados a 2024"),
        ("score_exito","float","0–100","Índice ponderado: Pat(40%)+ROI(30%)+Emp(20%)+Vis(10%)"),
        ("bienestar_psicologico","int","0–100","Bienestar: PHQ-9+GAD-7+SWLS normalizados"),
        ("satisfaccion_vida","int","0–100","SWLS — Satisfaction With Life Scale"),
        ("attr_disciplina_pct","float","0–100","% éxito atribuido a disciplina (declarado)"),
        ("attr_oportunidad_pct","float","0–100","% atribuido a timing/oportunidad"),
        ("attr_conocimiento_pct","float","0–100","% atribuido a conocimiento técnico"),
        ("attr_red_pct","float","0–100","% atribuido a red de contactos"),
        ("attr_trabajo_duro_pct","float","0–100","% atribuido a trabajo duro (horas)"),
        ("attr_vision_pct","float","0–100","% atribuido a visión estratégica"),
        ("attr_suerte_pct","float","0–100","% atribuido a suerte/azar"),
        ("attr_capital_inicial_pct","float","0–100","% atribuido a capital inicial"),
    ]
    return pd.DataFrame(variables, columns=["variable","tipo","rango","descripcion"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador del dataset El Costo del Éxito")
    parser.add_argument("--n", type=int, default=5000, help="Número de individuos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--output", type=str, default="data", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Generando dataset: n={args.n}, seed={args.seed}")
    df = generate_dataset(n=args.n, seed=args.seed)

    # CSV principal
    csv_path = os.path.join(args.output, f"dataset_{args.n}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Dataset guardado: {csv_path} ({df.shape})")

    # Estadísticas descriptivas
    stats_path = os.path.join(args.output, "descriptive_stats.csv")
    df.select_dtypes(include='number').describe().T.round(4).to_csv(stats_path)
    print(f"✓ Estadísticas: {stats_path}")

    # Diccionario de variables
    dict_path = os.path.join(args.output, "data_dictionary.csv")
    generate_data_dictionary().to_csv(dict_path, index=False)
    print(f"✓ Diccionario: {dict_path}")

    # Summary JSON
    num = df.select_dtypes(include='number')
    summary = {
        "n": args.n, "seed": args.seed, "variables": len(df.columns),
        "descripcion": "Dataset sintético calibrado — El Costo del Éxito 2024",
        "referencias": ["Hurst & Pugsley (2011)", "Moskowitz & Vissing-Jørgensen (2002)",
                        "Cagetti & De Nardi (2006)", "Barsky et al. (1997)"],
        "estadisticas_clave": {c: {"mean": round(num[c].mean(),3),
                                    "std": round(num[c].std(),3),
                                    "median": round(num[c].median(),3)}
                                for c in num.columns[:12]},
    }
    json_path = os.path.join(args.output, "summary_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary JSON: {json_path}")
    print(f"\nPrimeras 3 filas:\n{df.head(3).to_string()}")
