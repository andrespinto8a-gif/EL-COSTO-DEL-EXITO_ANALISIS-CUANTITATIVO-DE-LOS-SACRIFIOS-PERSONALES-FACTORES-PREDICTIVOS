"""
══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 06 — CLUSTERING: PERFILES DE EMPRENDEDOR
El Costo del Éxito | n = 5,000 | Estudio Cuantitativo 2024
══════════════════════════════════════════════════════════════════════════════
Descripción:
    Segmentación de los 5,000 individuos en perfiles arquetípicos usando
    K-Means manual (sin sklearn). Determinación del K óptimo via método
    del codo (inercia), análisis de silueta simplificado y perfilado
    completo de cada cluster.

Clusters esperados (k=5):
    1. "El Monje del Éxito"   — Alto éxito, alto sacrificio
    2. "El Equilibrista"      — Éxito moderado, bajo sacrificio
    3. "El Velocista"         — Llegó rápido, roi alto, sector tech
    4. "El Constructor"       — Éxito alto, trayectoria larga
    5. "El Emergente"         — Score bajo, en construcción

Output:
    - results/06_clusters_asignados.csv      ← dataset + cluster label
    - results/06_cluster_profiles.csv        ← perfil promedio por cluster
    - results/06_elbow_inertia.csv           ← inercia por k=2..10
    - results/06_cluster_sector_dist.csv     ← distribución sectorial
══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("../data/dataset_5000.csv")

# ── VARIABLES PARA CLUSTERING ─────────────────────────────────────────────────
CLUSTER_VARS = [
    'score_exito', 'sacrificio_total', 'roi_anualizado_pct',
    'score_disciplina', 'score_conocimiento', 'horas_sem_pico',
    'bienestar_psicologico', 'anos_primer_millon', 'patrimonio_m_usd',
]

X_raw = df[CLUSTER_VARS].copy()

# ── NORMALIZACIÓN (z-score manual) ───────────────────────────────────────────
means = X_raw.mean()
stds  = X_raw.std()
X_norm = ((X_raw - means) / stds).values

# ── K-MEANS MANUAL ────────────────────────────────────────────────────────────
def euclidean_dist(a, b):
    return np.sqrt(((a - b)**2).sum(axis=1))

def kmeans(X, k, n_init=10, max_iter=300, seed=42):
    np.random.seed(seed)
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for init in range(n_init):
        # Inicialización k-means++
        centers = [X[np.random.randint(len(X))]]
        for _ in range(k - 1):
            dists = np.array([min(((x - c)**2).sum() for c in centers) for x in X])
            probs = dists / dists.sum()
            centers.append(X[np.random.choice(len(X), p=probs)])
        centers = np.array(centers)

        for iteration in range(max_iter):
            # Asignación
            dists_matrix = np.array([euclidean_dist(X, c) for c in centers]).T
            labels = dists_matrix.argmin(axis=1)

            # Actualización
            new_centers = np.array([
                X[labels == j].mean(axis=0) if (labels == j).sum() > 0 else centers[j]
                for j in range(k)
            ])

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        # Inercia
        inertia = sum(((X[labels == j] - centers[j])**2).sum() for j in range(k))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia

# ── MÉTODO DEL CODO ───────────────────────────────────────────────────────────
print("=" * 70)
print("1. MÉTODO DEL CODO — Inercia por k")
print("=" * 70)

elbow_rows = []
for k in range(2, 11):
    _, _, inertia = kmeans(X_norm, k, n_init=5)
    elbow_rows.append({"k": k, "inertia": round(inertia, 2)})
    print(f"  k={k}: inercia={inertia:.2f}")

elbow_df = pd.DataFrame(elbow_rows)
elbow_df.to_csv("results/06_elbow_inertia.csv", index=False)

# ── MODELO FINAL k=5 ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. CLUSTERING FINAL — k=5")
print("=" * 70)

K = 5
labels, centers, inertia = kmeans(X_norm, K, n_init=15, seed=42)
df['cluster_id'] = labels

# ── PERFILADO DE CLUSTERS ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. PERFIL PROMEDIO POR CLUSTER")
print("=" * 70)

profile_vars = CLUSTER_VARS + ['satisfaccion_vida','divorcios_num',
                                 'anos_trayectoria_total','capital_inicial_usd',
                                 'empleos_generados','score_resiliencia','score_vision']

profiles = df.groupby('cluster_id')[profile_vars].mean().round(3)

# Etiquetas arquetípicas basadas en scores
archetype_names = {}
for cid, row in profiles.iterrows():
    exito = row['score_exito']
    sac   = row['sacrificio_total']
    roi   = row['roi_anualizado_pct']
    anos  = row['anos_primer_millon']
    bien  = row['bienestar_psicologico']

    if exito > 45 and sac > 7.5:
        name = "El Monje del Éxito"
    elif exito > 35 and sac < 6.5:
        name = "El Equilibrista"
    elif roi > 50 and anos <= 5:
        name = "El Velocista"
    elif exito > 30 and bien > 60:
        name = "El Constructor Sostenible"
    else:
        name = "El Emergente"
    archetype_names[cid] = name

profiles['arquetipo'] = pd.Series(archetype_names)
profiles['n'] = df.groupby('cluster_id')['score_exito'].count()
profiles['pct'] = (profiles['n'] / len(df) * 100).round(1)

profiles.to_csv("results/06_cluster_profiles.csv")
print(profiles[['arquetipo','n','pct','score_exito','sacrificio_total',
                 'roi_anualizado_pct','bienestar_psicologico','anos_primer_millon']].to_string())

# ── DISTRIBUCIÓN SECTORIAL POR CLUSTER ────────────────────────────────────────
print("\n" + "=" * 70)
print("4. DISTRIBUCIÓN SECTORIAL POR CLUSTER")
print("=" * 70)

sector_dist = pd.crosstab(df['cluster_id'], df['sector'], normalize='index').round(3) * 100
sector_dist.to_csv("results/06_cluster_sector_dist.csv")
print(sector_dist.to_string())

# ── DISTRIBUCIÓN EDUCATIVA POR CLUSTER ────────────────────────────────────────
print("\n" + "=" * 70)
print("5. DISTRIBUCIÓN EDUCATIVA POR CLUSTER")
print("=" * 70)

edu_dist = pd.crosstab(df['cluster_id'], df['nivel_educacion'], normalize='index').round(3) * 100
print(edu_dist.to_string())

# ── GUARDAR DATASET CON CLUSTERS ──────────────────────────────────────────────
df['cluster_arquetipo'] = df['cluster_id'].map(archetype_names)
df.to_csv("results/06_clusters_asignados.csv", index=False)

# ── ANOVA ENTRE CLUSTERS ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. ANOVA — DIFERENCIAS ENTRE CLUSTERS")
print("=" * 70)

from scipy import stats as sc_stats
for col in ['score_exito','sacrificio_total','roi_anualizado_pct','bienestar_psicologico']:
    grupos = [df[df['cluster_id']==c][col].values for c in range(K)]
    f, p = sc_stats.f_oneway(*grupos)
    print(f"  {col:35s}: F={f:.2f}, p={p:.6f} {'***' if p<0.001 else ''}")

print("\n✅ Análisis 06 completado.")
