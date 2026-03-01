# 📱 Assistant de Choix de Téléphone

Application web de recommandation de smartphones basée sur le **clustering non supervisé des caractéristiques techniques**, indépendamment des marques et des prix.

🔗 **Application en ligne** : [phone-choice-assistant.streamlit.app](https://phone-choice-assistant.streamlit.app)

---

## Objectif

Aider l'utilisateur à trouver le téléphone **optimal pour son budget** en se basant uniquement sur les specs techniques (RAM, caméra, batterie, écran) — sans biais de marque. Le clustering permet d'identifier des téléphones aux performances similaires mais à des prix très différents.

---

## Structure du projet

```
phone_choice_assistant/
├── segmentation_mobile.ipynb   # Pipeline complet : nettoyage → clustering → export
├── app.py                      # Application Streamlit
├── phones_clustered.csv        # Données pré-calculées (généré par le notebook)
├── requirements.txt            # Dépendances Python
└── README.md
```

---

## Pipeline du Notebook (`segmentation_mobile.ipynb`)

### Étape 1 — Chargement des données
- Lecture du fichier `Mobiles Dataset (2025).csv` (encodage `latin1`, ~930 lignes)
- Suppression des colonnes de prix régionaux (Pakistan, Inde, Chine, Dubaï)

### Étape 2 — Extraction des features numériques
Conversion des colonnes texte en valeurs numériques via des fonctions utilitaires :

| Colonne originale | Colonne créée | Exemple |
|---|---|---|
| Mobile Weight | `Mobile_Weight_g` | `"174g"` → `174` |
| RAM | `RAM_GB` | `"6GB"` → `6` |
| Front Camera | `FrontCam_MP` | `"12MP"` → `12` |
| Back Camera | `BackCam_MP` | `"48MP"` → `48` |
| Battery Capacity | `Battery_mAh` | `"3,600mAh"` → `3600` |
| Screen Size | `Screen_inch` | `"6.1 inches"` → `6.1` |
| Launched Price (USA) | `Price_USD` | `"USD 799"` → `799` |

### Étape 3 — Nettoyage des données
- Suppression des doublons exacts
- Détection et affichage des valeurs manquantes
- Correction manuelle de 4 lignes avec `Screen_inch` manquant → valeur `6.9` imputée
- Suppression de l'entrée Nokia T21 (prix incorrect dans le dataset)

### Étape 4 — Analyse exploratoire (EDA)
- Statistiques descriptives de toutes les colonnes numériques
- Matrice de corrélation pour visualiser les relations entre variables

### Étape 5 — Preprocessing pour le clustering
Le clustering est réalisé **uniquement sur les specs techniques**, sans inclure le prix :

```
Features : Mobile_Weight_g, RAM_GB, FrontCam_MP, BackCam_MP, Battery_mAh, Screen_inch
```

**Stratégie de pondération** (appliquée avant la standardisation) :

| Feature | Poids | Justification |
|---|---|---|
| Mobile_Weight_g | 0.1 | Peu pertinent pour les performances |
| RAM_GB | 1.5 | Critère principal de performance |
| FrontCam_MP | 1.2 | Important pour les selfies |
| BackCam_MP | 1.2 | Important pour la photo |
| Battery_mAh | 1.3 | Autonomie = critère majeur |
| Screen_inch | 0.5 | Secondaire |

Pipeline : `SimpleImputer (médiane)` → `pondération` → `StandardScaler`

### Étape 6 — Benchmark de 6 algorithmes de clustering
Chaque algorithme est testé sur plusieurs valeurs de `k` (entre 4 et 7 clusters) :

| Algorithme | Remarque |
|---|---|
| **KMeans** | Clusters sphériques, rapide |
| **Agglomerative (Ward)** | Hiérarchique, bonne forme de clusters |
| **DBSCAN** | Détecte les outliers — **exclu de la sélection** |
| **Birch** | Efficient sur grands datasets |
| **SpectralClustering** | Structures non-convexes |
| **GMM** | Probabiliste, souple |

> ⚠️ **DBSCAN est exclu** de la sélection finale car il génère trop de micro-clusters (>15) et des outliers non classifiables — inutilisable pour une application de recommandation.

**Métriques d'évaluation :**
- Silhouette Score ↑ (max=1, mesure la cohésion des clusters)
- Davies-Bouldin Score ↓ (min=0, mesure la séparation)
- Calinski-Harabasz Score ↑ (densité relative des clusters)

**Score composite** pour sélectionner le meilleur modèle :
$$\text{Score} = 0.6 \times \text{Silhouette normalisé} + 0.4 \times (1 - \text{Davies-Bouldin normalisé})$$

### Étape 7 — Visualisation PCA 2D
Réduction de dimension (2 composantes principales) pour visualiser les 3 meilleurs modèles éligibles dans un plan 2D.

### Étape 8 — Analyse du meilleur modèle
Pour le modèle retenu :
- Caractéristiques moyennes par cluster (RAM, caméras, batterie, écran, prix)
- Liste complète des téléphones par cluster avec marque, modèle et specs
- Distribution des prix par segment (boxplot)
- Scatter plot : prix moyen vs RAM moyenne (taille = batterie moyenne)

### Étape 9 — Nommage intelligent des segments
1. **Détection des tablettes** : tout cluster dont la taille d'écran moyenne > 7.0 pouces est nommé `"Tablettes"`
2. **Nommage des segments phones** : les clusters restants sont ordonnés par prix moyen croissant et nommés :

| N clusters phones | Noms attribués |
|---|---|
| 4 | Essentiel → Confort → Performance → Expert |
| 5 | Essentiel → Confort → Performance → Expert → Pro |
| 6 | Essentiel → Confort → Performance → Expert → Pro → Ultra Pro |

### Étape 10 — Export
Génération du fichier `phones_clustered.csv` avec les colonnes `Cluster` (entier) et `Segment` (nom) ajoutées au dataset original. Ce fichier est utilisé directement par `app.py`.

---

## Application Streamlit (`app.py`)

### Score Qualité/Prix
Chaque téléphone reçoit un score de 0 à 10 calculé comme suit :

$$\text{Score Q/P} = \frac{\sum w_i \cdot \text{spec}_i^{\text{normalisé}}}{\log(1 + \text{Prix})}$$

Avec les poids : RAM×1.5, Caméra arrière×1.2, Caméra frontale×0.8, Batterie×1.3, Écran×0.4

Un score élevé signifie **de bonnes specs pour un prix bas** — l'iPhone 15 aura un score faible malgré ses bonnes specs car son prix est très élevé par rapport à des Android similaires.

### Filtres (barre latérale)
- 💰 Budget maximum (USD)
- 🧠 RAM minimale (GB)
- 🔋 Batterie minimale (mAh)
- 📷 Caméra arrière minimale (MP)
- 🤳 Caméra frontale minimale (MP)
- 📐 Taille d'écran minimale (pouces)
- 📲 Inclure les tablettes (décoché par défaut)
- 📊 Critère de tri : Score Q/P / Prix / RAM

### Barre de recherche
Recherche textuelle par marque ou nom de modèle (ex: "Samsung", "Redmi Note 13").

### Onglet 1 — 🎯 Recommandations
Affiche le **top 10** des téléphones les mieux notés (score Q/P) parmi tous les résultats correspondant aux filtres. Inclut le classement 1→10, les specs complètes et le score.

### Onglet 2 — 📊 Explorer par segment
Affiche **tous** les téléphones d'un segment sélectionné avec leurs stats moyennes (prix, RAM, batterie, caméra).

### Onglet 3 — ℹ️ À propos des segments
Tableau récapitulatif de tous les segments : nombre de téléphones, prix min/max/moyen, specs moyennes.

---

## Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/mondher-ben-youssef/phone-choice-assistant.git
cd phone-choice-assistant

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

> Le fichier `phones_clustered.csv` est déjà inclus dans le repo. Pour le régénérer, exécuter toutes les cellules du notebook `segmentation_mobile.ipynb`.

---

## Technologies utilisées

| Outil | Usage |
|---|---|
| Python 3.11 | Langage principal |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Clustering, preprocessing, métriques |
| Matplotlib | Visualisations (EDA, PCA, boxplots) |
| Streamlit | Application web interactive |
| Jupyter Notebook | Pipeline d'analyse et d'expérimentation |
