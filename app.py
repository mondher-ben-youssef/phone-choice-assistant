import streamlit as st
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Assistant Choix Téléphone",
    page_icon="📱",
    layout="wide",
)

# ─────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────
DATA_FILE = "phones_clustered.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(
            f"❌ Fichier `{DATA_FILE}` introuvable. "
            "Veuillez d'abord exécuter toutes les cellules du notebook (jusqu'à la cellule Export)."
        )
        st.stop()
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()

# Colonnes attendues
SPEC_COLS = ["RAM_GB", "BackCam_MP", "FrontCam_MP", "Battery_mAh", "Screen_inch"]
DISPLAY_COLS = ["Company Name", "Model Name", "Segment", "RAM_GB", "BackCam_MP",
                "FrontCam_MP", "Battery_mAh", "Screen_inch", "Price_USD"]

# Segments triés par niveau (Cluster ordonné par prix moyen croissant)
segments_ordres = sorted(
    df["Segment"].dropna().unique(),
    key=lambda s: df[df["Segment"] == s]["Price_USD"].mean()
)
n_segments = df["Cluster"].nunique()

# ─────────────────────────────────────────────
# Score qualité/prix
# ─────────────────────────────────────────────
def compute_score(df_in: pd.DataFrame) -> pd.Series:
    """
    Score qualité/prix normalisé 0–10.
    Specs pondérées normalisées / log(prix).
    Plus le score est élevé, meilleur est le rapport qualité/prix.
    """
    df_s = df_in.copy()
    weights = {"RAM_GB": 1.5, "BackCam_MP": 1.2, "FrontCam_MP": 0.8,
               "Battery_mAh": 1.3, "Screen_inch": 0.4}
    score = pd.Series(0.0, index=df_s.index)
    for col, w in weights.items():
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            score += w * (df_s[col].fillna(col_min) - col_min) / (col_max - col_min)
    prix = df_s["Price_USD"].clip(lower=1)
    score = score / np.log1p(prix)
    s_min, s_max = score.min(), score.max()
    if s_max > s_min:
        score = (score - s_min) / (s_max - s_min) * 10
    return score.round(2)

df["Score_QP"] = compute_score(df)

# ─────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────
st.title("📱 Assistant de choix de téléphone")
st.markdown(
    "Trouvez le téléphone **optimal pour votre budget** basé sur les caractéristiques techniques "
    "— indépendamment de la marque."
)

# ─────────────────────────────────────────────
# Barre de recherche (marque ou modèle)
# ─────────────────────────────────────────────
search_query = st.text_input(
    "🔎 Rechercher une marque ou un modèle",
    placeholder="Ex: Samsung, iPhone, Xiaomi, Redmi Note...",
)
st.markdown("---")

# ─────────────────────────────────────────────
# Barre latérale – Filtres
# ─────────────────────────────────────────────
st.sidebar.title("🔍 Vos critères")
st.sidebar.markdown("Ajustez les filtres pour affiner votre recherche.")

# Budget
prix_min_global = int(df["Price_USD"].dropna().min())
prix_max_global = int(df["Price_USD"].dropna().max())
budget = st.sidebar.slider(
    "💰 Budget maximum (USD)",
    min_value=prix_min_global,
    max_value=prix_max_global,
    value=min(600, prix_max_global),
    step=10,
)

# RAM minimale
ram_values = sorted(df["RAM_GB"].dropna().unique().astype(int).tolist())
min_ram = st.sidebar.select_slider(
    "🧠 RAM minimale (GB)",
    options=ram_values,
    value=ram_values[0] if ram_values else 4,
)

# Batterie minimale
battery_min = st.sidebar.slider(
    "🔋 Batterie minimale (mAh)",
    min_value=int(df["Battery_mAh"].dropna().min()),
    max_value=int(df["Battery_mAh"].dropna().max()),
    value=3000,
    step=100,
)

# Caméra arrière minimale
cam_back_min = st.sidebar.slider(
    "📷 Caméra arrière minimale (MP)",
    min_value=int(df["BackCam_MP"].dropna().min()),
    max_value=int(df["BackCam_MP"].dropna().max()),
    value=12,
    step=1,
)

# Caméra frontale minimale
cam_front_min = st.sidebar.slider(
    "🤳 Caméra frontale minimale (MP)",
    min_value=int(df["FrontCam_MP"].dropna().min()),
    max_value=int(df["FrontCam_MP"].dropna().max()),
    value=int(df["FrontCam_MP"].dropna().min()),
    step=1,
)

# Taille d'écran minimale
screen_min = st.sidebar.slider(
    "📐 Taille d'écran minimale (pouces)",
    min_value=float(df["Screen_inch"].dropna().min()),
    max_value=float(df["Screen_inch"].dropna().max()),
    value=5.5,
    step=0.1,
)

include_tablettes = st.sidebar.checkbox(
    "📲 Inclure les tablettes",
    value=False,
    help="Le segment 'Tablettes' regroupe les appareils à grand écran/prix élevé (tablettes et ultra-premium). Décoché par défaut."
)

st.sidebar.markdown("---")
sort_by = st.sidebar.radio(
    "📊 Trier les résultats par",
    options=["Score qualité/prix ↑", "Prix croissant", "Prix décroissant", "RAM ↑"],
)

# ─────────────────────────────────────────────
# Filtrage
# ─────────────────────────────────────────────
mask = (
    (df["Price_USD"].fillna(99999) <= budget) &
    (df["RAM_GB"].fillna(0) >= min_ram) &
    (df["Battery_mAh"].fillna(0) >= battery_min) &
    (df["BackCam_MP"].fillna(0) >= cam_back_min) &
    (df["FrontCam_MP"].fillna(0) >= cam_front_min) &
    (df["Screen_inch"].fillna(0) >= screen_min)
)
if not include_tablettes:
    mask = mask & (df["Segment"] != "Tablettes")
df_filtered = df[mask].copy()

# Appliquer la recherche textuelle (marque ou modèle)
if search_query.strip():
    q = search_query.strip().lower()
    text_mask = (
        df_filtered["Company Name"].fillna("").str.lower().str.contains(q) |
        df_filtered["Model Name"].fillna("").str.lower().str.contains(q)
    )
    df_filtered = df_filtered[text_mask].copy()

# Tri
sort_map = {
    "Score qualité/prix ↑": ("Score_QP", False),
    "Prix croissant": ("Price_USD", True),
    "Prix décroissant": ("Price_USD", False),
    "RAM ↑": ("RAM_GB", False),
}
sort_col, sort_asc = sort_map[sort_by]
df_filtered = df_filtered.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# ─────────────────────────────────────────────
# Tabs principales
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Recommandations", "📊 Explorer par segment", "ℹ️ À propos des segments"])

# ── Tab 1 : Recommandations ──────────────────
with tab1:
    st.subheader("Top 10 des meilleures recommandations")

    if df_filtered.empty:
        st.warning("⚠️ Aucun téléphone ne correspond à vos critères. Essayez d'élargir vos filtres.")
    else:
        # Toujours trier par score Q/P pour le top 10
        top10 = df_filtered.sort_values("Score_QP", ascending=False).head(10).reset_index(drop=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("📱 Résultats trouvés", len(df_filtered))
        col_b.metric("💰 Meilleur prix", f"${top10['Price_USD'].min():.0f}")
        col_c.metric("⭐ Meilleur score Q/P", f"{top10['Score_QP'].max():.1f}/10")
        col_d.metric("🧠 RAM max (top 10)", f"{top10['RAM_GB'].max():.0f} GB")

        st.markdown("---")
        st.caption(f"Les 10 meilleurs téléphones selon le score qualité/prix parmi les {len(df_filtered)} résultats correspondant à vos critères.")

        display_df = top10[DISPLAY_COLS + ["Score_QP"]].copy()
        display_df.index = range(1, len(display_df) + 1)  # Classement 1→10
        display_df.columns = [
            "Marque", "Modèle", "Segment", "RAM (GB)", "Caméra arr. (MP)",
            "Caméra av. (MP)", "Batterie (mAh)", "Écran (\")", "Prix (USD)", "Score Q/P"
        ]
        st.dataframe(display_df, use_container_width=True)

# ── Tab 2 : Explorer par segment ────────────
with tab2:
    st.subheader("Explorer tous les téléphones par segment")

    seg_choice = st.selectbox("Choisir un segment", options=segments_ordres)
    df_seg = df[df["Segment"] == seg_choice].copy().sort_values("Price_USD")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📱 Total", len(df_seg))
    col2.metric("💰 Prix moy.", f"${df_seg['Price_USD'].mean():.0f}")
    col3.metric("🧠 RAM moy.", f"{df_seg['RAM_GB'].mean():.0f} GB")
    col4.metric("🔋 Batterie moy.", f"{df_seg['Battery_mAh'].mean():.0f} mAh")
    col5.metric("📷 Caméra arr. moy.", f"{df_seg['BackCam_MP'].mean():.0f} MP")

    st.markdown("---")
    display_seg = df_seg[DISPLAY_COLS + ["Score_QP"]].copy()
    display_seg.columns = [
        "Marque", "Modèle", "Segment", "RAM (GB)", "Caméra arr. (MP)",
        "Caméra av. (MP)", "Batterie (mAh)", "Écran (\")", "Prix (USD)", "Score Q/P"
    ]
    st.dataframe(display_seg.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── Tab 3 : À propos des segments ───────────
with tab3:
    st.subheader("Profil de chaque segment")
    st.markdown(
        "Les segments sont calculés automatiquement par clustering sur les specs techniques "
        "**(RAM, caméra, batterie — pas la marque ni le prix)**."
    )
    summary_rows = []
    for seg in segments_ordres:
        sub = df[df["Segment"] == seg]
        summary_rows.append({
            "Segment": seg,
            "Nb téléphones": len(sub),
            "Prix moyen ($)": round(sub["Price_USD"].mean(), 0),
            "Prix min ($)": round(sub["Price_USD"].min(), 0),
            "Prix max ($)": round(sub["Price_USD"].max(), 0),
            "RAM moy. (GB)": round(sub["RAM_GB"].mean(), 1),
            "Caméra arr. moy. (MP)": round(sub["BackCam_MP"].mean(), 1),
            "Caméra av. moy. (MP)": round(sub["FrontCam_MP"].mean(), 1),
            "Batterie moy. (mAh)": round(sub["Battery_mAh"].mean(), 0),
            "Écran moy. (\")": round(sub["Screen_inch"].mean(), 1),
        })
    st.dataframe(pd.DataFrame(summary_rows).reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"📊 Dataset : {len(df)} téléphones  |  {n_segments} segments  |  Modèle : clustering sur specs techniques")
