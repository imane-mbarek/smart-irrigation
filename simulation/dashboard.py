import streamlit as st
import pandas as pd
import time
import requests
import plotly.graph_objects as go
from datetime import datetime
import random

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="AgroSmart - Dashboard", layout="wide")

# --- STYLE DARK, NOIR ET VERT CLAIR ---
st.markdown("""
    <style>
    /* Fond principal noir */
    .main { background-color: #0E1117; color: #FFFFFF; }
    
    /* Titre principal vert émeraude */
    h1 { color: #2EA043 !important; font-family: 'Arial'; }

    /* Style des cartes de métriques (les carrés) */
    div[data-testid="stMetric"] {
        background-color: #1A1C23; 
        border: 2px solid #2EA043; 
        padding: 20px; 
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Couleur vert clair pour les chiffres des métriques */
    [data-testid="stMetricValue"] {
        color: #90EE90 !important; 
        font-size: 1.8rem !important;
    }

    /* Couleur blanche pour les labels des métriques */
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0D1117;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DONNÉES DU MODÈLE ---
crop_coeff = {
    'Maize': 0.55, 'Sugarcane': 0.65, 'Cotton': 0.60, 'Tobacco': 0.45, 
    'Paddy': 0.70, 'Barley': 0.45, 'Wheat': 0.50, 'Millets': 0.40, 
    'Oil seeds': 0.42, 'Pulses': 0.38, 'Ground Nuts': 0.48
}

soil_coeff = {
    'Sandy': 0.20, 'Loamy': 0.00, 'Black': -0.10, 
    'Red': 0.05, 'Clayey': -0.15
}

# --- INITIALISATION DE L'HISTORIQUE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- BARRE LATÉRALE (LOGO AGROSMART) ---
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; border-bottom: 2px solid #2EA043; margin-bottom: 20px;'>
        <h1 style='color: #2EA043; margin: 0; font-size: 35px;'>Agro<span style='color: #FFFFFF;'>Smart</span></h1>
        <p style='color: #90EE90; font-size: 14px;'>Intelligence Agricole</p>
    </div>
    """, unsafe_allow_html=True)

soil_type = st.sidebar.selectbox("Type de Sol", list(soil_coeff.keys()))
crop_type = st.sidebar.selectbox("Culture", list(crop_coeff.keys()))
speed = st.sidebar.slider("Fréquence de simulation (s)", 1, 10, 3)

st.sidebar.markdown("---")
if st.sidebar.button("Réinitialiser les données"):
    st.session_state.history = []

# --- TITRE PRINCIPAL ---
st.markdown("<h1 style='text-align: center;'>🌿 Dashboard d'Irrigation Intelligente</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8B949E;'>Monitoring en temps réel des cultures</p>", unsafe_allow_html=True)
st.markdown("---")

# --- AFFICHAGE DES MÉTRIQUES ---
col1, col2, col3, col4 = st.columns(4)
metric_moisture = col1.empty()
metric_temp = col2.empty()
metric_status = col3.empty()
metric_water = col4.empty()

st.markdown("---")
chart_moisture = st.empty()
chart_water = st.empty()

# --- BOUCLE DE SIMULATION ---
try:
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        
        val_moisture = round(random.uniform(15, 55), 2)
        val_temp = round(random.uniform(25, 42), 2)

        payload = {
            "soil_moisture": val_moisture,
            "temperature": val_temp,
            "humidity": 45.0,
            "rainfall": 0.0,
            "soil_type": soil_type,
            "crop_type": crop_type
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=payload).json()
            
            st.session_state.history.append({
                "Time": now,
                "Moisture": val_moisture,
                "Temperature": val_temp,
                "Litres": response.get('litres_eau', 0),
                "Status": response.get('statut_pompe', 'OFF')
            })
            
            if len(st.session_state.history) > 20:
                st.session_state.history.pop(0)
            
            df = pd.DataFrame(st.session_state.history)

            metric_moisture.metric("Humidité Sol", f"{val_moisture}%")
            metric_temp.metric("Température", f"{val_temp}°C")
            
            p_status = response.get('statut_pompe', 'OFF')
            status_emoji = "ON" if p_status == "ON" else "🚫 OFF"
            metric_status.metric("État Pompe", f"{status_emoji}")
            
            metric_water.metric("Débit Prédit", f"{response.get('litres_eau', 0)} L")

            # 5. Graphiques avec Titres
            with chart_moisture.container():
                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(x=df["Time"], y=df["Moisture"], name="Humidité", line=dict(color='#90EE90', width=3)))
                # Ajout du titre ici
                fig_m.update_layout(
                    title=" Évolution de l'Humidité du Sol en temps réel",
                    template="plotly_dark", 
                    height=300,
                    margin=dict(t=50) # Espace pour le titre
                )
                st.plotly_chart(fig_m, use_container_width=True)

            with chart_water.container():
                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(x=df["Time"], y=df["Litres"], marker_color='#2EA043', name="Litres"))
                # Ajout du titre ici
                fig_w.update_layout(
                    title=" Volume d'Eau Prédit par le Modèle (Litres)",
                    template="plotly_dark", 
                    height=300,
                    margin=dict(t=50) # Espace pour le titre
                )
                st.plotly_chart(fig_w, use_container_width=True)

        except Exception:
            st.warning("🔄 Connexion à l'API AgroSmart-Core...")

        time.sleep(speed)

except Exception as e:
    st.error(f"Erreur : {e}")