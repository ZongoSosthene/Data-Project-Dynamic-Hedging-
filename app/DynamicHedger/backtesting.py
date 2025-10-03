import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Définir l'URL du backend pour le backtesting
BACKEND_URL = "https://opti-hedge-backend.onrender.com/compare_strategies"
def show():

    # Ajout d'un peu de style custom via Markdown
    st.markdown(
        """
        <style>
        body {
            background-color: #001F3F;  /* bleu nuit */
            color: white;
        }
        .stButton>button {
            background-color: #0074D9;  /* bleu ciel */
            color: white;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            background-color: white;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True
    )

    def format_date_for_api(date_obj):
        """Convertit un objet date en chaîne de caractères au format MM/DD/YYYY."""
        return date_obj.strftime("%m/%d/%Y")

    st.header("Backtesting")
    st.write("Ici, vous pouvez tester vos stratégies de hedging sur des données historiques.")

    # Création d'un formulaire pour regrouper les inputs
    with st.form("backtesting_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", value="AAPL")
            start_date = st.date_input("Date de début", value=datetime(2023, 1, 1))
            quantity = st.number_input("Quantité", value=150, step=1)
            risk_free_rate = st.number_input("Taux sans risque", value=0.05, step=0.01, format="%.2f")
            strike = st.number_input("Prix d'exercice", value=100.0, step=0.1)
        with col2:
            maturity_date = st.date_input("Maturité", value=datetime(2024, 1, 1))
            rebalancing_freq = st.number_input("Fréquence de rééquilibrage", value=12, step=1)
            current_underlying_weight = st.number_input("Poids actuel de l'underlying", value=0.0, step=0.01, format="%.2f")
            current_cash = st.number_input("Cash actuel", value=0.0, step=1.0)

        submit_button = st.form_submit_button("Lancer le backtest")

    if submit_button:
        # Préparation du payload selon le format attendu par le backend
        payload = {
            "ticker": ticker.strip(),
            "start_date": format_date_for_api(start_date),
            "maturity_date": format_date_for_api(maturity_date),
            "quantity": int(quantity),
            "risk_free_rate": float(risk_free_rate),
            "strike": float(strike),
            "rebalance_freq": int(rebalancing_freq),
            "initial_weights": [float(current_underlying_weight), float(current_cash)]
        }
        
        st.info("Lancement du backtest en cours...")

        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=60)
            result = response.json()
        except Exception as e:
            st.error(f"Erreur lors de la requête vers le backend: {e}")
            result = None

        if result is None or result.get("alert"):
            st.error(f"Alert: {result.get('alert') if result else 'Aucune réponse'}")
        else:
            # Extraction des données de résultat
            results = result.get("results")
            
            # Affichage du tableau comparatif
            st.subheader("Tableau comparatif")
            df_comp = pd.DataFrame(results["comparison_data"])
            st.dataframe(df_comp)
            
            # Affichage des métriques
            st.subheader("Performance Metrics")
            df_metrics = pd.DataFrame(results["metrics"]).T
            st.table(df_metrics)
            
            # Plot de la performance
            st.subheader("Evolution de la performance")
            dates = [d["Date"] for d in results["comparison_data"]]
            lstm_values = [d["LSTM_Value"] for d in results["comparison_data"]]
            bs_values = [d["BS_Value"] for d in results["comparison_data"]]
            underlying_prices = [d["Underlying_Price"] for d in results["comparison_data"]]
            
            fig1, ax1 = plt.subplots(figsize=(15,5))
            ax1.plot(dates, lstm_values, label="LSTM", marker="o")
            ax1.plot(dates, bs_values, label="Black-Scholes", marker="o")
            ax1.plot(dates, underlying_prices, label="Underlying", alpha=0.5, marker="o")
            ax1.set_title("Comparison of Hedging Strategies")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Portfolio Value")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
            
            # Plot de l'évolution des deltas
            st.subheader("Evolution des Deltas")
            delta_dates = results["deltas_plot_data"]["dates"]
            lstm_deltas = results["deltas_plot_data"]["lstm_deltas"]
            bs_deltas = results["deltas_plot_data"]["bs_deltas"]
            
            fig2, ax2 = plt.subplots(figsize=(15,5))
            ax2.plot(delta_dates, lstm_deltas, label="LSTM Delta", marker="o")
            ax2.plot(delta_dates, bs_deltas, label="BS Delta", marker="o")
            ax2.set_title("Delta Evolution")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Delta")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
