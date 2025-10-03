import streamlit as st

def show():
    # Injection de CSS personnalisé pour un design moderne
    st.markdown(
        """
        <style>
        /* Fond global avec un dégradé léger */
        .stApp {
            background: linear-gradient(135deg, #f6f9fc, #e9eff5);
        }
        /* Conteneur principal avec ombre et bord arrondi */
        .main-container {
            background: #ffffff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 20px auto;
            max-width: 800px;
        }
        /* Style du titre principal */
        .main-header {
            text-align: center;
            font-size: 2.5em;
            color: #131CC9;
            margin-bottom: 20px;
        }
        /* Style pour les sous-titres importants (en bleu nuit) */
        .important-subtitle {
            color: #000080;
            font-size: 1.8em;
            font-weight: bold;
            border-bottom: 2px solid #E3B505;
            padding-bottom: 5px;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Conteneur principal
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown('<div class="main-header">Accueil</div>', unsafe_allow_html=True)
    st.write("Bienvenue sur la page d'accueil de l'application **Dynamic Hedger**.")
    st.write("Cette application permet d'obtenir les quantités d'actifs sous-jacents et d'actifs sans risques pour constituer un portefeuille de réplication, afin de couvrir une position directionnelle sur un call européen.")

    st.markdown('<p class="important-subtitle">Description du Modèle</p>', unsafe_allow_html=True)
    st.write("""
    Ce modèle utilise un réseau de neurones LSTM (Long Short-Term Memory) pour optimiser les stratégies de couverture (hedging) d'options financières. 
    L'objectif principal est de **minimiser le risque** de la stratégie de couverture en optimisant le CVaR (Conditional Value at Risk) plutôt que la variance.
    """)

    st.markdown('<p class="important-subtitle">Avantages par rapport au modèle Black-Scholes</p>', unsafe_allow_html=True)
    st.markdown(
    """
    - **Meilleure performance dans des marchés non-gaussiens** : Le modèle LSTM capture des dynamiques de marché plus complexes et non linéaires.
    - **Adaptation aux frictions du marché** : Il prend en compte les coûts de transaction et autres contraintes réelles du marché.
    - **Optimisation directe de la métrique de risque** : Le modèle optimise directement le CVaR, offrant ainsi une meilleure gestion du risque.
    - **Absence d'hypothèses restrictives** : Contrairement à Black-Scholes, il ne nécessite pas d'hypothèses sur la distribution des rendements.
    """, unsafe_allow_html=True)

    st.markdown('<p class="important-subtitle">Architecture du Modèle</p>', unsafe_allow_html=True)
    st.write(
    """
    Le modèle repose sur un réseau LSTM multicouche qui reçoit en entrée l'historique des prix. 
    En sortie, il génère les ratios de couverture optimaux (deltas) pour ajuster dynamiquement le portefeuille de réplication.
    """)

    st.write("Utilisez le menu latéral pour naviguer entre les différentes fonctionnalités de l'application.")

    st.markdown('</div>', unsafe_allow_html=True)
