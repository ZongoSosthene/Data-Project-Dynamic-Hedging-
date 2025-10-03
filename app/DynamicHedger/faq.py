import streamlit as st

def show():
    # Injecter du CSS personnalisé pour styliser la FAQ
    st.markdown(
        """
        <style>
        .faq-header {
            color: #131CC9;
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        .faq-container {
            margin: 20px 0;
        }
        .faq-question {
            background-color: #ECECFF;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            color: #131CC9;
        }
        .faq-answer {
            background-color: #FAFAFF;
            padding: 10px;
            border-left: 4px solid #E3B505;
            margin-bottom: 15px;
            border-radius: 5px;
            color: #000;
        }
        .faq-footer {
            text-align: center;
            margin-top: 30px;
            color: #131CC9;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='faq-header'>FAQ - Guide d'utilisation de Hedger</div>", unsafe_allow_html=True)
    st.write("Retrouvez ci-dessous quelques questions fréquentes pour vous aider à utiliser la page Hedger.")

    # Question 1 : Comment utiliser la page Hedger ?
    with st.expander("Comment utiliser la page Hedger ?"):
        st.markdown(
            """
            <div class="faq-answer">
            La page Hedger vous permet de simuler la couverture d'une position sur un call européen. Pour l'utiliser :<br>
            - Saisissez le <strong>ticker</strong> de l'actif (ex: AAPL). Un indicateur vert apparaît si le ticker est validé.<br>
            - Remplissez les champs <strong>Quantité</strong>, <strong>Date</strong>, <strong>Taux sans risque</strong>, <strong>Maturité</strong>, <strong>Prix d'exercice</strong>, <strong>Fréquence de rééquilibrage</strong>, <strong>Poids actuel de l'underlying</strong> et <strong>Cash actuel</strong>.<br>
            - Une fois tous les champs valides, cliquez sur le bouton <strong>"Obtenir son portefeuille de couverture"</strong> pour lancer la simulation.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Question 2 : Que faire si le ticker n'est pas validé ?
    with st.expander("Que faire si le ticker n'est pas validé ?"):
        st.markdown(
            """
            <div class="faq-answer">
            Si le ticker n'est pas validé, assurez-vous qu'il comporte au moins deux caractères et qu'il est correct. L'API vérifie le ticker via Yahoo Finance et affichera un indicateur vert (✓) en cas de succès.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Question 3 : Que représentent les différents champs ?
    with st.expander("Que représentent les différents champs de la simulation ?"):
        st.markdown(
            """
            <div class="faq-answer">
            <strong>Quantité :</strong> Nombre d'options ou d'actifs à couvrir.<br>
            <strong>Date :</strong> Date de début de la simulation.<br>
            <strong>Taux sans risque :</strong> Taux d'intérêt sans risque utilisé dans la simulation.<br>
            <strong>Maturité :</strong> Date d'expiration de l'option.<br>
            <strong>Prix d'exercice :</strong> Prix auquel l'option peut être exercée.<br>
            <strong>Fréquence de rééquilibrage :</strong> Nombre de fois par an que le portefeuille est ajusté.<br>
            <strong>Poids actuel de l'underlying :</strong> Proportion de l'actif sous-jacent dans le portefeuille.<br>
            <strong>Cash actuel :</strong> Montant en liquidités disponibles dans le portefeuille.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Question 4 : Que faire en cas d'erreur de simulation ?
    with st.expander("Que faire en cas d'erreur de simulation ?"):
        st.markdown(
            """
            <div class="faq-answer">
            Si une erreur survient lors de la simulation, un message d'erreur sera affiché. Vérifiez que tous les champs sont correctement remplis et que votre connexion à l'API est stable. Vous pouvez réinitialiser la simulation en cliquant sur le bouton <strong>"Faire une nouvelle simulation"</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.markdown(
        """
        <div class="faq-footer">
            Pour toute question supplémentaire, n'hésitez pas à contacter notre support.
        </div>
        """,
        unsafe_allow_html=True
    )
