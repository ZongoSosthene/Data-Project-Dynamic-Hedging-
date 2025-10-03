import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

def show():
    today_str = datetime.today().strftime("%Y-%m-%d")
    html_code = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <title>Hedger App</title>
      <style>
        /* Styles de base */
        body {{
          margin: 0;
          font-family: Arial, sans-serif;
          background-color: #FAFAFF;
          color: #000;
        }}
        .container {{
          display: flex;
          flex-direction: column;
          min-height: 100vh;
        }}
        /* Section des inputs */
        .input-section {{
          padding: 20px;
        }}
        .input-container {{
          max-width: 600px;
          margin: auto;
        }}
        h1 {{
          text-align: center;
          font-size: 2em;
          margin-bottom: 20px;
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
        }}
        .input-group {{
          display: flex;
          flex-direction: column;
        }}
        label {{
          margin-bottom: 5px;
          font-size: 0.9em;
        }}
        input {{
          padding: 10px;
          font-size: 1em;
          border: 1px solid #131CC9;
          border-radius: 4px;
          width: 100%;
          box-sizing: border-box;
        }}
        input.valid {{
          border-color: green;
        }}
        input[type="date"] {{}}
        .relative {{
          position: relative;
        }}
        .checkmark {{
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          color: green;
          font-weight: bold;
        }}
        .error {{
          color: red;
          font-size: 0.9em;
          margin-top: 10px;
          text-align: center;
        }}
        /* Divider jaune */
        .divider {{
          height: 2px;
          background-color: #E3B505;
          margin: 40px 0;
        }}
        /* Section de simulation et résultats */
        .simulation-section {{
          padding: 40px 20px;
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
        }}
        .simulate-view,
        .results-view {{
          width: 100%;
          max-width: 600px;
          text-align: center;
        }}
        .button {{
          display: inline-block;
          background-color: #131CC9;
          color: white;
          padding: 15px 30px;
          font-size: 1.1em;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          margin-top: 20px;
        }}
        .button:disabled {{
          opacity: 0.5;
          cursor: not-allowed;
        }}
        .spinner {{
          display: inline-block;
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: white;
          animation: spin 1s ease-in-out infinite;
          margin-right: 10px;
        }}
        @keyframes spin {{
          to {{ transform: rotate(360deg); }}
        }}
        /* Résultats */
        .result {{
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 6px rgba(0,0,0,0.2);
          margin: 20px auto;
          max-width: 400px;
        }}
        .result-item {{
          padding: 10px;
          border-left: 4px solid #E3B505;
          margin-bottom: 10px;
          text-align: left;
        }}
        .reset-button {{
          background-color: #E3B505;
          color: black;
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          margin-top: 20px;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <!-- Section des inputs -->
        <div class="input-section">
          <div class="input-container">
            <h1>Hedger</h1>
            <div class="grid">
              <div class="input-group">
                <label for="ticker">Ticker</label>
                <div class="relative">
                  <input type="text" id="ticker" placeholder="ex: AAPL">
                  <span id="ticker-valid" class="checkmark" style="display:none;">✓</span>
                </div>
              </div>
              <div class="input-group">
                <label for="quantity">Quantité</label>
                <input type="number" id="quantity" placeholder="ex: 100">
              </div>
              <div class="input-group">
                <label for="date">Date</label>
                <div class="relative">
                  <input type="date" id="date" value="{today_str}">
                </div>
              </div>
              <div class="input-group">
                <label for="riskFreeRate">Taux sans risque</label>
                <input type="number" id="riskFreeRate" placeholder="ex: 0.05" step="0.01">
              </div>
              <div class="input-group">
                <label for="maturityDate">Maturité</label>
                <div class="relative">
                  <input type="date" id="maturityDate" value="{today_str}">
                </div>
              </div>
              <div class="input-group">
                <label for="strike">Prix d'exercice</label>
                <input type="number" id="strike" placeholder="ex: 150.0" step="0.01">
              </div>
              <div class="input-group">
                <label for="rebalancing_freq">Fréquence de rééquilibrage</label>
                <input type="number" id="rebalancing_freq" placeholder="ex: 12" step="1">
              </div>
              <div class="input-group">
                <label for="current_underlying_weight">Poids actuel de l'underlying</label>
                <input type="number" id="current_underlying_weight" placeholder="ex: 0.5" step="0.01">
              </div>
              <div class="input-group">
                <label for="current_cash">Cash actuel</label>
                <input type="number" id="current_cash" placeholder="ex: 1000" step="any">
              </div>
            </div>
            <div id="error-message" class="error" style="display:none;">Veuillez remplir tous les champs correctement</div>
          </div>
        </div>
        <!-- Divider jaune -->
        <div class="divider"></div>
        <!-- Section de simulation et résultats -->
        <div class="simulation-section">
          <div id="simulate-view" class="simulate-view">
            <button id="simulate-button" class="button" disabled>Obtenir son portefeuille de couverture</button>
          </div>
          <div id="results-view" class="results-view" style="display:none;">
            <div id="results-container"></div>
            <button id="reset-button" class="reset-button">Faire une nouvelle simulation</button>
          </div>
        </div>
      </div>
      <script>
        document.addEventListener("DOMContentLoaded", function() {{
          document.getElementById("date").value = "{today_str}";
          document.getElementById("maturityDate").value = "{today_str}";
        }});

        const apiBaseUrl = "https://opti-hedge-backend.onrender.com";
        const tickerInput = document.getElementById("ticker");
        const quantityInput = document.getElementById("quantity");
        const dateInput = document.getElementById("date");
        const riskFreeRateInput = document.getElementById("riskFreeRate");
        const maturityDateInput = document.getElementById("maturityDate");
        const strikeInput = document.getElementById("strike");
        const rebalancingInput = document.getElementById("rebalancing_freq");
        const underlyingWeightInput = document.getElementById("current_underlying_weight");
        const cashInput = document.getElementById("current_cash");

        const simulateButton = document.getElementById("simulate-button");
        const tickerValidIndicator = document.getElementById("ticker-valid");
        const errorMessage = document.getElementById("error-message");

        let isTickerValid = false;
        let tickerTimeout;

        tickerInput.addEventListener("input", function() {{
          clearTimeout(tickerTimeout);
          tickerTimeout = setTimeout(() => {{
            if(tickerInput.value.trim().length >= 2) {{
              // Appel à l'API pour valider le ticker
              fetch(apiBaseUrl + "/validate_ticker/" + encodeURIComponent(tickerInput.value.trim()))
                .then(response => response.json())
                .then(data => {{
                  console.log("Réponse API pour le ticker:", data);
                  if(data.valid) {{
                    isTickerValid = true;
                    tickerInput.classList.add("valid");
                    tickerValidIndicator.style.display = "inline";
                  }} else {{
                    isTickerValid = false;
                    tickerInput.classList.remove("valid");
                    tickerValidIndicator.style.display = "none";
                  }}
                  validateForm();
                }})
                .catch(err => {{
                  console.error("Erreur lors de la validation du ticker:", err);
                  isTickerValid = false;
                  tickerInput.classList.remove("valid");
                  tickerValidIndicator.style.display = "none";
                  validateForm();
                }});
            }} else {{
              isTickerValid = false;
              tickerInput.classList.remove("valid");
              tickerValidIndicator.style.display = "none";
              validateForm();
            }}
          }}, 500);
        }});

        [quantityInput, dateInput, riskFreeRateInput, maturityDateInput, strikeInput, rebalancingInput, underlyingWeightInput, cashInput].forEach(input => {{
          input.addEventListener("input", validateForm);
        }});

        function validateForm() {{
          const isQuantityValid = quantityInput.value.trim() !== "";
          const isDateValid = dateInput.value.trim() !== "";
          const isRiskFreeRateValid = riskFreeRateInput.value.trim() !== "";
          const isMaturityValid = maturityDateInput.value.trim() !== "";
          const isStrikeValid = strikeInput.value.trim() !== "";
          const isRebalancingValid = rebalancingInput.value.trim() !== "";
          const isUnderlyingWeightValid = underlyingWeightInput.value.trim() !== "";
          const isCashValid = cashInput.value.trim() !== "";
          const isFormValid = isTickerValid && isQuantityValid && isDateValid && isRiskFreeRateValid &&
                              isMaturityValid && isStrikeValid && isRebalancingValid && isUnderlyingWeightValid && isCashValid;
          simulateButton.disabled = !isFormValid;
          errorMessage.style.display = isFormValid ? "none" : "block";
        }}

        function formatDate(dateStr) {{
          // Convertir "YYYY-MM-DD" en "MM/DD/YYYY"
          const parts = dateStr.split("-");
          return parts[1] + "/" + parts[2] + "/" + parts[0];
        }}

        simulateButton.addEventListener("click", function() {{
          if(simulateButton.disabled) return;
          simulateButton.disabled = true;
          simulateButton.innerHTML = '<span class="spinner"></span> Simulation en cours...';

          // Préparation des données à envoyer
          const payload = {{
            ticker: tickerInput.value.trim(),
            quantity: parseInt(quantityInput.value),
            riskFreeRate: parseFloat(riskFreeRateInput.value),
            date: formatDate(dateInput.value),
            maturityDate: formatDate(maturityDateInput.value),
            strike: parseFloat(strikeInput.value),
            rebalancing_freq: parseFloat(rebalancingInput.value),
            current_underlying_weight: parseFloat(underlyingWeightInput.value),
            current_cash: parseFloat(cashInput.value)
          }};

          fetch(apiBaseUrl + "/simulate", {{
            method: "POST",
            headers: {{
              "Content-Type": "application/json"
            }},
            body: JSON.stringify(payload)
          }})
          .then(response => response.json())
          .then(data => {{
            simulateButton.innerHTML = 'Obtenir son portefeuille de couverture';
            if(data.error || data.Error) {{
              displayResults({{ error: data.error || data.Error }});
            }} else if(data["prediction "]) {{
              displayResults(data["prediction "]);
            }} else {{
              displayResults({{ error: "Réponse inattendue de l'API" }});
            }}
          }})
          .catch(err => {{
            console.error("Erreur lors de la simulation:", err);
            simulateButton.innerHTML = 'Obtenir son portefeuille de couverture';
            displayResults({{ error: "Erreur lors de la simulation" }});
          }});
        }});

        function displayResults(resultData) {{
          document.getElementById("simulate-view").style.display = "none";
          document.getElementById("results-view").style.display = "block";
          const resultsContainer = document.getElementById("results-container");
          resultsContainer.innerHTML = "";
          if(resultData.error) {{
            const errorDiv = document.createElement("div");
            errorDiv.className = "result";
            errorDiv.textContent = resultData.error;
            resultsContainer.appendChild(errorDiv);
          }} else {{
            for(const key in resultData) {{
              const itemDiv = document.createElement("div");
              itemDiv.className = "result-item";
              itemDiv.textContent = key + ": " + resultData[key];
              resultsContainer.appendChild(itemDiv);
            }}
          }}
        }}

        document.getElementById("reset-button").addEventListener("click", function() {{
          document.getElementById("simulate-view").style.display = "block";
          document.getElementById("results-view").style.display = "none";
          simulateButton.innerHTML = 'Obtenir son portefeuille de couverture';
          simulateButton.disabled = false;
          validateForm();
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=800, scrolling=True)
