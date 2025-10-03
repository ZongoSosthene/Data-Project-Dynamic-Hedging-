import codecs
import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from utils.simulate import apply_model
from utils.backtesting import compare_strategies   # Import de la fonction compare_strategies
import uvicorn
import datetime
import yfinance as yf
from utils.validate_ticker import is_valid_ticker

# Import des métriques Prometheus
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST, make_asgi_app

# Déclaration des compteurs
TOTAL_REQUESTS = Counter('total_requests', 'Nombre total de requêtes')
REQUESTS_BY_ENDPOINT = Counter('requests_by_endpoint', 'Nombre de requêtes par endpoint', ['endpoint'])
SIMULATE_AAPL = Counter('simulate_aapl_requests', 'Nombre de requêtes de simulation pour AAPL')
SIMULATE_AMZN = Counter('simulate_amzn_requests', 'Nombre de requêtes de simulation pour AMZN')
COMPARE_STRATEGIES_COUNTER = Counter('compare_strategies_requests', 'Nombre de requêtes pour compare_strategies')

app = FastAPI()
app.mount("/metrics", make_asgi_app())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À ajuster en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationInput(BaseModel):
    ticker: str
    quantity: int
    riskFreeRate: float
    date: str
    maturityDate: str
    strike: float
    rebalancing_freq: float
    current_underlying_weight: float
    current_cash: float

# Nouveau modèle de données pour l'endpoint compare_strategies
class CompareStrategiesInput(BaseModel):
    ticker: str
    start_date: str
    maturity_date: str
    quantity: int
    risk_free_rate: float
    strike: float
    rebalance_freq: int
    initial_weights: tuple[float, float]

@app.get("/", response_class=HTMLResponse)
async def front_root():
    """
    Cet endpoint permet de retourner une page html qui est une 
    application frontend utilisant notre API. Nous n'utilisons pas streamlit pour des choix de design.
    NB : Cette route est toujours en cours de développement.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PRESENTATION_HTML_PATH = os.path.join(BASE_DIR, 'presentation.html')
    with codecs.open(PRESENTATION_HTML_PATH, 'r', 'utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/validate_ticker/{ticker}")
def validate_ticker(ticker: str):
    # Incrémenter les compteurs
    TOTAL_REQUESTS.inc()
    REQUESTS_BY_ENDPOINT.labels(endpoint="/validate_ticker").inc()
    
    valid = is_valid_ticker(ticker)
    return {
        "ticker": ticker,
        "valid": valid
    }

@app.post("/simulate")
def simulate(params: SimulationInput):
    # Incrémenter les compteurs
    TOTAL_REQUESTS.inc()
    REQUESTS_BY_ENDPOINT.labels(endpoint="/simulate").inc()

    # Compter spécifiquement les requêtes de simulation pour AAPL et AMZN
    ticker_upper = params.ticker.upper()
    if ticker_upper == "AAPL":
        SIMULATE_AAPL.inc()
    if ticker_upper == "AMZN":
        SIMULATE_AMZN.inc()

    # Vérification du ticker
    if not is_valid_ticker(params.ticker):
        return {"error": "Ticker invalide"}

    # Conversion des dates
    today = datetime.datetime.strptime(params.date, "%m/%d/%Y")
    maturity_dt = datetime.datetime.strptime(params.maturityDate, "%m/%d/%Y")

    # Appel de la fonction de simulation
    prediction = apply_model(
        ticker=params.ticker,
        start_date=today.strftime("%m/%d/%Y"),
        maturity_date=maturity_dt.strftime("%m/%d/%Y"),
        option_quantity=params.quantity,
        strike=params.strike,
        rebalancing_freq=12,
        current_weights={params.ticker: params.current_underlying_weight},  
        cash_account=params.current_cash,
        trained_model_path=""
    )
    if "error" in prediction:
        return {"Error": prediction["error"]}
    else:
        return {"prediction": prediction}

# Nouveau endpoint pour compare_strategies
@app.post("/compare_strategies")
def compare_strategies_route(params: CompareStrategiesInput):
    # Incrémenter les compteurs
    TOTAL_REQUESTS.inc()
    REQUESTS_BY_ENDPOINT.labels(endpoint="/compare_strategies").inc()
    COMPARE_STRATEGIES_COUNTER.inc()

    # Vérification du ticker
    if not is_valid_ticker(params.ticker):
        return {"error": "Ticker invalide"}

    # Conversion des dates
    start_date_dt = datetime.datetime.strptime(params.start_date, "%m/%d/%Y")
    maturity_date_dt = datetime.datetime.strptime(params.maturity_date, "%m/%d/%Y")

    # Préparation des paramètres pour compare_strategies
    compare_params = {
        "ticker": params.ticker,
        "start_date": start_date_dt.strftime("%m/%d/%Y"),
        "maturity_date": maturity_date_dt.strftime("%m/%d/%Y"),
        "quantity": params.quantity,
        "risk_free_rate": params.risk_free_rate,
        "strike": params.strike,
        "rebalance_freq": params.rebalance_freq,
        "initial_weights": params.initial_weights
    }

    # Appel de la fonction compare_strategies
    results, alert = compare_strategies(compare_params)
    return {"results": results, "alert": alert}

@app.get("/metrics")
def metrics():
    # Retourne les métriques Prometheus
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
