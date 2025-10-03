# import yfinance as yf

from utils.parquetage import afficher_donnees_ticker

def is_valid_ticker(ticker: str) -> bool:
    """
    Vérifie si un ticker est valide en récupérant son historique
    via yfinance. Si l'historique est vide ou qu'une erreur survient,
    on considère le ticker comme invalide.
    """
    try:
        # On récupère l'historique sur 1 jour
        data = afficher_donnees_ticker(ticker)
        # print(data)
        # Si le DataFrame est vide, on suppose que le ticker est invalide
        return not data.empty
    except Exception:
        print(Exception)
        return False
