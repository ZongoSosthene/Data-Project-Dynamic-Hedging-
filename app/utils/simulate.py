#!pip install tensorflow==2.12.0
# from utils.parquetage import afficher_donnees_ticker
from utils.parquetage import afficher_donnees_ticker
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import joblib  # Import de joblib pour la sérialisation
# Import keras from tensorflow
import keras
import yfinance as yf
from datetime import date, timedelta, datetime # Import date and timedelta



# ==================== GÉNÉRATION DE DONNÉES ====================
def monte_carlo_paths(S_0, time_to_expiry, sigma, drift, seed, n_sims, n_timesteps):
    """Génère des trajectoires de prix en 3D : (n_timesteps+1, n_sims, 1)."""
    if seed is not None:
        np.random.seed(seed)

    dt_val = time_to_expiry / n_timesteps
    paths = np.zeros((n_timesteps + 1, n_sims, 1))
    paths[0, :, 0] = S_0
    for t in range(1, n_timesteps + 1):
        rand = np.random.normal(0, 1, n_sims)
        paths[t, :, 0] = paths[t-1, :, 0] * np.exp(
            (drift - 0.5 * sigma**2) * dt_val +
            sigma * np.sqrt(dt_val) * rand
        )
    return paths
# ==================== MODÈLE LSTM AVEC NOUVEAUX INPUTS ====================
# Use tf.keras.utils.register_keras_serializable instead of keras.saving.register_keras_serializable
# @tf.keras.utils.register_keras_serializable()
# class Agent(tf.keras.Model):
#     def __init__(self,
#                  time_steps,
#                  batch_size,
#                  features,  # doit être égal à 7
#                  T,
#                  r=0.05,
#                  sigma=0.2,
#                  nodes=[64, 48, 32, 1],
#                  lambda_skew=0.1,
#                  lambda_var=0.1,
#                  name='model'):
#         """
#         :param features: nombre total de features (ici 7 : [Sₜ, T_remaining, delta_prev, cash_prev, call_price_t_minus, call_price_t, simulation_summary])
#         :param T: maturité totale
#         :param sigma: volatilité du sous-jacent (pour Black-Scholes et la simulation)
#         """
#         super().__init__(name=name)
#         self.time_steps = time_steps
#         self.batch_size = batch_size
#         self.features = features
#         self.T = T
#         self.r = r
#         self.sigma = sigma
#         self.lambda_skew = lambda_skew
#         self.lambda_var = lambda_var

#         # Construction de l'architecture LSTM
#         self.lstm_layers = [
#             tf.keras.layers.LSTM(
#                 units,
#                 return_sequences=True,
#                 activation='tanh',
#                 kernel_initializer='glorot_uniform'
#             )
#             for units in nodes
#         ]
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

#     def call(self, inputs):
#         """
#         Passage avant du modèle.
#         Les inputs doivent avoir la forme (1, batch_size, features).
#         """
#         x = inputs
#         for lstm in self.lstm_layers:
#             x = lstm(x)
#         # On renvoie un tenseur de forme (batch_size,)
#         return tf.squeeze(x, axis=[0, -1])
@tf.keras.utils.register_keras_serializable()
class Agent(tf.keras.Model):
    def __init__(self,
                 time_steps,
                 batch_size,
                 features,  # doit être égal à 7
                 T,
                 r=0.05,
                 sigma=0.2,
                 nodes=[64, 48, 32, 1],
                 lambda_skew=0.1,
                 lambda_var=0.1,
                 name='model',
                 **kwargs):  # Accepte les kwargs pour ignorer les paramètres supplémentaires
        super().__init__(name=name, **kwargs)
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.features = features
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lambda_skew = lambda_skew
        self.lambda_var = lambda_var
        self.nodes = nodes  # On sauvegarde la liste des nodes pour la config

        # Construction de l'architecture LSTM
        self.lstm_layers = [
            tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                activation='tanh',
                kernel_initializer='glorot_uniform'
            )
            for units in nodes
        ]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

    def call(self, inputs):
        """
        Passage avant du modèle.
        Les inputs doivent avoir la forme (1, batch_size, features).
        """
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x)
        # On renvoie un tenseur de forme (batch_size,)
        return tf.squeeze(x, axis=[0, -1])

    def get_config(self):
        config = super().get_config()
        config.update({
            'time_steps': self.time_steps,
            'batch_size': self.batch_size,
            'features': self.features,
            'T': self.T,
            'r': self.r,
            'sigma': self.sigma,
            'nodes': self.nodes,
            'lambda_skew': self.lambda_skew,
            'lambda_var': self.lambda_var,
        })
        return config

    def black_scholes_call_price(self, S, K, T, r, sigma):
        """
        Calcule le prix d'un call européen via Black-Scholes de manière différentiable.
        :param S: prix du sous-jacent (tensor)
        :param K: strike (scalar ou tensor)
        :param T: temps jusqu'à maturité (tensor ou scalaire)
        """
        eps = 1e-8
        T = tf.maximum(T, eps)
        d1 = (tf.math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * tf.sqrt(T))
        d2 = d1 - sigma * tf.sqrt(T)
        cdf_d1 = 0.5 * (1.0 + tf.math.erf(d1 / tf.sqrt(2.0)))
        cdf_d2 = 0.5 * (1.0 + tf.math.erf(d2 / tf.sqrt(2.0)))
        call = S * cdf_d1 - K * tf.exp(-r * T) * cdf_d2
        return call

    def calculate_hedging_pnl(self, S_t_input, K, delta_init=None, cash_init=None):
        """
        Calcule le PnL de la stratégie de couverture de manière récursive.

        :param S_t_input: trajectoires de prix, tenseur de forme (time_steps, batch_size, 1)
        :param K: strike de l'option (scalar ou tensor)
        :param delta_init: (optionnel) delta initial pour chaque échantillon, tenseur de forme (batch_size,)
        :param cash_init: (optionnel) cash initial pour chaque échantillon, tenseur de forme (batch_size,)
        :return: tuple (pnl, decisions) où :
          - pnl est un tenseur de forme (batch_size,) représentant le PnL final,
          - decisions est un tenseur contenant les deltas calculés à chaque instant.
        """
        time_steps = tf.shape(S_t_input)[0]
        batch_size = tf.shape(S_t_input)[1]
        dt_val = self.T / tf.cast(time_steps - 1, tf.float32)

        # Initialisation du delta et du cash
        if delta_init is None:
            delta_prev = tf.zeros((batch_size,), dtype=tf.float32)
        else:
            delta_prev = delta_init

        if cash_init is None:
            cash_prev = tf.zeros((batch_size,), dtype=tf.float32)
        else:
            cash_prev = cash_init

        # Initialisation du call à t=0
        S0_tensor = tf.squeeze(S_t_input[0, :, :], axis=-1)
        T_remaining_0 = self.T
        call_price_t = self.black_scholes_call_price(S0_tensor, K, T_remaining_0, self.r, self.sigma)
        call_price_t_minus = call_price_t  # pour t = 0, aucune valeur antérieure

        decisions = tf.TensorArray(tf.float32, size=time_steps, dynamic_size=False)

        for t in tf.range(time_steps):
            T_remaining = self.T - tf.cast(t, tf.float32) * dt_val
            S_t = tf.squeeze(S_t_input[t, :, :], axis=-1)
            call_price_t = self.black_scholes_call_price(S_t, K, T_remaining, self.r, self.sigma)
            simulation_summary = S_t * tf.exp(self.r * T_remaining)
            T_vec = T_remaining * tf.ones_like(S_t)
            x_t = tf.stack([
                S_t,
                T_vec,
                delta_prev,
                cash_prev,
                call_price_t_minus,
                call_price_t,
                simulation_summary
            ], axis=-1)
            x_t_expanded = tf.expand_dims(x_t, axis=0)  # forme (1, batch_size, 7)
            delta_t = self(x_t_expanded)
            decisions = decisions.write(t, delta_t)

            cash_t = tf.cond(
                tf.equal(t, 0),
                lambda: -(delta_t - delta_prev) * S_t,
                lambda: cash_prev * tf.exp(self.r * dt_val) - (delta_t - delta_prev) * S_t
            )
            delta_prev = delta_t
            cash_prev = cash_t
            call_price_t_minus = call_price_t

        cash_final = cash_prev * tf.exp(self.r * dt_val)
        S_T = tf.squeeze(S_t_input[-1, :, :], axis=-1)
        Pi_T = delta_prev * S_T + cash_final
        pnl = Pi_T - tf.maximum(S_T - K, 0)
        decisions = decisions.stack()
        return pnl, decisions

    def calculate_cvar(self, pnl, alpha):
        sorted_pnl = tf.sort(pnl)  # ordre croissant
        n = tf.cast(tf.shape(pnl)[0], tf.float32)
        var_index = tf.cast((1 - alpha) * n, tf.int32)
        return -tf.reduce_mean(sorted_pnl[:var_index])

    def calculate_skewness(self, pnl):
        mean_pnl = tf.reduce_mean(pnl)
        std_pnl = tf.math.reduce_std(pnl) + 1e-8
        skew = tf.reduce_mean(((pnl - mean_pnl) / std_pnl) ** 3)
        return skew

    def calculate_variance(self, pnl):
        return tf.math.reduce_variance(pnl)

    @tf.function
    def train_step(self, S_t_input, K, alpha):
        with tf.GradientTape() as tape:
            pnl, decisions = self.calculate_hedging_pnl(S_t_input, K)
            cvar = self.calculate_cvar(pnl, alpha)
            skew = self.calculate_skewness(pnl)
            var = self.calculate_variance(pnl)
            loss = cvar - self.lambda_skew * skew + self.lambda_var * var
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, pnl, decisions

    def training(self, paths, strikes, riskaversion, epochs):
        sample_size = paths.shape[1]
        idx = np.arange(sample_size)
        for epoch in range(epochs):
            np.random.shuffle(idx)
            pnls = []
            for i in range(0, sample_size, self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                batch = paths[:, batch_idx, :]  # forme : (time_steps, batch_size, 1)
                loss, pnl, _ = self.train_step(
                    tf.cast(batch, tf.float32),
                    tf.cast(strikes[batch_idx], tf.float32),
                    tf.constant(riskaversion, tf.float32)
                )
                pnls.append(pnl.numpy())
            if epoch % 10 == 0:
                all_pnls = np.concatenate(pnls)
                current_cvar = np.mean(-np.sort(all_pnls)[:int((1 - riskaversion) * sample_size)])
                print(f"Epoch {epoch} | Loss: {loss.numpy():.4f} | CVaR: {current_cvar:.4f}")
        return self
# ==================== ÉVALUATION ====================
class HedgingTest:
    def __init__(self, S_0=100, K=100, r=0.05, T=1/12, vol=0.2, timesteps=15):
        self.params = {
            'S_0': S_0,
            'K': K,
            'r': r,
            'T': T,
            'vol': vol,
            'timesteps': timesteps
        }
        self.dt = T / timesteps

    def black_scholes_delta(self, S, t):
        if t <= 1e-6:
            return np.where(S > self.params['K'], 1.0, 0.0)
        d1 = (np.log(S/self.params['K']) +
              (self.params['r'] + 0.5*self.params['vol']**2)*t) / (self.params['vol']*np.sqrt(t))
        return norm.cdf(d1)

    def calculate_bs_pnl(self, paths):
        n_sims = paths.shape[1]
        pnls = np.zeros(n_sims)
        for i in range(n_sims):
            path = paths[:, i, 0]
            cash = 0.0
            delta_prev = 0.0
            for t in range(len(path)-1):
                time_left = self.params['T'] - t*self.dt
                delta = self.black_scholes_delta(path[t], time_left)
                cash = cash * np.exp(self.params['r'] * self.dt) - (delta - delta_prev)*path[t]
                delta_prev = delta
            cash_final = cash * np.exp(self.params['r'] * self.dt)
            Pi_T = delta_prev * path[-1] + cash_final
            pnls[i] = Pi_T - max(path[-1] - self.params['K'], 0)
        return pnls

    def compare_strategies(self, model, n_paths=5000):
        path_params = {
            'S_0': self.params['S_0'],
            'time_to_expiry': self.params['T'],
            'sigma': self.params['vol'],
            'drift': self.params['r'],
            'seed': 42,
            'n_sims': n_paths,
            'n_timesteps': self.params['timesteps']
        }
        paths = monte_carlo_paths(**path_params)
        bs_pnl = self.calculate_bs_pnl(paths)
        pnl_tensor, _ = model.calculate_hedging_pnl(tf.constant(paths, tf.float32),
                                                    tf.constant(self.params['K'], tf.float32))
        lstm_pnl = pnl_tensor.numpy()
        results = {
            'Black-Scholes': {
                'Moyenne': bs_pnl.mean(),
                'Écart-type': bs_pnl.std(),
                'CVaR 95%': self._calculate_cvar(bs_pnl, 0.95)
            },
            'LSTM': {
                'Moyenne': lstm_pnl.mean(),
                'Écart-type': lstm_pnl.std(),
                'CVaR 95%': self._calculate_cvar(lstm_pnl, 0.95)
            }
        }
        self._plot_results(bs_pnl, lstm_pnl)
        return results

    def _calculate_cvar(self, pnl, alpha):
        sorted_pnl = np.sort(pnl)
        n_worst = int((1 - alpha) * len(pnl))
        return -sorted_pnl[:n_worst].mean()

    def _plot_results(self, bs_pnl, lstm_pnl):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        sns.kdeplot(bs_pnl, label='Black-Scholes', fill=True)
        sns.kdeplot(lstm_pnl, label='LSTM', fill=True)
        plt.title('Distribution des PnL')
        plt.xlabel('PnL')
        plt.legend()
        plt.subplot(2, 2, 2)
        pnl_data = pd.DataFrame({'Black-Scholes': bs_pnl, 'LSTM': lstm_pnl})
        sns.boxplot(data=pnl_data)
        plt.title('Box Plot des PnL')
        plt.ylabel('PnL')
        plt.subplot(2, 2, 3)
        plt.bar(['Black-Scholes', 'LSTM'],
                [self._calculate_cvar(bs_pnl, 0.95), self._calculate_cvar(lstm_pnl, 0.95)])
        plt.title('CVaR 95% Comparaison')
        plt.ylabel('CVaR')
        plt.subplot(2, 2, 4)
        stats_df = pd.DataFrame({
            'Black-Scholes': [np.mean(bs_pnl), np.median(bs_pnl), np.std(bs_pnl),
                              np.percentile(bs_pnl, 5), np.percentile(bs_pnl, 95)],
            'LSTM': [np.mean(lstm_pnl), np.median(lstm_pnl), np.std(lstm_pnl),
                     np.percentile(lstm_pnl, 5), np.percentile(lstm_pnl, 95)]
        }, index=['Mean', 'Median', 'Std', '5th Percentile', '95th Percentile'])
        plt.table(cellText=stats_df.round(4).values,
                  rowLabels=stats_df.index,
                  colLabels=stats_df.columns,
                  cellLoc='center', loc='center')
        plt.axis('off')
        plt.title('Statistical Summary')
        plt.tight_layout()
        plt.show()

# ==================== EXÉCUTION ====================
def run_training():
    params = {
        'S_0': 100,
        'T': 1/12,  # Par exemple 1 mois de maturité pour l'entraînement
        'r': 0.05,
        'vol': 0.2,
        'timesteps': 15,
        'n_sims': 100,
        'batch_size': 50,
        'epochs': 50,
        'alpha': 0.95,
        'lambda_skew': 0.4,
        'lambda_var': 0.1
    }
    print("Génération des paths...")
    paths = monte_carlo_paths(
        S_0=params['S_0'],
        time_to_expiry=params['T'],
        sigma=params['vol'],
        drift=params['r'],
        seed=42,
        n_sims=params['n_sims'],
        n_timesteps=params['timesteps']
    )
    # Ici, features = 7 :
    # [Sₜ, T_remaining, delta_prev, cash_prev, call_price_t_minus, call_price_t, simulation_summary]
    model = Agent(
        time_steps=paths.shape[0],
        batch_size=params['batch_size'],
        features=7,
        T=params['T'],
        r=params['r'],
        sigma=params['vol'],
        lambda_skew=params['lambda_skew'],
        lambda_var=params['lambda_var']
    )
    print("Début de l'entraînement...")
    model.training(
        paths,
        np.full(params['n_sims'], 100),  # Strike K = 100
        params['alpha'],
        params['epochs']
    )
    return model

def apply_model(ticker, start_date, maturity_date, option_quantity, strike,
                rebalancing_freq=12,
                current_weights=None, cash_account=0,
                trained_model_path="utils/trained_model.keras"):
    """
    Applique le modèle entraîné pour calculer la stratégie de couverture.

    Les arguments :
      - ticker : le symbole du sous-jacent
      - start_date : date de début ("mm/jj/aaaa")
      - maturity_date : date de maturité ("mm/jj/aaaa")
      - option_quantity : quantité d'options (nombre)
      - strike : strike de l'option (nombre)
      - rebalancing_freq : fréquence de rebalancement (nombre de simulations, 12 par défaut)
      - current_weights : dictionnaire contenant les poids actuels (ex. {ticker: delta_initial})
      - cash_account : montant du cash dans le portefeuille de réplication
      - trained_model_path : chemin vers le modèle entraîné enregistré avec joblib

    La fonction récupère les données marché, calcule le temps restant T,
    génère des trajectoires via la méthode de Monte Carlo et prépare les inputs
    du modèle (les 7 features attendues à chaque instant).
    """
    if current_weights is None:
        current_weights = {ticker: 0.0}

    # Chargement du modèle via keras
    try:
      model = tf.keras.models.load_model("utils/trained_model.keras", safe_mode=False)
    except Exception as e:
        return {"error": f"Erreur de chargement: {str(e)}"}

    # Calcul du temps restant en années à partir des dates
    maturity_dt = datetime.strptime(maturity_date, "%m/%d/%Y").date()
    start_dt = datetime.strptime(start_date, "%m/%d/%Y").date()
    T = (maturity_dt - start_dt).days / 365.0
    if T <= 0:
        return {"error": "La maturité doit être dans le futur"}

    # Récupération des données marché
    try:
        stock = afficher_donnees_ticker(ticker)
        S0 = float(stock['Close'].iloc[-1])

        # On considère ici le strike tel que fourni (possiblement ATM)
        # strike est passé en argument

        # Récupérer le taux sans risque (exemple simplifié)
        treasury = afficher_donnees_ticker("^TNX")
        r = float(treasury['Close'].iloc[-1] / 100.0)


        # Estimation simplifiée de la volatilité (à remplacer par une estimation plus précise)
        sigma = 0.3

    except Exception as e:
        return {"error": f"Erreur lors de la récupération des données marché: {str(e)}"}

    # Générer les trajectoires de prix sur l'horizon T
    # Le modèle a été entraîné avec un nombre de timesteps défini par model.time_steps
    n_timesteps = model.time_steps
    paths = monte_carlo_paths(
        S_0=S0,
        time_to_expiry=T,
        sigma=sigma,
        drift=r,
        seed=42,
        n_sims=rebalancing_freq,  # On génère autant de trajectoires que la fréquence de rebalancement
        n_timesteps=n_timesteps - 1
    )
    
    # Mise à jour éventuelle des paramètres du modèle
    # model.r = r
    # model.sigma = sigma

    # Préparation des inputs initiaux pour le modèle
    # Ici, paths a la forme (time_steps, n_sims, 1) donc paths.shape[1] est valide
    delta_init = tf.constant(np.full((paths.shape[1],), current_weights.get(ticker, 0.0)), dtype=tf.float32)
    cash_init = tf.constant(np.full((paths.shape[1],), cash_account), dtype=tf.float32)

    # Calcul de la stratégie et du PnL
    try:
        pnl_tensor, decisions = model.calculate_hedging_pnl(
            tf.constant(paths, tf.float32),
            tf.constant(strike, tf.float32),
            delta_init=delta_init,
            cash_init=cash_init
        )
    except Exception as e:
        return {"error": f"Erreur lors du calcul de la stratégie: {str(e)}"}

    pnl = pnl_tensor.numpy()

    # Pour l'exemple, on calcule le delta moyen au premier instant et on le multiplie par la quantité d'options
    predicted_delta = np.mean(decisions[0].numpy()) * option_quantity

    # Le reporting du cash final est ici simplifié
    cash_final = cash_account
    pnl_mean = np.mean(pnl)
    print("cash :", cash_final)
    print("delta :", predicted_delta)

    return {
        "Quantité d'actif sous-jacents nécessaire": round(predicted_delta, 2),
        "Quantité d'actif sans risque nécessaire": round(cash_final, 2),
        # "details": {
        #     "strike": round(strike, 2),
        #     "T": round(T, 4),
        #     "risk_free_rate": round(r, 4),
        #     "volatility": round(sigma, 4),
        #     "pnl_mean": round(pnl_mean, 2),
        #     "all_predicted_deltas": decisions.numpy().tolist()
        # }
    }

# # ==================== EXÉCUTION PRINCIPALE ====================
# if __name__ == "__main__":
#     # Exemple d'utilisation de la fonction apply_model
#     ticker = "AAPL"

#     # Date de début et de maturité au format "mm/jj/aaaa"
#     today = date.today()
#     maturity_dt = today + timedelta(days=90)  # maturité dans 90 jours

#     print(f"Prédiction de la stratégie de couverture pour {ticker}...")
#     prediction = apply_model(
#         ticker=ticker,
#         start_date=today.strftime("%m/%d/%Y"),
#         maturity_date=maturity_dt.strftime("%m/%d/%Y"),
#         option_quantity=100,
#         strike=100,  # Exemple de strike fourni
#         rebalancing_freq=12,
#         current_weights={ticker: 50},  # Par exemple, un delta initial de 50
#         cash_account=5000,
#         trained_model_path="trained_model.joblib"
#     )

#     if "error" in prediction:
#         print(prediction["error"])
#     else:
#         print("\nRésultats de la prédiction:")
#         print(f"Delta optimal pour {ticker} : {prediction[ticker]}")
#         print(f"Cash account : {prediction['cash_account']}")
#         print("\nDétails :")
#         for key, value in prediction["details"].items():
#             if key != "all_predicted_deltas":
#                 print(f"{key} : {value}")

# if __name__ == "__main__":
#     trained_model = run_training()
#     trained_model.save("utils/trained_model.keras")



# # ==================== EXÉCUTION PRINCIPALE ====================
# if __name__ == "__main__":
#     # Exemple d'utilisation de la fonction apply_model
#     ticker = "AAPL"

#     # Date de début et de maturité au format "mm/jj/aaaa"
#     today = date.today()
#     maturity_dt = today + timedelta(days=90)  # maturité dans 90 jours

#     print(f"Prédiction de la stratégie de couverture pour {ticker}...")
#     prediction = apply_model(
#         ticker=ticker,
#         start_date=today.strftime("%m/%d/%Y"),
#         maturity_date=maturity_dt.strftime("%m/%d/%Y"),
#         option_quantity=100,
#         strike=100,  # Exemple de strike fourni
#         rebalancing_freq=12,
#         current_weights={ticker: 50},  # Par exemple, un delta initial de 50
#         cash_account=5000,
#         trained_model_path="trained_model.joblib"
#     )

#     if "error" in prediction:
#         print(prediction["error"])
#     else:
#         print("\nRésultats de la prédiction:")
#         print(f"Delta optimal pour {ticker} : {prediction[ticker]}")
#         print(f"Cash account : {prediction['cash_account']}")
#         print("\nDétails :")
#         for key, value in prediction["details"].items():
#             if key != "all_predicted_deltas":
#                 print(f"{key} : {value}")

# if __name__ == "__main__":
#     trained_model = run_training()
#     trained_model.save("utils/trained_model.keras")

