from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import pickle

class HalfCheetahDiscretized:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        bins_per_dim: int = 3  # Número de divisiones por cada dimensión del estado
    ):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Parámetros de exploración
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Discretización del ESPACIO DE ACCIONES
        # HalfCheetah tiene 6 motores continuos. Vamos a simplificarlo a un número discreto de acciones.
        # Por ejemplo: cada motor puede estar en -1.0, 0.0 o 1.0. 
        # Para no tener 3^6 = 729 acciones (demasiadas para Q-learning), definiremos un conjunto pequeño de acciones combinadas:
        self.action_mapping = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),       # Quieto
            np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),       # Adelante suave
            np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4]), # Atrás suave
            np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5]),   # Balanceo sutil A
            np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),   # Balanceo sutil B
        ]
        self.n_actions = len(self.action_mapping)

        # Q-table basada en el número de acciones discretas que creamos
        self.q_values = defaultdict(lambda: np.zeros(self.n_actions))

        # Discretización del ESPACIO DE ESTADOS
        # HalfCheetah-v5 tiene 17 dimensiones continuas. 
        self.bins_per_dim = bins_per_dim
        # Acotamos los límites para la discretización (ya que algunos van de -inf a inf)
        self.low_bounds = np.array([-3.0] * 17)
        self.high_bounds = np.array([3.0] * 17)

        self.training_error = []

    def discretize_state(self, obs: np.ndarray) -> tuple:
        # Seleccionamos solo las primeras, por ejemplo, 6 o 8 dimensiones críticas
        # Evita que el exponente de la combinación explote
        reduced_obs = obs[:6] 
        
        clipped_obs = np.clip(reduced_obs, self.low_bounds[:6], self.high_bounds[:6])
        ratios = (clipped_obs - self.low_bounds[:6]) / (self.high_bounds[:6] - self.low_bounds[:6])
        discretized = (ratios * (self.bins_per_dim - 1)).astype(int)
        
        return tuple(discretized)

    def get_action(self, obs: np.ndarray, evaluation: bool = False) -> int:
        """Elige una acción usando epsilon-greedy (devuelve el índice discreto)."""
        state_key = self.discretize_state(obs)
        
        # En evaluación no exploramos
        if not evaluation and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return int(np.argmax(self.q_values[state_key]))

    def update(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        state_key = self.discretize_state(obs)
        next_state_key = self.discretize_state(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_state_key])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state_key][action_idx]

        self.q_values[state_key][action_idx] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



# --- CONFIGURACIÓN Y ENTRENAMIENTO ---

# Reducimos los episodios porque Q-learning tabular sufre mucho con la maldición de la dimensionalidad
learning_rate = 1     
n_episodes = 5000        # 100k es demasiado para una tabla en este entorno continuo
start_epsilon = 1.0         
epsilon_decay = start_epsilon / (n_episodes * 0.8)  # Decaer durante el primer 80% de episodios
final_epsilon = 0.05         

# Crear entorno HalfCheetah (Asegúrate de tener v4 o v5 instalado, típicamente 'HalfCheetah-v4' o 'HalfCheetah-v5')
env = gym.make("HalfCheetah-v5") 
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = HalfCheetahDiscretized(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    bins_per_dim=3 # 3 bins por 17 dimensiones = 3^17 estados posibles. ¡Ya es un número enorme!
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        # 1. Obtener índice de la acción discreta
        action_idx = agent.get_action(obs)
        
        # 2. Convertir el índice a la acción continua real que entiende HalfCheetah
        continuous_action = agent.action_mapping[action_idx]

        # 3. Ejecutar en el entorno
        next_obs, reward, terminated, truncated, info = env.step(continuous_action)

        # 4. Actualizar la tabla usando el índice discreto
        agent.update(obs, action_idx, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

# --- GUARDAR MODELO ---

filename = "half_cheetah_q_table.pkl"
with open(filename,"wb") as f:
    pickle.dump(dict(agent.q_values), f)
print(f"Modelo guardado en {filename}")

# --- PLOTTING DE RESULTADOS ---
# Graficar la recompensa promedio por episodio

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()
