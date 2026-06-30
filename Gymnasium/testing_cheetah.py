from matplotlib import pyplot as plt
import time
import pickle
from collections import defaultdict
import gymnasium as gym
import numpy as np

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

def test_agent_visual(agent, num_episodes=3):
    """Prueba al agente visualizando su comportamiento en tiempo real."""
    print("\nIniciando visualización en tiempo real...")
    
    # Creación de un entorno específico para renderizar
    # Usamos render_mode="human" para que abra la ventana gráfica
    visual_env = gym.make("HalfCheetah-v5")
    
    # Desactivamos la exploración para ver su mejor comportamiento aprendido
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  
    rewards = []
    for episode in range(num_episodes):
        obs, info = visual_env.reset()
        episode_reward = 0
        done = False
        
        print(f"Jugando episodio {episode + 1}...")

        while not done:
            # 1. El agente elige la acción según lo aprendido
            action_idx = agent.get_action(obs, evaluation=True)
            continuous_action = agent.action_mapping[action_idx]
            
            # 2. Se ejecuta la acción en el entorno visual
            obs, reward, terminated, truncated, info = visual_env.step(continuous_action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
            
            # 3. Controlar los FPS (opcional)
            # MuJoCo suele encargarse del framerate, pero si va "demasiado rápido", 
            # puedes descomentar la siguiente línea para ralentizarlo a ~60 FPS:
            # time.sleep(1/60)

        print(f"Episodio {episode + 1} terminado. Recompensa: {episode_reward:.2f}")

    # Restauramos el epsilon original del agente y cerramos la ventana
    agent.epsilon = old_epsilon
    visual_env.close()
    return rewards

# 1. Instancias el entorno y el agente desde cero
env = gym.make("HalfCheetah-v5")
agent = HalfCheetahDiscretized(env=env, learning_rate=0.05, initial_epsilon=0.0, epsilon_decay=0.0, final_epsilon=0.0)
num_episodes = 50
list_num_episodes = list(range(num_episodes))

# 2. Cargas los pesos guardados
filename = "half_cheetah_q_table.pkl"
with open(filename, "rb") as f:
    raw_dict = pickle.load(f)

# 3. Asignas los valores cargados reconstruyendo el defaultdict
agent.q_values = defaultdict(lambda: np.zeros(agent.n_actions), raw_dict)
print(type(agent.q_values))
print("¡Pesos cargados con éxito!")

# 4. Ejecutas el test visual directamente
rewards = test_agent_visual(agent, num_episodes=num_episodes)
print(rewards)
fig, axs = plt.subplots(ncols=1, figsize=(12, 5))
axs.set_title("Recompensa por episodio durante la prueba visual")
axs.plot(list_num_episodes, rewards, label="Recompensa por paso")
axs.set_xlabel("Paso")
axs.set_ylabel("Recompensa")
plt.xticks(np.arange(0, num_episodes+1, 1))
axs.legend()
plt.show()