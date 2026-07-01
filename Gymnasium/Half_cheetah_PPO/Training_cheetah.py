import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- CONFIGURACIÓN DE HIPERPARÁMETROS ---
APRENDER_PASOS = 1_000_000
TASA_APRENDIZAJE = 3e-4       # Prueba: 1e-4 (más lento/estable) o 5e-4 (más rápido)
BUFFER_PASOS = 4096           # Prueba: 2048 o 4096 (mejor para dinámicas complejas)
LOTE_OPTIMIZACION = 128       # Prueba: 64 o 128
LIMITE_RECORTE = 10.0

# 1. Configurar entorno
env = gym.make("HalfCheetah-v5")
env = DummyVecEnv([lambda: env])
env = VecNormalize(
    env, 
    norm_obs=True, 
    norm_reward=True, 
    clip_obs=LIMITE_RECORTE
)

# 2. Instanciar el agente con las variables configuradas
model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=TASA_APRENDIZAJE, 
    n_steps=BUFFER_PASOS, 
    batch_size=LOTE_OPTIMIZACION, 
    verbose=1
)

# 3. Entrenar
model.learn(total_timesteps=APRENDER_PASOS)

# 4. Guardar
model.save("ppo_half_cheetah_custom")
env.save("vec_normalize_custom.pkl")