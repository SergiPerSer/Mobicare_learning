import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def test_agent():
    print("Cargando el entorno y el agente entrenado...")
    
    # 1. Recrear el entorno de Gymnasium
    # 'render_mode="human"' abrirá la ventana gráfica para que lo veas correr
    env = gym.make("HalfCheetah-v5", render_mode="human")
    env = DummyVecEnv([lambda: env])
    
    # 2. CARGAR LA NORMALIZACIÓN (CRÍTICO)
    # La red neuronal se entrenó con estados normalizados. Si le pasas los estados
    # nativos de MuJoCo sin normalizar, el robot hará movimientos caóticos.
    env = VecNormalize.load("vec_normalize.pkl", env)
    
    # En testeo NO queremos que siga actualizando las medias de normalización
    env.training = False  
    # En testeo NO queremos que altere las recompensas visuales
    env.norm_reward = False  

    # 3. Cargar el modelo neuronal entrenado
    model = PPO.load("ppo_half_cheetah", env=env)
    print("¡Modelo cargado con éxito! Iniciando simulación...")

    num_episodes = 5
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 4. Predicción Determinista
            # 'deterministic=True' apaga la aleatoriedad/exploración.
            # El actor elegirá la acción matemáticamente óptima (la media de la distribución).
            action, _states = model.predict(obs, deterministic=True)
            
            # 5. Ejecutar en el entorno
            obs, reward, done, info = env.step(action)
            
            # Al usar VecNormalize, el wrapper devuelve la recompensa original 
            # sin alterar en 'info' para que podamos medir el rendimiento real
            episode_reward += info[0].get('episode', {}).get('r', reward[0])

        print(f"Episodio {episode + 1} terminado. Recompensa real lograda: {episode_reward:.2f}")

    env.close()
    print("Simulación finalizada.")

if __name__ == "__main__":
    test_agent()