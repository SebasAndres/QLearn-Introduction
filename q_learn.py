import numpy as np
import matplotlib.pyplot as plt
import time

class Q_Model:

    def __init__(self, env_train, env_test):
        # Inicializa la Q-Tabla
        self.env_train = env_train
        self.env_test = env_test
        # Se supone que son el mismo entorno, solo varia el render mode
        self.STATES = env_train.observation_space.n
        self.ACTIONS = env_test.action_space.n
        self.Q = np.zeros((self.STATES, self.ACTIONS)) 

    def train_q_table(self, EPISODES, MAX_STEPS, LEARNING_RATE, GAMMA, epsilon, RENDER=False):
        # Entrena la Q-Tabla con los siguientes hiperparametros:

        # EPISODES: Cantidad de episodios
        # MAX_STEPS: Cantidad de pasos máximos por episodio
        # LEARNING_RATE: Tasa de aprendizaje
        # GAMMA: Factor de descuento
        # RENDER: Si se desea ver el entrenamiento
        # epsilon: Probabilidad de exploración

        rewards = []
        for episode in range(EPISODES):

            # estado inicial
            state, _ = self.env_train.reset()

            for _ in range(MAX_STEPS):
            
                if RENDER:
                    # muestro en pantalla
                    self.env_train.render()

                # elijo una accion, donde epsilon es la probabilidad de exploracion
                if np.random.uniform(0, 1) < epsilon:
                    # accion aleatoria
                    action = self.env_train.action_space.sample()  
                else:
                    # accion segun la tabla Q
                    action = np.argmax(self.Q[state, :])

                # ejecuto la accion y avanzo al siguiente estado
                next_state, reward, terminated, truncated, _ = self.env_train.step(action)
                done = terminated or truncated

                # actualizo la tabla Q
                self.Q[state, action] += LEARNING_RATE * (reward + GAMMA * np.max(self.Q[next_state, :]) - self.Q[state, action])

                # actualizo el estado
                state = next_state
                
                if done: 
                    rewards.append(reward)
                    epsilon -= 0.001
                    break  # reached goal
            
        return rewards
    
    def plot_rewards(self, rewards):
        # Muestro el promedio de rewards
        mean_reward = np.mean(rewards)
        print("Mean reward: ", mean_reward)

        avg_rewards = []
        get_average = lambda x: sum(x)/len(x)

        for i in range(0, len(rewards), 100):
            avg_rewards.append(get_average(rewards[i:i+100])) 
        
        plt.plot(avg_rewards)
        plt.ylabel('average reward')
        plt.xlabel('episodes (100\'s)')
        plt.title("Average Reward vs Episodes")
        plt.show()
    
        return mean_reward

    def play(self, SLEEP=0.5):
        state, _ = self.env_test.reset()
        done = False

        while not done:
            action = np.argmax(self.Q[state,:])
            state, reward, terminated, truncated, info = self.env_test.step(action)
            done = terminated or truncated
            self.env_test.render()
            time.sleep(SLEEP)