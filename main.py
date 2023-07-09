import gym
from q_learn import Q_Model

env_train = gym.make('FrozenLake-v1')
env_test = gym.make('FrozenLake-v1', render_mode='human')

model = Q_Model(env_train, env_test)
rewards = model.train_q_table(EPISODES=20000,
                              MAX_STEPS=100,
                              LEARNING_RATE=0.81,
                              GAMMA=0.99,
                              RENDER=False,
                              epsilon=0.9)

print("Model Summary")
model.plot_rewards(rewards)
model.play()