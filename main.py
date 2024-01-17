import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt


class FrozenLakeAgent:
    def __init__(self, env, learning_rate=0.9, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.0001):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng()

     #Aceasta este metoda de alegere a acțiunilor pe baza stării curente.În cazul în care un număr aleatoriu este mai mic decât epsilon,
     # se efectuează o acțiune aleatorie; în caz contrar, se alege cea mai bună acțiune pe baza tabelului Q.
    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state, :])

    # Aceasta metoda actualizeaza tabelul Q cu noile valori calculate pe baza recompensei primite pentru actiunea anterioara si estimarile Q
    # ale starii urmatoare. Acesta aplica regula de invatare Q-learning.
    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.learning_rate * (
                reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
        )

    #Actualizeaza epsilon prin reducerea acestuia cu rata de descrestere. Daca epsilon devine zero, rata de invatare este redusa la 0.0001
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
        if self.epsilon == 0:
            self.learning_rate = 0.0001


def load_q_table(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_q_table(q_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

#este o functie care antreneaza agentul pe baza unui numar specificat de episoade. Aceasta initializeaza agentul,
# ruleaza episoadele, actualizeaza tabelul Q si returneaza recompensele obtinute in fiecare episod.
def train_agent(episodes, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    agent = FrozenLakeAgent(env)

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0]
        terminated, truncated = False, False

        while not terminated and not truncated:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            rewards_per_episode[episode] = reward

        agent.update_epsilon()

    env.close()
    save_q_table(agent.q_table, "frozen_lake8x8.pkl")

    return rewards_per_episode


def plot_training_metrics(rewards_per_episode):
    moving_avg_rewards = np.convolve(rewards_per_episode, np.ones(100) / 100, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, label='Reward per Episode')
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Metrics')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    rewards = train_agent(1000, render=True)
    plot_training_metrics(rewards)
