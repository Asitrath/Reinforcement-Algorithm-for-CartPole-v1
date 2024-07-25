import gym
import numpy as np
import matplotlib.pyplot as plt

def policy(state, theta):
    """ Return probabilities for actions under softmax action selection """
    h = np.dot(state, theta)
    exp_h = np.exp(h - np.max(h))  # Subtract max for numerical stability
    return exp_h / np.sum(exp_h)

def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states,
        the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)
    return states, rewards, actions

def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    alpha = 0.01  # learning rate
    gamma = 0.99  # discount factor
    episode_lengths = []
    mean_lengths = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        episode_length = len(states)
        episode_lengths.append(episode_length)
        if len(episode_lengths) > 100:
            episode_lengths.pop(0)
        mean_length = np.mean(episode_lengths)
        mean_lengths.append(mean_length)

        print(f"Episode: {e}, Length: {episode_length}, Mean Length: {mean_length:.2f}")

        # Implement the REINFORCE algorithm
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            action = actions[t]

            # Compute gradient
            probs = policy(state, theta)
            grad = np.outer(state, probs)
            grad[:, action] -= state

            # Update policy parameters
            theta += alpha * G * grad

        if mean_length >= 495:
            print(f"Solved in {e} episodes!")
            break

    return mean_lengths

def REINFORCE_with_baseline(env):
    theta = np.random.rand(4, 2)  # policy parameters
    w = np.random.rand(4)  # baseline parameters
    alpha = 0.01  # learning rate for policy
    beta = 0.01  # learning rate for baseline
    gamma = 0.99  # discount factor
    episode_lengths = []
    mean_lengths = []

    for e in range(10000):
        env.render()
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        episode_length = len(states)
        episode_lengths.append(episode_length)
        if len(episode_lengths) > 100:
            episode_lengths.pop(0)
        mean_length = np.mean(episode_lengths)
        mean_lengths.append(mean_length)

        print(f"Episode: {e}, Length: {episode_length}, Mean Length: {mean_length:.2f}")

        # Implement the REINFORCE with baseline algorithm
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            action = actions[t]

            # Calculate baseline value
            V = np.dot(w, state)

            # Compute gradients
            probs = policy(state, theta)
            policy_grad = np.outer(state, probs)
            policy_grad[:, action] -= state

            # Update policy parameters
            theta += alpha * (G - V) * policy_grad

            # Update baseline parameters
            w += beta * (G - V) * state

        if mean_length >= 495:
            print(f"Solved in {e} episodes!")
            break

    return mean_lengths

def main():
    env = gym.make('CartPole-v1')
    
    reinforce_results = REINFORCE(env)
    reinforce_baseline_results = REINFORCE_with_baseline(env)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(reinforce_results)), reinforce_results, label='REINFORCE')
    plt.plot(range(len(reinforce_baseline_results)), reinforce_baseline_results, label='REINFORCE with Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Length (last 100 episodes)')
    plt.title('REINFORCE vs REINFORCE with Baseline on CartPole-v1')
    plt.legend()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()