# REINFORCE Algorithm for CartPole-v1

This project implements the REINFORCE algorithm and REINFORCE with baseline for the CartPole-v1 environment from OpenAI Gym.

![Recording 2024-07-25 at 12 51 18](https://github.com/user-attachments/assets/3c58356e-0066-44fa-95c5-83d94fef9e9b)

## Description

This project explores policy gradient methods in reinforcement learning, specifically:

1. REINFORCE algorithm
2. REINFORCE with baseline

Both algorithms are applied to the CartPole-v1 problem, where the goal is to keep a pole balanced on a moving cart.

## Features

- Implementation of the REINFORCE algorithm
- Implementation of REINFORCE with baseline
- Visualization of learning progress
- Comparison of both algorithms' performance

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenAI Gym

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/reinforce-cartpole.git
cd reinforce-cartpole
pip install numpy matplotlib gym
```

## Usage

Run the main script to execute both algorithms and see the comparison:

```bash
python reinforce_cartpole.py
```

## Implementation Details

- `policy()`: Implements softmax action selection
- `generate_episode()`: Generates an episode using the current policy
- `REINFORCE()`: Implements the REINFORCE algorithm
- `REINFORCE_with_baseline()`: Implements the REINFORCE with baseline algorithm
- `main()`: Runs both algorithms and plots the results

## Results

The script outputs:
- Episode-by-episode updates on episode length and mean length
- A notification when the environment is considered solved (mean length >= 495)
- A plot comparing the learning curves of REINFORCE and REINFORCE with baseline
  ![image](https://github.com/user-attachments/assets/82d327e8-41a6-477f-94a3-e1efefb0cb7b)


## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project uses the CartPole-v1 environment from OpenAI Gym and implements reinforcement learning algorithms as described in the literature.
```

This README provides an overview of your REINFORCE implementation for the CartPole-v1 problem, including what it does, how to set it up, how to use it, and what results to expect. You may want to customize it further based on any specific details or additional information you'd like to include.
