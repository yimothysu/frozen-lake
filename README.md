# Frozen Lake

Tabular Q-learning on OpenAI Gym's Frozen Lake.
Samples from the observation space, updating the Q-value of each state/action pair.
Starts by exploring the observation space through taking random actions, then over time exploits the known Q-values by taking the argmax at the current state.
