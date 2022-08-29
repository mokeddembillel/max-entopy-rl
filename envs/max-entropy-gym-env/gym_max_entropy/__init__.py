from gym.envs.registration import register

register(
    id='max-entropy-v0',
    entry_point='gym_max_entropy.envs:MaxEntropyEnv',
)
