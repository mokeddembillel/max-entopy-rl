# max-entopy-rl

Our approach: Stein Soft Actor-Critic (STAC), a model-free RL algorithm that aims at learning policies that can represent arbitrary action distributions without compromising efficiency. STAC uses Stein Variational Gradient Descent (SVGD) as the underlying policy to generate action samples from distributions represented using EBMs, and adopts the policy iteration procedure like SAC that maintains sample efficiency.
