"""Solving the CartPole-v1 environment with the DDQN method """

import gym
from foo import *

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

# PARAMETERS
# Environment
number_of_episodes = 2500
render_bool = False
penalize_bad_states = (True, -100)

# DDQN solver
policy_from_online = True  # if True -> epsilon greedy policy is derived from online network, if False -> target network
param_dict = {'gamma': 1,
              'batch_size': 64,
              'HL_1_size': 24,
              'HL_2_size': 24,
              'memory_size': 2000,
              'min_memory_size': 100,
              'learning_rate_adam': 0.001,
              'epsilon_all': {'initial': 1, 'decay': 999 / 1000, 'min': 0.01}}

# Algorithm termination
number_of_consecutive_episodes = 5
threshold_average = 490

# INITIALIZATION
my_solver = DDQN_Solver(**param_dict)
env = gym.make('CartPole-v1')
solved = False
results = []

# MAIN ALGORITHM
for e in range(number_of_episodes):
    s_old = env.reset()
    done = False
    t = 0
    while not done:
        t += 1

        if render_bool:
            env.render()

        # Choose action
        a = my_solver.choose_action(s_old, policy_from_online)

        # Take action
        s_new, r, done, _ = env.step(a)

        # Penalize
        if penalize_bad_states[0] and done and t < 500:
            r = penalize_bad_states[1]

        # Memorize
        my_solver.memorize(s_old, a, r, s_new, done and t < 500)

        # Train
        if not solved:
            my_solver.train()

        # Move forward
        s_old = s_new

    # Append results and check if solved
    results.append(t)

    if np.mean(results[-min(number_of_consecutive_episodes, e):]) > threshold_average:
        solved = True
        print('Stable solution found - no more training!!!!!!!!!!!!')
    else:
        solved = False

    print('The episode %s lasted for %s steps' % (e, t))
    my_solver.feed_most_recent_score(t)
    my_solver.update_target_network()
