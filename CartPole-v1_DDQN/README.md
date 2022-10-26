# CartPole_v1_DDQN
TensorFlow implementation of a Double Deep Q Network (**DDQN**) solving the problem of balancing a pole on cart.
Environment provided by the OpenAI gym.

## Code

### Running
```
python Main.py
```

### Dependencies
*  collections
*  gym
*  numpy
*  random
*  tensorflow

## Detailed Description
### Problem Statement and Environment
The same environment as CartPole-v0 but now the average number of timesteps is required to be 475.

---
### DDQN
DDQN tries to eliminate the inherent problem of DQN - **overestimation**. The way it does it is through using a different 
target value than DQN. Namely its the following:

![screen shot 2017-09-19 at 1 31 32 am](https://user-images.githubusercontent.com/18519371/30569222-ee9b217c-9cd9-11e7-8bb1-77ddb85f2f39.png)


* to choose the action we use the **online** network weights
* to evaluate the Q function we use the **target** network weights


## Results and discussion
DDQN seems to find the right solution irrespective of the initialization. See below an evolution of the score for one run:
![screen shot 2017-09-19 at 1 43 23 am](https://user-images.githubusercontent.com/18519371/30569493-84f81002-9cdb-11e7-9d3a-e699c351f912.png)


## Resources and links
* https://arxiv.org/abs/1509.06461 - Original paper from Hado van Hasselt
* https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/2-double-dqn - Similar algorithm in Keras and same hyperparameters


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
