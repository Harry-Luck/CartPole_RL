""" Helper functions and classes"""

import random
from collections import deque

import numpy as np
import tensorflow as tf


class DDQN_Solver():
    def __init__(self, gamma, memory_size, min_memory_size, learning_rate_adam, HL_1_size, HL_2_size, batch_size,
                 epsilon_all):
        # Load important parameters
        self.gamma = gamma  # discount rate
        self.memory_size = memory_size  # size of the memory buffer
        self.HL_1_size = HL_1_size  # number of nodes in the first hidden layer
        self.HL_2_size = HL_2_size  # number of nodes in the second hidden layer
        self.learning_rate_adam = learning_rate_adam  # learning rate for Adam optimizer
        self.batch_size = batch_size  # batch size for training
        self.min_memory_size = max(self.batch_size, min_memory_size)  # minimal memory size before we start training
        self.epsilon_initial = epsilon_all['initial']  # epsilon-greedy policy - initial value
        self.epsilon_decay = epsilon_all['decay']  # decay after each time step
        self.epsilon_min = epsilon_all['min']  # minimal value of epsilon

        # Initialize attributes
        self.replay_buffer = deque()
        self.global_step = 0  # counts the number of times we have trained our model = sum_{episode} timesteps_episode
        self.most_recent_score = tf.Variable(0, dtype=tf.int32)  # most recent score - visualized in tensorboard
        tf.summary.scalar('most_recent_score', self.most_recent_score)
        self.epsilon = self.epsilon_initial  # we initialize our epsilon
        self.epsilon_tensor = tf.Variable(self.epsilon, dtype=tf.float32)  # for tensorboard
        tf.summary.scalar('epsilon', self.epsilon_tensor)

        # Build online and target networks
        self.__build_Q_net()

        # Merge summaries
        self.overall_summary = tf.summary.merge_all()

        # Initialize variables and summary writer
        self.__init_session()
        self.summary_writer = tf.summary.FileWriter('/Users/jankrepl/Desktop/ddqn_summaries', self.session.graph)

        # Synchronize Online and Target Network
        self.update_target_network()

    def __build_Q_net(self):
        # Placeholders
        self.input_state = tf.placeholder(tf.float32, [None, 4], 'Input_state')
        self.input_action = tf.placeholder(tf.float32, [None, 2], 'Input_action')
        self.target = tf.placeholder(tf.float32, [None], 'Target')

        # Variables - Online Network
        self.W1_on = tf.Variable(tf.truncated_normal([4, self.HL_1_size]))
        self.b1_on = tf.Variable(tf.constant(0.1, shape=[self.HL_1_size]))
        self.HL_1_on = tf.nn.relu(tf.matmul(self.input_state, self.W1_on) + self.b1_on, )
        self.W2_on = tf.Variable(tf.truncated_normal([self.HL_1_size, self.HL_2_size]))
        self.b2_on = tf.Variable(tf.constant(0.1, shape=[self.HL_2_size]))
        self.HL_2_on = tf.nn.relu(tf.matmul(self.HL_1_on, self.W2_on) + self.b2_on)
        self.W3_on = tf.Variable(tf.truncated_normal([self.HL_1_size, 2]))
        self.b3_on = tf.Variable(tf.constant(0.1, shape=[2]))
        self.Q_ohr_on = tf.matmul(self.HL_2_on, self.W3_on) + self.b3_on

        # Variables - Target Network
        self.W1_tn = tf.Variable(tf.truncated_normal([4, self.HL_1_size]))
        self.b1_tn = tf.Variable(tf.constant(0.1, shape=[self.HL_1_size]))
        self.HL_1_tn = tf.nn.relu(tf.matmul(self.input_state, self.W1_tn) + self.b1_tn, )
        self.W2_tn = tf.Variable(tf.truncated_normal([self.HL_1_size, self.HL_2_size]))
        self.b2_tn = tf.Variable(tf.constant(0.1, shape=[self.HL_2_size]))
        self.HL_2_tn = tf.nn.relu(tf.matmul(self.HL_1_tn, self.W2_tn) + self.b2_tn)
        self.W3_tn = tf.Variable(tf.truncated_normal([self.HL_1_size, 2]))
        self.b3_tn = tf.Variable(tf.constant(0.1, shape=[2]))
        self.Q_ohr_tn = tf.matmul(self.HL_2_tn, self.W3_tn) + self.b3_tn

        # Q function and loss
        self.Q_on = tf.reduce_sum(tf.multiply(self.Q_ohr_on, self.input_action), reduction_indices=1)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.target - self.Q_on), name='loss')
        tf.summary.scalar("loss", self.loss)

        # Train operations
        self.train_op = tf.train.AdamOptimizer(self.learning_rate_adam).minimize(self.loss)

    def __init_session(self):
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def train(self):
        """ Samples a minibatch from the memory and based on it trains the network
        """
        self.global_step += 1

        # Just make sure that it breaks at the beginning when memory is not big enough < min_memory_size
        if len(self.replay_buffer) < self.min_memory_size:
            print('The memory is too small to train')
            return

        # Sample from memory

        mini_batch = random.sample(self.replay_buffer, self.batch_size)  # sampling without replacement
        batch_s_old = [element[0] for element in mini_batch]
        batch_a = [element[1] for element in mini_batch]
        batch_r = [element[2] for element in mini_batch]
        batch_s_new = [element[3] for element in mini_batch]
        batch_d = [element[4] for element in mini_batch]

        # Generating targets
        Q_new_on = self.Q_ohr_on.eval(feed_dict={self.input_state: batch_s_new})  # forward pass - ONLINE NETWORK
        Q_new_tn = self.Q_ohr_tn.eval(feed_dict={self.input_state: batch_s_new})  # forward pass - TARGET NETWORK
        argmax = np.argmax(Q_new_on, axis=1)
        Q_target = np.reshape(np.array([Q_new_tn[i][argmax[i]] for i in range(self.batch_size)]),
                              newshape=self.batch_size)

        # Generate targets
        batch_target = []
        for i in range(self.batch_size):
            if batch_d[i]:
                # The new state is the end game - its target Q value is definitely 0
                batch_target.append(batch_r[i])
            else:
                batch_target.append(batch_r[i] + self.gamma * Q_target[i])

        # Train and write summary
        _, summary_str = self.session.run([self.train_op, self.overall_summary], feed_dict={
            self.target: batch_target,
            self.input_state: batch_s_old,
            self.input_action: batch_a,
        })
        self.summary_writer.add_summary(summary_str, self.global_step)

        # Decay epsilon
        self.__decay_epsilon()

    def update_target_network(self):
        # We simply copy online network values into the target network
        ops_list = []
        ops_list.append(self.W1_tn.assign(self.W1_on))
        ops_list.append(self.b1_tn.assign(self.b1_on))
        ops_list.append(self.W2_tn.assign(self.W2_on))
        ops_list.append(self.b2_tn.assign(self.b2_on))
        ops_list.append(self.W3_tn.assign(self.W3_on))
        ops_list.append(self.b3_tn.assign(self.b3_on))

        self.session.run(ops_list)


    def __decay_epsilon(self, printme=False):
        """ Decays epsilon based on epsilon_decay

        :param printme: print current value of epsilon
        :type printme: bool
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if printme:
            print('The current value of epsilon is ' + str(self.epsilon))

    def memorize(self, s_old, action, reward, s_new, done):
        """ Inserts the most recent SARS and done into the memory - a is saved in the one hot representation

        :param s_old: old state
        :type s_old: ndarray
        :param action: 0 or 1
        :type action: int
        :param reward: reward
        :type reward: int
        :param s_new: new state
        :type s_new: ndarray
        :param done: is finished
        :type done: bool
        """
        # Convert action to one hot representation
        a_ohr = np.zeros(2)
        a_ohr[action] = 1

        # Make sure they have the right dimensions
        s_old.shape = (4,)
        a_ohr.shape = (2,)
        s_new.shape = (4,)

        # Add into replay_buffer and if necessary pop oldest memory
        memory_element = tuple((s_old, a_ohr, reward, s_new, done))
        self.replay_buffer.append(memory_element)
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def choose_action(self, s_old, policy_from_online):
        """ Epsilon greedy policy
        
        :param s_old: old observation
        :type s_old: ndarray
        :param policy_from_online: if True, online network as the policy, if False, target network as the policy
        :type policy_from_online: bool
        :return: 0 or 1
        :rtype: int
        """
        # just a forward pass and max
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.choice([0, 1], 1)[0]
        else:
            # Exploit
            s_old.shape = (1, 4)  # make sure it matches the placeholder shape (None, 4)
            if policy_from_online:
                return np.argmax(self.Q_ohr_on.eval(feed_dict={self.input_state: s_old}))
            else:
                return np.argmax(self.Q_ohr_tn.eval(feed_dict={self.input_state: s_old}))

    def feed_most_recent_score(self, score):
        """ Feeds the most recent score into our solver class so that we can visualize it in tensorboard together
        with epsilon

        :param score: most recent score
        :type score: int
        """
        op1 = self.most_recent_score.assign(score)
        op2 = self.epsilon_tensor.assign(self.epsilon)
        self.session.run([op1, op2])
