import gym
import random
import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q);
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-Learning Demo')
    parser.add_argument('-n', '--nepisodes', type=int, default=1500,
                        help='number of episodes to train agent')
    parser.add_argument('-p', '--plotfreq', type=int, default=150,
                        help='frequency of rendering game')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')

    plot_freq = args.plotfreq
    n_episodes = args.nepisodes

    n_bins = 10
    n_bins_angle = 10
    goal_average_steps = 195
    max_number_of_steps = 300

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    # number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    # q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)

    for i_episode in range(n_episodes):
        observation = env.reset()

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

        # obtain initial state
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])

        for t in range(max_number_of_steps):
            if i_episode % plot_freq == 0:
                env.render()

            # pick an action based on the current state
            action = qlearn.chooseAction(state)

            # execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

            # obtain next state
            nextState = build_state([to_bin(cart_position, cart_position_bins),
                                     to_bin(pole_angle, pole_angle_bins),
                                     to_bin(cart_velocity, cart_velocity_bins),
                                     to_bin(angle_rate_of_change, angle_rate_bins)])

            # perform q-learning
            if not done:
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                reward = -200
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                print(f'Episode: {i_episode}, Duration: {int(t+1)}')
                break

    env.close()
    print(f'Overall score: {last_time_steps.mean()}')


    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    last_time_smooth = smooth(last_time_steps, 20)

    plt.figure()
    plt.plot(last_time_steps, label='Q-Learning')
    plt.plot(last_time_smooth, label='Q-Learning Smooth')
    plt.ylabel('Episode Reward')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()
