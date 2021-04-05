import gym
from RL_brain import DeepQNetwork

# 建立环境
env = gym.make('CartPole-v0')
print(env.action_space)            # 打印所有的动作
print(env.observation_space)       # observation的数量
print(env.observation_space.high)  # observation的最高值
print(env.observation_space.low)   # observation的最低值

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0

for i_episode in range(1000):

    observation = env.reset()

    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation_

        # the smaller theta and closer to center the better

        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()
            print('episode: ', i_episode,
                  'cost: ', round(RL.cost, 4),
                  'epsilon: ', round(RL.epsilon, 2))

        if done:
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()