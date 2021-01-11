import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import policy_saver
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from scipy.integrate import solve_ivp
from tf_agents.trajectories import time_step as ts
import os


# SHIP ENVIRONMENT CLASS
# starts here


class ship_environment(py_environment.PyEnvironment):

    def __init__(self):
        self.yaw_rate = 0
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int64, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=[-20,-np.pi,0], maximum=[20,np.pi,20], name='observation')
        self.state = np.array([0, 0, 0])
        self.episode_ended = False
        self.counter = 0
        self.random_x=0
        self.random_y=0
        self.distance=0
        self.coord_x = [0]
        self.coord_y = [0]


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action_no):

        action_num = action_no
        action_set = [-0.2, 0, 0.2]
        epsilon = np.maximum(1 - (self.counter / 9000), 0)

        if (np.random.random() < epsilon):
            action_num = np.random.randint(0, 3)

        delta = action_set[action_num]
        obs = self.state

        # SOLVING NOMOTO'S EQUATION
        K = 2.5
        T = 0.5
        u = 1

        yinit = [obs[2], self.yaw_rate, obs[0], obs[1]]
        tspan = (0, 0.4)

        def f(t, y):
            dydt = [y[1], (K * delta - y[1]) / T, np.cos(y[0]) * u, np.sin(y[0]) * u]
            return dydt

        solpsi = solve_ivp(lambda t, y: f(t, y), [tspan[0], tspan[-1]], yinit, t_eval=tspan)
        obs_new = [0] * 3
        rad = solpsi.y[0][-1]
        obs_new[0] = solpsi.y[2][-1]
        obs_new[1] = solpsi.y[3][-1]
        self.yaw_rate = solpsi.y[1][-1]

        # LIMIT psi in range 0 to 2pi

        while (rad < 0):
            rad = rad + 2 * np.pi

        obs_new[2] = rad % (2 * np.pi)

        self.state = np.array([obs_new[0], obs_new[1], obs_new[2]])
        x = obs_new[0]
        y = obs_new[1]
        self.coord_x.append(x)
        self.coord_y.append(y)
        psi = obs_new[2]

        # REWARD FUNCTIONS

        x_init = 0
        y_init = 0
        x_goal = self.random_x
        y_goal = self.random_y


        psip = np.arctan2(y_goal - y_init, x_goal - x_init)

        slope_WP = np.tan(psip)

        distance = ((x - x_goal) ** 2 + (y - y_goal) ** 2) ** 0.5

        side_track_error = (y - slope_WP * x - (y_init - slope_WP * x_init)) / (1 + slope_WP ** 2) ** 0.5

        if psi> np.pi:
            psi=psi-2*np.pi

        course_angle_err = psip - psi
        if course_angle_err>np.pi:
            course_angle_err=course_angle_err-2*np.pi
        if course_angle_err<-np.pi:
            course_angle_err=course_angle_err+2*np.pi


        if np.absolute(side_track_error)>=0.5:
            R1=-2
        else:
            R1 = 2*np.exp(-10 *(side_track_error ** 2))

        if np.absolute(course_angle_err*180/np.pi)>=20:
            R2=-1
        else:
            R2 = np.exp(-10 * (course_angle_err ** 2))

        if distance<=0.3:
            R3=3
        else:
            R3= -2*(distance-self.distance)


        reward= R1+R2+R3
        self.distance=distance

        observation = [side_track_error, course_angle_err, distance]

        # DESTINATION CHECK

        if distance <= 0.3:
            print("Destination reached")
            return ts.termination(np.array(observation, dtype=np.float32), reward)

        # BOUNDARY CHECK
        if (abs(x) >= abs(x_goal)+0.5 or abs(y) >= abs(y_goal)+0.5):

            self.episode_ended = True

            return ts.termination(np.array(observation, dtype=np.float32), reward)

        else:
            return ts.transition(np.array(observation, dtype=np.float32), reward, discount=0.8)

    def _reset(self):
        self.yaw_rate = 0
        self.state = np.array([0, 0, 0])
        self.episode_ended = False
        print("RESETTING ENVIRONMENT")
        self.distance=0

        self.random_x = np.random.randint(5,19) * np.random.choice([1, -1])
        self.random_y = np.random.randint(5,19) * np.random.choice([1, -1])
        self.distance = (self.random_x**2+self.random_y**2)**0.5
        print(self.random_x, self.random_y)
        x_goal = self.random_x
        y_goal = self.random_y
        self.coord_x = [0]
        self.coord_y = [0]

        psip = np.arctan2(y_goal, x_goal)

        observation = [0,psip,self.distance]
        self.counter = self.counter + 1
        return ts.restart(np.array(observation, dtype=np.float32))


# ENDS HERE


env = ship_environment()

tf_env = tf_py_environment.TFPyEnvironment(env)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.005)
q_net = q_network.QNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=(32, 32),activation_fn=tf.keras.layers.LeakyReLU())

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


replay_buffer_max_length = 10000
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length)


def collect_data(envr, policy, buffer):
    time_step = envr.reset()

    i = 0
    episode_return = 0
    time = 0

    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)

        next_time_step = envr.step(action_step)

        time = time + 0.4
        if time > 100:
            print("time crossed")
            break

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        episode_return = next_time_step.reward.numpy()[0] + episode_return
        buffer.add_batch(traj)
        time_step = next_time_step

    print("EPISODE RETURN")
    print(episode_return)
    return episode_return



dataset = replay_buffer.as_dataset(
    num_parallel_calls=2,
    sample_batch_size=640,
    num_steps=2).prefetch(3)

# Reset the train step
agent.train_step_counter.assign(0)

episodes = 10000
returns = []

for _ in range(episodes):

    step = agent.train_step_counter.numpy()

    ereturn = collect_data(tf_env, agent.policy, replay_buffer)
    returns.append(ereturn)

    # X = env.coord_x
    # Y = env.coord_y
    #
    # plt.plot(X,Y)
    # plt.grid()
    # plt.show()

    # Sample a batch of data from the buffer and update the agent's network.
    iterator = iter(dataset)
    experience, unused_info = next(iterator)

    train_loss = agent.train(experience).loss

    log_interval = 5
    eval_interval = 5

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % 150==0 and step>=9000:
        policy_dir = os.path.join('waypoints', 'DOUBLE REWARDS_POLICY')
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)

        checkpoint_dir = os.path.join('waypoints', 'Double rewards_CP')
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
        )




plt.plot(returns)
plt.grid()
plt.show()

