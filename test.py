import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from scipy.integrate import solve_ivp
from tf_agents.trajectories import time_step as ts


# SHIP ENVIRONMENT CLASS
# starts here

class ship_environment(py_environment.PyEnvironment):

    def __init__(self):
        self.yaw_rate = 0
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int64, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=[-20,-np.pi,0], maximum=[20, np.pi,20], name='observation')
        self.state = np.array([0, 0, 0])
        self.episode_ended = False
        self.counter = 0
        self.distance= 0
        self.coord_x=[0]
        self.coord_y=[0]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action_no):

        action_num = action_no
        action_set = [-0.2, 0, 0.2]

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
        psi = obs_new[2]

        self.coord_x.append(x)
        self.coord_y.append(y)

        # REWARD FUNCTIONS

        x_init = 0
        y_init = 0
        x_goal = 10
        y_goal = -3

        psip = np.arctan2(y_goal - y_init, x_goal - x_init)

        slope_WP = np.tan(psip)

        distance = ((x - x_goal) ** 2 + (y - y_goal) ** 2) ** 0.5

        side_track_error = (y - slope_WP * x - (y_init - slope_WP * x_init)) / (1 + slope_WP ** 2) ** 0.5

        if psi> np.pi:
            psi=psi-2*np.pi

        course_angle_err = psip - psi
        if course_angle_err > np.pi:
            course_angle_err = course_angle_err - 2 * np.pi
        if course_angle_err < -np.pi:
            course_angle_err = course_angle_err + 2 * np.pi


        if np.absolute(side_track_error)>=0.5:
            R1=-1
        else:
            R1 = np.exp(-10 * (side_track_error ** 2))


        if np.absolute(course_angle_err*180/np.pi)>=30:
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
        print(observation,[R1,R2,R3])

        # DESTINATION CHECK

        if distance <= 0.5:
            print("Destination reached")
            return ts.termination(np.array(observation, dtype=np.float32), reward)

        # BOUNDARY CHECK
        if (abs(x) >= abs(x_goal)+0.5 or abs(y) >= abs(y_goal)+0.5):

            self.episode_ended = True

            return ts.termination(np.array(observation, dtype=np.float32), reward)

        else:
            return ts.transition(np.array(observation, dtype=np.float32), reward, discount=0.99)

    def _reset(self):
        self.yaw_rate = 0
        self.state = [0, 0, 0]
        self.episode_ended = False
        print("RESETTING ENVIRONMENT")

        # self.random_x = np.random.randint(0, 16) * np.random.choice([1, -1])
        # self.random_y = np.random.randint(0, 16) * np.random.choice([1, -1])
        # print(self.random_x,self.random_y)
        # self.distance = (self.random_x**2+self.random_y**2)**0.5
        # x_goal = self.random_x
        # y_goal = self.random_y

        # psip = np.arctan2(y_goal, x_goal)
        self.coord_x = [0]
        self.coord_y = [0]

        observation = [0,np.pi/4,7.07]
        self.counter = self.counter + 1
        return ts.restart(np.array(observation, dtype=np.float32))

# ENDS HERE


env = ship_environment()

tf_env = tf_py_environment.TFPyEnvironment(env)


train_step_counter = tf.Variable(0)

# Reset the train step

episodes = 1
returns = []

saved_policy = tf.compat.v2.saved_model.load('waypoints\\DOUBLE REWARDS_POLICY')


for _ in range(episodes):

    time_step = tf_env.reset()
    time = 0

    while not np.equal(time_step.step_type, 2):
        action_step = saved_policy.action(time_step)
        time_step = tf_env.step(action_step.action)

        # time = time + 0.4
        # if (time > 16):
        #     print("time crossed")
        #     break


X=env.coord_x
Y=env.coord_y
plt.plot(X,Y)
plt.grid()
plt.show()



