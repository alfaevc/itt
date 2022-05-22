import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import PolynomialFeatures
from itertools import product
from scipy.optimize import minimize
from scipy.optimize import Bounds

def egreedy(x, e=0.01):
    k = x.size
    pi = (e / k) * np.ones(k)
    a_min = np.argmin(x)
    pi[a_min] += 1-e
    return np.random.choice(k, p=pi)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def action_energy_loss(latent_actions, latent_states):
    # loss = -K.mean(tf.exp(-tf.einsum('ij,ij->i', latent_actions, latent_states)))
    loss = K.mean(tf.einsum('ij,ij->i', latent_actions, latent_states))
    return loss


class Log(object):  
    def __init__(self, env):
        self.env = env

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_action(self, s, theta):
        w = theta[:4]
        b = theta[4]
        p_left = self.sigmoid(w @ s + b)
        a = np.random.choice(2, p=[p_left, 1 - p_left])
        return a

    def F(self, theta, gamma=0.99):
        # START HIDE
        done = False
        s = self.env.reset()
        total_reward = 0
        discount = 1
        while not done:
          a = self.get_action(s, theta)
          s, r, done, _ = self.env.step(a)
          total_reward += discount * r
          discount *= gamma
        # END HIDE
        return total_reward
    
    def evaluate(self, theta):
        # START HIDE
        done = False
        s = self.env.reset()
        total_reward = 0
        while not done:
          a = self.get_action(s, theta)
          s, r, done, _ = self.env.step(a)
          total_reward += r
        # END HIDE
        return total_reward

class Gaus(object):
    def __init__(self, env, env_name, state_dim, nA, min_logvar=1, max_logvar=5):
        self.env = env
        self.env_name = env_name
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.nA = nA
        self.state_dim = state_dim


    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the actual output.
        """
        means = output[:, 0:self.nA]    
        raw_vs = output[:, self.nA:]
        logvars = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_vs)
        logvars = self.min_logvar + tf.nn.softplus(logvars - self.min_logvar)
        return means, tf.exp(logvars).numpy()

    def F(self, theta, gamma=1, max_step=5e3):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        discount = 1
        steps = 0
        # while not done:
        while not done and (steps < max_step):
            # WRITE CODE HERE
            fn = lambda a: [theta[2*a*(self.state_dim+1)] + state @ theta[2*a*(self.state_dim+1)+1: (2*a+1)*(self.state_dim+1)], 
                            theta[(2*a+1)*(self.state_dim+1)] + state @ theta[(2*a+1)*(self.state_dim+1)+1: (2*a+2)*(self.state_dim+1)]]
            mvs = np.array(list(map(fn, a_dim))).flatten()
            a_mean, a_v  = self.get_output(np.expand_dims(mvs, 0))
            action = np.random.normal(a_mean[0], a_v[0])
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, theta):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        steps = 0
        while not done:
            # WRITE CODE HERE
            fn = lambda a: [theta[2*a*(self.state_dim+1)] + state @ theta[2*a*(self.state_dim+1)+1: (2*a+1)*(self.state_dim+1)], 
                            theta[(2*a+1)*(self.state_dim+1)] + state @ theta[(2*a+1)*(self.state_dim+1)+1: (2*a+2)*(self.state_dim+1)]]
            mvs = np.array(list(map(fn, a_dim))).flatten()
            a_mean, a_v  = self.get_output(np.expand_dims(mvs, 0))
            action = np.random.normal(a_mean[0], a_v[0])
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward
            steps += 1
        # print("The length of the trajectory is {}".format(steps))
        return G


    def nnF(self, nn, gamma=.99, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        discount = 1
        steps = 0
        while not done:
        # while not done and (steps < max_step):
            a_mean, a_v  = self.get_output(nn(np.expand_dims(state, 0)).numpy())
            action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G
    
    def nneval(self, nn, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        while not done:
        # while not done and (steps < max_step):
            a_mean, a_v  = self.get_output(nn(np.expand_dims(state, 0)).numpy())
            action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward

        return G


class GausNN(object):
    def __init__(self, env, nn, state_dim, nA, min_logvar=1, max_logvar=3):
        self.env = env
        self.nn = nn
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.nA = nA
        self.input_dim = state_dim
        self.theta_len = self.nn.nnparams2theta().size

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the actual output.
        """
        means = output[:, :self.nA]
        raw_vs = output[:, self.nA:]
        logvars = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_vs)
        logvars = self.min_logvar + tf.nn.softplus(logvars - self.min_logvar)
        return means, tf.exp(logvars).numpy()

    def sample_action(self, nn, state):
        a_mean, a_v  = self.get_output(nn(np.expand_dims(state,0)).numpy())
        mu, var = a_mean[0], a_v[0]
        return np.random.normal(mu, var)

    def attention_action(self, nn, state, K=10):
        a_mean, a_v  = self.get_output(nn(np.expand_dims(state,0)).numpy())
        mu, var = a_mean[0], a_v[0]
        actions = np.tanh(np.random.normal(mu, var, (K*self.nA, self.nA)))
        fn = lambda a: np.dot(np.multiply(a-mu, 1/var), a-mu)
        energies = np.array(list(map(fn, actions)))
        return actions[np.argmax(energies)]

    def F(self, theta, gamma=1, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        self.nn.update_params(self.nn.theta2nnparams(theta, self.input_dim, self.nn.output_dim))
        while not done:
        # while not done and (steps < max_step):
            # a_mean, a_v  = self.get_output(self.nn(np.expand_dims(state, 0)).numpy())
            # action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])
            # action = self.sample_action(self.nn, state)
            action = self.attention_action(self.nn, state)
            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, nn):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            # a_mean, a_v  = self.get_output(nn(np.expand_dims(state, 0)).numpy())
            # action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])
            # action = self.sample_action(nn, state)
            action = self.attention_action(nn, state)

            state, reward, done, _ = self.env.step(action)
            G += reward
        return G



class Energy(object):
    def __init__(self, env, nn, state_dim, nA, min_logvar=1, max_logvar=3):
        self.env = env
        self.nn = nn
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.input_dim = nA + state_dim
        self.nA=nA

    def energy_action(self, actor, state, K):
        sample_actions = np.random.uniform(low=-2.0, high=2.0, size=(K,self.nA))
        #states = np.repeat(state, K).reshape((K,state.size))#this gives a wrong matrix
        states = np.tile(state,(K,1))
        sas = np.concatenate((states, sample_actions), axis=1)
        energies = actor(sas).numpy().reshape(-1)
        return sample_actions[np.argmin(energies)]

    
    def F(self, theta, gamma=.99, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        self.nn.update_params(self.nn.theta2nnparams(theta, self.input_dim, 1))
        while not done:
        # while not done and (steps < max_step):
            action = self.energy_action(self.nn, state, K=self.nA*10)
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, nn):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            action = self.energy_action(nn, state, K=self.nA*10)

            state, reward, done, _ = self.env.step(action)
            G += reward
        return G

class Energy_polyn(object):
    def __init__(self, env, state_dim, nA, min_logvar=1, max_logvar=3):
        self.env = env
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.input_dim = nA + state_dim
        self.nA = nA

    def energy_action(self, theta, state, K):
        #sample_actions=np.array([i for i in product([-1,0,1],repeat=2)])
        sample_actions = np.random.uniform(low=-1.0, high=1.0, size=(K,self.nA))
        #states = np.repeat(state, K).reshape((K,state.size))#this gives a wrong matrix
        states = np.tile(state,(K,1))
        sas = np.concatenate((states, sample_actions), axis=1)
        sas_Matrix = PolynomialFeatures(degree=2, include_bias=False).fit_transform(sas)
        energies=sas_Matrix@theta
        return(sample_actions[np.argmin(energies)])
        
    
    def F(self, theta, gamma=1, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        while not done:
        # while not done and (steps < max_step):
            action = self.energy_action(theta, state, K=self.nA*10)
            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, theta):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            action = self.energy_action(theta, state, K=self.nA*10)
            state, reward, done, _ = self.env.step(action)
            G += reward
        return G

class Energy_twin(object):
    def __init__(self, env, actor, critic, state_dim, nA, min_logvar=1, max_logvar=3):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.state_dim = state_dim
        self.nA = nA
        self.actor_theta_len = actor.nnparams2theta().size
        self.critic_theta_len = critic.nnparams2theta().size
    
    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the actual output.
        """
        means = output[:, :self.nA]
        raw_vs = output[:, self.nA:]
        logvars = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_vs)
        logvars = self.min_logvar + tf.nn.softplus(logvars - self.min_logvar)
        return means, tf.exp(logvars).numpy()

    def energy_actions(self, actor, critic, state, K=10):
        #sample_actions = np.array(list(product([-1,0,1], repeat=self.nA)))
        #K = min(len(sample_actions), K)
        #ind=np.random.choice(np.arange(len(sample_actions)),K,replace=False)
        #sample_actions=sample_actions[ind]
        sample_actions = np.random.uniform(low=-1.0, high=1.0, size=(K,self.nA))
        #states = np.repeat(state, K).reshape((K,state.size))#this gives a wrong matrix
        latent_actions, latent_states = actor(sample_actions).numpy(), np.tile(critic(np.expand_dims(state,0)).numpy().reshape(-1), (K,1))
        energies = np.einsum('ij,ij->i', latent_actions, latent_states)
        # return sample_actions[np.argmin(energies)]
        return energies, sample_actions

    '''
    def gd_energy_action(self, actor, critic, state, K=20, lr=0.05):
        action = np.random.uniform(low=-1.0, high=1.0, size=(1,self.nA))
        latent_state = np.tile(critic(np.expand_dims(state,0)).numpy().reshape(-1), (1,1))
        # actor_temp = tf.keras.models.clone_model(actor)

        for _ in range(K):
            with tf.GradientTape() as tape:
                # V_update_vecs = tf.convert_to_tensor(V_update_vecs, dtype=tf.float32)
                latent_action = actor(action, training=False)
                actor_loss = action_energy_loss(latent_action, latent_state)

            gradient = tape.gradient(actor_loss, action).numpy()
            action += lr*gradient
            action = np.clip(action, -1., 1.)
        
        return action[0]
    
    
    def energy_min_action(self, actor, critic, state):
        param1 = actor.get_layer_i_param(0)
        param2 = actor.get_layer_i_param(1)
        latent_state = critic(np.expand_dims(state,0)).numpy()
        return np.dot(np.dot(param1, param2), latent_state.T)
    '''
    
    def blackbox(self, action, actor, critic, state):#input action, output objective and gradient
        #latent_state = np.tile(critic(np.expand_dims(state,0)).numpy().reshape(-1), (1,1))
        state_tensor = tf.reshape(state, shape=(1,self.state_dim))
        latent_state = critic(state_tensor)
        a = tf.Variable(tf.reshape(action, shape=(1,self.nA)))
        with tf.GradientTape() as tape:
            tape.watch(a)
            latent_action = actor(a, training=False)
            actor_loss = action_energy_loss(latent_action, latent_state)
        # print('objective value: ', actor_loss)
        gradient = tape.gradient(actor_loss, a)
        # print('objective value: ', actor_loss,'gradient: ', gradient)
        # print('objective value: ', actor_loss.numpy(),'gradient: ', gradient[0].numpy())
        return actor_loss.numpy(), gradient[0].numpy()
    

    def gd_energy_action_trust_region(self, actor, critic, state):
        bounds = Bounds([-1.0]*(self.nA), [1.0]*(self.nA))
        action = tf.random.uniform(shape=(self.nA,), minval=-1.0, maxval=1.0)
        # print(action)
        # action = np.random.uniform(low=-1.0, high=1.0, size=(1,self.nA))
        res = minimize(self.blackbox, action, args=(actor, critic, state), method='trust-constr', jac=True,tol=0.1,
                   options={'verbose': 0,'maxiter':10}, bounds=bounds)#default maxIter=1000
        opt_action = res.x
        return opt_action


    def F(self, theta, gamma=1, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        theta_action = theta[:self.actor_theta_len]
        theta_state = theta[self.actor_theta_len:]
        self.actor.update_params(self.actor.theta2nnparams(theta_action, self.nA, self.nA))
        self.critic.update_params(self.critic.theta2nnparams(theta_state, self.state_dim, self.nA))

        while not done:
        # while not done and (steps < max_step):
            # energies, actions = self.energy_actions(self.actor, self.critic, state, K=self.nA*10)
            # action = actions[egreedy(energies)]
            # action = actions[np.argmin(energies)]
            # action = self.energy_min_action(self.actor, self.critic, state)
            action = self.gd_energy_action_trust_region(self.actor, self.critic, state)

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, actor, critic):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            # energies, actions = self.energy_actions(actor, critic, state, K=self.nA*10)
            # action = actions[np.argmin(energies)]
            # action = self.energy_min_action(actor, critic, state)
            action = self.gd_energy_action_trust_region(actor, critic, state)

            state, reward, done, _ = self.env.step(action)
            G += reward
        return G


