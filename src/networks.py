import tensorflow as tf
import numpy as np


class NN(tf.keras.Model):
    
  def __init__(self, input_size, output_size, layers=[32,16,8], actor_lr=1e-4):
    super().__init__()
    initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="normal", seed=None)

    self.dense1 = tf.keras.layers.Dense(layers[0], activation=tf.identity, kernel_initializer=initializer)
    # self.dense2 = tf.keras.layers.Dense(layers[1], activation=tf.nn.relu, kernel_initializer=initializer)
    #self.dense3 = tf.keras.layers.Dense(layers[2], activation=tf.nn.relu, kernel_initializer=initializer)

    self.output_layer = tf.keras.layers.Dense(output_size, activation=tf.identity, kernel_initializer=initializer)

    self.ls = layers

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    self.loss = tf.keras.losses.MeanSquaredError()
    self.input_dim = input_size
    self.output_dim = output_size

    self.theta_len = self.calc_theta_len(self.input_dim, self.output_dim)


  def call(self, inputs):
    x = self.dense1(inputs)
    # x = self.dense2(x)
    #x = self.dense3(x)
    x = self.output_layer(x)
    return x

  def nnparams2theta(self):
    theta = []
    for p in self.trainable_weights:
      theta.append(p.numpy().reshape(-1))
    theta = np.concatenate(tuple(theta))
    return theta
  
  def calc_theta_len(self, input_dim, output_dim):
    end_index = input_dim * self.ls[0]
    
    for i in range(len(self.ls)-1):
      end_index += self.ls[i] + self.ls[i] * self.ls[i+1]
    
    end_index += self.ls[-1] + self.ls[-1] * output_dim

    return end_index + output_dim
    

  def theta2nnparams(self, theta, input_dim, output_dim):
    params = []
    # self.print_params()

    end_index = input_dim * self.ls[0]
    params.append(theta[:end_index].reshape((input_dim, self.ls[0])))
    start_index = end_index
    
    for i in range(len(self.ls)-1):
      end_index += self.ls[i]
      params.append(theta[start_index:end_index])
      start_index = end_index
      end_index += self.ls[i] * self.ls[i+1]
      params.append(theta[start_index:end_index].reshape(self.ls[i], self.ls[i+1]))
      start_index = end_index
    
    end_index += self.ls[-1]
    params.append(theta[start_index:end_index])
    start_index = end_index
    end_index += self.ls[-1] * output_dim
    params.append(theta[start_index:end_index].reshape(self.ls[-1], output_dim))
    params.append(theta[-output_dim:])

    assert theta.size == end_index + output_dim

    return params

  def get_layer_i_param(self, i):
    return self.trainable_weights[2*i].numpy()

  def update_params(self, new_params):
    for i in range(len(new_params)):
      self.trainable_weights[i].assign(new_params[i])

  def print_params(self):
    for p in self.trainable_weights:
      print(p.numpy().shape)

    
  

  