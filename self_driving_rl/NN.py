""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

Here, DDPG is used to achieve Full-Speed-ACC

"""


import tensorflow as tf
import numpy as np
import tflearn
import argparse
import pprint as pp
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer

#=======================================
#   	Actor Neural Network
#=======================================
class actorNetwork(object):
	"""
	Input to the Critic Network is the environment state, 
	Output is the action under a deterministic policy
	The output layer activation is a tanh to keep the action in its range.
	
	"""


	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.batch_size = batch_size
		print self.action_bound

		#Actor Network
		self.inputs, self.out, self.scaled_out = self.create_actor_network()
		self.network_params = tf.trainable_variables() #get the trainable variables list

		#Target Network
		self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
		self.target_network_params = tf.trainable_variables()[len(self.network_params):]

		#Periodically updating target network with online network weights
		self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + 
																				  tf.multiply(self.target_network_params[i], 1 - self.tau))
											for i in range(len(self.target_network_params))]

		self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
		self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
		self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

		#Optimization
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))
		self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

	def create_actor_network(self):
		inputs = tflearn.input_data(shape = [None, self.s_dim])
		net = tflearn.fully_connected(inputs, 40)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		net = tflearn.fully_connected(net, 30)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		#Initialize the final layer NN weights
		w_init = tflearn.initializations.uniform(minval = -0.003, maxval = 0.003)
		out = tflearn.fully_connected(net, self.a_dim, activation = 'tanh', weights_init = w_init)
		#Scale the output of Actor NN to -action_bound to action_bound
		scaled_out = tf.multiply(out, self.action_bound)
		print self.action_bound
		return inputs, out, scaled_out

	def train(self, inputs, a_gradient):
		self.sess.run(self.optimize, feed_dict = {self.inputs: inputs,
												self.action_gradient: a_gradient})
	def predict(self, inputs):
		return self.sess.run(self.scaled_out, feed_dict = {self.inputs: inputs})

	def predict_target(self, inputs):
		return self.sess.run(self.target_scaled_out, feed_dict = {self.target_inputs: inputs})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars


#=======================================
#   	Critic Neural Network
#=======================================
class critciNetwork(object):
	"""
	Input to the Critic Network is the environment state and action, output is the Q value
	The action must be obtained from the output of the Actor Network

	"""
	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau
		self.gamma = gamma

		#Critic Network
		self.inputs, self.action, self.out = self.create_critic_network()
		self.network_params = tf.trainable_variables()[num_actor_vars:]

		#Target Network
		self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
		self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars): ]

		#Periodically updating target network with online network weights with regularization
		self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + 
																			  tf.multiply(self.target_network_params[i], 1 - self.tau))
										for i in range(len(self.target_network_params))]

		#Network target y_i
		self.predict_q_value = tf.placeholder(tf.float32, [None, 1])

		#Define loss and optimization
		self.loss = tflearn.mean_square(self.predict_q_value, self.out)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		self.action_grads = tf.gradients(self.out, self.action)

	def create_critic_network(self):
		inputs = tflearn.input_data(shape = [None, self.s_dim])
		action = tflearn.input_data(shape = [None, self.a_dim])
		net = tflearn.fully_connected(inputs, 40)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)

		#Add action tensor in the 2nd hidden layer
		#Use two temp layers to get the corresponding weights and biases
		t1 = tflearn.fully_connected(net, 30)
		t2 = tflearn.fully_connected(action, 30)

		# linear layer connected to 1 output representing Q(s,a)
		# Weights are init to Uniform[-3e-3, 3e-3]
		net =  tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation = 'relu')
		w_init = tflearn.initializations.uniform(minval = -0.003, maxval = 0.003)
		out = tflearn.fully_connected(net, 1 , weights_init = w_init)
		return inputs, action, out

	def train(self, inputs, action, predict_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict = {
			self.inputs: inputs,
			self.action: action,
			self.predict_q_value: predict_q_value
			})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict = {
			self.inputs: inpusts,
			self.action: action
			})

	def predict_target(self, inputs, action):
		return self.sess.run(self.target_out, feed_dict = {
			self.target_inputs: inputs,
			self.target_action: action
			})

	def action_gradients(self, inputs, actions):
		return self.sess.run(self.action_grads, feed_dict = {
			self.inputs: inputs,
			self.action: actions
			})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

class OrnsteinUhlenbeckActionNoise:
	def __init__(self, mu, sigma = 0.3, theta = 0.15, dt = 0.02, x0 =None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()
	def __call__(self):
		x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
			self.sigma*np.sqrt(self.dt)*np.random.normal(size = self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu = {}, sigma = {})'.format(self.mu, self.sigma)

#=======================================
#   	Tensorflow Summary Options
#=======================================

def build_summaries():
	episode_reward = tf.Variable(0.)
	tf.summary.scalar("Reward", episode_reward)
	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("Qmax Value", episode_ave_max_q)

	summary_vars = [episode_reward, episode_ave_max_q]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars


#=======================================
#   		Agent Training
#=======================================
def train(sess, args, actor, critic, actor_noise):
	summary_ops, summary_vars = build_summaries()

	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

	#Initilize target network weights
	actor.update_target_network()
	critic.update_target_network()

	#Initilize replay memory
	replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

	for i in range(int(args['max_episodes'])):
		#Initilize the start states of the subject vehicle and the CIPV
		#The state: CIPV_speed, CIPV_acceleration, distance, subject_speed
		#The control variable: subject_acceleration
		CIPV_speed = 10
		CIPV_acceleration = -0.01	#store the CIPV acceleration at each time
		subject_speed = 12
		distance = 18
		s = [CIPV_speed, CIPV_acceleration, subject_speed, distance] 

		ep_reward = 0
		ep_ave_max_q = 0
		terminal = False
		CIPV_speed_list = [10]
		subject_speed_list = [12]
		CIPV_acceleration_list = [-0.01]
		distance_list = [18]
		subject_action_list=[0]
		headway_list = [1.5]
		desired_headway_list=[1.5]
		if i==20:
			replay_buffer.clear()

		for j in range(int(args['max_episode_len'])):
			# Add exploration noise
			a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()/100

			#sample_time = 0.02s
			sample_time = 0.02
			CIPV_speed_ = CIPV_speed + CIPV_acceleration * sample_time
			subject_speed_ = subject_speed + a * sample_time
			distance_ = distance + CIPV_speed * sample_time + 0.5 * CIPV_acceleration * sample_time * sample_time - \
								   subject_speed * sample_time + 0.5 * a * sample_time * sample_time
			headway = distance_ / subject_speed_

			#desired headway = 1.5s, threshold = 0.1s
			desired_headway = 1.5
			if headway >= desired_headway and headway < desired_headway + 0.1:
				r = 4 * (desired_headway + 0.1 - headway)
			if headway > desired_headway - 0.1 and headway < desired_headway:
				r = 4 * (headway - desired_headway + 0.1)
			if headway >= desired_headway + 0.1:
				r = -5 * (headway - desired_headway - 0.1)
			if terminal == False and headway <= desired_headway - 0.1:
				r = -3 * (desired_headway + 0.1 - headway)

			#Is collision or not, if true, terminal = true
			if distance_ <= 0:
				terminal = True
			else:
				terminal = False
			if(i == 20):
				#
				CIPV_speed_list.append(CIPV_speed_)
				subject_speed_list.append(subject_speed_[0][0])
				distance_list.append(distance_[0][0])
				subject_action_list.append(a[0][0])
				headway_list.append(headway[0][0])
				desired_headway_list.append(1.5)


			#The next envirnoment state
			s2 = [CIPV_speed_, CIPV_acceleration, subject_speed_, distance_] 

			#add to buffer
			replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))
			
            # Keep adding experience to the memory until
            # there are at least minibatch size samples
			if replay_buffer.size() >= int(args['minibatch_size']):
				s_batch, a_batch, r_batch, t_batch, s2_batch = \
					replay_buffer.sample_batch(int(args['minibatch_size']))

				# Calculate targets
				target_q = critic.predict_target(
					s2_batch, actor.predict_target(s2_batch))

				y_i = []
				for k in range(int(args['minibatch_size'])):
					if t_batch[k]:
						y_i.append(r_batch[k])
					else:
						y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
				predicted_q_value, _ = critic.train(
					s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

				ep_ave_max_q += np.amax(predicted_q_value)

				# Update the actor policy using the sampled gradient
				a_outs = actor.predict(s_batch)
				grads = critic.action_gradients(s_batch, a_outs)
				actor.train(s_batch, grads[0])

				# Update target networks
				actor.update_target_network()
				critic.update_target_network()

				#replay_buffer.clear()

				print('a:{:.4f} | Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Headway: {:.4f}'.format(float(a), int(ep_reward), \
						i, (ep_ave_max_q / float(j)), float(headway)))	

			s = s2
			CIPV_speed = CIPV_speed_
			subject_speed =subject_speed_
			distance = distance_
			ep_reward += r

			if terminal:
				print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Headway: {:.4f}'.format(int(ep_reward), \
						i, (ep_ave_max_q / float(j)), float(headway)))
		
				break
					
		if i == 20:
			#print subject_speed_list

			plt.plot(range(0,len(subject_speed_list)), subject_speed_list, label='subject speed')
			plt.plot(range(0,len(CIPV_speed_list)), CIPV_speed_list, label='CIPV speed')
			plt.ylim(min(CIPV_speed_list) * 0.9, max(subject_speed_list) * 1.1)
			plt.legend(['subject speed','CIPV speed'])
			plt.xlabel('point')
			plt.figure()
			plt.plot(range(0,len(headway_list)), headway_list, label='headway')
			plt.plot(range(0,len(desired_headway_list)), desired_headway_list, label='disered headway')
			plt.ylim(min(headway_list) * 0.9, max(desired_headway_list) * 1.1)
			plt.legend(['headway','disered headway'])
			plt.xlabel('point')
			plt.figure()
			plt.plot(range(0,len(subject_action_list)), subject_action_list, label='action')
			#plt.ylim(min(subject_action_list) * 0.9, max(subject_action_list) * 1.1)
			plt.legend(['action'])
			plt.xlabel('point')
			plt.show()


def main(args):

	 with tf.Session() as sess:
		state_dim = 4
		action_dim = 1
		action_bound = [5.0]

		actor = actorNetwork(sess, state_dim, action_dim, action_bound,
							float(args['actor_lr']), float(args['tau']),
							float(args['minibatch_size']))
		critic = critciNetwork(sess, state_dim, action_dim, float(args['critic_lr']), 
								float(args['tau']), float(args['gamma']), 
								actor.get_num_trainable_vars())
		actor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(action_dim))

		train(sess, args, actor, critic, actor_noise)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

	# agent parameters
	parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
	parser.add_argument('--critic-lr', help='critic network learning rate', default=0.0001)
	parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
	parser.add_argument('--tau', help='soft target update parameter', default=0.001)
	parser.add_argument('--buffer-size', help='max size of the replay buffer', default=300)
	parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=200)

    # run parameters
	parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
	parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
	parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
	parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
	parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

	parser.set_defaults(render_env=True)
	parser.set_defaults(use_gym_monitor=True)
    
	args = vars(parser.parse_args())
    
	pp.pprint(args)

	main(args)
