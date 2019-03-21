import numpy as np
import itertools
import math

class E_greedy:
	def __init__(self, e, estimator, discount, lmbda):
		self.e = e
		self.est = estimator
		self.discount = discount
		self.lmbda = lmbda

	def get_optimal_action(self, s, env):
		q = np.zeros(env.action_space.n)
		for a in range(env.action_space.n):
			q[a] = self.est.estimate(s, a)
		return np.argmax(q)



	def run(self, T, env):
		'''
		T: int, maximum number of timesteps in 1 episode
		env: gym.env object, MountainCar-v0		
		'''
		s = env.reset()
		res = -1

		for t in range(T):
			if np.random.binomial(1, self.e):
				action = env.action_space.sample()
			else:
				action = self.get_optimal_action(s, env)
			
			n_s, r, d, i = env.step(action)


			target = r + self.discount * self.est.estimate(n_s, self.get_optimal_action(n_s, env)) - self.est.estimate(s, action) # target = r + q(s_{t+1}, a_{t+1}) - q(s_t, a_t)
			self.est.update_w(target, s, action, self.discount, self.lmbda) 

			if d:
				res = t
				break				

			s = n_s
		
		env.close()
		return res



	

class U_e_greedy:
	def __init__(self, e):
		self.e = e


class Estimator:
	def __init__(self, order, learn_rate, env):
		self.order = order
		self.alpha = learn_rate
		self.dim_count = len(env.observation_space.sample())
		self.action_count = env.action_space.n
		self.env = env

		self.w = np.zeros((self.order+1) ** self.dim_count * self.action_count) # Konindaris et al.
		self.z = np.zeros((self.order+1) ** self.dim_count * self.action_count) # lambda

	def gather_c(self):
		cart_prod = list(itertools.product(np.arange(0, self.order+1), repeat=self.dim_count))
		return [list(elem) for elem in cart_prod]

	def basis(self, s, a):

		B = np.zeros((self.action_count, (self.order + 1) ** self.dim_count))
		
		c = self.gather_c()
		basis = np.ndarray((self.order + 1) ** self.dim_count)
		
		for i in range(len(basis)):
			B[a][i] = math.cos(math.pi * np.dot(c[i], self.interp_s(s)))
			
		return B.flatten()

	def estimate(self, s, a):
		features = self.basis(s, a) 
		return np.dot(self.w, features)

	def update_w(self, target, s, a, dc, lmbda):
		base = self.basis(s, a)		
		self.z = dc * lmbda * self.z + base
		self.w =  self.w + (self.alpha * target) * self.z

	def interp_s(self, s):
		s_ = np.empty(len(self.env.observation_space.sample()))		
		for i in range(len(s)):
			s_[i] = np.interp(s[i], (self.env.observation_space.low[i], self.env.observation_space.high[i]), (0, 1))
		return s_
