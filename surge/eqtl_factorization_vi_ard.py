import numpy as np 
import sys
import pdb
import scipy.special as special
from sklearn.linear_model import LinearRegression
import time
import sklearn.decomposition

def sigmoid_function(x):
	return 1.0/(1.0 + np.exp(-x))

def run_linear_model_for_initialization(Y, G, cov, z):
	num_tests = Y.shape[1]
	F_betas = []
	C_betas = []
	residual_varz = []
	for test_number in range(num_tests):
		y_vec = Y[:,test_number]
		g_vec = G[:,test_number]
		X = np.hstack((np.transpose(np.asmatrix(g_vec)), cov))
		#dd = {'y':y_vec, 'g':g_vec, 'z':z}
		#df = pd.DataFrame(dd)
		#model = Lmer('y ~ g  + (1|z)', data=df)
		#pdb.set_trace()
		reg = LinearRegression(fit_intercept=False).fit(X, np.transpose(np.asmatrix(y_vec)))
		F_betas.append(reg.coef_[0][0])
		C_betas.append(reg.coef_[0][1:])
		pred_y = reg.predict(X)[:,0]
		resid_y = y_vec - pred_y
		residual_varz.append(np.var(resid_y))
	return np.asarray(F_betas), np.asarray(C_betas), np.asarray(residual_varz)


def compute_kl_divergence_of_gaussian_bernoulli(S, W_mu, W_var, W_var_s_0, gamma_expected, theta_a, theta_b, K):
	num_feat = W_mu.shape[1]
	# Relevent expectations
	#log_gamma_expected = special.digamma(gamma_alpha) - np.log(gamma_beta)
	#gamma_expected = gamma_alpha/gamma_beta
	log_theta_expected_val = special.digamma(theta_a) - special.digamma(theta_a + theta_b)
	log_1_minus_theta_expected_val = special.digamma(theta_b) - special.digamma(theta_a + theta_b)
	#W_var_s_0_temp = np.dot(np.transpose([(gamma_beta/gamma_alpha)]),np.ones((1,num_feat)))
	W_squared_expected_val = (S*(np.square(W_mu) + W_var)) + ((1.0-S)*W_var_s_0)

	# Initialize variables
	likelihood_term_a = 0
	likelihood_term_b = 0
	likelihood_term_c = 0
	likelihood_term_d = 0
	entropy_term_a = 0
	entropy_term_b = 0

	for k in range(K):
		#likelihood_term_a = likelihood_term_a + (num_feat/2.0)*log_gamma_expected[k]
		likelihood_term_b = likelihood_term_b - (np.sum(W_squared_expected_val[k,:])*gamma_expected/2.0)
		likelihood_term_c = likelihood_term_c + (np.sum(S[k,:])*log_theta_expected_val[k])
		likelihood_term_d = likelihood_term_d + (np.sum(1.0-S[k,:])*log_1_minus_theta_expected_val[k])
		#entropy_term_a = entropy_term_a + (.5)*np.sum(np.log((S[k,:]*W_var[k,:]) + ((1-S[k,:])*W_var_s_0[k,:])))
		entropy_term_a = entropy_term_a - (.5)*np.sum(S[k,:]*np.log(W_var[k,:]) + (1.0-S[k,:])*np.log(W_var_s_0[k,:]))

		temp_term_b = ((1.0-S[k,:])*np.log(1.0-S[k,:])) + (S[k,:]*np.log(S[k,:]))
		temp_term_b[np.isnan(temp_term_b)] = 0.
		entropy_term_b = entropy_term_b + np.sum(temp_term_b)
		#entropy_term_b = entropy_term_b + np.sum(((1.0-S[k,:])*np.log(log_const+1.0-S[k,:])) - (S[k,:]*np.log(log_const+S[k,:])))
	
	kl_divergence = entropy_term_a + entropy_term_b - likelihood_term_a - likelihood_term_b - likelihood_term_c - likelihood_term_d

	return kl_divergence

def compute_kl_divergence_of_gaussian(W_mu, W_var, gamma_alpha, gamma_beta, K):
	num_feat = W_mu.shape[1]
	# Relevent expectations
	log_gamma_expected = special.digamma(gamma_alpha) - np.log(gamma_beta)
	gamma_expected = gamma_alpha/gamma_beta
	#log_theta_expected_val = special.digamma(theta_a) - special.digamma(theta_a + theta_b)
	#log_1_minus_theta_expected_val = special.digamma(theta_b) - special.digamma(theta_a + theta_b)
	#W_var_s_0_temp = np.dot(np.transpose([(gamma_beta/gamma_alpha)]),np.ones((1,num_feat)))
	W_squared_expected_val = ((np.square(W_mu) + W_var))

	# Initialize variables
	likelihood_term_a = 0
	likelihood_term_b = 0
	likelihood_term_c = 0
	likelihood_term_d = 0
	entropy_term_a = 0
	entropy_term_b = 0

	for k in range(K):
		likelihood_term_a = likelihood_term_a + (num_feat/2.0)*log_gamma_expected[k]
		likelihood_term_b = likelihood_term_b - (np.sum(W_squared_expected_val[k,:])*gamma_expected[k]/2.0)
		#likelihood_term_c = likelihood_term_c + (np.sum(S[k,:])*log_theta_expected_val[k])
		#likelihood_term_d = likelihood_term_d + (np.sum(1.0-S[k,:])*log_1_minus_theta_expected_val[k])
		#entropy_term_a = entropy_term_a + (.5)*np.sum(np.log((S[k,:]*W_var[k,:]) + ((1-S[k,:])*W_var_s_0[k,:])))
		entropy_term_a = entropy_term_a - (.5)*np.sum(np.log(W_var[k,:]))
		#temp_term_b = ((1.0-S[k,:])*np.log(1.0-S[k,:])) + (S[k,:]*np.log(S[k,:]))
		#temp_term_b[np.isnan(temp_term_b)] = 0.
		#entropy_term_b = entropy_term_b + np.sum(temp_term_b)
		#entropy_term_b = entropy_term_b + np.sum(((1.0-S[k,:])*np.log(log_const+1.0-S[k,:])) - (S[k,:]*np.log(log_const+S[k,:])))
	
	kl_divergence = entropy_term_a  - likelihood_term_a - likelihood_term_b - likelihood_term_c - likelihood_term_d

	return kl_divergence

def compute_kl_divergence_of_gaussian_fixed_variance(W_mu, W_var, gamma_expected, K):
	num_feat = W_mu.shape[1]
	# Relevent expectations
	#log_gamma_expected = special.digamma(gamma_alpha) - np.log(gamma_beta)
	#gamma_expected = gamma_alpha/gamma_beta
	#log_theta_expected_val = special.digamma(theta_a) - special.digamma(theta_a + theta_b)
	#log_1_minus_theta_expected_val = special.digamma(theta_b) - special.digamma(theta_a + theta_b)
	#W_var_s_0_temp = np.dot(np.transpose([(gamma_beta/gamma_alpha)]),np.ones((1,num_feat)))
	W_squared_expected_val = ((np.square(W_mu) + W_var))

	# Initialize variables
	likelihood_term_a = 0
	likelihood_term_b = 0
	likelihood_term_c = 0
	likelihood_term_d = 0
	entropy_term_a = 0
	entropy_term_b = 0

	for k in range(K):
		#likelihood_term_a = likelihood_term_a + (num_feat/2.0)*log_gamma_expected[k]
		likelihood_term_b = likelihood_term_b - (np.sum(W_squared_expected_val[k,:])*gamma_expected/2.0)
		#likelihood_term_c = likelihood_term_c + (np.sum(S[k,:])*log_theta_expected_val[k])
		#likelihood_term_d = likelihood_term_d + (np.sum(1.0-S[k,:])*log_1_minus_theta_expected_val[k])
		#entropy_term_a = entropy_term_a + (.5)*np.sum(np.log((S[k,:]*W_var[k,:]) + ((1-S[k,:])*W_var_s_0[k,:])))
		entropy_term_a = entropy_term_a - (.5)*np.sum(np.log(W_var[k,:]))
		#temp_term_b = ((1.0-S[k,:])*np.log(1.0-S[k,:])) + (S[k,:]*np.log(S[k,:]))
		#temp_term_b[np.isnan(temp_term_b)] = 0.
		#entropy_term_b = entropy_term_b + np.sum(temp_term_b)
		#entropy_term_b = entropy_term_b + np.sum(((1.0-S[k,:])*np.log(log_const+1.0-S[k,:])) - (S[k,:]*np.log(log_const+S[k,:])))
	
	kl_divergence = entropy_term_a  - likelihood_term_b - likelihood_term_c - likelihood_term_d

	return kl_divergence



def compute_kl_divergence_of_gamma(alpha_prior, beta_prior, gamma_alpha, gamma_beta):
	# Relevent expectations
	log_gamma_expected = special.digamma(gamma_alpha) - np.log(gamma_beta)
	gamma_expected = gamma_alpha/gamma_beta
	# Compute kl divergence
	likelihood_term = np.sum(alpha_prior*np.log(beta_prior) + (alpha_prior-1.0)*log_gamma_expected - beta_prior*gamma_expected - special.gammaln(alpha_prior))
	entropy_term = np.sum(gamma_alpha*np.log(gamma_beta) + (gamma_alpha-1.0)*log_gamma_expected - gamma_beta*gamma_expected - special.gammaln(gamma_alpha))
	kl_divergence = entropy_term - likelihood_term
	return kl_divergence

def compute_kl_divergence_of_beta(a_prior, b_prior, theta_a, theta_b):
	# Relevent expectations
	ln_theta_expected_val = special.digamma(theta_a) - special.digamma(theta_a + theta_b)
	ln_1_minus_theta_expected_val = special.digamma(theta_b) - special.digamma(theta_a + theta_b)
	# Compuate kl divergence
	likelihood_term = np.sum((a_prior-1.0)*ln_theta_expected_val + (b_prior-1.0)*ln_1_minus_theta_expected_val - special.betaln(a_prior, b_prior))
	entropy_term = np.sum((theta_a-1.0)*ln_theta_expected_val + (theta_b-1.0)*ln_1_minus_theta_expected_val - special.betaln(theta_a, theta_b))
	kl_divergence = entropy_term - likelihood_term
	return kl_divergence


def outside_update_U_n(U_mu, U_var, G_slice, G_fe_slice, Y_slice, K, V_S_expected_val, V_S_squared_expected_val, F_S_expected_val, covariate_predicted_slice, gamma_u, tau_expected_val, alpha_i_expected_val):
	for k in range(K):
		# Compute relevent expectations
		U_S_expected_val = U_mu
		V_k_S_k_expected_val = V_S_expected_val[k,:]
		# Compute expectations on other components
		other_components_expected = (U_S_expected_val@V_S_expected_val) - U_S_expected_val[k]*V_S_expected_val[k,:]
		# Update variance of q(U|s=1)
		a_term = np.sum(tau_expected_val*np.square(G_slice)*V_S_squared_expected_val[k,:]) + gamma_u[k]
		U_var[k] = 1.0/a_term
		# Update mean of q(U|s=1)
		resid = Y_slice - covariate_predicted_slice - alpha_i_expected_val - G_fe_slice*F_S_expected_val - G_slice*other_components_expected
		b_term = np.sum(tau_expected_val*G_slice*V_k_S_k_expected_val*resid)
		U_mu[k] = U_var[k]*b_term
	return np.hstack((U_mu, U_var))

def outside_update_V_t(V_mu, V_var, G_slice, G_fe_slice, Y_slice, K, U_S_expected_val, U_S_squared_expected_val, F_S_t_expected_val, covariate_predicted_slice, alpha_t_mu, gamma_v, tau_t_expected_val):
	for k in range(K):
		# Compute expectations on other components
		other_components_expected = (U_S_expected_val@V_mu) - U_S_expected_val[:, k]*V_mu[k]
		# Update variance of q(V|s=1)
		a_term = gamma_v + (tau_t_expected_val*np.sum(np.square(G_slice)*U_S_squared_expected_val[:,k]))
		# Update mean of q(U|s=1)
		resid = Y_slice - alpha_t_mu - covariate_predicted_slice - G_slice*other_components_expected - G_fe_slice*F_S_t_expected_val
		b_term = np.sum(tau_t_expected_val*G_slice*U_S_expected_val[:,k]*resid)

		new_var = 1.0/a_term
		new_mu = new_var*b_term
		V_var[k] = new_var
		V_mu[k] = new_mu
	return np.hstack((V_mu, V_var))

def outside_update_intercept_t(intercept_mu, intercept_var, G_slice, Y_slice, N, U_S_expected_val, V_S_t_expected_val, F_S_t_expected_val, tau_t_expected_val, alpha_t_mu, sample_batch_fraction, step_size, SVI):
	# Compute relevent expectations
	# Compute expectations on other components
	other_components_expected = U_S_expected_val@V_S_t_expected_val
	resid = Y_slice - alpha_t_mu - G_slice*(F_S_t_expected_val + other_components_expected)

	new_var = 1.0/((1.0/sample_batch_fraction)*N*tau_t_expected_val)
	new_mu = new_var*tau_t_expected_val*np.sum(resid)*(1.0/sample_batch_fraction)

	if SVI == False:
		intercept_var = new_var
		intercept_mu = new_mu
	elif SVI == True:
		intercept_var = weighted_SVI_updated(intercept_var, new_var, step_size)
		intercept_mu = weighted_SVI_updated(intercept_mu, new_mu, step_size)
	return np.hstack((intercept_mu, intercept_var))


def outside_update_C_t(C_t_mu, C_t_var, G_slice, G_fe_slice, Y_slice, N, U_S_expected_val, V_S_t_expected_val, F_S_t_expected_val, tau_t_expected_val, alpha_t_mu, cov, cov_squared):
	num_cov = len(C_t_mu)
	components_expected = U_S_expected_val@V_S_t_expected_val
	base_resid = Y_slice - alpha_t_mu - G_fe_slice*F_S_t_expected_val - G_slice*components_expected
	for cov_num in range(num_cov):
		other_covariates = (cov@C_t_mu) - (cov[:, cov_num]*C_t_mu[cov_num])
		b_term = tau_t_expected_val*np.sum(cov[:, cov_num]*(base_resid - other_covariates))
		a_term = 0.0 + tau_t_expected_val*np.sum(cov_squared[:,cov_num])
		new_var = 1.0/a_term
		new_mu = new_var*b_term
		C_t_mu[cov_num] = new_mu
		C_t_var[cov_num] = new_var
	return np.hstack((C_t_mu, C_t_var))

def outside_update_alpha_t(alpha_mu_copy, alpha_var_copy, G_slice, G_fe_slice, Y_slice, I, U_S_expected_val, V_S_t_expected_val, F_S_t_expected_val, covariate_predicted_t, tau_t_expected_val, psi_t_expected_val, individual_to_sample_indices, individual_to_number_full_indices):
	other_components_expected = U_S_expected_val@V_S_t_expected_val
	resid = Y_slice - covariate_predicted_t - G_fe_slice*F_S_t_expected_val - G_slice*other_components_expected
	# Loop through individuals
	for individual_index in range(I):
		# Indices of samples corresponding to this label
		sample_indices = individual_to_sample_indices[individual_index]
		# Number of indices corresponding to this sample (w/o subsetting)
		num_full_indices = individual_to_number_full_indices[individual_index]
		# Number of indices corresponding to this individaul
		n_i = len(sample_indices)
		individual_batch_fraction = n_i/num_full_indices
		if individual_batch_fraction != 1.0:
			print('assumption error')
			pdb.set_trace()
		# Update variance of q(alpha_it)
		new_var = 1.0/((1.0/individual_batch_fraction)*n_i*tau_t_expected_val + psi_t_expected_val)
		new_mu = new_var*tau_t_expected_val*np.sum(resid[sample_indices])*(1.0/individual_batch_fraction)
		alpha_var_copy[individual_index] = new_var
		alpha_mu_copy[individual_index] = new_mu
	return np.hstack((alpha_mu_copy, alpha_var_copy))

def outside_update_F_t(F_mu, F_var, G_slice, G_fe_slice, Y_slice, U_S_expected_val, V_S_t_expected_val, covariate_predicted_t, gamma_f_expected_val, tau_t_expected_val, alpha_t_mu):
	# Compute expectations on other components
	other_components_expected = U_S_expected_val@V_S_t_expected_val

	# Update variance of q(F|s=1)
	a_term = gamma_f_expected_val + tau_t_expected_val*np.sum(np.square(G_fe_slice))
	# Update mean of q(F|s=1)
	resid = Y_slice - alpha_t_mu - covariate_predicted_t - G_slice*(other_components_expected)
	b_term = np.sum(tau_t_expected_val*G_fe_slice*resid)
	new_var = 1.0/a_term
	new_mu = new_var*b_term
	F_var = new_var
	F_mu = new_mu

	return np.hstack((F_mu, F_var))

def outside_update_tau_t(tau_alpha, tau_beta, G_slice, G_fe_slice, Y_slice, N, U_S, V_S_t, F_S_t, C_t, V_S_t_squared, F_S_t_squared, U_S_squared, C_t_squared, alpha_mu_t, alpha_var_t, cov, cov_squared, alpha_prior, beta_prior):
	# Compute Relevent expectations
	squared_factor_terms = U_S_squared@V_S_t_squared
	factor_terms = U_S@V_S_t


	alpha_t_squared = np.square(alpha_mu_t) + alpha_var_t

	squared_covariate_terms = cov_squared@C_t_squared
	covariate_terms = cov@C_t

	# First add together square terms
	resid = np.square(Y_slice) + alpha_t_squared + squared_covariate_terms + np.square(G_fe_slice)*F_S_t_squared + np.square(G_slice)*squared_factor_terms
	# Now add terms with Y
	resid = resid - (2.0*Y_slice*(covariate_terms + alpha_mu_t + G_slice*factor_terms + G_fe_slice*F_S_t))

	resid = resid + 2.0*covariate_terms*(alpha_mu_t + G_slice*factor_terms + G_fe_slice*F_S_t)
	resid = resid + 2.0*alpha_mu_t*(G_slice*factor_terms + G_fe_slice*F_S_t)
	# Now add terms with factors
	resid = resid + 2.0*G_slice*factor_terms*G_fe_slice*F_S_t
	# Now add terms with interactions between factors
	resid = resid + (np.square(G_slice)*(factor_terms*factor_terms - np.sum(np.square(U_S*V_S_t),axis=1)))
	#resid = resid + ((covariate_terms*covariate_terms - np.sum(np.square(cov*C_t),axis=1)))
	resid = resid + (covariate_terms*covariate_terms - cov_squared@np.square(C_t))

	# Make Updates
	new_alpha = alpha_prior + (N/2.0) + 0.0
	new_beta = beta_prior + (np.sum(resid)/2.0) + 0.0

	tau_alpha = new_alpha
	tau_beta = new_beta

	return np.hstack((tau_alpha, tau_beta))


class EQTL_FACTORIZATION_VI(object):
	def __init__(self, K=25, alpha=1e-16, beta=1e-16, ard_alpha=1e-16, ard_beta=1e-16, gamma_v=1.0, max_iter=10, delta_elbo_threshold=.01, warmup_iterations=0,re_boolean=False, output_root=''):
		# Prior on gamma distributions defining variances
		self.alpha_prior = alpha
		self.beta_prior = beta
		# Prior on gamma distribution defining ARD variance
		self.ard_alpha_prior = ard_alpha
		self.ard_beta_prior = ard_beta
		# Maximum allowed iterations
		self.max_iter = max_iter
		# Number of latent factors
		self.K = K
		# Variance prior placed on elements of V
		self.gamma_v = gamma_v
		# Initialize iteration number
		self.iter = 0
		# Threshold on elbo change (for convergence detection)
		self.delta_elbo_threshold = delta_elbo_threshold
		# Output root (directory) to save intermediate results
		self.output_root = output_root
		# Number of iterations before ARD prior on U is started to be learned.
		self.warmup_iterations = warmup_iterations
		# Whether or not to include a random effects intercept
		self.re_boolean = re_boolean
	def fit(self, G, G_fe, Y, z, cov):
		""" Fit the model.
			Args:
			G: A genotype matrix of floats with shape [num_samples, num_tests].
			G_fe: A genotype matrix (used for fixed effect of genotype) of floats with shape [num_samples, num_tests]. Will be the same as G unless in a particular permutation scheme
			Y: An expression matrix of floats with shape [num_samples, num_tests].
			z: groupings of length num_samples
			cov: A covariate matrix of floats with shape [num_samples, num_covariates]  ... we assume this contains an intercept term
		"""
		self.print_logo()
		self.G = G
		self.G_fe = G_fe
		self.Y = Y
		self.z = np.asarray(z)
		self.cov = cov
		# Initialize variables
		print('Initialize variables')
		self.initialize_variables()
		# Elbo after initialization 
		self.update_elbo()
		print('##############')
		print('##############')
		# Loop through VI iterations
		for vi_iter in range(self.max_iter):
			print('Variational Inference iteration: ' + str(vi_iter))
			start_time = time.time()
			# Update parameter estimaters via coordinate ascent
			print('U update')
			self.update_U()
			print('V update')
			self.update_V()
			if self.re_boolean and vi_iter >= self.warmup_iterations:
				print('alpha update')
				self.update_alpha()
			print('C update')
			self.update_C()
			print('F update')
			self.update_F()
			# Only run gammaU update after X warmup iterations
			if vi_iter >= self.warmup_iterations: 
				print('gammaU update')
				self.update_gamma_U()
				if self.re_boolean:
					print('psi update')
					self.update_psi()
			print('tau update')
			self.update_tau()
			self.iter = self.iter + 1

			####################
			# Compute ELBO after update
			self.update_elbo()
			current_elbo = self.elbo[len(self.elbo)-1]
			delta_elbo = (current_elbo - self.elbo[len(self.elbo)-2])
			print('delta ELBO: ' + str(delta_elbo))
			####################
			# Print gamma parameter
			print('Gamma parameters: ')
			print(self.gamma_U_alpha/self.gamma_U_beta)
			#####################
			# Compute Genetic PVE
			self.shared_genetic_pve, self.factor_genetic_pve = self.compute_variance_explained_of_factors('genetic_pve')
			self.shared_pve, self.factor_pve = self.compute_variance_explained_of_factors('pve')
			print('Shared PVE: ')
			print(self.shared_pve)
			print('Factor PVE: ')
			print(self.factor_pve)
			#####################
			# Print runtime
			end_time = time.time()
			print('Iteration runtime: ' + str(end_time-start_time))
			print('##############')
			print('##############')

			# Print temporary results to output every X iterations
			if np.mod(vi_iter, 5) == 0 and vi_iter > 0:
				np.savetxt(self.output_root + 'temper_U_S.txt', (self.U_mu), fmt="%s", delimiter='\t')
				np.savetxt(self.output_root + 'temper_factor_genetic_pve.txt', (self.factor_genetic_pve), fmt="%s", delimiter='\t')
				np.savetxt(self.output_root + 'temper_factor_pve.txt', (self.factor_pve), fmt="%s", delimiter='\t')
				np.save(self.output_root + 'temper_U_S.npy', self.U_mu)
				np.save(self.output_root + 'temper_gamma_U.npy', self.gamma_U_alpha/self.gamma_U_beta)
				np.save(self.output_root + 'temper_V.npy', (self.V_mu))
				np.save(self.output_root + 'temper_F.npy', (self.F_mu))
				np.save(self.output_root + 'temper_alpha.npy', self.alpha_mu)
				np.save(self.output_root + 'temper_tau.npy', (self.tau_alpha/self.tau_beta))
				np.save(self.output_root + 'temper_psi.npy', (self.psi_alpha/self.psi_beta))
				np.save(self.output_root + 'temper_C.npy', (self.C_mu))
				np.savetxt(self.output_root + 'temper_iter.txt', np.asmatrix(vi_iter), fmt="%s", delimiter='\t')
				np.savetxt(self.output_root + 'temper_elbo.txt', np.asarray(self.elbo), fmt="%s", delimiter='\n')
	def print_logo(self):
		print('*********************************************************')
		print('SURGE Training')
		print('Single cell Unsupervised Regulation of Gene Expression')
		print('*********************************************************')
	def update_step_size(self):
		# Only needs to be done for SVI
		if self.SVI == True:
			self.step_size = self.learning_rate/(np.power((1.0 + (self.forgetting_rate*self.iter)), .75))
	def remove_irrelevent_factors(self):
		#shared_pve, factor_pve = self.compute_variance_explained_of_factors()
		theta_U = self.theta_U_a/(self.theta_U_a + self.theta_U_b)

		#ordered_indices = np.argsort(-theta_U)
		#num_indices = sum(theta_U > .01)
		#factor_ordering = ordered_indices[:num_indices]

		factor_ordering = np.where(theta_U > 0.01)[0]
		print(factor_ordering)
		self.U_mu = self.U_mu[:, factor_ordering]
		self.U_var = self.U_var[:, factor_ordering]
		self.S_U = self.S_U[:, factor_ordering]

		self.V_mu = self.V_mu[factor_ordering, :]
		self.V_var = self.V_var[factor_ordering, :]
		self.theta_U_a = self.theta_U_a[factor_ordering]
		self.theta_U_b = self.theta_U_b[factor_ordering]
		self.K = len(factor_ordering)
	def compute_variance_explained_of_factors(self, version):
		# Based on bottom of P21 of https://arxiv.org/pdf/1802.06931.pdf
		variance_effect = self.N*np.sum(self.tau_beta/self.tau_alpha)
	

		F_terms = self.G*np.dot(np.ones((self.N,1)),[self.F_mu])
		shared_genetic_effect = np.sum(np.square(F_terms))

		# Initailize array to keep track of variance explained from each factor
		U_S = self.U_mu
		V_S = self.V_mu
		factor_genetic_effects = []
		for k in range(self.K):
			componenent_effects = np.sum(np.square(self.G*(np.dot(np.transpose([U_S[:,k]]), [V_S[k,:]]))))
			factor_genetic_effects.append(componenent_effects)
		if version == 'genetic_pve':
			denominator = np.sum(factor_genetic_effects) + shared_genetic_effect
		elif version == 'pve':
			denominator = np.sum(factor_genetic_effects) + shared_genetic_effect + variance_effect
		else:
			print(version + ' currently not implemented for computing variance explained of factors')
			pdb.set_trace()
		shared_pve = shared_genetic_effect/denominator
		factor_pve = factor_genetic_effects/denominator
		return shared_pve, factor_pve
	def update_V(self):
		###################
		# UPDATE V
		###################
		# Precompute quantities
		U_S_expected_val = self.U_mu
		U_S_squared_expected_val = (np.square(self.U_mu) + self.U_var)
		tau_expected_val = self.tau_alpha/self.tau_beta
		V_mu_copy = np.copy(self.V_mu)
		V_var_copy = np.copy(self.V_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)

		# Keep track of variables
		V_update_data = []

		for test_index in range(self.T):
			V_update_data.append(outside_update_V_t(V_mu_copy[:, test_index], V_var_copy[:, test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.K, U_S_expected_val, U_S_squared_expected_val, self.F_mu[test_index], covariate_predicted[:, test_index], self.alpha_big_mu[:, test_index], self.gamma_v, tau_expected_val[test_index]))

		# Convert to array
		V_update_data = np.asarray(V_update_data).T
		# Fill in data structures
		self.V_mu = V_update_data[(self.K*0):(1*self.K), :]
		self.V_var = V_update_data[(self.K*1):(2*self.K), :]


	def update_U(self):
		###################
		# UPDATE U
		###################
		V_S_squared_expected_val = (np.square(self.V_mu) + self.V_var)
		U_mu_copy = np.copy(self.U_mu)
		U_var_copy = np.copy(self.U_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)
		U_update_data = []
		gamma_u = self.gamma_U_alpha/self.gamma_U_beta

		for sample_index in range(self.N):
			U_update_data.append(outside_update_U_n(U_mu_copy[sample_index,:], U_var_copy[sample_index,:], self.G[sample_index, :], self.G_fe[sample_index,:], self.Y[sample_index, :], self.K, self.V_mu, V_S_squared_expected_val, self.F_mu, covariate_predicted[sample_index, :], gamma_u, self.tau_alpha/self.tau_beta, self.alpha_big_mu[sample_index, :]))

		# Convert to array
		U_update_data = np.asarray(U_update_data)
		# Fill in data structures
		self.U_mu = U_update_data[:,(self.K*0):(1*self.K)]
		self.U_var = U_update_data[:,(self.K*1):(2*self.K)]
		#self.U_var = U_update_data[:,(self.K*2):(3*self.K)]

	def update_F(self):
		U_S_expected_val = self.U_mu

		tau_expected_val = self.tau_alpha/self.tau_beta
		F_mu_copy = np.copy(self.F_mu)
		F_var_copy = np.copy(self.F_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)
		F_update_data = []
		gamma_f = 0.001
		for test_index in range(self.T):
			F_update_data.append(outside_update_F_t(F_mu_copy[test_index], F_var_copy[test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], U_S_expected_val, self.V_mu[:,test_index], covariate_predicted[:, test_index], gamma_f, tau_expected_val[test_index], self.alpha_big_mu[:, test_index]))
		F_update_data = np.asarray(F_update_data)
		self.F_mu = F_update_data[:,0]
		self.F_var = F_update_data[:,1]

	def update_alpha(self):
		U_S_expected_val = self.U_mu
		tau_expected_val = self.tau_alpha/self.tau_beta
		psi_expected_val = self.psi_alpha/self.psi_beta
		alpha_mu_copy = np.copy(self.alpha_mu)
		alpha_var_copy = np.copy(self.alpha_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)

		alpha_update_data = []
		for test_index in range(self.T):
			alpha_update_data.append(outside_update_alpha_t(alpha_mu_copy[:, test_index], alpha_var_copy[:, test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.I, U_S_expected_val, self.V_mu[:, test_index], self.F_mu[test_index], covariate_predicted[:, test_index], tau_expected_val[test_index], psi_expected_val[test_index], self.individual_to_sample_indices, self.individual_to_number_full_indices))

		alpha_update_data = np.transpose(np.asarray(alpha_update_data))
		self.alpha_mu = alpha_update_data[:(self.I),:]
		self.alpha_var = alpha_update_data[(self.I):, :]
		# Now fill in big matrix
		self.alpha_big_mu = np.zeros((self.N, self.T))
		self.alpha_big_var = np.zeros((self.N, self.T))
		for sample_num, z_label in enumerate(self.z):
			self.alpha_big_mu[sample_num,:] = self.alpha_mu[self.z_mapping[z_label], :]
			self.alpha_big_var[sample_num,:] = self.alpha_var[self.z_mapping[z_label], :]

	def update_C(self):
		U_S_expected_val = self.U_mu
		tau_expected_val = self.tau_alpha/self.tau_beta
		C_mu_copy = np.copy(self.C_mu)
		C_var_copy = np.copy(self.C_var)


		C_update_data = []
		for test_index in range(self.T):
			C_update_data.append(outside_update_C_t(C_mu_copy[:, test_index], C_var_copy[:, test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.N, U_S_expected_val, self.V_mu[:,test_index], self.F_mu[test_index], tau_expected_val[test_index], self.alpha_big_mu[:, test_index], self.cov, self.cov_squared))
		C_update_data = np.transpose(np.asarray(C_update_data))

		# Fill in data structures
		self.C_mu = C_update_data[(self.num_cov*0):(1*self.num_cov), :]
		self.C_var = C_update_data[(self.num_cov*1):(2*self.num_cov), :]

	def update_gamma_U(self):
		# Loop through factors
		for k in range(self.K):
			#U_squared_k_expected_val = ((np.square(self.U_mu[:,k]) + self.U_var[:,k])*self.S_U[:,k]) + (1.0-self.S_U[:,k])*(self.gamma_U_beta[k]/self.gamma_U_alpha[k])
			U_squared_k_expected_val = np.square(self.U_mu[:,k]) + self.U_var[:,k]
			self.gamma_U_alpha[k] = self.ard_alpha_prior + (self.N/2.0)
			self.gamma_U_beta[k] = self.ard_beta_prior + (np.sum(U_squared_k_expected_val)/2.0)
	def update_theta_U(self):
		# Loop through factors
		for k in range(self.K):
			#self.theta_U_a[k] = self.a_prior + np.sum(self.S_U[:,k])
			#self.theta_U_b[k] = self.b_prior + self.N - np.sum(self.S_U[:,k])
			new_theta_U_a = self.a_prior + np.sum(self.S_U[:,k])
			new_theta_U_b = self.b_prior + (self.N - np.sum(self.S_U[:,k]))
			self.theta_U_a[k] = new_theta_U_a
			self.theta_U_b[k] = new_theta_U_b
	def update_psi(self):
		alpha_squared_expected_value = np.square(self.alpha_mu) + self.alpha_var
		# Loop through tests
		for test_index in range(self.T):
			self.psi_alpha[test_index] = self.alpha_prior + (self.I/2.0)
			self.psi_beta[test_index] = self.beta_prior + (np.sum(alpha_squared_expected_value[:,test_index])/2.0)
	def update_tau(self):
		tau_alpha_copy = np.copy(self.tau_alpha)
		tau_beta_copy = np.copy(self.tau_beta)

		# Precompute quantities
		F_S_squared = np.square(self.F_mu) + self.F_var
		V_S_squared = np.square(self.V_mu) + self.V_var
		U_S_squared = ((np.square(self.U_mu) + self.U_var))
		U_S = (self.U_mu)
		C_squared = np.square(self.C_mu) + self.C_var
		# Loop through tests
		tau_update_data = []
		for test_index in range(self.T):
			tau_update_data.append(outside_update_tau_t(tau_alpha_copy[test_index], tau_beta_copy[test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.N, U_S, self.V_mu[:,test_index], self.F_mu[test_index], self.C_mu[:, test_index], V_S_squared[:, test_index], F_S_squared[test_index], U_S_squared, C_squared[:, test_index], self.alpha_big_mu[:, test_index], self.alpha_big_var[:, test_index], self.cov, self.cov_squared, self.alpha_prior, self.beta_prior))
		tau_update_data = np.asarray(tau_update_data)
		self.tau_alpha = tau_update_data[:,0]
		self.tau_beta = tau_update_data[:,1]

	def update_elbo(self):
		data_likelihood_term = self.compute_elbo_log_likelihood_term()
		kl_V_S = self.compute_kl_divergence_of_V_S()
		kl_U_S = self.compute_kl_divergence_of_U_S()
		kl_F_S = self.compute_kl_divergence_of_F_S()
		kl_tau = self.compute_kl_divergence_of_tau()
		if self.re_boolean:
			kl_psi = self.compute_kl_divergence_of_psi()
			kl_alpha = self.compute_kl_divergence_of_alpha()
		kl_theta_u = self.compute_kl_divergence_of_gamma_u()
		kl_C = self.compute_kl_divergence_of_C()
		
		if self.re_boolean:
			kl_divergence = kl_V_S + kl_U_S + kl_F_S + kl_tau + kl_theta_u + kl_C + kl_psi + kl_alpha
		else:
			kl_divergence = kl_V_S + kl_U_S + kl_F_S + kl_tau + kl_theta_u + kl_C
		elbo = data_likelihood_term - kl_divergence
		self.elbo.append(elbo)

	def compute_kl_divergence_of_theta_u(self):
		a_prior = self.a_prior
		b_prior = self.b_prior
		theta_a = self.theta_U_a 
		theta_b = self.theta_U_b
		kl_divergence = compute_kl_divergence_of_beta(a_prior, b_prior, theta_a, theta_b)
		return kl_divergence
	def compute_kl_divergence_of_tau(self):
		alpha_prior = self.alpha_prior
		beta_prior = self.beta_prior
		gamma_alpha = self.tau_alpha
		gamma_beta = self.tau_beta
		kl_divergence = compute_kl_divergence_of_gamma(alpha_prior, beta_prior, gamma_alpha, gamma_beta)
		return kl_divergence
	def compute_kl_divergence_of_psi(self):
		alpha_prior = self.alpha_prior
		beta_prior = self.beta_prior
		gamma_alpha = self.psi_alpha
		gamma_beta = self.psi_beta
		kl_divergence = compute_kl_divergence_of_gamma(alpha_prior, beta_prior, gamma_alpha, gamma_beta)
		return kl_divergence

	def compute_kl_divergence_of_gamma_u(self):
		alpha_prior = self.ard_alpha_prior
		beta_prior = self.ard_beta_prior
		gamma_alpha = self.gamma_U_alpha
		gamma_beta = self.gamma_U_beta
		kl_divergence = compute_kl_divergence_of_gamma(alpha_prior, beta_prior, gamma_alpha, gamma_beta)
		return kl_divergence
	def compute_kl_divergence_of_C(self):
		W_mu = self.C_mu
		W_var = self.C_var
		expected_gamma = 0.0
		kl_divergence = compute_kl_divergence_of_gaussian_fixed_variance(W_mu, W_var, expected_gamma, self.C_mu.shape[0])
		return kl_divergence

	def compute_kl_divergence_of_F_S(self):
		W_mu = np.asarray([self.F_mu])
		W_var = np.asarray([self.F_var])
		expected_gamma = 0.001
		kl_divergence = compute_kl_divergence_of_gaussian_fixed_variance(W_mu, W_var, expected_gamma, 1)
		return kl_divergence
	def compute_kl_divergence_of_alpha(self):
		kl_divergence = compute_kl_divergence_of_gaussian(np.transpose(self.alpha_mu), np.transpose(self.alpha_var), self.psi_alpha, self.psi_beta, self.T)
		return kl_divergence
	def compute_kl_divergence_of_V_S(self):
		W_mu = self.V_mu
		W_var = self.V_var
		expected_gamma_v = self.gamma_v
		kl_divergence = compute_kl_divergence_of_gaussian_fixed_variance(W_mu, W_var, expected_gamma_v, self.K)
		return kl_divergence
	def compute_kl_divergence_of_U_S(self):
		W_mu = np.transpose(self.U_mu)
		W_var = np.transpose(self.U_var)
		kl_divergence = compute_kl_divergence_of_gaussian(W_mu, W_var, self.gamma_U_alpha, self.gamma_U_beta, self.K)
		return kl_divergence
	def compute_elbo_log_likelihood_term(self):
		# Compute expectation of log of gamma variables
		log_tau_expected = special.digamma(self.tau_alpha) - np.log(self.tau_beta)
		# Compute expectation of gamma variable
		tau_expected = self.tau_alpha/self.tau_beta
		# Other relevent expectations
		U_S = (self.U_mu)
		V_S = (self.V_mu)
		F_S = (self.F_mu)

		# alpha_squared = np.square(self.alpha_big_mu) + self.alpha_big_var
		# alpha = self.alpha_big_mu
		F_S_squared = ((np.square(self.F_mu) + self.F_var))
		V_S_squared = ((np.square(self.V_mu) + self.V_var))
		U_S_squared = ((np.square(self.U_mu) + self.U_var))
		C_squared = np.square(self.C_mu) + self.C_var

		component_squared_terms = np.dot(U_S_squared, V_S_squared)
		F_squared_terms = np.dot(np.ones((self.N,1)),[F_S_squared])
		component_terms = np.dot(U_S, V_S)
		F_terms = np.dot(np.ones((self.N,1)),[F_S])
		covariate_terms = np.dot(self.cov, self.C_mu)
		squared_covariate_terms = np.dot(np.square(self.cov), C_squared)
		alpha_squared_terms = np.square(self.alpha_big_mu) + self.alpha_big_var


		# Terms of interest in likelihood
		term_a = -np.log(2.0*np.pi)*(self.N*self.T/2.0)
		term_b = (self.N/2.0)*np.sum(log_tau_expected)
		# Compute residual matrix
		squared_residual_mat = np.square(self.Y) + alpha_squared_terms + squared_covariate_terms + np.square(self.G)*component_squared_terms + np.square(self.G_fe)*F_squared_terms
		squared_residual_mat = squared_residual_mat - 2.0*self.Y*(self.alpha_big_mu + covariate_terms + self.G*component_terms + self.G_fe*F_terms)
		squared_residual_mat = squared_residual_mat + 2.0*self.alpha_big_mu*(covariate_terms + self.G*component_terms + self.G_fe*F_terms)
		squared_residual_mat = squared_residual_mat + 2.0*covariate_terms*(self.G*component_terms + self.G_fe*F_terms)
		squared_residual_mat = squared_residual_mat + 2.0*self.G*self.G_fe*component_terms*F_terms

		squared_residual_mat = squared_residual_mat + (covariate_terms*covariate_terms - np.dot(np.square(self.cov), np.square(self.C_mu)))
		squared_residual_mat = squared_residual_mat + np.square(self.G)*(component_terms*component_terms - np.dot(np.square(U_S), np.square(V_S)))

		term_c = np.sum(squared_residual_mat*tau_expected)/2.0
		data_likelihood_term = term_a + term_b - term_c
		return data_likelihood_term
	# Randomly generate indices
	def get_svi_sample_indices(self):
		svi_sample_indices = []
		for individual_index in range(self.I):
			individual_indices = self.individual_to_sample_indices_full[individual_index]
			n_i = len(individual_indices)
			subset_n_i = int(np.ceil(n_i*self.sample_batch_fraction))
			randos = np.random.choice(individual_indices, size=int(subset_n_i), replace=False)
			for rando in randos:
				svi_sample_indices.append(rando)
		svi_sample_indices = np.asarray(svi_sample_indices)
		if len(svi_sample_indices) > self.N:
			svi_sample_indices = np.random.choice(svi_sample_indices,size=self.N, replace=False)
		if len(svi_sample_indices) != self.N:
			print('svi_sample_indices assumption error1')
			pdb.set_trace()
		if len(np.unique(svi_sample_indices)) != self.N:
			print('svi_sample_indices assumption error2')
			pdb.set_trace()
		return svi_sample_indices
		# return np.random.choice(self.N_full, size=self.N, replace=False)
	def initialize_variables(self):
		# Initialize array to keep track of ELBO
		self.elbo = []

		# Add model dimensions to object
		self.N = self.Y.shape[0]
		self.T = self.Y.shape[1]
		self.num_cov = self.cov.shape[1]

		# Initialze U
		pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
		pca.fit(np.random.randn(self.N, 9999).T)
		self.U_mu = pca.components_.T
		for k in range(self.K):
			self.U_mu[:,k] = ((self.U_mu[:,k]-np.mean(self.U_mu[:,k]))/np.std(self.U_mu[:,k]))
		#self.U_mu = self.U_mu*0.0
		self.U_var = np.ones((self.N, self.K))*(1.0/self.gamma_v) 
		self.gamma_U_alpha = np.ones(self.K)
		self.gamma_U_beta = np.ones(self.K)
		#self.S_U = np.ones((self.N,self.K))

		# Random effects
		self.z_mapping = {}
		self.z_inverse_mapping = {}
		# Create mapping from grouping to index
		_, idx = np.unique(self.z, return_index=True)
		unique_groups = np.asarray(self.z)[np.sort(idx)]
		for i, label in enumerate(unique_groups):
			self.z_mapping[label] = i
			self.z_inverse_mapping[i] = label
		self.I = len(np.unique(self.z))
		self.individual_to_sample_indices = []
		self.individual_to_sample_indices_full = []
		self.individual_to_number_full_indices = []
		for ii in range(self.I):
			# z_label corresponding to this individual
			z_label = self.z_inverse_mapping[ii]
			sample_indices = np.where(np.asarray(self.z) == z_label)[0]
			self.individual_to_sample_indices.append(sample_indices)
			self.individual_to_sample_indices_full.append(sample_indices)
			self.individual_to_number_full_indices.append(float(len(sample_indices)))

		# Random effects variances
		self.psi_alpha = np.ones(self.T)*self.alpha_prior
		self.psi_beta = np.ones(self.T)*self.beta_prior*.0001 # Initialize random effects variance to be samller than residual variance (tau)

		# Random effects
		self.alpha_mu = np.zeros((self.I, self.T))
		if self.re_boolean == True:
			self.alpha_var = (np.zeros((self.I, self.T)) + 1.0)*1e-12
		elif self.re_boolean == False:
			self.alpha_var = np.zeros((self.I, self.T))
		# Convert random effects matrix to samplesXtests instead of groupsXtest
		self.alpha_big_mu = np.zeros((self.N, self.T))
		self.alpha_big_var = np.zeros((self.N, self.T))
		for sample_num, z_label in enumerate(self.z):
			self.alpha_big_mu[sample_num,:] = self.alpha_mu[self.z_mapping[z_label], :]
			self.alpha_big_var[sample_num,:] = self.alpha_var[self.z_mapping[z_label], :]

		# Intialize V
		pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
		pca.fit(np.random.randn(self.T, 9999).T)
		self.V_mu = pca.components_
		for k in range(self.K):
			self.V_mu[k,:] = ((self.V_mu[k,:]-np.mean(self.V_mu[k,:]))/np.std(self.V_mu[k,:]))
		self.V_var = np.ones((self.K, self.T))*(1.0/self.gamma_v)

		# Initialize C and F
		F_betas, C_betas, residual_varz = run_linear_model_for_initialization(self.Y, self.G_fe, self.cov, self.z)
		self.F_mu = F_betas
		self.F_var = np.ones(self.T)
		self.C_mu = np.transpose(C_betas)
		self.C_var = np.ones(self.C_mu.shape)

		self.cov_squared = np.square(self.cov)
		# Variances
		self.tau_alpha = np.ones(self.T)*self.alpha_prior
		self.tau_beta = np.ones(self.T)*self.beta_prior
		self.print_diagnostic_data()
	def print_diagnostic_data(self):
		print(str(self.N) + ' samples detected')
		print(str(self.T) + ' tests detected')
		print(str(self.K) + ' latent factors detected')