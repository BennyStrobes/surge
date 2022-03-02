import numpy as np 
import sys
import pdb
import scipy.special as special
from sklearn.linear_model import LinearRegression
import time
import sklearn.decomposition


def run_linear_model_for_initialization(Y, G, cov, z):
	num_tests = Y.shape[1]
	F_betas = []
	C_betas = []
	residual_varz = []
	for test_number in range(num_tests):
		y_vec = Y[:,test_number]
		g_vec = G[:,test_number]
		X = np.hstack((np.transpose(np.asmatrix(g_vec)), cov))
		reg = LinearRegression(fit_intercept=False).fit(X, np.transpose(np.asmatrix(y_vec)))
		F_betas.append(reg.coef_[0][0])
		C_betas.append(reg.coef_[0][1:])
		pred_y = reg.predict(X)[:,0]
		resid_y = y_vec - pred_y
		residual_varz.append(np.var(resid_y))
	return np.asarray(F_betas), np.asarray(C_betas), np.asarray(residual_varz)



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
	# Loop through latent contexts
	for k in range(K):
		# Compute relevent expectations
		U_S_expected_val = U_mu
		V_k_S_k_expected_val = V_S_expected_val[k,:]
		# Compute expectations on other components (!= k)
		other_components_expected = (U_S_expected_val@V_S_expected_val) - U_S_expected_val[k]*V_S_expected_val[k,:]
		# Update variance of q(U) according to VI update
		a_term = np.sum(tau_expected_val*np.square(G_slice)*V_S_squared_expected_val[k,:]) + gamma_u[k]
		U_var[k] = 1.0/a_term
		# Update mean of q(U) according to VI update
		resid = Y_slice - covariate_predicted_slice - alpha_i_expected_val - G_fe_slice*F_S_expected_val - G_slice*other_components_expected
		b_term = np.sum(tau_expected_val*G_slice*V_k_S_k_expected_val*resid)
		U_mu[k] = U_var[k]*b_term
	return np.hstack((U_mu, U_var))

def outside_update_V_t(V_mu, V_var, G_slice, G_fe_slice, Y_slice, K, U_S_expected_val, U_S_squared_expected_val, F_S_t_expected_val, covariate_predicted_slice, alpha_t_mu, gamma_v, tau_t_expected_val):
	# Loop through latent contexts
	for k in range(K):
		# Compute expectations on other components (!= k)
		other_components_expected = (U_S_expected_val@V_mu) - U_S_expected_val[:, k]*V_mu[k]
		# Update variance of q(V) according to VI update
		a_term = gamma_v + (tau_t_expected_val*np.sum(np.square(G_slice)*U_S_squared_expected_val[:,k]))
		# Update mean of q(U) according to VI update
		resid = Y_slice - alpha_t_mu - covariate_predicted_slice - G_slice*other_components_expected - G_fe_slice*F_S_t_expected_val
		b_term = np.sum(tau_t_expected_val*G_slice*U_S_expected_val[:,k]*resid)
		new_var = 1.0/a_term
		new_mu = new_var*b_term
		V_var[k] = new_var
		V_mu[k] = new_mu
	return np.hstack((V_mu, V_var))


def outside_update_C_t(C_t_mu, C_t_var, G_slice, G_fe_slice, Y_slice, N, U_S_expected_val, V_S_t_expected_val, F_S_t_expected_val, tau_t_expected_val, alpha_t_mu, cov, cov_squared):
	# Precompute quantities relavent to VI updates
	num_cov = len(C_t_mu)
	components_expected = U_S_expected_val@V_S_t_expected_val
	base_resid = Y_slice - alpha_t_mu - G_fe_slice*F_S_t_expected_val - G_slice*components_expected

	# Loop through covariates
	for cov_num in range(num_cov):
		# Expectation of all other coviates (!= cov_num)
		other_covariates = (cov@C_t_mu) - (cov[:, cov_num]*C_t_mu[cov_num])

		# Update mean and variance of Q(C) according to VI updates
		b_term = tau_t_expected_val*np.sum(cov[:, cov_num]*(base_resid - other_covariates))
		a_term = 0.0 + tau_t_expected_val*np.sum(cov_squared[:,cov_num])
		new_var = 1.0/a_term
		new_mu = new_var*b_term
		C_t_mu[cov_num] = new_mu
		C_t_var[cov_num] = new_var
	return np.hstack((C_t_mu, C_t_var))

def outside_update_alpha_t(alpha_mu_copy, alpha_var_copy, G_slice, G_fe_slice, Y_slice, I, U_S_expected_val, V_S_t_expected_val, F_S_t_expected_val, covariate_predicted_t, tau_t_expected_val, psi_t_expected_val, individual_to_sample_indices, individual_to_number_full_indices):
	# Precompute quanties relevative to VI updates
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

		# Update variance of q(alpha_it) according to VI updates
		new_var = 1.0/((1.0/individual_batch_fraction)*n_i*tau_t_expected_val + psi_t_expected_val)
		# Update mean of q(alpha_it) according to VI updates
		new_mu = new_var*tau_t_expected_val*np.sum(resid[sample_indices])*(1.0/individual_batch_fraction)
		alpha_var_copy[individual_index] = new_var
		alpha_mu_copy[individual_index] = new_mu
	return np.hstack((alpha_mu_copy, alpha_var_copy))

def outside_update_F_t(F_mu, F_var, G_slice, G_fe_slice, Y_slice, U_S_expected_val, V_S_t_expected_val, covariate_predicted_t, gamma_f_expected_val, tau_t_expected_val, alpha_t_mu):
	# Compute expectations from G(UXV)
	other_components_expected = U_S_expected_val@V_S_t_expected_val

	# Update variance of q(F) according to VI update
	a_term = gamma_f_expected_val + tau_t_expected_val*np.sum(np.square(G_fe_slice))
	# Update mean of q(F) according to VI update
	resid = Y_slice - alpha_t_mu - covariate_predicted_t - G_slice*(other_components_expected)
	b_term = np.sum(tau_t_expected_val*G_fe_slice*resid)
	new_var = 1.0/a_term
	new_mu = new_var*b_term
	F_var = new_var
	F_mu = new_mu

	return np.hstack((F_mu, F_var))

def outside_update_tau_t(tau_alpha, tau_beta, G_slice, G_fe_slice, Y_slice, N, U_S, V_S_t, F_S_t, C_t, V_S_t_squared, F_S_t_squared, U_S_squared, C_t_squared, alpha_mu_t, alpha_var_t, cov, cov_squared, alpha_prior, beta_prior):
	# Compute expectations relevent to VI Update
	squared_factor_terms = U_S_squared@V_S_t_squared
	factor_terms = U_S@V_S_t
	alpha_t_squared = np.square(alpha_mu_t) + alpha_var_t
	squared_covariate_terms = cov_squared@C_t_squared
	covariate_terms = cov@C_t

	# First add together square terms
	resid = np.square(Y_slice) + alpha_t_squared + squared_covariate_terms + np.square(G_fe_slice)*F_S_t_squared + np.square(G_slice)*squared_factor_terms
	# Now add terms with Y
	resid = resid - (2.0*Y_slice*(covariate_terms + alpha_mu_t + G_slice*factor_terms + G_fe_slice*F_S_t))
	# Now add terms with cov (w/o Y)
	resid = resid + 2.0*covariate_terms*(alpha_mu_t + G_slice*factor_terms + G_fe_slice*F_S_t)
	# Now add terms with alpha (w/o cov and Y)
	resid = resid + 2.0*alpha_mu_t*(G_slice*factor_terms + G_fe_slice*F_S_t)
	# Now add terms with factors
	resid = resid + 2.0*G_slice*factor_terms*G_fe_slice*F_S_t
	# Now add terms with interactions between factors
	resid = resid + (np.square(G_slice)*(factor_terms*factor_terms - np.sum(np.square(U_S*V_S_t),axis=1)))
	resid = resid + (covariate_terms*covariate_terms - cov_squared@np.square(C_t))

	# Make Updates
	new_alpha = alpha_prior + (N/2.0)
	new_beta = beta_prior + (np.sum(resid)/2.0)

	return np.hstack((new_alpha, new_beta))


class SURGE_VI(object):
	def __init__(self, K=20, alpha=1e-3, beta=1e-3, ard_alpha=1e-3, ard_beta=1e-3, gamma_v=1.0, max_iter=1000, delta_elbo_threshold=1e-2, warmup_iterations=5, re_boolean=False, verbose=False, output_root=''):
		# Prior on gamma distributions defining residual variance and
		self.alpha_prior = alpha
		self.beta_prior = beta
		# Prior on gamma distribution defining automatic relavence determination variance (ie variance on U ie variance on surge latent contexts)
		self.ard_alpha_prior = ard_alpha
		self.ard_beta_prior = ard_beta
		# Maximum allowed iterations
		self.max_iter = max_iter
		# Number of initial latent factors
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
		# Whether or not to be verbose about optimization
		self.verbose = verbose
	def fit(self, G, Y, cov, G_fe=None, z=None):
		""" Fit the model.
			Args:
			G: A genotype matrix of floats with shape [num_samples, num_tests].
			G_fe: A genotype matrix (used for fixed effect of genotype) of floats with shape [num_samples, num_tests]. Will be the same as G unless using a particular permutation scheme
			Y: An expression matrix of floats with shape [num_samples, num_tests].
			z: groupings vector of length num_samples
			cov: A covariate matrix of floats with shape [num_samples, num_covariates]  ... we assume this contains an intercept term
		"""
		# Allow for user to input only G, and not G_fe
		if G_fe is None:
			G_fe = np.copy(G)
		# Allow for user to not input z (iff re_boolean==False)
		if z is None:
			z = np.arange(Y.shape[0])
			if self.re_boolean:
				print('Error: must provide random effects groups if running surge with re_boolean==True\nQuitting now.')
				return

		# Print SURGE logo
		self.print_logo()

		# Add data to SURGE_VI object
		self.G = G
		self.G_fe = G_fe
		self.Y = Y
		self.z = np.asarray(z)
		self.cov = cov
		self.converged = False  # Variable keeping track of whether optimization converges

		#####################
		# Initialize variables
		print('###############################')
		print('Initialize variables')
		print('###############################')
		self.initialize_variables()
		# Elbo after initialization 
		self.update_elbo()

		#####################
		# Loop through VI iterations
		print('###############################')
		print('Begin Coordinate ascent variational inference (CAVI) iterative algorithm')
		print('###############################')
		for vi_iter in range(self.max_iter):
			#####################
			# CAVI updates in one iteration
			print('Variational Inference iteration: ' + str(vi_iter))
			start_time = time.time()
			# Update parameter estimaters via coordinate ascent
			self.update_U()
			self.update_V()
			# Only run alpha update if fitting random effects intercept to model sample repeat structure (re_boolean==True)
			if self.re_boolean:
				self.update_alpha()
			self.update_C()
			self.update_F()
			# Only run gammaU update after X warmup iterations (X=self.warmup_iterations)
			if vi_iter >= self.warmup_iterations: 
				self.update_gamma_U()
			# Only run psi update if fitting random effects intercept to model sample repeat structure (re_boolean==True)
			if self.re_boolean:
				self.update_psi()
			self.update_tau()
			self.iter = self.iter + 1

			#####################
			# Compute Genetic PVE
			self.shared_genetic_pve, self.factor_genetic_pve = self.compute_variance_explained_of_factors('genetic_pve')
			self.shared_pve, self.factor_pve = self.compute_variance_explained_of_factors('pve')
			

			####################
			# Compute ELBO after update
			self.update_elbo()
			current_elbo = self.elbo[len(self.elbo)-1]
			delta_elbo = (current_elbo - self.elbo[len(self.elbo)-2])
			
			####################
			# Print change in elbo
			print('delta ELBO: ' + str(delta_elbo))

			####################
			# Print things if we are being verbose
			end_time = time.time()
			if self.verbose:
				self.verbose_printing(start_time, end_time, vi_iter)

			####################
			# Check for convergence
			if delta_elbo <= self.delta_elbo_threshold:
				self.converged = True
				break

		print('###############################')
		if self.converged == False:
			print('SURGE did not converge after maximum number (' + str(self.iter) + ') iterations.')
		else:
			print('SURGE converged after ' + str(self.iter) + ' iterations.')
		print('###############################')

	def verbose_printing(self, start_time, end_time, vi_iter):
		####################
		# Print gamma parameter
		print('Gamma parameters: ')
		print(self.gamma_U_alpha/self.gamma_U_beta)
		print('Shared PVE: ')
		print(self.shared_pve)
		print('Factor PVE: ')
		print(self.factor_pve)
		#####################
		# Print runtime
		print('Iteration runtime: ' + str(end_time-start_time))
		print('##############')
		print('##############')

		# Print temporary results to output every X iterations
		if np.mod(vi_iter, 10) == 0 and vi_iter > 0:
			np.savetxt(self.output_root + 'temp_U_S.txt', (self.U_mu), fmt="%s", delimiter='\t')
			np.savetxt(self.output_root + 'temp_factor_genetic_pve.txt', (self.factor_genetic_pve), fmt="%s", delimiter='\t')
			np.savetxt(self.output_root + 'temp_factor_pve.txt', (self.factor_pve), fmt="%s", delimiter='\t')
			np.save(self.output_root + 'temp_U_S.npy', self.U_mu)
			np.save(self.output_root + 'temp_gamma_U.npy', self.gamma_U_alpha/self.gamma_U_beta)
			np.save(self.output_root + 'temp_V.npy', (self.V_mu))
			np.save(self.output_root + 'temp_F.npy', (self.F_mu))
			np.save(self.output_root + 'temp_alpha.npy', self.alpha_mu)
			np.save(self.output_root + 'temp_tau.npy', (self.tau_alpha/self.tau_beta))
			np.save(self.output_root + 'temp_psi.npy', (self.psi_alpha/self.psi_beta))
			np.save(self.output_root + 'temp_C.npy', (self.C_mu))
			np.savetxt(self.output_root + 'temp_iter.txt', np.asmatrix(vi_iter), fmt="%s", delimiter='\t')
			np.savetxt(self.output_root + 'temp_elbo.txt', np.asarray(self.elbo), fmt="%s", delimiter='\n')

	def print_logo(self):
		print('*********************************************************')
		print('SURGE')
		print('Single cell Unsupervised Regulation of Gene Expression')
		print('*********************************************************')

	def compute_variance_explained_of_factors(self, version):
		##################################################################
		# Based on bottom of P21 of https://arxiv.org/pdf/1802.06931.pdf
		##################################################################

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
		shared_pve = shared_genetic_effect/denominator
		factor_pve = factor_genetic_effects/denominator
		return shared_pve, factor_pve
	def update_V(self):
		###################
		# UPDATE V
		###################
		# Precompute quantities relavent to VI updates
		U_S_expected_val = self.U_mu
		U_S_squared_expected_val = (np.square(self.U_mu) + self.U_var)
		tau_expected_val = self.tau_alpha/self.tau_beta
		V_mu_copy = np.copy(self.V_mu)
		V_var_copy = np.copy(self.V_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)

		# Keep track of updated variables
		V_update_data = []

		# Loop through each test
		for test_index in range(self.T):
			# Update V for a particular test
			V_update_data.append(outside_update_V_t(V_mu_copy[:, test_index], V_var_copy[:, test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.K, U_S_expected_val, U_S_squared_expected_val, self.F_mu[test_index], covariate_predicted[:, test_index], self.alpha_big_mu[:, test_index], self.gamma_v, tau_expected_val[test_index]))

		# Convert V_update_data to np array
		V_update_data = np.asarray(V_update_data).T

		# Update SURGE_VI data structures
		self.V_mu = V_update_data[(self.K*0):(1*self.K), :]
		self.V_var = V_update_data[(self.K*1):(2*self.K), :]


	def update_U(self):
		###################
		# UPDATE U
		###################
		# Precompute quantities relavent to VI updates
		V_S_squared_expected_val = (np.square(self.V_mu) + self.V_var)
		U_mu_copy = np.copy(self.U_mu)
		U_var_copy = np.copy(self.U_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)
		gamma_u = self.gamma_U_alpha/self.gamma_U_beta

		# Keep track of updated variables
		U_update_data = []

		# Loop through each sample
		for sample_index in range(self.N):
			# Update U for a particular sample (ok because all samples are independent)
			U_update_data.append(outside_update_U_n(U_mu_copy[sample_index,:], U_var_copy[sample_index,:], self.G[sample_index, :], self.G_fe[sample_index,:], self.Y[sample_index, :], self.K, self.V_mu, V_S_squared_expected_val, self.F_mu, covariate_predicted[sample_index, :], gamma_u, self.tau_alpha/self.tau_beta, self.alpha_big_mu[sample_index, :]))

		# Convert U_update_data to np array
		U_update_data = np.asarray(U_update_data)

		# Update SURGE_VI data structures
		self.U_mu = U_update_data[:,(self.K*0):(1*self.K)]
		self.U_var = U_update_data[:,(self.K*1):(2*self.K)]

	def update_F(self):
		###################
		# UPDATE F
		###################
		# Precompute quantities relavent to VI updates
		U_S_expected_val = self.U_mu
		tau_expected_val = self.tau_alpha/self.tau_beta
		F_mu_copy = np.copy(self.F_mu)
		F_var_copy = np.copy(self.F_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)

		# Fix precision on F to be relatively small
		gamma_f = 0.001

		# Keep track of updated variables
		F_update_data = []

		# loop through each test
		for test_index in range(self.T):
			# Update F for a particular test
			F_update_data.append(outside_update_F_t(F_mu_copy[test_index], F_var_copy[test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], U_S_expected_val, self.V_mu[:,test_index], covariate_predicted[:, test_index], gamma_f, tau_expected_val[test_index], self.alpha_big_mu[:, test_index]))
		
		# Convert F_update_data to np array
		F_update_data = np.asarray(F_update_data)

		# Update SURGE_VI data structures
		self.F_mu = F_update_data[:,0]
		self.F_var = F_update_data[:,1]

	def update_alpha(self):
		###################
		# UPDATE alpha
		###################
		# Precompute quantities relavent to VI updates
		U_S_expected_val = self.U_mu
		tau_expected_val = self.tau_alpha/self.tau_beta
		psi_expected_val = self.psi_alpha/self.psi_beta
		alpha_mu_copy = np.copy(self.alpha_mu)
		alpha_var_copy = np.copy(self.alpha_var)
		covariate_predicted = np.dot(self.cov, self.C_mu)

		# Keep track of updated variables
		alpha_update_data = []

		# Loop through each test
		for test_index in range(self.T):
			# Update alpha for a particular test
			alpha_update_data.append(outside_update_alpha_t(alpha_mu_copy[:, test_index], alpha_var_copy[:, test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.I, U_S_expected_val, self.V_mu[:, test_index], self.F_mu[test_index], covariate_predicted[:, test_index], tau_expected_val[test_index], psi_expected_val[test_index], self.individual_to_sample_indices, self.individual_to_number_full_indices))

		# Convert alpha_update_data to np array
		alpha_update_data = np.transpose(np.asarray(alpha_update_data))

		# Update SURGE_VI data structures
		self.alpha_mu = alpha_update_data[:(self.I),:]
		self.alpha_var = alpha_update_data[(self.I):, :]
		# Convert random effects matrix to samplesXtests from  individualsXtests
		self.alpha_big_mu = np.zeros((self.N, self.T))
		self.alpha_big_var = np.zeros((self.N, self.T))
		for sample_num, z_label in enumerate(self.z):
			self.alpha_big_mu[sample_num,:] = self.alpha_mu[self.z_mapping[z_label], :]
			self.alpha_big_var[sample_num,:] = self.alpha_var[self.z_mapping[z_label], :]

	def update_C(self):
		###################
		# UPDATE C
		###################
		# Precompute quantities relavent to VI updates
		U_S_expected_val = self.U_mu
		tau_expected_val = self.tau_alpha/self.tau_beta
		C_mu_copy = np.copy(self.C_mu)
		C_var_copy = np.copy(self.C_var)

		# Keep track of updated variables
		C_update_data = []

		# Loop through each test
		for test_index in range(self.T):
			# Update C for a particular test
			C_update_data.append(outside_update_C_t(C_mu_copy[:, test_index], C_var_copy[:, test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.N, U_S_expected_val, self.V_mu[:,test_index], self.F_mu[test_index], tau_expected_val[test_index], self.alpha_big_mu[:, test_index], self.cov, self.cov_squared))
		
		# Convert C_update_data to np array
		C_update_data = np.transpose(np.asarray(C_update_data))

		# Updaet SURGE_VI data structures
		self.C_mu = C_update_data[(self.num_cov*0):(1*self.num_cov), :]
		self.C_var = C_update_data[(self.num_cov*1):(2*self.num_cov), :]

	def update_gamma_U(self):
		###################
		# UPDATE gamma_U
		###################
		# Loop through latent contexts
		for k in range(self.K):
			# Precompute quantity to VI updates
			U_squared_k_expected_val = np.square(self.U_mu[:,k]) + self.U_var[:,k]
			# VI updates
			self.gamma_U_alpha[k] = self.ard_alpha_prior + (self.N/2.0)
			self.gamma_U_beta[k] = self.ard_beta_prior + (np.sum(U_squared_k_expected_val)/2.0)
	def update_psi(self):
		###################
		# UPDATE psi
		###################
		# Precompute quantities relavent to VI updates
		alpha_squared_expected_value = np.square(self.alpha_mu) + self.alpha_var
		# Loop through tests
		for test_index in range(self.T):
			# VI Updates
			self.psi_alpha[test_index] = self.alpha_prior + (self.I/2.0)
			self.psi_beta[test_index] = self.beta_prior + (np.sum(alpha_squared_expected_value[:,test_index])/2.0)
	def update_tau(self):
		###################
		# UPDATE tau
		###################

		# Precompute quantities relavent to VI updates
		tau_alpha_copy = np.copy(self.tau_alpha)
		tau_beta_copy = np.copy(self.tau_beta)
		F_S_squared = np.square(self.F_mu) + self.F_var
		V_S_squared = np.square(self.V_mu) + self.V_var
		U_S_squared = ((np.square(self.U_mu) + self.U_var))
		U_S = (self.U_mu)
		C_squared = np.square(self.C_mu) + self.C_var

		# Keep track of updated data
		tau_update_data = []

		# Loop through tests
		for test_index in range(self.T):
			# Update tau for a particular test
			tau_update_data.append(outside_update_tau_t(tau_alpha_copy[test_index], tau_beta_copy[test_index], self.G[:, test_index], self.G_fe[:, test_index], self.Y[:, test_index], self.N, U_S, self.V_mu[:,test_index], self.F_mu[test_index], self.C_mu[:, test_index], V_S_squared[:, test_index], F_S_squared[test_index], U_S_squared, C_squared[:, test_index], self.alpha_big_mu[:, test_index], self.alpha_big_var[:, test_index], self.cov, self.cov_squared, self.alpha_prior, self.beta_prior))
		
		# Convert tau_update_data to np array
		tau_update_data = np.asarray(tau_update_data)

		# Update SURGE_VI data structures
		self.tau_alpha = tau_update_data[:,0]
		self.tau_beta = tau_update_data[:,1]

	def update_elbo(self):
		# Elbo is equal to likelihood minus KL divergence from all variational parameters
		data_likelihood_term = self.compute_elbo_log_likelihood_term()
		kl_V_S = self.compute_kl_divergence_of_V_S()
		kl_U_S = self.compute_kl_divergence_of_U_S()
		kl_F_S = self.compute_kl_divergence_of_F_S()
		kl_tau = self.compute_kl_divergence_of_tau()
		if self.re_boolean:
			kl_psi = self.compute_kl_divergence_of_psi()
			kl_alpha = self.compute_kl_divergence_of_alpha()
		kl_gamma_u = self.compute_kl_divergence_of_gamma_u()
		kl_C = self.compute_kl_divergence_of_C()
		
		if self.re_boolean:
			kl_divergence = kl_V_S + kl_U_S + kl_F_S + kl_tau + kl_gamma_u + kl_C + kl_psi + kl_alpha
		else:
			kl_divergence = kl_V_S + kl_U_S + kl_F_S + kl_tau + kl_gamma_u + kl_C
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

	def initialize_variables(self):
		# Initialize array to keep track of ELBO
		self.elbo = []

		# Add model dimensions to SURGE_VI object
		self.N = self.Y.shape[0]
		self.T = self.Y.shape[1]
		self.num_cov = self.cov.shape[1]

		# Initialze U (parameterized in variational approximation by by U ~ N(U_mu, U_var))
		pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
		pca.fit(np.random.randn(self.N, 9999).T)
		self.U_mu = pca.components_.T
		# Standardize each of components of U
		for k in range(self.K):
			self.U_mu[:,k] = ((self.U_mu[:,k]-np.mean(self.U_mu[:,k]))/np.std(self.U_mu[:,k]))
		self.U_var = np.ones((self.N, self.K))*(1.0/self.gamma_v) 
		
		# Initialize gamma_U (parameterized in variational approximation by gamma_U ~ gamma(gamma_U_alpha, gamma_U_beta))
		# gamma_U is precision of prior on U s.t. U_nk ~ N(0, 1.0/gamma_U_k)
		self.gamma_U_alpha = np.ones(self.K)
		self.gamma_U_beta = np.ones(self.K)
		

		# Organize sample repeat structure in the data (ie having multiple samples from the same individual)
		# This makes it faster to update variational parameters down the road.
		self.I = len(np.unique(self.z))  # Number of individual ids
		self.z_mapping = {}  # Create dictionary mapping from individual id (as provided by user in z) to individual index (defined by range(I))
		self.z_inverse_mapping = {}  # Create dictionary mapping from individual index (range(I)) to individual id (as provided by user through z)
		# Create mapping from grouping to index
		_, idx = np.unique(self.z, return_index=True)
		unique_groups = np.asarray(self.z)[np.sort(idx)]
		for i, label in enumerate(unique_groups):
			self.z_mapping[label] = i
			self.z_inverse_mapping[i] = label
		# Create mapping from individual index to list of sample indices corresponding to that individual
		self.individual_to_sample_indices = []
		self.individual_to_number_full_indices = []
		for ii in range(self.I):
			# z_label corresponding to this individual
			z_label = self.z_inverse_mapping[ii]
			sample_indices = np.where(np.asarray(self.z) == z_label)[0]
			self.individual_to_sample_indices.append(sample_indices)
			self.individual_to_number_full_indices.append(float(len(sample_indices)))


		# Initialize psi (parameterized in variational approximation by psi ~ gamma(psi_alpha, psi_beta))
		# psi_t is precision of prior on alpha s.t. alpha_it ~ N(0, 1/psi_t)
		self.psi_alpha = np.ones(self.T)*self.alpha_prior
		self.psi_beta = np.ones(self.T)*self.beta_prior*.001 # Initialize random effects variance to be samller than residual variance (tau)

		# Initialize alpha (parameterized in variational approximation by alpha ~ N(alpha_mu, alpha_var))
		self.alpha_mu = np.zeros((self.I, self.T))  # Initialize to zero random effect mean
		if self.re_boolean == True:
			self.alpha_var = (np.zeros((self.I, self.T)) + 1.0)*.001  # Initialize to have same variance as that initialized by psi
		elif self.re_boolean == False:
			self.alpha_var = np.zeros((self.I, self.T))  # If re_boolean == False, set alpha ~ N(0,0) under variational approximation.

		# Convert random effects matrix to samplesXtests from  individualsXtests
		self.alpha_big_mu = np.zeros((self.N, self.T))
		self.alpha_big_var = np.zeros((self.N, self.T))
		for sample_num, z_label in enumerate(self.z):
			self.alpha_big_mu[sample_num,:] = self.alpha_mu[self.z_mapping[z_label], :]
			self.alpha_big_var[sample_num,:] = self.alpha_var[self.z_mapping[z_label], :]

		# Intialize V (parameterized in variational approximation by by V ~ N(V_mu, V_var))
		pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
		pca.fit(np.random.randn(self.T, 9999).T)
		self.V_mu = pca.components_
		for k in range(self.K):
			self.V_mu[k,:] = ((self.V_mu[k,:]-np.mean(self.V_mu[k,:]))/np.std(self.V_mu[k,:]))
		self.V_var = np.ones((self.K, self.T))*(1.0/self.gamma_v)

		# Initialize C and F according to estimates from linear model (Y ~ F_beta*self.G_fe + C_betas*self.cov)
		F_betas, C_betas, residual_varz = run_linear_model_for_initialization(self.Y, self.G_fe, self.cov, self.z)
		# Initialize F (parameterized in variational approximation by F ~ N(F_mu, F_var))
		self.F_mu = F_betas
		self.F_var = np.ones(self.T)
		# Initialize C (parameterized in variational approximation by C ~ N(C_mu, C_var))
		self.C_mu = np.transpose(C_betas)
		self.C_var = np.ones(self.C_mu.shape)

		# Initialize tau (residual variance)
		# Parameterized in variational approximation by tau ~ gamma(tau_alpha, tau_beta)
		# tau_t is the residual precision for test t
		self.tau_alpha = np.ones(self.T)*self.alpha_prior
		self.tau_beta = np.ones(self.T)*self.beta_prior

		# Pre-compute a relevent quantity (that will be used downstream many times)
		self.cov_squared = np.square(self.cov)

		# Print summary of data
		self.print_diagnostic_data()
	def print_diagnostic_data(self):
		print(str(self.N) + ' samples detected')
		print(str(self.T) + ' tests detected')
		print(str(self.K) + ' latent factors detected')