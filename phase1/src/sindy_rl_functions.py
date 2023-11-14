import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.helper_functions import *
import pysindy as ps
from pysindy.feature_library import GeneralizedLibrary, FourierLibrary, PolynomialLibrary, CustomLibrary, TensoredLibrary
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from pysindy.deeptime import deeptime
from sklearn.metrics import mean_squared_error
import math
import itertools

def fixed_r(T, q0):
    T=T+1
    rate = [q0] * T
    return rate

# create a randomly drifting reward rate across T trials, starting at rate q0, with an upper limit of 'upper' and lower
# limit of 'lower'. The drift amount/direction is sampled from a 0-centered gaussian with a standard deviation of sd. 
def random_walk_r(T, q0, upper=0.9, lower=0.1, sd=0.1, seed=101):
    T=T+1
    rate = np.zeros((T), dtype = float)
    rate[0] = q0
    np.random.seed(seed)
    for t in range(T-1):
        rate[t+1] = rate[t] + np.random.normal(0, sd)
        if rate[t+1] > upper:
            rate[t+1] = upper
        if rate[t+1] < lower:
            rate[t+1] = lower
    return list(rate)

def random_walk_varying_intercept_r(T, q0, upper=0.9, lower=0.1, sd=0.1, seed=101):
    T=T+1
    rate = np.zeros((T), dtype = float)
    np.random.seed(seed)
    rate[0] = np.random.uniform(upper, lower)
    for t in range(T-1):
        rate[t+1] = rate[t] + np.random.normal(0, sd)
        if rate[t+1] > upper:
            rate[t+1] = upper
        if rate[t+1] < lower:
            rate[t+1] = lower
    return list(rate)
    
def preSINDYsim(RLparams):

    subs = RLparams['subs']
    T = RLparams['T']
    mu0 = RLparams['mu0']
    mu_behavior = RLparams['mu_behavior']
    bandits = RLparams['bandits']
    t = T # number of trials per subject
    bandits = bandits # number of slot machines
    mu = []
    seed_loop = 101
    for s in range(subs+1):
        seed_loop +=1
        subject_mu = []
        for x in range(bandits):
            if mu_behavior=='random_walk':
                subject_mu.append(random_walk_r(t, mu0, seed=seed_loop+x)) # this assigns a randomly drifting reward rate to each bandit. Use fixed_r() instead if you want fixed rates
            elif mu_behavior=='fixed':
                subject_mu.append(fixed_r(t, mu0, seed=seed_loop+x)) # this assigns a fixed reward rate
            elif mu_behavior=='random_walk_varying_intercept':
                subject_mu.append(random_walk_varying_intercept_r(t, mu0, seed=seed_loop+x))
        mu.append(subject_mu)
    RLparams['mu'] = mu
    
    # add a param for how to name saved files. Just pulling info from the above dictionary to use as identifiers in naming
    RLparams['save_name'] = "%s - eta%s - beta[%s] - bandits[%s] - q_noise[%s] - eta_noise[%s]" % (RLparams['eta_function'].__name__, [x for x in RLparams['eta_params']],
                                             RLparams['beta_params'], bandits, RLparams['q_noise'], RLparams['eta_noise'])

    # run the simulations!
    truevalslist, qvalslist, rewardslist, X_test, U, C, L, qvals = create_train_test_data(RLparams)

    control_inputs_training = rewardslist.copy() # training data from sims
    control_inputs_testing = U.copy() # testing data from sims

    # Next we add columns to training/testing for each possible bandit, indicating whether it was chosen or not for each trial
    # we need to add these for each participant in training, hence the loop through each training ptp
    for i, x in enumerate(control_inputs_training):
        for y in range(bandits): # add one column per bandit
            control_inputs_training[i] = np.c_[ control_inputs_training[i], (y == control_inputs_training[i][:, 2])*1 ]  # convert choice N choice labels to N columns of binary response (1 if chosen, 0 if not)
        control_inputs_training[i] = np.delete(control_inputs_training[i], 2, 1) # remove redundant columns

    for y in range(bandits): # since there is only one testing ptp, we only need to loop through bandits here
        control_inputs_testing = np.c_[ control_inputs_testing, (y == control_inputs_testing[:, 2])*1 ] # like before, convert the choice labels
    control_inputs_testing = np.delete(control_inputs_testing, 2, 1) # and remove redundant columns

    return qvals, qvalslist, X_test, control_inputs_training, control_inputs_testing, truevalslist, C, L

def SINDYlibrary(qvals, bandits, rw=False):

    # Get feature names for SINDYc model
    qvalnames = []; choicenames = []
    for x in range(len(qvals)):
        qvalnames.append('qvals[%s]' % x) # we want feature names for each qval (only one for single bandit)
        choicenames.append('choice[%s]' % x) # we want feature names for each possible choice (only one for single bandit)
    feat_names = qvalnames + ['rewards', 'time'] + choicenames # these are all of the features we will have SINDy consider

    # Define all of the functions you want to consider (Don't worry about what features pair with which
    # functions, we'll handle that later, but list all of the needed functions)
    library_functions = [
        lambda x: x,
        lambda x: np.power(x, 2),
        lambda x,y: x*y,
        lambda x: np.exp(-x/30),
        lambda x: np.exp(-x/20),
        lambda x: np.exp(-x/10)
    ]

    library_functions_rw = [
        lambda y,x: x-y,
    ]

    # List all of the names that pair with each function
    library_names = [
        lambda x: x,
        lambda x: x+'^2',
        lambda x,y: x+'*'+y,
        lambda x: "exp(-" + x + "/30)",
        lambda x: "exp(-" + x + "/20)",
        lambda x: "exp(-" + x + "/10)"
    ]

    library_names_rw = [
        lambda y,x: '('+x+'-'+y+')',
    ]

    # Define the features that pair with each function (IMPORTANT: this needs to be modified if you include more bandits, features, functions, etc.)
    # There is one row for each library/function, and there is one column for each possible variable
    # Note that with single bandit, we always drop variable 3, choice, because it is always 1. Choice becomes important for multi-bandit problems
    library_features = np.array([
        [0, 1, 2, 2], # this means that we're considering the first function for variables 0, 1, and 2
        [0, 0, 2, 2], # this means we're considering the second function for only variables 0 and 2
        [0, 1, 2, 2], # same variables as first row but for the third function
        [2, 2, 2, 2], # consider only variable 2 for the fourth function
        [2, 2, 2, 2], # same as above, but for fifth function
        [2, 2, 2, 2]
    ])
    library_features_rw = np.array([
        [0, 1, 1, 1],
    ])

    # Here is an example array for a 2-armed bandit; SINDy will not use this if you are using a single bandit
    library_features_2bandit = np.array([
        [0, 1, 2, 3, 3],
        [0, 0, 1, 1, 3],
        [0, 1, 2, 3, 4],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
    ])
    library_features_2bandit_rw = np.array([
        [0, 1, 1, 1, 1],
    ])

    # Here is an example array for a 3-armed bandit; SINDy will not use this if you are using a single bandit
    library_features_3bandit = np.array([
        [0, 1, 2, 3, 4, 4, 4],
        [0, 0, 1, 1, 2, 2, 4],
        [0, 1, 2, 3, 4, 5, 6],
        [4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4],
    ])
    library_features_3bandit_rw = np.array([
        [0, 1, 1, 1, 1, 1, 1],
    ])

    # Select the inputs_per_library array conditional on the number of bandits
    if rw == False:
        if bandits == 1:
            inputs_per_library = library_features
        elif bandits == 2:
            inputs_per_library = library_features_2bandit
        elif bandits == 3:
            inputs_per_library = library_features_3bandit
    else:
        if bandits == 1:
            inputs_per_library = library_features_rw
        elif bandits == 2:
            inputs_per_library = library_features_2bandit_rw
        elif bandits == 3:
            inputs_per_library = library_features_3bandit_rw

    # Put all of the libraries into one list, and then use that list of libraries and their feature pairings for a single generalized library
    libraries = []
    for i, x in enumerate(library_functions):
        libraries.append(ps.CustomLibrary(library_functions=[x], function_names=[library_names[i]]))
    libraries_rw = []
    for i, x in enumerate(library_functions_rw):
        libraries_rw.append(ps.CustomLibrary(library_functions=[x], function_names=[library_names_rw[i]]))


    # Code below is commented out, but uncomment if you want to tensor libraries
    tensor_array = [
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1]
    ]

    if rw == False:
        generalized_library = ps.GeneralizedLibrary(
            libraries,
            tensor_array=tensor_array,
            inputs_per_library=inputs_per_library,
        )
    else:
        generalized_library = ps.GeneralizedLibrary(
            libraries_rw,
            inputs_per_library=inputs_per_library,
        )

    return generalized_library, feat_names

def SINDYfit(generalized_library, feat_names, qvalslist, X_test, control_inputs_training, control_inputs_testing, RLparams, r2cap):

    optimizer = ps.SSR(alpha=0.05, max_iter=1000, criteria="model_residual", verbose=False) # using a greedy algorithm here to hopefully find the most parsimonious model

    model = ps.SINDy(feature_library = generalized_library,
                     optimizer=optimizer,
                     differentiation_method = FiniteDifference(order=1),
                     feature_names=feat_names,
                     discrete_time=True)

    model.fit(qvalslist,
              u=control_inputs_training,
              t=1,
              multiple_trajectories=True)

    # Iterate backwards from the most parsimonious model to most complex. This tries to identify a model that meets some R2
    # criterion `r2cap` that is most parsimonius. If none of the models meet this criterion, it lowers the criterion and starts
    # the search again. The results is the best fitting, most parsimonius model.
    r2=0; idx=0
    t=1
    while r2 < r2cap:
        try:
            idx+=1
            optimizer.coef_ = np.asarray(optimizer.history_)[len(optimizer.history_)-idx, :, :]
            r2 = model.score(X_test, u=control_inputs_testing, t=1)
        except:
            r2cap-=0.01
            idx=0

    return model, t, str(r2)

def SINDYsim(model, X_test, control_inputs_testing, truevalslist, C, L, RLparams, r2):

    # Simulate and plot SINDYc model expected values
    try:
        test_shape = X_test.shape[1]
    except:
        test_shape = 1
    sindy_simulation = model.simulate([0.5]*test_shape, u=control_inputs_testing, t=len(X_test))
    tlist = np.linspace(0, RLparams['T'], RLparams['T']+1)
    
    plot_simulations(test_shape, truevalslist, X_test, sindy_simulation, control_inputs_testing, C, L, RLparams, tlist)
    return float(r2)