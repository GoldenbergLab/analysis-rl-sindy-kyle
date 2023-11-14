import pickle
import numpy as np
import pandas as pd
import cmdstanpy
import warnings
warnings.filterwarnings("ignore")

with open('data/sindy_empirical_study1.pkl', 'rb') as f:
    data = pickle.load(f)

nsubs = data['N']
print(nsubs)

# compile C++ code for model
model_sm = cmdstanpy.CmdStanModel(
    stan_file='models/rw.stan',
    cpp_options={"STAN_THREADS": True},
)

initdict = {
        'alpha': [.2]*nsubs,
        'initmu': [0]*nsubs,
        'noise': [.2]*nsubs,
    }

model_fit = model_sm.optimize(data, inits=initdict, seed=101, refresh=1,
                            output_dir='./output/%s' % 'rw')
model_fit.optimized_params_pd.to_csv('fits/rw_fit.csv')

model_sm = cmdstanpy.CmdStanModel(
    stan_file='models/rw_decay.stan',
    cpp_options={"STAN_THREADS": True},
)

initdict = {
        'lambda': [15]*nsubs,
        'initmu': [0]*nsubs,
        'noise': [.2]*nsubs,
    }

model_fit = model_sm.optimize(data, inits=initdict, seed=101, refresh=1,
                            output_dir='./output/%s' % 'rw')
model_fit.optimized_params_pd.to_csv('fits/rw_decay_fit.csv')

model_sm = cmdstanpy.CmdStanModel(
    stan_file='models/qqw.stan',
    cpp_options={"STAN_THREADS": True},
)

initdict = {
        'eta': [.2]*nsubs,
        'phi_minus_eta': [.1]*nsubs,
        'initmu': [0]*nsubs,
        'noise': [.2]*nsubs,
    }

model_fit = model_sm.optimize(data, inits=initdict, seed=101, refresh=1,
                            output_dir='./output/%s' % 'rw')
model_fit.optimized_params_pd.to_csv('fits/qqw_fit.csv')

model_sm = cmdstanpy.CmdStanModel(
    stan_file='models/rw_asym.stan',
    cpp_options={"STAN_THREADS": True},
)

initdict = {
        'etap': [.2]*nsubs,
        'etan': [.2]*nsubs,
        'initmu': [0]*nsubs,
        'noise': [.2]*nsubs,
    }

model_fit = model_sm.optimize(data, inits=initdict, seed=101, refresh=1,
                            output_dir='./output/%s' % 'rw')
model_fit.optimized_params_pd.to_csv('fits/rw_asym_fit.csv')

model_sm = cmdstanpy.CmdStanModel(
    stan_file='models/kalman.stan',
    cpp_options={"STAN_THREADS": True},
)

initdict = {
        'initvolat': [1]*nsubs,
        'lambda': [.2]*nsubs,
        'omega': [1]*nsubs,
        'initmu': [0]*nsubs,
        'noise': [.2]*nsubs,
    }

model_fit = model_sm.optimize(data, inits=initdict, seed=101, refresh=1,
                            output_dir='./output/%s' % 'rw')
model_fit.optimized_params_pd.to_csv('fits/kalman_fit.csv')