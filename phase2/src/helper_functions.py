import numpy as np
import matplotlib.pyplot as plt

def simulate_RL(RWfunction, qnoise, etafunction, etaparams, etanoise, betaparams, T, mu, reversal=None, single=None, subject=0):
    def softmax(vector, beta):
        e = np.exp([x*beta for x in vector])
        return e / e.sum()
    
    subject_mu = mu[subject]
    T=T+1
    
    c = np.zeros((T), dtype = int)
    r = np.zeros((T), dtype = int)  
    trial = np.zeros((T), dtype = int)
    Q_stored = np.zeros((len(subject_mu), T), dtype = float)
    eta = np.zeros((T), dtype = float)
    Q = [0.5]*len(subject_mu)
    reward_rates = subject_mu

    for t in range(T):
        
        if reversal != None:
            if t == reversal:
                for i, x in enumerate(subject_mu):
                    if (type(x) == int) or (type(x) == float):
                        reward_rates[i] = 1-x
                    elif type(x) == list:
                        reward_rates[i] = [1-y if j > reversal else y for j, y in enumerate(x)]

        # store trial number
        trial[t] = t+1
        
        # store values for Q_{t+1}
        Q_stored[:,t] = Q
        
        # compute choice probabilities
        p = softmax(Q, betaparams)
        
        # make choice according to choice probababilities
        # as weighted coin flip to make a choice
        # choose stim 0 if random number is in the [0 p0] interval
        # and 1 otherwise
        if len(subject_mu)>1:
            c[t] = np.random.choice(range(0, len(subject_mu)), 1, p=p)
            
        else: # make choice without noise
            c[t] = np.argmax(p)
            
        if single != None:
            c[t] = single
        
        # generate reward based on reward probability
        if (type(reward_rates[c[t]]) == int) or (type(reward_rates[c[t]]) == float):
            r[t] = np.random.rand() < reward_rates[c[t]]
        elif type(reward_rates[c[t]]) == list:
            r[t] = np.random.rand() < reward_rates[c[t]][t]
        
        # specify trial-specific learning-rate
        eta[t] = etafunction(etaparams, t, r[t]) + np.random.normal(0, etanoise)

        # update values
        Q = RWfunction(r[t], Q, c[t], eta[t])
        for i, x in enumerate(Q):
            Q[i] += np.random.normal(0, qnoise)
            if x < 0:
                Q[i] = 0
            elif x >1:
                Q[i] = 1
    return c, r, trial, Q_stored, eta, reward_rates

def create_train_test_data(p):
    # Control input
    qvalslist=[]; rewardslist=[]
    for s in range(1,p['subs']+2): # simulate N+1 subjects
        np.random.seed(seed=p['seed']+s)
        choices, rewards, trials, qvals, eta, reward_rates= simulate_RL(
            p['RW_function'], # RW function
            p['q_noise'], # noise around updating of expected values
            p['eta_function'], # updating function
            p['eta_params'], # parameters for updating function; see updating functions above for params
            p['eta_noise'], # noise around learning rate eta
            p['beta_params'], # inverse temperature
            p['T'], # number of trials
            p['mu'], # probability of reward from each bandit
            reversal=p['reversal'], # point in task where reward probabilities reverse
            subject = s-1
            )
        if s < p['subs']+1: # training data
            qvalslist.append(np.stack((qvals), axis=-1))
            rewardslist.append(np.stack((rewards, trials, choices), axis=-1))
        else: # testing data from last subject
            truevalslist = reward_rates
            X_test = np.stack((qvals), axis=-1)
            U = np.stack((rewards, trials, choices), axis=-1)
            C = np.stack((choices), axis=-1)
            L = np.stack((eta), axis=-1)
    return truevalslist, qvalslist, rewardslist, X_test, U, C, L, qvals

def plot_simulations(test_shape, truevals, X_test, sindy_simulation, U, C, L, RLparams, t):
    if test_shape % 2 == 0:
        fig, ax = plt.subplots(test_shape//2, 2, figsize=(12, 4*(test_shape//2)), constrained_layout=True)
    else:
        fig, ax = plt.subplots(test_shape, 1, figsize=(12, 3*test_shape), constrained_layout=True)
    fig.patch.set_facecolor('white')
    if test_shape>1 and (test_shape % 2 == 1 or test_shape==2):
        for i in np.arange(test_shape):
            ax[i].plot(t, truevals[i], color='grey', label='True Reward Rate')
            ax[i].plot(t, X_test[:,i], color='r', label=r'Simulated $Q$')
            ax[i].plot(t, sindy_simulation[:,i], '--', color='k', label=r'SINDy Recovered $Q$')
            ax[i].plot(range(1, RLparams['T']+1), [0 if (x==i and U[:,0][j]==1) else -1 for j, x in enumerate(C[:])][:-1], 'b+', label='Trial with Reward')
            ax[i].plot(range(1, RLparams['T']+1), [0 if (x==i and U[:,0][j]==0 and RLparams['bandits']>1) else -1 for j, x in enumerate(C[:])][:-1], 'b^', label='Trial with No Reward')
            ax[i].set(xlabel="Trial", ylabel='Expected Value $Q_{t}$', xlim=[0, 100], ylim=[-0.05, 1.05], yticks=[0, 0.25, 0.5, 0.75, 1])
    elif test_shape>1 and test_shape % 2 == 0:
        for i in np.arange(test_shape):
            ax[i//2][i % 2].plot(t, truevals[i], color='grey', label='True Reward Rate')
            ax[i//2][i % 2].plot(t, X_test[:,i], color='r', label=r'Simulated $Q$')
            ax[i//2][i % 2].plot(t, sindy_simulation[:,i], '--', color='k', label=r'SINDy Recovered $Q$')
            ax[i//2][i % 2].plot(range(1, RLparams['T']+1), [0 if (x==i and U[:,0][j]==1) else -1 for j, x in enumerate(C[:])][:-1], 'b+', label='Trial with Reward')
            ax[i//2][i % 2].plot(range(1, RLparams['T']+1), [0 if (x==i and U[:,0][j]==0 and RLparams['bandits']>1) else -1 for j, x in enumerate(C[:])][:-1], 'b^', label='Trial with No Reward')
            ax[i//2][i % 2].set(xlabel="Trial", ylabel='Expected Value $Q_{t}$', xlim=[0, 100], ylim=[-0.05, 1.05], yticks=[0, 0.25, 0.5, 0.75, 1])
    else:
        ax.plot(t[:], truevals[0], color='grey', label='True Reward Rate')
        ax.plot(t[:], X_test[:], color='r', label=r'Simulated $Q$')
        ax.plot(t, sindy_simulation[:,0], '--', color='k', label=r'SINDy Recovered $Q$')
        ax.plot(range(1, RLparams['T']+1), [0 if (x==0 and U[:,0][j]==1) else -1 for j, x in enumerate(C[:])][:-1], 'b+', label='Trial with Reward')
        ax.plot(range(1, RLparams['T']+1), [0 if (x==0 and U[:,0][j]==0 and RLparams['bandits']>1) else -1 for j, x in enumerate(C[:])][:-1], 'b^', label='Trial with No Reward')
        ax.set(xlabel="Trial", ylabel='Expected Value $Q_{t}$', xlim=[0, 100], ylim=[-0.05, 1.05], yticks=[0, 0.25, 0.5, 0.75, 1])
    plt.show()
