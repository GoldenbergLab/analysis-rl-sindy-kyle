function new_results = wrapper(groupdata)

% simulation parameters
N = 200;        % number of trials per subject

data = struct;

nstarts = 100;

for j = 1:length(groupdata.i)
    
    i = groupdata.i(j);
    
    subdata = groupdata.subdata{i};
    
    data(j).win = subdata.win;
    data(j).choice1 = subdata.choice1;
    data(j).choice2 = subdata.choice2;
    data(j).state2 = subdata.state2;
    data(j).stake = subdata.stake;
    data(j).stim_1_left = subdata.stim_1_left;
    data(j).N = N;
    
end

% run optimization
[params] = set_params;
f = @(x,data) MB_MF_rllik(x,data);
new_results = mfit_optimize(f,params,data,nstarts);

beta = [prctile(est_rw.beta, 25); prctile(est_rw.beta, 50); prctile(est_rw.beta, 75)];
alpha = [prctile(est_rw.alpha, 25); prctile(est_rw.alpha, 50); prctile(est_rw.alpha, 75)];
lambda = [prctile(est_rw.lambda, 25); prctile(est_rw.lambda, 50); prctile(est_rw.lambda, 75)];
w_low = [prctile(est_rw.w_low, 25); prctile(est_rw.w_low, 50); prctile(est_rw.w_low, 75)];
w_high = [prctile(est_rw.w_high, 25); prctile(est_rw.w_high, 50); prctile(est_rw.w_high, 75)];
pi = [prctile(est_rw.pi, 25); prctile(est_rw.pi, 50); prctile(est_rw.pi, 75)];
rho = [prctile(est_rw.rho, 25); prctile(est_rw.rho, 50); prctile(est_rw.rho, 75)];
writetable(table(beta,alpha,lambda,pi,rho,w_low,w_high), 'rwprint.csv')

beta = [prctile(est_sindy.beta, 25); prctile(est_sindy.beta, 50); prctile(est_sindy.beta, 75)];
eta = [prctile(est_sindy.reward_coef, 25); prctile(est_sindy.reward_coef, 50); prctile(est_sindy.reward_coef, 75)];
phi = [prctile(est_sindy.qval_coef, 25); prctile(est_sindy.qval_coef, 50); prctile(est_sindy.qval_coef, 75)];
lambda = [prctile(est_sindy.lambda, 25); prctile(est_sindy.lambda, 50); prctile(est_sindy.lambda, 75)];
w_low = [prctile(est_sindy.w_low, 25); prctile(est_sindy.w_low, 50); prctile(est_sindy.w_low, 75)];
w_high = [prctile(est_sindy.w_high, 25); prctile(est_sindy.w_high, 50); prctile(est_sindy.w_high, 75)];
pi = [prctile(est_sindy.pi, 25); prctile(est_sindy.pi, 50); prctile(est_sindy.pi, 75)];
rho = [prctile(est_sindy.rho, 25); prctile(est_sindy.rho, 50); prctile(est_sindy.rho, 75)];
writetable(table(beta,eta,phi,lambda,pi,rho,w_low,w_high), 'sindyprint.csv')