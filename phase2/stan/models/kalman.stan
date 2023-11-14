functions {
    row_vector vkf(real r, real ev, real vari, real volat, real omega, real lambda) {
        real PE; // reward prediction error
        real K; // Kalman gain
        real rate; // learning generated
        real auto_cov; // covariance between the mean reward of trial and trial-1
        real new_ev;
        real new_vari;
        real new_volat;
        PE = r - (1 / (1 + exp(-1 * ev)));
        K = (vari + volat) / (vari + volat + omega + 1e-10); // Eq 14
        rate = sqrt(vari + volat); // Eq 15
        new_ev = ev + rate * PE; // Eq 16
        new_vari = (1-K)*(vari + volat); // Eq 17
        auto_cov = vari - K * vari; // Eq 18
        new_volat = volat + lambda * (pow(new_ev - ev, 2) + vari + new_vari - (2 * auto_cov) - volat); // Eq 19
        return [ new_ev, new_vari, new_volat ];
    }
  
  real update_function(real[,] qval, int start, int end, int[] Tsub, int[,] reward, real[] initvolat, real[] lambda, real[] omega, real[] initmu, real[] noise){

        real lp = 0;
        int counter = 1;

        for (n in 1:(end-start+1)) {
            row_vector[3] out;
            int i = start + (n - 1);
            real ev = initmu[i]; // expected value
            real vari = omega[i]; // variance
            real volat = initvolat[i]; // volatility
            for (t in 1:(Tsub[i])) {
                if (t == 1) {
                    out = vkf(reward[i, t], initmu[i], vari, volat, omega[i], lambda[i]);
                }
                else {
                    out = vkf(reward[i, t], ev, vari, volat, omega[i], lambda[i]);
                }
                ev = out[1];
                vari = out[2];
                volat = out[3];
                if (qval[counter, t] != -1.0){
                    lp = lp + normal_lpdf(qval[counter, t] | (1 / (1 + exp(-1 * ev))), noise[i]+1e-10);
                }
            }
            counter += 1;
        } // end of subject loop
        return lp;
    }
}

data {
  int<lower=1> N;      // Number of subjects
  int<lower=1> T;      // Number of trials
  int<lower=1, upper=T> Tsub[N]; // number of trials per subject
  real<lower=-1, upper=1> qval[N, T]; // reported probability of reward
  int<lower=-1, upper=1> reward[N, T]; // reward
}

parameters {
  // Subject-level parameters
  real<lower=0, upper=10> initvolat[N];
  real<lower=0, upper=1> lambda[N];
  real<lower=0> omega[N];
  real initmu[N];
  real<lower=0> noise[N];
}

model {
  for (k in 1:N) {
            initvolat[k] ~ normal(1, .1)T[0,10];
            lambda[k] ~ normal(.2, .1)T[0,1];
            omega[k] ~ gamma(1, 1);
            initmu[k] ~ normal(0, .1);
            noise[k] ~ gamma(1, .2);
        }

  target += reduce_sum(update_function, qval, 1, Tsub, reward, initvolat, lambda, omega, initmu, noise);
}

generated quantities {
  real log_lik[N];

  // Begin subject loop
  for (i in 1:N) {
    row_vector[3] out;
    real ev = initmu[i]; // expected value
    real vari = omega[i]; // variance
    real volat = initvolat[i]; // volatility
    log_lik[i] = 0;
    for (t in 1:(Tsub[i])) {
        if (t == 1) {
            out = vkf(reward[i, t], initmu[i], vari, volat, omega[i], lambda[i]);
        }
        else {
            out = vkf(reward[i, t], ev, vari, volat, omega[i], lambda[i]);
        }
        ev = out[1];
        vari = out[2];
        volat = out[3];
        if (qval[i, t] != -1.0){
            log_lik[i] += normal_lpdf(qval[i, t] | (1 / (1 + exp(-1 * ev))), noise[i]+1e-10);
        }
    }
  }
}