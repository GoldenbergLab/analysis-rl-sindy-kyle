functions {
  real update_function(real[,] qval, int start, int end, int[] Tsub, int[,] reward, real[] lambda, real[] initmu, real[] noise){

        real lp = 0;
        int counter = 1;

        for (n in 1:(end-start+1)) {
          int i = start + (n - 1);
          real ev = initmu[i]; // expected value
          real Q;
          for (t in 1:(Tsub[i])) {
            Q = (1 / (1 + exp(-1 * ev)));
            ev += exp(-t / lambda[i]) * (reward[i, t] - Q);
            if (qval[counter, t] != -1.0) { // skip evaluation if the trial is missing (e.g., an attention check)
              lp = lp + normal_lpdf(qval[counter, t] | (1 / (1 + exp(-1 * ev))), noise[i]+1e-10);
            }
          }
          counter += 1;
        }
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
  real<lower=0> lambda[N];
  real initmu[N];
  real<lower=0> noise[N];
}

model {
  for (k in 1:N) {
            lambda[k] ~ normal(15, 10)T[0,];
            initmu[k] ~ normal(0, .1);
            noise[k] ~ gamma(1, .2);
        }

  target += reduce_sum(update_function, qval, 1, Tsub, reward, lambda, initmu, noise);
}

generated quantities {
  real log_lik[N];

  // Begin subject loop
  for (i in 1:N) {
    real ev; // expected value
    real Q;
    ev = initmu[i];
    log_lik[i] = 0;
    for (t in 1:(Tsub[i])) {
      Q = (1 / (1 + exp(-1 * ev)));
      ev += exp(-t / lambda[i]) * (reward[i, t] - Q);
      if (qval[i, t] != -1.0){
        log_lik[i] += normal_lpdf(qval[i, t] | (1 / (1 + exp(-1 * ev))), noise[i]+1e-10);
      }
    }
  } // end of subject loop
}