functions {
  real update_function(real[,] qval, int start, int end, int[] Tsub, int[,] reward, real[] eta, real[] phi, real[] initmu, real[] noise){

        real lp = 0;
        int counter = 1;

        for (n in 1:(end-start+1)) {
          int i = start + (n - 1);
          real ev = initmu[i]; // expected value
          real Q;
          for (t in 1:(Tsub[i])) {
            Q = (1 / (1 + exp(-1 * ev)));
            ev += (eta[i] * reward[i, t]) - (phi[i] * pow(Q, 2));
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
  real<lower=0, upper=1> eta[N];
  real<lower=0> phi_minus_eta[N];
  real initmu[N];
  real<lower=0> noise[N];
}

transformed parameters {
  real<lower=0> phi[N];
  for (i in 1:N) {
    phi[i] = phi_minus_eta[i] + eta[i];
  }
}

model {
  for (k in 1:N) {
            eta[k] ~ normal(.2, .1)T[0,1];
            phi_minus_eta[k] ~ gamma(1, .2);
            initmu[k] ~ normal(0, .1);
            noise[k] ~ gamma(1, .2);
        }

  target += reduce_sum(update_function, qval, 1, Tsub, reward, eta, phi, initmu, noise);
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
      ev += (eta[i] * reward[i, t]) - (phi[i] * pow(Q, 2));
      if (qval[i, t] != -1.0){
        log_lik[i] += normal_lpdf(qval[i, t] | (1 / (1 + exp(-1 * ev))), noise[i]+1e-10);
      }
    }
  } // end of subject loop
}