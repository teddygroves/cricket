data {
  int<lower=1> N;
  int<lower=1> N_bowl;
  int<lower=1> N_bat;
  int<lower=1> N_innings;
  int<lower=1,upper=N_bowl> bowler[N];
  int<lower=1,upper=N_bat> batter[N];
  int<lower=1,upper=N_innings> innings[N];
  int<lower=0> runs[N];
  int<lower=0> wickets[N];
  int<lower=1> balls[N];
}
transformed data {
  vector[N] log_balls;
  for (n in 1:N) log_balls[n] = log(balls[n] * 1.0);
}
parameters {
  vector[N_bowl] bowl_ability_z[2];
  vector[N_bat] bat_ability_z[2];
  vector[N_innings] mu[2];
  vector<lower=0>[2] sigma_bat;
  vector<lower=0>[2] sigma_bowl;
  real<lower=0> phi;  // controls overdispersion for runs
}
transformed parameters {
  vector[N_bowl] bowl_ability[2];
  vector[N_bat] bat_ability[2]; 
  for (i in 1:2){
    bowl_ability[i] = bowl_ability_z[i] * sigma_bowl[i];
    bat_ability[i] = bat_ability_z[i] * sigma_bat[i];
  }
}
model {
  vector[N] log_lambda_runs = log_balls
                            + mu[1][innings]
                            + bat_ability[1][batter]
                            - bowl_ability[1][bowler];
  vector[N] logit_wicket_prob = mu[2][innings]
                              + bowl_ability[2][bowler]
                              - bat_ability[2][batter];
  runs ~ neg_binomial_2_log(log_lambda_runs, phi);
  wickets ~ binomial_logit(balls, logit_wicket_prob);
  for (i in 1:2) {
    mu[i] ~ normal(-1, 3);
    bowl_ability_z[i] ~ normal(0, 1);
    bat_ability_z[i] ~ normal(0, 1);
    sigma_bowl[i] ~ cauchy(0, 2);
    sigma_bat[i] ~ cauchy(0, 2);
  }
  phi ~ gamma(1, 0.1);
}
generated quantities {
  vector[N] runs_rep;
  vector[N] wickets_rep;
  {
    vector[N] log_lambda_runs = log_balls
                              + mu[1][innings]
                              + bat_ability[1][batter]
                              - bowl_ability[1][bowler];
    vector[N] logit_wicket_prob = mu[2][innings]
                                + bowl_ability[2][bowler]
                                - bat_ability[2][batter];
    for (n in 1:N){
      runs_rep[n] = neg_binomial_2_log_rng(log_lambda_runs[n], phi);
      wickets_rep[n] = binomial_rng(balls[n], inv_logit(logit_wicket_prob[n]));
    }
  }
}
