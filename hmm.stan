functions{
  vector stationary_distribution(matrix transitions){
    int N = rows(transitions);
    matrix [N + 1, N] A;
    vector [N+1] b;
    matrix [N, N + 1] At;
    A[1:N,1:N] = transitions' - diag_matrix(rep_vector(1,N));
    A[N+1, :] = rep_row_vector(1,N);
    At = A';
    b[1:N] = rep_vector(0,N);
    b[N+1] = 1;
    return (At * A) \ (At * b);
  }
  matrix to_matrix(vector [] x){
    int N = rows(x[1]);
    matrix [N,N] M;
    for (i in 1:N)
      M[i,:] = x[i]';
    return M;
  }
}

data {
  int optimizing;
  int ignore_prior;
  int state_count;
  real mean_scale;
  real mean_df;
  real var_scale;
  real transition_scale;
  real scale_factor;
  int N_1;
  row_vector [N_1] observations;
}

transformed data {
  int N = N_1 + 1;
  row_vector [N_1] observations_scale = observations * scale_factor;
}

parameters {
  vector  [state_count] means_raw;
  ordered [state_count] log_sds_raw;
  simplex [state_count] transitions [state_count];
}

transformed parameters {  
  vector [state_count] inv_sds_raw = exp(-log_sds_raw);
  matrix [state_count,N_1] log_latent_probs;
  matrix [state_count,N_1] conditional_llikelihoods;
  matrix [state_count,N_1] llikelihoods;
  matrix [state_count,N] unconditional_l_state_prob;
  matrix [state_count,state_count] transitions_ = to_matrix(transitions);
  matrix [state_count,state_count] log_transitions = log(transitions_);  
  vector [state_count] pi_ = stationary_distribution(transitions_);
  real total_llikelihood=0;
  real step_llikelihood;

  for(j in 1:state_count){
    conditional_llikelihoods[j,:] = -0.5 * square((observations_scale - means_raw[j])*inv_sds_raw[j]) - log_sds_raw[j];
  }
  
  unconditional_l_state_prob[:,1] = log(pi_);  
  for(i in 1:N){
    if(i > 1){
      for(j in 1:state_count){
        unconditional_l_state_prob[j,i] = log_sum_exp(log_latent_probs[:, i-1] + log_transitions[:, j]);
      }
    }
    if(i < N){
      llikelihoods[:,i] = unconditional_l_state_prob[:,i] + conditional_llikelihoods[:,i];
      step_llikelihood = log_sum_exp(llikelihoods[:,i]);
      total_llikelihood += step_llikelihood;
      log_latent_probs[:,i] = llikelihoods[:,i] - step_llikelihood;
    }
  }

}

model{
  if(!optimizing){
    target += 2*log_sds_raw;
  }
  if(!ignore_prior){
    means_raw ~ student_t(mean_df,0,mean_scale * scale_factor);
    // var_raw is inv_gamma distributed
    target += -4 * 2*log_sds_raw - (var_scale*square(scale_factor)) * exp(-2*log_sds_raw);
    for(i in 1:state_count){
      vector [state_count] p = rep_vector(1,state_count);
      p[i]=transition_scale;
      transitions[i] ~ dirichlet(p);
    }
  }
  target += total_llikelihood;
}
generated quantities{
  vector [state_count] means = means_raw / scale_factor;
  vector [state_count] sds = exp(log_sds_raw) / scale_factor ;  
  matrix [N_1,state_count] latent_probs = exp(log_latent_probs');
  matrix [N,state_count] unconditional_probs = exp(unconditional_l_state_prob');
}

