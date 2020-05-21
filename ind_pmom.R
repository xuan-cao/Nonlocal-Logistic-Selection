pmom_laplace <- function(X.ind, y, r, tau) {
  
  n_k <- ncol(X.ind)
  if(length(n_k) == 0) return(-100000)
  obj_func <- function(beta) {
    inner <- X.ind %*% beta
    obj <- sum(inner * y - log(1 + exp(inner))) - n_k * log(prod(seq(1, 2 * r - 1, 2))) - n_k / 2 * log(2 * pi) - (r + 0.5) * n_k * log(tau) - sum(beta ^ 2) / 2 / tau + 2 * r * sum(log(abs(beta)))
    return(-obj)
  }
  
  obj_grad <- function(beta) {
    inner <- X.ind %*% beta
    grad <- colSums(X.ind * as.vector(y - exp(inner) / (1 + exp(inner)))) - beta / tau + 2 * r / beta
    return(-grad)
  }
  
  #print(ncol(X.ind))
  res <- optim(rep(0.5, ncol(X.ind)), obj_func, obj_grad, method = "BFGS")
  # return(res$par)
  
  get_V <- function(beta) {
    inner <- X.ind %*% beta
    V <- - t(X.ind) %*% (X.ind * as.vector(exp(inner) / ((1 + exp(inner)) ^ 2))) - diag(n_k) / tau - diag(2 * r / (beta ^ 2))
    return(V)
  }
  
  get_log_marginal <- function(beta) {
    V <- get_V(beta)
    log_marginal <- n_k / 2 * log(2 * pi) - obj_func(beta) - 0.5 * sum(log(abs(eigen(V)$values)))
    return(log_marginal)
  }
  #print(get_log_marginal(res$par))
  return(get_log_marginal(res$par))
}

