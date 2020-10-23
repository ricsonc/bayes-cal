
import torch
import torch.nn.functional as F
from torch.distributions import constraints
import torch.distributions as tdist
from torch.nn import Parameter
import numpy as np

from scipy.stats import beta

import pyro
from pyro import sample, param
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.models import GPRegression
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.contrib.gp.util import conditional
from pyro.params import param_with_module_name
from pyro.nn import PyroModule, PyroSample, PyroParam
from statsmodels.stats.proportion import proportion_confint
from pyro.infer import MCMC, NUTS, Predictive, SVI, TraceMeanField_ELBO


import matplotlib.pyplot as plt
from ipdb import set_trace as st
from munch import Munch as M_


def response(x):
    # eps = 1e-3
    return torch.exp(x) #doesnt even have to be negative now
    # return F.elu(x)+1+1e-3

def quantile_normalbeta(mu_alpha, mu_beta, sd_alpha, sd_beta, q):
    # given a, b ~ Normal
    # and z ~ Beta(a, b)
    # compute z such that CDF(z) = q
    # if we define a loss L = 1/2 (CDF(z) - q)^2, then the gradient is
    # (CDF(z) - q) * PDF(z)
    # also, CDF(z) = E_ab[CDF(z|a,b)] (the inner term is beta cdf)
    # PDF(z) = E_ab[PDF(z|a,b)] (the inner term is beta pdf)
    # thus we can solve this by SGD

    B = 100
    z = torch.ones_like(q) * 0.5 #0.5 is as good an initialization as any..
    iters = 500
    rate = 1E-2

    mu_alpha_ = mu_alpha.unsqueeze(0)
    mu_beta_ = mu_beta.unsqueeze(0)
    sd_alpha_ = sd_alpha.unsqueeze(0)
    sd_beta_ = sd_beta.unsqueeze(0)

    for i in range(iters):
        
        a = response(mu_alpha_ + torch.randn(B, mu_alpha.shape[0]) * sd_alpha_)
        b = response(mu_beta_ + torch.randn(B, mu_beta.shape[0]) * sd_beta_)

        pdf = tdist.Beta(a, b).log_prob(z).exp()
        cdf = torch.Tensor(beta.cdf(z.numpy(), a.numpy(), b.numpy()))

        grad = ((cdf-q)*pdf).mean(axis = 0)
        z -= rate*grad
        #clamp to valid range
        z = torch.clamp(z, 1E-6, 1-1E-6)
        
        #resort to scipy
    return z
    
def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, likelihood_fn = None, n_test=200, samples=1000000, bars = None):

    RANGE_LOW = np.log(0.01) - np.log(0.5)
    RANGE_HIGH = 0#-RANGE_LOW
    
    plt.figure(figsize=(12, 6))
    
    if plot_observed_data is not False:
        plt.plot(plot_observed_data.X.numpy(), plot_observed_data.y.numpy(), 'kx')
        
    if plot_predictions:
        Xtest = torch.linspace(RANGE_LOW, RANGE_HIGH, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP or type(model) == gp.models.VariationalGP:
                mean, cov = model(Xtest, full_cov=False)
            else:
                mean, cov = model(Xtest, full_cov=False, noiseless=False)

        # st()
        # sd = cov.diag().sqrt()  # standard deviation at each input point x
        sd = cov.sqrt()

        # normal likelihood...
        # plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        # plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
        #                  (mean - 2.0 * sd).numpy(),
        #                  (mean + 2.0 * sd).numpy(),
        #                  color='C0', alpha=0.3)

        # f_res = likelihood.response_function(mean).numpy()
        # f_res_low = likelihood.response_function(mean-2*sd).numpy()
        # f_res_high = likelihood.response_function(mean+2*sd).numpy()

        # f_res = mean[0] / mean.sum(0)

        #NUMERICAL APPROACH
        A, B = response(torch.FloatTensor(2, 1, samples).cuda().normal_() * sd.unsqueeze(-1).cuda() + mean.unsqueeze(-1).cuda())
        P = tdist.beta.Beta(A,B).sample()
        P = P.cpu().numpy()

        # f_res = np.percentile(P, 50, axis = 1)
        f_res = P.mean(axis = 1)
        
        f_plus1 = np.percentile(P, 12.5, axis = 1)
        f_minus1 = np.percentile(P, 87.5, axis = 1)
        
        f_plus2 = np.percentile(P, 2.5, axis = 1)
        f_minus2 = np.percentile(P, 97.5, axis = 1)

        #compare with a slightly smarter numerical approach...
        # f_res2 = quantile_normalbeta(mean[0], mean[1], sd[0], sd[1], torch.ones_like(mean[0])*0.5)
        # f_res_low2 = quantile_normalbeta(mean[0], mean[1], sd[0], sd[1], torch.ones_like(mean[0])*0.975)
        # f_res_high2 = quantile_normalbeta(mean[0], mean[1], sd[0], sd[1], torch.ones_like(mean[0])*0.025)

        #oh this is quite tricky actually...
        # f_res_low = None
        # f_res_high = None

        X_ = unstretch(Xtest, 'torch').numpy()
        
        plt.plot(X_, f_res, 'r', lw=2)  # plot the mean
        # plt.plot(X_, f_res2, 'b', lw=2)  # plot the mean
        plt.fill_between(X_, f_minus2, f_plus2, color='C0', alpha=0.3)
        plt.fill_between(X_, f_minus1, f_plus1, color='C0', alpha=0.5)
        # plt.fill_between(X_,  # plot the two-sigma uncertainty about the mean
        #                  f_res_low2, f_res_high2,
        #                  color='C1', alpha=0.25)

        bar_x = [bar.bin_pos for bar in bars]
        heights = [bar.interval[1]-bar.interval[0] for bar in bars]
        bottom = [bar.interval[0] for bar in bars]
        centers = [bar.mean for bar in bars]
        h_widths = [bar.upper-bar.lower for bar in bars]
        plt.bar(bar_x, heights, 0, bottom=bottom, alpha=1.0, edgecolor='black', linewidth=1)
        plt.barh(centers, width = h_widths, height = 0, left = [bar.lower for bar in bars], edgecolor='black', linewidth=1)
        plt.scatter(bar_x, centers, color='green')

        plt.plot([0,1], [0,1], linestyle='--', color='green')
        
        
        # st() #plot some bars...


    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    # plt.xlim(RANGE_LOW, RANGE_HIGH)#0.0, 1.0)
    plt.xlim(0.0, 0.5)


class Binary(Likelihood):
    def __init__(self, response_function=None):
        super().__init__()
        self.response_function = (response_function if response_function is not None
                                  else F.sigmoid)

    def forward(self, f_loc, f_var, y=None):
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        f = dist.Normal(f_loc, f_var)()
        f_res = self.response_function(f)

        y_dist = dist.Bernoulli(f_res)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).independent(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)

class BetaBinomial(Likelihood):
    def __init__(self):
        super().__init__()

    def forward(self, f_loc, f_var, y=None):
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        alpha, beta = dist.Normal(f_loc, f_var)()
        eps = 1E-3
        alpha = response(alpha)
        beta = response(beta)
        p = dist.Beta(alpha, beta)()
        y_dist = dist.Bernoulli(p)
        
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-p.dim()]).independent(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)            

def stretch(X, back):
    def stretch_(X):
        #input: 0 to 0.5
        #ouput: -inf to 0
        if back == 'torch':
            return torch.log(X) - np.log(0.5)
        else:
            return np.log(X) - np.log(0.5)

    if back == 'torch':
        return torch.where(X<=0.5, stretch_(X), -stretch_(1-X))
    else:
        return np.where(X<=0.5, stretch_(X), -stretch_(1-X))

def unstretch(X, back):
    def unstretch_(X):
        if back == 'torch':
            return np.exp(X+np.log(0.5))
        else:
            return torch.exp(X+np.log(0.5))

    if back == 'torch':
        return torch.where(X<=0, unstretch_(X), 1-unstretch_(-X))
    else:
        return np.where(X<=0, unstretch_(X), 1-unstretch_(-X))

    
if __name__ == '__main__':
    torch.manual_seed(1)

    N = 300
    M = 100
    # X = torch.rand(N)*0.98 + 0.01 #avoid edges..
    X = torch.rand(N)*0.49+0.01
    X_orig = X.clone()
    
    X = stretch(X, 'torch')

    RANGE_LOW = np.log(0.01) - np.log(0.5)
    # RANGE_HIGH = -RANGE_LOW
    RANGE_HIGH = 0
    Xu = torch.linspace(RANGE_LOW, RANGE_HIGH, M)
    
    p_ = unstretch(X, 'torch') #torch.clamp(X+torch.rand(N)*0.1, 0.01, 0.99)
    #suppose all values i think are VERY unlikely are actually a bit more likely...
    p_ = torch.clamp(p_, 0.1, 1.0)
    y = tdist.Bernoulli(p_).sample()

    #1. historgram comparison #jeffreys interval... bins?? let's go with 5% as sanity..., and 1% on the ends?
    num_bins = 20
    breakpoints = np.linspace(0.0, 1.0, num_bins+1) #5%
    bars = []
    for i in range(num_bins):
        lower = breakpoints[i]
        upper = breakpoints[i+1]
        bin_mask = torch.min(lower < X_orig, X_orig < upper)
        if not bin_mask.max().item():
            continue
        vals_mean = X_orig[bin_mask].mean().item()
        vals_positive = y[bin_mask].sum().item()
        bin_count = bin_mask.sum().item()
        interval = proportion_confint(vals_positive, bin_count, method='jeffreys', alpha = 0.05)
        bars.append(M_(bin_pos = vals_mean, interval = interval, mean = vals_positive/bin_count, lower = lower, upper = upper))

    # plt.scatter(X, y)
    # plt.show()

    # pyro.enable_validation(True)

    kernel = gp.kernels.RBF(input_dim=1)
    # kernel.lengthscale = PyroParam(torch.tensor(0.2), dist.constraints.greater_than(0.01))
    # kernel.variance = PyroParam(torch.tensor(0.1), dist.constraints.greater_than(0.01))

    #uh... does this actually do ANYTHING?...
    # kernel.lengthscale = PyroSample(dist.TransformedDistribution(dist.LogNormal(0.0, 2.0), [tdist.AffineTransform(loc=1, scale=1)])) #setting this to 1 works though..

    #there's some sort of bug here...
    kernel.lengthscale = PyroSample(dist.LogNormal(0.0, 2.0))
    kernel.variance = PyroSample(dist.LogNormal(1.0, 2.0))

    #scale X axis?
    
    # likelihood=gp.likelihoods.Gaussian()
    # likelihood = Binary() # gp.likelihoods.Binary()
    likelihood = BetaBinomial()
    #1. 1d -> 2d GP, with beta likelihood uhh i am confused...
    #gpr = gp.models.VariationalSparseGP(X, y, kernel, Xu=Xu, likelihood=likelihood, whiten=True, jitter=1E-4)

    #seems a bit...conservative...

    args = [X, y, kernel, likelihood]
    kwargs = M_(
        mean_function = lambda x: 1.0, 
        latent_shape = (2,),
        whiten=True,
        jitter=1E-4,
    )
    if N > 100:
        model = gp.models.VariationalSparseGP
        args.insert(3, Xu)
    else:
        model = gp.models.VariationalGP

    gpr = model(*args, **kwargs)

    #params include posterior AND parameters

    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.001)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    num_steps = 1000
    
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        print(i, loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plot(
        model=gpr,
        plot_observed_data=M_(X=unstretch(X, 'torch'), y =y),
        plot_predictions=True,
        likelihood_fn = likelihood,
        bars = bars,
    )
    
    plt.show()
    
    st()

    
    #2. test cases
    #3. FAQ
    #4. investigate stretch behavior...
