
People often make estimations or predictions. People often assign confidences to those predictions, such as "I am 95% sure that the actual number of X is within this range", or "I am 70% sure that this will happen".

To "calibrate" oneself is to collect a sample of such predictions, along with the actual outcome (i.e was the true value within the quoted range? did the predicted outcome happen?), and then determine whether one is over or underconfident. It's considered quite difficult to be well-calibrated, since a large number of cognitive biases often get in the way of logical reasoning.

Unlike many other statistical or machine learning applications, calibrating a person usually needs to be done with a fairly small sample size (one tends to get bored after answering hundreds of trivia questions), and the statistical significance of results is often unclear. The simplest strategies (binning) also has strange behavior near the tails (probability 0 or 1) -- in some sense, probability 0 is infinitely farm from probability 0.001.

In this repository, I've put together a simple bayesian model to evaluate and verify calibration in a more "sane" way. The model puts a beta prior for the actual probability of an event happening for every quoted confidence value. The alpha and beta parameters of the prior themselves are parameterized by a gaussian process. Tweaking the (hyper)parameters in the kernel of the gaussian process roughly correspond to providing prior beliefs about ones own calibration such as...

1. I believe my calibratedness varies {this much} across different confidences.
2. I believe I am miscalibrated by about {this much} on average.

Most of the challenges I'm working on solving/fixing with the current model is that relatively small changes in these hyperparameters tend to have significant effects in the posterior distribution, which is mildly disturbing as the goal here is to be more "rigorous". 

Two other challenges I hope to tackle are:

1. the interpretability of various hyperparameters on the prior -- the gaussian process parameters affect beta and alpha instead of the expectation of Beta(a,b), which leads to rather tricky interpretation.
2. confidence distance -- in "information-space", 0.9999 and 0.9995 are as far apart as 0.9 and 0.8. currently the gaussian process uses this "log space" domain, but it leads to very wide posteriors near the tails (maybe this is the "correct" thing to do though -- anything else would be putting too much faith in the model -- or alternatively, maybe there's some axis along which our priors should be made much stronger (i.e the confidence curve is very likely monotonic)). 

In the current state, the code is mostly just a playground of different ideas and half-baked implementations -- a lot more testing and fine-tuning is needed before it's really usable.