import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound

from gpytorch.mlls import ExactMarginalLogLikelihood
