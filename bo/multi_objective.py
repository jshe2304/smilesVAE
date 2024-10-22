import torch

import os
import sys
import random
from rdkit.Chem import MolFromSmiles, MolToSmiles, RDConfig
from rdkit.Chem.QED import qed as QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from sascorer import calculateScore as SAS

from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

# ==================
# Objective Function
# ==================

def filter_and_score(candidates, decoder, from_hot):
    '''
    Validity-aware scoring objective. 
    Removes invalid candidates. 
    Valid candidates are scored by true score. 
    '''

    smiles = from_hot(decoder(candidates))
    
    x = []
    scores = []
    for i, smile in enumerate(smiles):
        mol = MolFromSmiles(smile)
        if mol is None: continue
        
        x.append(candidates[i].unsqueeze(0))
        scores.append([QED(mol), -SAS(mol)])

    if len(x) > 0:
        return torch.cat(x), torch.Tensor(scores)
    else:
        return torch.Tensor(), torch.Tensor()

# ===============
# Dataset Sampler
# ===============

def get_samples(encoder, dataiter):
    x, logp, qed, sas = next(dataiter)
    z, _, _ = encoder(x)
    return z, torch.cat((qed, -sas), dim=1)
    
def get_restricted_samples(encoder, dataiter, n=16, restrictions=((0, 1), (0, 10))):
    qed_lim, sas_lim = restrictions
    
    samples, scores = [], []
    while len(z) < n:
        # Get sample
        x, logp, qed, sas = next(dataiter)
        
        if (qed > qed_lim[0] and qed < qed_lim[1]) and (sas > sas_lim[0] and sas < sas_lim[1]):
            z, _, _ = encoder(x)
            samples.append(z)
            scores.append(torch.cat([qed, -sas]).unsqueeze(0))

    return torch.cat(samples), torch.cat(scores)

# ===============
# Surrogate Model
# ===============

def fit_surrogate(z, scores, state_dict):
    models = [
        SingleTaskGP(
            train_X=z, 
            train_Y=scores[..., i:i+1], 
            outcome_transform=Standardize(m=1)
        ) for i in range(scores.size(-1))
    ]
    model = ModelListGP(*models)
    
    if state_dict is not None:
        model.load_state_dict(state_dict)

    mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    fit_gpytorch_mll(mll)
    
    return model

# ====================
# Acquisition Function
# ====================

def optimize_qehvi(model, z, scores, ref_point, unit_bounds):
    
    with torch.no_grad():
        pred = model.posterior(z).mean

    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=pred
    )
    
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
    
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=unit_bounds,
        q=8, 
        num_restarts=4, 
        raw_samples=8
    )

    return candidates.detach()

def optimize_qnparego(model, z, scores, unit_bounds):
    
    with torch.no_grad():
        pred = model.posterior(z).mean
    
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    
    acq_func_list = []

    for _ in range(8):
        weights = sample_simplex(scores.size(-1)).squeeze()
        
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        
        acq_func = qNoisyExpectedImprovement(
            model=model,
            objective=objective,
            X_baseline=z,
            sampler=sampler,
            prune_baseline=True,
        )
        
        acq_func_list.append(acq_func)

    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=unit_bounds,
        num_restarts=8, 
        raw_samples=64,  # used for intialization heuristic
    )

    return candidates.detach()

