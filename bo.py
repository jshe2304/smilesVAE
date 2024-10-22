from utils.utils import *
from utils.dataset import *
from bo.multi_objective import *
from models.RNN0 import Encoder, Decoder, Predictor

import sys 

import torch
from torch.utils.data import DataLoader

from botorch.utils.transforms import normalize, unnormalize

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ====================
# Optimization Options
# ====================

sample_restrictions = None
acquisition_function = sys.argv[1]
n_samples = 8

# =====
# Setup
# =====

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATADIR = './data/gdb13/'
RUNDIR = './runs/train11/'
OUTDIR = sys.argv[2]
DATASPEC_FILE = os.path.join(DATADIR, 'spec.json')
RUNSPEC_FILE = os.path.join(RUNDIR, 'spec.json')

encoder_weights_file = os.path.join(RUNDIR, 'encoder_weights.pth')
decoder_weights_file = os.path.join(RUNDIR, 'decoder_weights.pth')
predictor_weights_file = os.path.join(RUNDIR, 'predictor_weights.pth')

samples_z_file = os.path.join(OUTDIR, 'samples_z.pt')
samples_scores_file = os.path.join(OUTDIR, 'samples_scores.pt')
candidates_z_file = os.path.join(OUTDIR, 'candidates_z.pt')
candidates_scores_file = os.path.join(OUTDIR, 'candidates_scores.pt')
state_dict_file = os.path.join(OUTDIR, 'state_dict.pt')

dataspec = fetch_params(DATASPEC_FILE)
runspec = fetch_params(RUNSPEC_FILE)

to_hot, from_hot = make_embedding_utils(dataspec)

# =====
# Model
# =====

encoder = Encoder(runspec.L)
encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
encoder.eval()

decoder = Decoder(runspec.L)
decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))
decoder.eval()

predictor = Predictor(runspec.L)
predictor.load_state_dict(torch.load(predictor_weights_file, map_location=device))
predictor.eval()

# ====
# Data
# ====

_, testset = make_data(DATADIR, to_hot, n=500)
dataiter = iter(DataLoader(
    testset, 
    batch_size=n_samples if sample_restrictions is None else 1, 
    shuffle=True
))

means, stds = get_latent_distributions(encoder, testset.hots)
strong_dims = get_important_dimensions(testset.hots, encoder, decoder)

compressed_means = means[strong_dims]
compressed_stds = stds[strong_dims]

z_bounds = torch.stack((
    (compressed_means - 4 * compressed_stds), 
    (compressed_means + 4 * compressed_stds)
))

score_bounds = torch.Tensor(
    [[0, -10], 
     [1, 0]]
)

unit_bounds = torch.zeros_like(z_bounds)
unit_bounds[1] = 1

# =============
# Load progress
# =============

if all(os.path.isfile(f) for f in (samples_z_file, samples_scores_file, candidates_z_file, candidates_scores_file)):
    samples_z = torch.load(samples_z_file)
    samples_scores = torch.load(samples_scores_file)
    candidates_z = torch.load(candidates_z_file)
    candidates_scores = torch.load(candidates_scores_file)
    state_dict = torch.load(state_dict_file)
else:
    samples_z, samples_scores = torch.Tensor(), torch.Tensor()
    candidates_z, candidates_scores = torch.Tensor(), torch.Tensor()
    state_dict = None

# =================
# Optimization Loop
# =================

for batch in range(16):

    # Sample from dataset
    if sample_restrictions is None:
        sample_z, sample_scores = get_samples(encoder, dataiter)
        sample_z = sample_z[:, strong_dims]
    else:
        sample_z, sample_scores = get_restricted_samples(encoder, dataiter, n=n_samples, restriction=sample_restrictions)

    # Append samples to progress
    samples_z = torch.cat([samples_z, sample_z])
    samples_scores = torch.cat([samples_scores, sample_scores])

    # Optimization baseline
    ref_point = samples_scores.min(0)[0]

    # Acquisition loop
    for _ in range(16):
        z = normalize(torch.cat([samples_z, candidates_z]), z_bounds).detach()
        scores = normalize(torch.cat([samples_scores, candidates_scores]), score_bounds).detach()

        # Fit GP surrogate
        surrogate = fit_surrogate(z, scores, state_dict)
        state_dict = surrogate.state_dict()
        
        # Acquire candidate samples
        if acquisition_function == 'qehvi':
            new_candidates = optimize_qehvi(surrogate, z, scores, ref_point, unit_bounds)
        elif acquisition_function == 'qnparego':
            new_candidates = optimize_qnparego(surrogate, z, scores, unit_bounds)

        # Unnormalize, decompress, filter, score, and recompress candidates
        new_candidates = unnormalize(new_candidates, z_bounds)
        new_candidates = decompress(new_candidates, strong_dims, runspec.L)
        new_z, new_scores = filter_and_score(new_candidates, decoder, from_hot)
        if new_z.dim() > 1: new_z = new_z[:, strong_dims]

        # Update samples
        candidates_z = torch.cat([candidates_z, new_z])
        candidates_scores = torch.cat([candidates_scores, new_scores])

    # Save progress
    torch.save(samples_z, samples_z_file)
    torch.save(samples_scores, samples_scores_file)
    torch.save(candidates_z, candidates_z_file)
    torch.save(candidates_scores, candidates_scores_file)
    torch.save(state_dict, state_dict_file)
