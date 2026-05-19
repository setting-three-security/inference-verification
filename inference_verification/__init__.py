"""
Inference Verification

Modules for verifying LLM-generated tokens using Gumbel Likelihood Score (GLS)
and Convolved Gaussian Score (CGS) methods.

Import scoring helpers from their submodules directly, e.g.

    from inference_verification.scoring_functions import (
        compute_gumbel_likelihood_score,
        compute_convolved_gaussian_score,
    )

The package `__init__` deliberately does no eager imports so that
torch-free CI lanes can import sibling submodules (e.g. webapp surfaces)
without pulling the GPU stack transitively.
"""
