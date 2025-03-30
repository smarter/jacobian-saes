import argparse
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Union

import einops
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from transformer_lens.utils import get_act_name

import wandb
from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.toolkit.pretrained_saes_directory import (
    get_pretrained_saes_directory,
)
from jacobian_saes.training.activations_store import ActivationsStore
from jacobian_saes.training.sparsity_metrics import _gini, _kurtosis

jacobian_sparsity_thresholds = [0.005, 0.01, 0.02, 0.04]
FuncDict = Dict[str, Callable[[torch.Tensor], torch.Tensor]]
jac_norms: FuncDict = {
    "": lambda _: 1,
    "l2b": lambda jac: jac.pow(2).mean().sqrt() + 1e-20,
    "l2t": lambda jac: jac.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-20,
    "l4b": lambda jac: jac.pow(4).mean().pow(0.25) + 1e-20,
    "l4t": lambda jac: jac.pow(4).mean(dim=(-2, -1), keepdim=True).pow(0.25) + 1e-20,
    "linfb": lambda jac: jac.abs().max() + 1e-20,
    "linft": lambda jac: jac.abs().flatten(start_dim=-2).max(dim=-1).values[..., None, None] + 1e-20,
}
jac_normed_metrics: FuncDict = {
    "hist": lambda jac: jac.flatten(start_dim=-2).sort(dim=-1).values,
    "abs_hist": lambda jac: jac.abs().flatten(start_dim=-2).sort(dim=-1).values,
    "abs_max": lambda jac: jac.abs().flatten(start_dim=-2).max(dim=-1).values,

    "abs_hist_summed_inputs": lambda jac: jac.abs().sum(dim=-1).sort(dim=-1).values,
    "abs_hist_summed_outputs": lambda jac: jac.abs().sum(dim=-2).sort(dim=-1).values,

    # This metric goes down when each output dimension only depends on a small number of input dimensions
    "abs_max_of_summed_inputs": lambda jac: jac.abs().sum(dim=-1).max(dim=-1).values,
    # This metric goes down when each input dimension only affects a small number of output dimensions
    "abs_max_of_summed_outputs": lambda jac: jac.abs().sum(dim=-2).max(dim=-1).values,

    **{
        f"above_{thresh}": lambda jac: (jac.abs() > thresh).sum(dim=(-1, -2)).float()
        for thresh in jacobian_sparsity_thresholds
    },

    # Use scaled thresholds for summed dimensions
    **{
        f"abs_above_summed_inputs_{thresh}": lambda jac: (jac.abs().sum(dim=-1) > thresh * jac.shape[-1]).sum(dim=-1).float()
        for thresh in jacobian_sparsity_thresholds
    },
    **{
        f"abs_above_summed_outputs_{thresh}": lambda jac: (jac.abs().sum(dim=-2) > thresh * jac.shape[-2]).sum(dim=-1).float()
        for thresh in jacobian_sparsity_thresholds
    },
}
jac_norm_t_metrics: FuncDict = {
    "hist": lambda jac_norm: jac_norm.flatten().sort().values,
    "max": lambda jac_norm: jac_norm.flatten().max().unsqueeze(dim=0),
}
jac_norm_b_metrics: FuncDict = {
    "": lambda jac_norm: jac_norm.unsqueeze(dim=0),
}

jacobian_sparsity_metrics = [
    "jac_l1",
    "jac_gini",
    "jac_kurtosis",
    *[
        f"jac{'_normed_' if norm != '' else ''}{norm}_{metric}"
        for norm in jac_norms.keys()
        for metric in jac_normed_metrics.keys()
    ],
    *[
        f"jac_{norm}_norm_{metric}"
        for norm in jac_norms.keys()
        if norm != "" and norm[-1] == "t"
        for metric in jac_norm_t_metrics.keys()
    ],
    *[
        f"jac_{norm}_norm_{metric}"
        for norm in jac_norms.keys()
        if norm != "" and norm[-1] == "b"
        for metric in jac_norm_b_metrics.keys()
    ],
]
# Anything with "2" in the name also refers to Jacobian SAEs
# (it refers to reconstructing using the second SAE)


def get_git_hash() -> str:
    """
    Retrieves the current Git commit hash.
    Returns 'unknown' if the hash cannot be determined.
    """
    try:
        # Ensure the command is run in the directory where .git exists
        git_dir = Path(__file__).resolve().parent.parent  # Adjust if necessary
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=git_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


# Everything by default is false so the user can just set the ones they want to true
@dataclass
class EvalConfig:
    batch_size_prompts: int | None = None

    # Reconstruction metrics
    n_eval_reconstruction_batches: int = 10
    compute_kl: bool = False
    compute_ce_loss: bool = False

    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches: int = 1
    compute_l2_norms: bool = False
    compute_sparsity_metrics: bool = False
    compute_variance_metrics: bool = False
    # compute featurewise density statistics
    compute_featurewise_density_statistics: bool = False

    # compute featurewise_weight_based_metrics
    compute_featurewise_weight_based_metrics: bool = False

    git_hash: str = field(default_factory=get_git_hash)


def get_eval_everything_config(
    batch_size_prompts: int | None = None,
    n_eval_reconstruction_batches: int = 10,
    n_eval_sparsity_variance_batches: int = 1,
) -> EvalConfig:
    """
    Returns an EvalConfig object with all metrics set to True, so that when passed to run_evals all available metrics will be run.
    """
    return EvalConfig(
        batch_size_prompts=batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
    )


@torch.no_grad()
def run_evals(
    sae: SAEPair,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    eval_config: EvalConfig = EvalConfig(),
    model_kwargs: Mapping[str, Any] = {},
    ignore_tokens: set[int | None] = set(),
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Main entry point to the file; calls 3 downstream functions to compute
    the metrics and adds all of them into a dict.
    """

    hook_name = sae.cfg.hook_name
    actual_batch_size = (
        eval_config.batch_size_prompts or activation_store.store_batch_size_prompts
    )

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    all_metrics = {
        "model_behavior_preservation": {},
        "model_performance_preservation": {},
        "reconstruction_quality": {},
        "shrinkage": {},
        "sparsity": {},
        "jacobian_sparsity": {},
        "token_stats": {},
    }

    if eval_config.compute_kl or eval_config.compute_ce_loss:
        assert eval_config.n_eval_reconstruction_batches > 0
        reconstruction_metrics = get_downstream_reconstruction_metrics(
            sae,
            model,
            activation_store,
            compute_kl=eval_config.compute_kl,
            compute_ce_loss=eval_config.compute_ce_loss,
            n_batches=eval_config.n_eval_reconstruction_batches,
            eval_batch_size_prompts=actual_batch_size,
            ignore_tokens=ignore_tokens,
            verbose=verbose,
        )

        if eval_config.compute_kl:
            all_metrics["model_behavior_preservation"].update(
                {
                    "kl_div_score": reconstruction_metrics["kl_div_score"],
                    "kl_div_with_ablation": reconstruction_metrics[
                        "kl_div_with_ablation"
                    ],
                    "kl_div_with_sae": reconstruction_metrics["kl_div_with_sae"],
                }
            )
            if sae.cfg.use_jacobian_loss:
                all_metrics["model_behavior_preservation"].update(
                    {
                        "kl_div_score2": reconstruction_metrics["kl_div_score2"],
                        "kl_div_with_sae2": reconstruction_metrics["kl_div_with_sae2"],
                        "kl_div_with_ablation2": reconstruction_metrics[
                            "kl_div_with_ablation2"
                        ],
                    }
                )

        if eval_config.compute_ce_loss:
            all_metrics["model_performance_preservation"].update(
                {
                    "ce_loss_score": reconstruction_metrics["ce_loss_score"],
                    "ce_loss_with_ablation": reconstruction_metrics[
                        "ce_loss_with_ablation"
                    ],
                    "ce_loss_with_sae": reconstruction_metrics["ce_loss_with_sae"],
                    "ce_loss_without_sae": reconstruction_metrics[
                        "ce_loss_without_sae"
                    ],
                }
            )
            if sae.cfg.use_jacobian_loss:
                all_metrics["model_performance_preservation"].update(
                    {
                        "ce_loss_with_sae2": reconstruction_metrics[
                            "ce_loss_with_sae2"
                        ],
                        "ce_loss_with_ablation2": reconstruction_metrics[
                            "ce_loss_with_ablation2"
                        ],
                        "ce_loss_score2": reconstruction_metrics["ce_loss_score2"],
                    }
                )

        activation_store.reset_input_dataset()

    if (
        eval_config.compute_l2_norms
        or eval_config.compute_sparsity_metrics
        or eval_config.compute_variance_metrics
    ):
        assert eval_config.n_eval_sparsity_variance_batches > 0
        sparsity_variance_metrics, feature_metrics = get_sparsity_and_variance_metrics(
            sae,
            model,
            activation_store,
            compute_l2_norms=eval_config.compute_l2_norms,
            compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
            compute_variance_metrics=eval_config.compute_variance_metrics,
            compute_featurewise_density_statistics=eval_config.compute_featurewise_density_statistics,
            n_batches=eval_config.n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=actual_batch_size,
            model_kwargs=model_kwargs,
            ignore_tokens=ignore_tokens,
            verbose=verbose,
        )

        if eval_config.compute_l2_norms:
            all_metrics["shrinkage"].update(
                {
                    "l2_norm_in": sparsity_variance_metrics["l2_norm_in"],
                    "l2_norm_out": sparsity_variance_metrics["l2_norm_out"],
                    "l2_ratio": sparsity_variance_metrics["l2_ratio"],
                    "relative_reconstruction_bias": sparsity_variance_metrics[
                        "relative_reconstruction_bias"
                    ],
                }
            )
            if sae.cfg.use_jacobian_loss:
                all_metrics["shrinkage"].update(
                    {
                        "l2_norm_in2": sparsity_variance_metrics["l2_norm_in2"],
                        "l2_norm_out2": sparsity_variance_metrics["l2_norm_out2"],
                        "l2_ratio2": sparsity_variance_metrics["l2_ratio2"],
                        "relative_reconstruction_bias2": sparsity_variance_metrics[
                            "relative_reconstruction_bias2"
                        ],
                    }
                )

        if eval_config.compute_sparsity_metrics:
            all_metrics["sparsity"].update(
                {
                    "l0": sparsity_variance_metrics["l0"],
                    "l1": sparsity_variance_metrics["l1"],
                }
            )
            if sae.cfg.use_jacobian_loss:
                all_metrics["jacobian_sparsity"].update(
                    {m: sparsity_variance_metrics[m] for m in jacobian_sparsity_metrics}
                )

        if eval_config.compute_variance_metrics:
            all_metrics["reconstruction_quality"].update(
                {
                    "explained_variance": sparsity_variance_metrics[
                        "explained_variance"
                    ],
                    "mse": sparsity_variance_metrics["mse"],
                    "cossim": sparsity_variance_metrics["cossim"],
                }
            )
            if sae.cfg.use_jacobian_loss:
                all_metrics["reconstruction_quality"].update(
                    {
                        "explained_variance2": sparsity_variance_metrics[
                            "explained_variance2"
                        ],
                        "mse2": sparsity_variance_metrics["mse2"],
                        "cossim2": sparsity_variance_metrics["cossim2"],
                    }
                )

    else:
        feature_metrics = {}

    if eval_config.compute_featurewise_weight_based_metrics:
        feature_metrics |= get_featurewise_weight_based_metrics(sae, False)
        if sae.cfg.use_jacobian_loss:
            feature_metrics |= get_featurewise_weight_based_metrics(sae, True)

    if len(all_metrics) == 0:
        raise ValueError(
            "No metrics were computed, please set at least one metric to True."
        )

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    total_tokens_evaluated_eval_reconstruction = (
        activation_store.context_size
        * eval_config.n_eval_reconstruction_batches
        * actual_batch_size
    )

    total_tokens_evaluated_eval_sparsity_variance = (
        activation_store.context_size
        * eval_config.n_eval_sparsity_variance_batches
        * actual_batch_size
    )

    all_metrics["token_stats"] = {
        "total_tokens_eval_reconstruction": total_tokens_evaluated_eval_reconstruction,
        "total_tokens_eval_sparsity_variance": total_tokens_evaluated_eval_sparsity_variance,
    }

    # Remove empty metric groups
    all_metrics = {k: v for k, v in all_metrics.items() if v}

    # Flatten the nested feature metrics
    all_metrics = flatten_dict(all_metrics)

    return all_metrics, feature_metrics


def get_featurewise_weight_based_metrics(
    sae: SAEPair, is_output_sae: bool
) -> dict[str, Any]:

    unit_norm_encoders = (
        sae.get_W_enc(is_output_sae)
        / sae.get_W_enc(is_output_sae).norm(dim=0, keepdim=True)
    ).cpu()
    unit_norm_decoder = (
        sae.get_W_dec(is_output_sae).T
        / sae.get_W_dec(is_output_sae).T.norm(dim=0, keepdim=True)
    ).cpu()

    encoder_norms = sae.get_W_enc(is_output_sae).norm(dim=-2).cpu().tolist()
    encoder_bias = sae.get_b_enc(is_output_sae).cpu().tolist()
    encoder_decoder_cosine_sim = (
        torch.nn.functional.cosine_similarity(
            unit_norm_decoder.T,
            unit_norm_encoders.T,
        )
        .cpu()
        .tolist()
    )

    return {
        f"encoder_bias{"2" if is_output_sae else ""}": encoder_bias,
        f"encoder_norm{"2" if is_output_sae else ""}": encoder_norms,
        f"encoder_decoder_cosine_sim{"2" if is_output_sae else ""}": encoder_decoder_cosine_sim,
    }


def get_downstream_reconstruction_metrics(
    sae: SAEPair,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    n_batches: int,
    eval_batch_size_prompts: int,
    ignore_tokens: set[int | None] = set(),
    verbose: bool = False,
):
    """
    Iterates over tokens, calls get_recons_loss on each batch, computes
    the CE loss score and the KL div score
    """

    metrics_dict = {}
    if compute_kl:
        metrics_dict["kl_div_with_sae"] = []
        metrics_dict["kl_div_with_ablation"] = []
        if sae.cfg.use_jacobian_loss:
            metrics_dict["kl_div_with_sae2"] = []
            metrics_dict["kl_div_with_ablation2"] = []
    if compute_ce_loss:
        metrics_dict["ce_loss_with_sae"] = []
        metrics_dict["ce_loss_without_sae"] = []
        metrics_dict["ce_loss_with_ablation"] = []
        if sae.cfg.use_jacobian_loss:
            metrics_dict["ce_loss_with_sae2"] = []
            metrics_dict["ce_loss_with_ablation2"] = []

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Reconstruction Batches")

    for _ in batch_iter:
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        for metric_name, metric_value in get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
            compute_kl=compute_kl,
            compute_ce_loss=compute_ce_loss,
        ).items():

            if len(ignore_tokens) > 0:
                mask = torch.logical_not(
                    torch.any(
                        torch.stack(
                            [batch_tokens == token for token in ignore_tokens], dim=0
                        ),
                        dim=0,
                    )
                )
                if metric_value.shape[1] != mask.shape[1]:
                    # ce loss will be missing the last value
                    mask = mask[:, :-1]
                metric_value = metric_value[mask]

            metrics_dict[metric_name].append(metric_value)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metrics_dict.items():
        metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

    if compute_kl:
        metrics["kl_div_score"] = (
            metrics["kl_div_with_ablation"] - metrics["kl_div_with_sae"]
        ) / metrics["kl_div_with_ablation"]
        if sae.cfg.use_jacobian_loss:
            metrics["kl_div_score2"] = (
                metrics["kl_div_with_ablation2"] - metrics["kl_div_with_sae2"]
            ) / metrics["kl_div_with_ablation2"]

    if compute_ce_loss:
        metrics["ce_loss_score"] = (
            metrics["ce_loss_with_ablation"] - metrics["ce_loss_with_sae"]
        ) / (metrics["ce_loss_with_ablation"] - metrics["ce_loss_without_sae"])
        if sae.cfg.use_jacobian_loss:
            metrics["ce_loss_score2"] = (
                metrics["ce_loss_with_ablation2"] - metrics["ce_loss_with_sae2"]
            ) / (metrics["ce_loss_with_ablation2"] - metrics["ce_loss_without_sae"])

    return metrics


def get_variance_metrics(flattened_sae_input, flattened_sae_out, flattened_mask):
    """
    Computes the MSE, explained variance, and cosine similarity between the
    input and output of the SAE.
    """

    resid_sum_of_squares = (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
    total_sum_of_squares = (
        (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
    )

    mse = resid_sum_of_squares / flattened_mask.sum()
    explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares

    x_normed = flattened_sae_input / torch.norm(
        flattened_sae_input, dim=-1, keepdim=True
    )
    x_hat_normed = flattened_sae_out / torch.norm(
        flattened_sae_out, dim=-1, keepdim=True
    )
    cossim = (x_normed * x_hat_normed).sum(dim=-1)

    return mse, explained_variance, cossim


def get_l2_norms(flattened_sae_input, flattened_sae_out):
    l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
    l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
    l2_norm_in_for_div = l2_norm_in.clone()
    l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
    l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

    # Equation 10 from https://arxiv.org/abs/2404.16014
    # https://github.com/saprmarks/dictionary_learning/blob/main/evaluation.py
    x_hat_norm_squared = torch.norm(flattened_sae_out, dim=-1) ** 2
    x_dot_x_hat = (flattened_sae_input * flattened_sae_out).sum(dim=-1)
    relative_reconstruction_bias = (
        x_hat_norm_squared.mean() / x_dot_x_hat.mean()
    ).unsqueeze(0)

    return l2_norm_in, l2_norm_out, l2_norm_ratio, relative_reconstruction_bias


def get_sparsity_and_variance_metrics(
    sae: SAEPair,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int,
    compute_l2_norms: bool,
    compute_sparsity_metrics: bool,
    compute_variance_metrics: bool,
    compute_featurewise_density_statistics: bool,
    eval_batch_size_prompts: int,
    model_kwargs: Mapping[str, Any],
    ignore_tokens: set[int | None] = set(),
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Iterates over the activations, runs the LLM, runs the activations through
    the SAE, computes metrics (L2 norms of sae ins and outs, L0 and L1 of
    sae acts, MSE, explained variance, cosine sim, activation density etc)
    """

    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    metric_dict = {}
    feature_metric_dict = {}

    if compute_l2_norms:
        metric_dict["l2_norm_in"] = []
        metric_dict["l2_norm_out"] = []
        metric_dict["l2_ratio"] = []
        metric_dict["relative_reconstruction_bias"] = []
        if sae.cfg.use_jacobian_loss:
            metric_dict["l2_norm_in2"] = []
            metric_dict["l2_norm_out2"] = []
            metric_dict["l2_ratio2"] = []
            metric_dict["relative_reconstruction_bias2"] = []
    if compute_sparsity_metrics:
        metric_dict["l0"] = []
        metric_dict["l1"] = []
        if sae.cfg.use_jacobian_loss:
            for metric_name in jacobian_sparsity_metrics:
                metric_dict[metric_name] = []
    if compute_variance_metrics:
        metric_dict["explained_variance"] = []
        metric_dict["mse"] = []
        metric_dict["cossim"] = []
        if sae.cfg.use_jacobian_loss:
            metric_dict["explained_variance2"] = []
            metric_dict["mse2"] = []
            metric_dict["cossim2"] = []
    if compute_featurewise_density_statistics:
        feature_metric_dict["feature_density"] = []
        feature_metric_dict["consistent_activation_heuristic"] = []

    total_feature_acts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_feature_prompts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_tokens = 0

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Sparsity and Variance Batches")

    for _ in batch_iter:
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

        if len(ignore_tokens) > 0:
            mask = torch.logical_not(
                torch.any(
                    torch.stack(
                        [batch_tokens == token for token in ignore_tokens], dim=0
                    ),
                    dim=0,
                )
            )
        else:
            mask = torch.ones_like(batch_tokens, dtype=torch.bool)
        flattened_mask = mask.flatten()

        # get cache
        has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
        cache = {}

        def reconstr_and_cache_hook(activations: torch.Tensor, hook: Any):
            is_output_sae = "mlp_out" in hook.name

            # we would include hook z, except that we now have base SAE's
            # which will do their own reshaping for hook z.
            if hook_head_index is not None:
                original_act = activations[:, :, hook_head_index]
            elif any(
                substring in hook_name for substring in has_head_dim_key_substrings
            ):
                original_act = activations.flatten(-2, -1)
            else:
                original_act = activations

            # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
            if activation_store.normalize_activations == "expected_average_only_in":
                original_act = activation_store.apply_norm_scaling_factor(original_act)

            # send the (maybe normalised) activations into the SAE
            if sae.cfg.use_jacobian_loss:
                sae_feature_activations, topk_indices = sae.encode(
                    original_act.to(sae.device), is_output_sae, return_topk_indices=True
                )
            else:
                sae_feature_activations = sae.encode(
                    original_act.to(sae.device), is_output_sae
                )
            sae_out = sae.decode(sae_feature_activations, is_output_sae).to(
                original_act.device
            )

            if activation_store.normalize_activations == "expected_average_only_in":
                sae_out = activation_store.unscale(sae_out)

            flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
            flattened_sae_feature_acts = einops.rearrange(
                sae_feature_activations, "b ctx d -> (b ctx) d"
            )
            flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")
            if sae.cfg.use_jacobian_loss:
                flattened_topk_indices = einops.rearrange(
                    topk_indices, "b ctx k -> (b ctx) k"
                )

            # TODO: Clean this up.
            # apply mask
            masked_sae_feature_activations = sae_feature_activations * mask.unsqueeze(
                -1
            )
            flattened_sae_input = flattened_sae_input[flattened_mask]
            flattened_sae_feature_acts = flattened_sae_feature_acts[flattened_mask]
            flattened_sae_out = flattened_sae_out[flattened_mask]
            if sae.cfg.use_jacobian_loss:
                flattened_topk_indices = flattened_topk_indices[flattened_mask]

            cache_dict = {
                "masked_sae_feature_activations": masked_sae_feature_activations,
                "flattened_sae_input": flattened_sae_input,
                "flattened_sae_feature_acts": flattened_sae_feature_acts,
                "flattened_sae_out": flattened_sae_out,
            }
            if sae.cfg.use_jacobian_loss:
                cache_dict["flattened_topk_indices"] = flattened_topk_indices

            if hook.name == hook_name:
                cache["hook"] = cache_dict
            elif "mlp_out" in hook.name:
                cache["mlp_out"] = cache_dict
            else:
                raise ValueError(f"Unexpected hook name: {hook.name}")

            return sae_out

        model.run_with_hooks(
            batch_tokens,
            prepend_bos=False,
            fwd_hooks=[(hook_name, reconstr_and_cache_hook)],
            stop_at_layer=sae.cfg.hook_layer + 1,
            **model_kwargs,
        )

        if sae.cfg.use_jacobian_loss:
            model.run_with_hooks(
                batch_tokens,
                prepend_bos=False,
                fwd_hooks=[
                    (
                        get_act_name("mlp_out", sae.cfg.hook_layer),
                        reconstr_and_cache_hook,
                    )
                ],
                stop_at_layer=sae.cfg.hook_layer + 1,
                **model_kwargs,
            )

        masked_sae_feature_activations = cache["hook"]["masked_sae_feature_activations"]
        flattened_sae_input = cache["hook"]["flattened_sae_input"]
        flattened_sae_feature_acts = cache["hook"]["flattened_sae_feature_acts"]
        flattened_sae_out = cache["hook"]["flattened_sae_out"]
        if sae.cfg.use_jacobian_loss:
            flattened_topk_indices = cache["hook"]["flattened_topk_indices"]
            flattened_sae_input_mlp_out = cache["mlp_out"]["flattened_sae_input"]
            flattened_sae_out_mlp_out = cache["mlp_out"]["flattened_sae_out"]

        if compute_l2_norms:
            l2_norm_in, l2_norm_out, l2_norm_ratio, relative_reconstruction_bias = (
                get_l2_norms(flattened_sae_input, flattened_sae_out)
            )

            metric_dict["l2_norm_in"].append(l2_norm_in)
            metric_dict["l2_norm_out"].append(l2_norm_out)
            metric_dict["l2_ratio"].append(l2_norm_ratio)
            metric_dict["relative_reconstruction_bias"].append(
                relative_reconstruction_bias
            )

            if sae.cfg.use_jacobian_loss:
                (
                    l2_norm_in2,
                    l2_norm_out2,
                    l2_norm_ratio2,
                    relative_reconstruction_bias2,
                ) = get_l2_norms(flattened_sae_input_mlp_out, flattened_sae_out_mlp_out)

                metric_dict["l2_norm_in2"].append(l2_norm_in2)
                metric_dict["l2_norm_out2"].append(l2_norm_out2)
                metric_dict["l2_ratio2"].append(l2_norm_ratio2)
                metric_dict["relative_reconstruction_bias2"].append(
                    relative_reconstruction_bias2
                )

        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            metric_dict["l0"].append(l0)
            metric_dict["l1"].append(l1)

            if sae.cfg.use_jacobian_loss:
                mlp_out, mlp_act_grads = sae.mlp(flattened_sae_out)
                _, _, topk_indices2 = sae.encode_with_hidden_pre_fn(
                    mlp_out, True, return_topk_indices=True
                )

                wd1 = sae.get_W_dec(False) @ sae.mlp.W_in  # (d_sae, d_mlp)
                w2e = sae.mlp.W_out @ sae.get_W_enc(True)  # (d_mlp, d_sae)
                jacobian = einops.einsum(
                    wd1[flattened_topk_indices],
                    mlp_act_grads,
                    w2e[:, topk_indices2],
                    "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
                    "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
                )

                metric_dict["jac_l1"].append(jacobian.abs().sum(dim=(-1, -2)))
                metric_dict["jac_gini"].append(_gini(jacobian))
                metric_dict["jac_kurtosis"].append(_kurtosis(jacobian))

                # compute the sparsity metrics for different normalizations of the Jacobian
                # (for each norm, compute each metric)
                for norm_name, norm_func in jac_norms.items():
                    norm_val = norm_func(jacobian)
                    jac_normed = jacobian / norm_val

                    for metric_name, metric_func in jac_normed_metrics.items():
                        key = f"jac{'_normed_' if norm_name != '' else ''}{norm_name}_{metric_name}"
                        metric_dict[key].append(metric_func(jac_normed))

                    # the metrics below are metrics of the norms; not relevant when we're skipping normalization
                    if norm_name == "":
                        continue
                    if norm_name[-1] == "t":
                        # metrics only relevant for per-token norms
                        for metric_name, metric_func in jac_norm_t_metrics.items():
                            key = f"jac_{norm_name}_norm_{metric_name}"
                            metric_dict[key].append(metric_func(norm_val))
                    elif norm_name[-1] == "b":
                        # metrics only relevant for per-batch norms
                        for metric_name, metric_func in jac_norm_b_metrics.items():
                            key = f"jac_{norm_name}_norm_{metric_name}"
                            metric_dict[key].append(metric_func(norm_val))

        if compute_variance_metrics:
            mse, explained_variance, cossim = get_variance_metrics(
                flattened_sae_input, flattened_sae_out, flattened_mask
            )
            metric_dict["explained_variance"].append(explained_variance)
            metric_dict["mse"].append(mse)
            metric_dict["cossim"].append(cossim)

            if sae.cfg.use_jacobian_loss:
                mse2, explained_variance2, cossim2 = get_variance_metrics(
                    flattened_sae_input_mlp_out,
                    flattened_sae_out_mlp_out,
                    flattened_mask,
                )
                metric_dict["mse2"].append(mse2)
                metric_dict["explained_variance2"].append(explained_variance2)
                metric_dict["cossim2"].append(cossim2)

        if compute_featurewise_density_statistics:
            sae_feature_activations_bool = (masked_sae_feature_activations > 0).float()
            total_feature_acts += sae_feature_activations_bool.sum(dim=1).sum(dim=0)
            total_feature_prompts += (sae_feature_activations_bool.sum(dim=1) > 0).sum(
                dim=0
            )
            total_tokens += mask.sum()

    # Aggregate scalar metrics
    metrics: dict[str, float | wandb.Histogram] = {}
    for metric_name, metric_values in metric_dict.items():
        if "hist" in metric_name:
            # The averaging here is hacky
            metrics[f"{metric_name}"] = wandb.Histogram(
                torch.stack(metric_values).sort(dim=1).values.mean(dim=0).cpu().numpy()
            )
        else:
            metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

    # Aggregate feature-wise metrics
    feature_metrics: dict[str, list[float]] = {}
    feature_metrics["feature_density"] = (total_feature_acts / total_tokens).tolist()
    feature_metrics["consistent_activation_heuristic"] = (
        total_feature_acts / total_feature_prompts
    ).tolist()

    return metrics, feature_metrics


# TODO for faster performance and less duplication, consider merging
# get_downstream_reconstruction_metrics, get_sparsity_and_variance_metrics, and
# get_recons_loss, that way we'd only iterate over the dataset once and do one
# clean LLM forward pass per batch


@torch.no_grad()
def get_recons_loss(
    sae: SAEPair,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:
    """
    Runs the model with the clean activations, SAE-reconstructed activations,
    and zero-ablated activations; computes the KL divergence between the logits
    and the CE loss score.
    """

    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    original_logits, original_ce_loss = model(
        batch_tokens, return_type="both", loss_per_token=True, **model_kwargs
    )

    metrics = {}

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        is_output_sae = "mlp_out" in hook.name

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        activations = sae.decode(
            sae.encode(activations, is_output_sae), is_output_sae
        ).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)

        return activations.to(original_device)

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
        is_output_sae = "mlp_out" in hook.name

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(
            sae.encode(activations.flatten(-2, -1), is_output_sae), is_output_sae
        ).to(activations.dtype)

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        return new_activations.to(original_device)

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
        is_output_sae = "mlp_out" in hook.name

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        new_activations = sae.decode(
            sae.encode(activations[:, :, head_index], is_output_sae), is_output_sae
        ).to(activations.dtype)
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        return activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        loss_per_token=True,
        **model_kwargs,
    )

    zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        loss_per_token=True,
        **model_kwargs,
    )

    if sae.cfg.use_jacobian_loss:
        recons_logits2, recons_ce_loss2 = model.run_with_hooks(
            batch_tokens,
            return_type="both",
            fwd_hooks=[(get_act_name("mlp_out", sae.cfg.hook_layer), replacement_hook)],
            loss_per_token=True,
            **model_kwargs,
        )

        zero_abl_logits2, zero_abl_ce_loss2 = model.run_with_hooks(
            batch_tokens,
            return_type="both",
            fwd_hooks=[(get_act_name("mlp_out", sae.cfg.hook_layer), zero_ablate_hook)],
            loss_per_token=True,
            **model_kwargs,
        )

    def kl(original_logits: torch.Tensor, new_logits: torch.Tensor, eps: float = 1e-8):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs + eps)
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        log_new_probs = torch.log(new_probs + eps)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    if compute_kl:
        recons_kl_div = kl(original_logits, recons_logits)
        zero_abl_kl_div = kl(original_logits, zero_abl_logits)
        metrics["kl_div_with_sae"] = recons_kl_div
        metrics["kl_div_with_ablation"] = zero_abl_kl_div
        if sae.cfg.use_jacobian_loss:
            recons_kl_div2 = kl(original_logits, recons_logits2)
            zero_abl_kl_div2 = kl(original_logits, zero_abl_logits2)
            metrics["kl_div_with_sae2"] = recons_kl_div2
            metrics["kl_div_with_ablation2"] = zero_abl_kl_div2

    if compute_ce_loss:
        metrics["ce_loss_with_sae"] = recons_ce_loss
        metrics["ce_loss_without_sae"] = original_ce_loss
        metrics["ce_loss_with_ablation"] = zero_abl_ce_loss
        if sae.cfg.use_jacobian_loss:
            metrics["ce_loss_with_sae2"] = recons_ce_loss2
            metrics["ce_loss_with_ablation2"] = zero_abl_ce_loss2

    return metrics


def flatten_dict(d: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    """
    Take a nested dictionary and flatten it to a single level dictionary,
    with keys separated by the given separator.

    Example:
    {
        "a": 1,
        "b": {
            "c": 2,
            "d": 3
        }
    }

    becomes

    {
        "a": 1,
        "b/c": 2,
        "b/d": 3
    }
    """

    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                result[key + sep + key2] = value2
        else:
            result[key] = value
    return result


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append(
                (release, sae_name, expected_var_explained, expected_l0)
            )

    return all_loadable_saes


def get_saes_from_regex(
    sae_regex_pattern: str, sae_block_pattern: str
) -> list[tuple[str, str, float, float]]:
    sae_regex_compiled = re.compile(sae_regex_pattern)
    sae_block_compiled = re.compile(sae_block_pattern)
    all_saes = all_loadable_saes()
    filtered_saes = [
        sae
        for sae in all_saes
        if sae_regex_compiled.fullmatch(sae[0]) and sae_block_compiled.fullmatch(sae[1])
    ]
    return filtered_saes


def nested_dict() -> defaultdict[Any, Any]:
    return defaultdict(nested_dict)


def dict_to_nested(flat_dict: dict[str, Any]) -> defaultdict[Any, Any]:
    nested = nested_dict()
    for key, value in flat_dict.items():
        parts = key.split("/")
        d = nested
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value
    return nested


def multiple_evals(
    sae_regex_pattern: str,
    sae_block_pattern: str,
    n_eval_reconstruction_batches: int,
    n_eval_sparsity_variance_batches: int,
    eval_batch_size_prompts: int = 8,
    datasets: list[str] = ["Skylion007/openwebtext", "lighteval/MATH"],
    ctx_lens: list[int] = [128],
    output_dir: str = "eval_results",
    verbose: bool = False,
) -> List[Dict[str, Any]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_saes = get_saes_from_regex(sae_regex_pattern, sae_block_pattern)

    assert len(filtered_saes) > 0, "No SAEs matched the given regex patterns"

    eval_results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_config = get_eval_everything_config(
        batch_size_prompts=eval_batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
    )

    current_model = None
    current_model_str = None
    print(filtered_saes)
    for sae_release_name, sae_id, _, _ in tqdm(filtered_saes):

        sae = SAEPair.from_pretrained(
            release=sae_release_name,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=sae_id,  # won't always be a hook point
            device=device,
        )[0]

        # move SAE to device if not there already
        sae.to(device)

        if current_model_str != sae.cfg.model_name:
            del current_model  # potentially saves GPU memory
            current_model_str = sae.cfg.model_name
            current_model = HookedTransformer.from_pretrained_no_processing(
                current_model_str, device=device, **sae.cfg.model_from_pretrained_kwargs
            )
        assert current_model is not None

        for ctx_len in ctx_lens:
            for dataset in datasets:

                activation_store = ActivationsStore.from_sae(
                    current_model, sae, context_size=ctx_len, dataset=dataset
                )
                activation_store.shuffle_input_dataset(seed=42)

                eval_metrics = nested_dict()
                eval_metrics["unique_id"] = f"{sae_release_name}-{sae_id}"
                eval_metrics["sae_set"] = f"{sae_release_name}"
                eval_metrics["sae_id"] = f"{sae_id}"
                eval_metrics["eval_cfg"]["context_size"] = ctx_len
                eval_metrics["eval_cfg"]["dataset"] = dataset
                eval_metrics["eval_cfg"][
                    "library_version"
                ] = eval_config.library_version
                eval_metrics["eval_cfg"]["git_hash"] = eval_config.git_hash

                scalar_metrics, feature_metrics = run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=current_model,
                    eval_config=eval_config,
                    ignore_tokens={
                        current_model.tokenizer.pad_token_id,  # type: ignore
                        current_model.tokenizer.eos_token_id,  # type: ignore
                        current_model.tokenizer.bos_token_id,  # type: ignore
                    },
                    verbose=verbose,
                )
                eval_metrics["metrics"] = scalar_metrics
                eval_metrics["feature_metrics"] = feature_metrics

                # Add SAE config
                eval_metrics["sae_cfg"] = sae.cfg.to_dict()

                # Add eval config
                eval_metrics["eval_cfg"].update(eval_config.__dict__)

                eval_results.append(eval_metrics)

    return eval_results


def run_evaluations(args: argparse.Namespace) -> List[Dict[str, Any]]:
    # Filter SAEs based on regex patterns
    filtered_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)

    num_sae_sets = len(set(sae_set for sae_set, _, _, _ in filtered_saes))
    num_all_sae_ids = len(filtered_saes)

    print("Filtered SAEs based on provided patterns:")
    print(f"Number of SAE sets: {num_sae_sets}")
    print(f"Total number of SAE IDs: {num_all_sae_ids}")

    eval_results = multiple_evals(
        sae_regex_pattern=args.sae_regex_pattern,
        sae_block_pattern=args.sae_block_pattern,
        n_eval_reconstruction_batches=args.n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=args.n_eval_sparsity_variance_batches,
        eval_batch_size_prompts=args.batch_size_prompts,
        datasets=args.datasets,
        ctx_lens=args.ctx_lens,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    return eval_results


def replace_nans_with_negative_one(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: replace_nans_with_negative_one(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nans_with_negative_one(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return -1
    else:
        return obj


def process_results(
    eval_results: List[Dict[str, Any]], output_dir: str
) -> Dict[str, Union[List[Path], Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Replace NaNs with -1 in each result
    cleaned_results = [
        replace_nans_with_negative_one(result) for result in eval_results
    ]

    # Save individual JSON files
    for result in cleaned_results:
        json_filename = f"{result['unique_id']}_{result['eval_cfg']['context_size']}_{result['eval_cfg']['dataset']}.json".replace(
            "/", "_"
        )
        json_path = output_path / json_filename
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save all results in a single JSON file
    with open(output_path / "all_eval_results.json", "w") as f:
        json.dump(cleaned_results, f, indent=2)

    # Convert to DataFrame and save as CSV
    df = pd.json_normalize(cleaned_results)
    df.to_csv(output_path / "all_eval_results.csv", index=False)

    return {
        "individual_jsons": list(output_path.glob("*.json")),
        "combined_json": output_path / "all_eval_results.json",
        "csv": output_path / "all_eval_results.csv",
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run evaluations on SAEs")
    arg_parser.add_argument(
        "sae_regex_pattern",
        type=str,
        help="Regex pattern to match SAE names. Can be an entire SAE name to match a specific SAE.",
    )
    arg_parser.add_argument(
        "sae_block_pattern",
        type=str,
        help="Regex pattern to match SAE block names. Can be an entire block name to match a specific block.",
    )
    arg_parser.add_argument(
        "--batch_size_prompts",
        type=int,
        default=16,
        help="Batch size for evaluation prompts.",
    )
    arg_parser.add_argument(
        "--n_eval_reconstruction_batches",
        type=int,
        default=10,
        help="Number of evaluation batches for reconstruction metrics.",
    )
    arg_parser.add_argument(
        "--compute_kl",
        action="store_true",
        help="Compute KL divergence.",
    )
    arg_parser.add_argument(
        "--compute_ce_loss",
        action="store_true",
        help="Compute cross-entropy loss.",
    )
    arg_parser.add_argument(
        "--n_eval_sparsity_variance_batches",
        type=int,
        default=1,
        help="Number of evaluation batches for sparsity and variance metrics.",
    )
    arg_parser.add_argument(
        "--compute_l2_norms",
        action="store_true",
        help="Compute L2 norms.",
    )
    arg_parser.add_argument(
        "--compute_sparsity_metrics",
        action="store_true",
        help="Compute sparsity metrics.",
    )
    arg_parser.add_argument(
        "--compute_variance_metrics",
        action="store_true",
        help="Compute variance metrics.",
    )
    arg_parser.add_argument(
        "--compute_featurewise_density_statistics",
        action="store_true",
        help="Compute featurewise density statistics.",
    )
    arg_parser.add_argument(
        "--compute_featurewise_weight_based_metrics",
        action="store_true",
        help="Compute featurewise weight-based metrics.",
    )
    arg_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Skylion007/openwebtext"],
        help="Datasets to evaluate on, such as 'Skylion007/openwebtext' or 'lighteval/MATH'.",
    )
    arg_parser.add_argument(
        "--ctx_lens",
        nargs="+",
        type=int,
        default=[128],
        help="Context lengths to evaluate on.",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with tqdm loaders.",
    )

    args = arg_parser.parse_args()
    eval_results = run_evaluations(args)
    output_files = process_results(eval_results, args.output_dir)

    print("Evaluation complete. Output files:")
    print(f"Individual JSONs: {len(output_files['individual_jsons'])}")  # type: ignore
    print(f"Combined JSON: {output_files['combined_json']}")
    print(f"CSV: {output_files['csv']}")
