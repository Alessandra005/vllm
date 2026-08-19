# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"


def test_mtp_draft_pipeline_parallel_defaults_to_one():
    """MTP-style drafters (which don't implement SupportsPP) should default
    to pipeline_parallel_size=1 for the draft model even when the target
    model uses pipeline parallelism, since vLLM fully replicates the draft
    model on the last PP rank rather than sharding it (see
    gpu_model_runner.py). Regression test for #52761."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100,
        trust_remote_code=True,
    )
    target_parallel_config = ParallelConfig(pipeline_parallel_size=3)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=target_parallel_config,
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=1,
    )

    assert speculative_config.draft_parallel_config.pipeline_parallel_size == 1
    assert speculative_config.draft_pipeline_parallel_size == 1