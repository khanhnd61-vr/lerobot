#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IMPACT invariants.

These pin the things the README calls out as easy to get wrong - the places where a
reasonable implementation is a *wrong* implementation and the model still trains, still
runs, and still produces plausible actions. None of them would be caught by a shape check
or by a loss that goes down.

The fixtures instantiate the real frozen T5-small tower, so the module needs
``transformers`` and a Hub fetch (cached after the first run).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.impact.configuration_impact import IMPACTConfig  # noqa: E402
from lerobot.policies.impact.film import FiLMHead, apply_film  # noqa: E402
from lerobot.policies.impact.modeling_impact import IMPACTPolicy  # noqa: E402
from lerobot.policies.impact.quantization import INT8_ALL  # noqa: E402
from lerobot.policies.impact.text import masked_mean  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

H, W = 120, 160
BATCH, STATE_DIM, ACTION_DIM = 2, 6, 6
N_REAL_TOKENS = 7


def make_config(**overrides) -> IMPACTConfig:
    kwargs = {
        "input_features": {
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "pretrained_backbone_weights": None,
        "device": "cpu",
        # Keep the test cheap; none of these invariants depend on depth.
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "chunk_size": 8,
        "n_action_steps": 8,
        "dim_feedforward": 64,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return IMPACTConfig(**kwargs)


def make_batch(config: IMPACTConfig, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    length = config.tokenizer_max_length
    mask = torch.zeros(BATCH, length, dtype=torch.long)
    mask[:, :N_REAL_TOKENS] = 1
    return {
        "observation.images.front": torch.rand(BATCH, 3, H, W, generator=g),
        "observation.images.wrist": torch.rand(BATCH, 3, H, W, generator=g),
        OBS_STATE: torch.randn(BATCH, STATE_DIM, generator=g),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (BATCH, length), generator=g),
        OBS_LANGUAGE_ATTENTION_MASK: mask,
        ACTION: torch.randn(BATCH, config.chunk_size, ACTION_DIM, generator=g),
        "action_is_pad": torch.zeros(BATCH, config.chunk_size, dtype=torch.bool),
    }


@pytest.fixture(scope="module")
def policy_and_batch():
    torch.manual_seed(0)
    config = make_config()
    policy = IMPACTPolicy(config)
    policy.eval()
    return policy, make_batch(config)


def test_chunk_shape_and_finiteness(policy_and_batch):
    policy, batch = policy_and_batch
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    assert actions.shape == (BATCH, policy.config.chunk_size, ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_film_is_identity_at_init():
    """``(1 + gamma) * x + beta`` with a zero-init head must leave the backbone alone.

    If the head were not zero-initialized, or the modulation were written ``gamma * x``,
    training would start from a mangled or annihilated feature map instead of from ACT.
    """
    head = FiLMHead(in_dim=16, channels=[4, 8])
    gammas, betas = head(torch.randn(3, 16))
    for g, b in zip(gammas, betas, strict=True):
        assert torch.equal(g, torch.zeros_like(g))
        assert torch.equal(b, torch.zeros_like(b))

    x = torch.randn(3, 4, 5, 5)
    assert torch.equal(apply_film(x, gammas[0], betas[0]), x)


def test_film_uses_one_plus_gamma():
    """A gamma of -1 must zero the feature map, not leave it unchanged.

    This is the test that fails if someone "simplifies" the modulation to ``gamma * x``:
    there, gamma = -1 negates rather than zeroes, and gamma = 0 zeroes rather than passes.
    """
    x = torch.randn(2, 3, 4, 4)
    minus_one = torch.full((2, 3), -1.0)
    zeros = torch.zeros(2, 3)
    assert torch.allclose(apply_film(x, minus_one, zeros), torch.zeros_like(x))

    beta = torch.randn(2, 3)
    out = apply_film(x, zeros, beta)
    assert torch.allclose(out, x + beta.unsqueeze(-1).unsqueeze(-1))


def test_film_head_splits_gamma_before_beta():
    """The flat head output is cut gamma-half first, then beta - the C++ port reads the
    same buffer with the same convention, so the order is part of the format."""
    head = FiLMHead(in_dim=4, channels=[2, 3])
    with torch.no_grad():
        head.net.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]))
    gammas, betas = head(torch.zeros(1, 4))
    assert torch.equal(gammas[0], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(gammas[1], torch.tensor([[3.0, 4.0, 5.0]]))
    assert torch.equal(betas[0], torch.tensor([[10.0, 20.0]]))
    assert torch.equal(betas[1], torch.tensor([[30.0, 40.0, 50.0]]))


def test_padded_text_rows_do_not_reach_the_output(policy_and_batch):
    """The single most load-bearing test in this file.

    T5 emits a real, non-zero hidden state for every padded position, so those rows are
    live tensors sitting in the encoder memory. They must be masked as keys in *both* the
    encoder self-attention and the decoder cross-attention. Miss either one and the actions
    depend on the tokenizer's padding - which changes with instruction length, so the same
    instruction in a differently-padded batch would command a different trajectory.

    Scrambling only the padded ids must leave the chunk bit-identical.
    """
    policy, batch = policy_and_batch
    with torch.no_grad():
        base = policy.predict_action_chunk(batch)

    scrambled = dict(batch)
    ids = batch[OBS_LANGUAGE_TOKENS].clone()
    ids[:, N_REAL_TOKENS:] = torch.randint(0, 1000, ids[:, N_REAL_TOKENS:].shape)
    scrambled[OBS_LANGUAGE_TOKENS] = ids
    with torch.no_grad():
        other = policy.predict_action_chunk(scrambled)

    assert torch.equal(base, other), (
        "padded text rows changed the action chunk: the key-padding mask is not reaching "
        "the encoder self-attention or the decoder cross-attention"
    )


def test_real_text_tokens_do_reach_the_output(policy_and_batch):
    """The converse. A policy that masks everything passes the test above trivially."""
    policy, batch = policy_and_batch
    with torch.no_grad():
        base = policy.predict_action_chunk(batch)

    changed = dict(batch)
    ids = batch[OBS_LANGUAGE_TOKENS].clone()
    ids[:, :N_REAL_TOKENS] = (ids[:, :N_REAL_TOKENS] + 137) % 1000
    changed[OBS_LANGUAGE_TOKENS] = ids
    with torch.no_grad():
        other = policy.predict_action_chunk(changed)

    assert not torch.equal(base, other), "the instruction has no effect on the action chunk"


def test_encoder_sequence_is_vision_then_text(policy_and_batch):
    """Text goes last, so the visual block layout stays identical to ACT's and the padding
    mask is a contiguous tail."""
    policy, batch = policy_and_batch
    model = policy.model
    config = policy.config

    captured = {}
    original = model.encoder.forward

    def spy(x, pos_embed=None, key_padding_mask=None):
        captured["seq"] = x.shape[0]
        captured["mask"] = key_padding_mask
        return original(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)

    model.encoder.forward = spy
    try:
        with torch.no_grad():
            policy.predict_action_chunk(batch)
    finally:
        model.encoder.forward = original

    n_1d = 2  # latent + robot state
    # Derived from the backbone rather than as H // 32: each stage floors independently, so
    # a 120x160 input lands on 4x5, not 3x5.
    with torch.no_grad():
        feat = model.backbone(batch["observation.images.front"][:1])
    n_vision = n_1d + 2 * feat.shape[-2] * feat.shape[-1]
    assert captured["seq"] == n_vision + config.tokenizer_max_length

    mask = captured["mask"]
    assert mask.shape == (BATCH, captured["seq"])
    assert not mask[:, :n_vision].any(), "a visual token was marked as padding"
    assert not mask[:, n_vision : n_vision + N_REAL_TOKENS].any()
    assert mask[:, n_vision + N_REAL_TOKENS :].all(), "padded text was left readable"


def test_masked_mean_ignores_padding():
    """FiLM is driven by this pool. An unmasked mean would make the backbone's modulation
    depend on instruction *length*, since the pad rows are real T5 outputs."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0, 3.0]]))

    all_pad = torch.zeros(1, 3, dtype=torch.long)
    assert torch.isfinite(masked_mean(hidden, all_pad)).all()


def test_language_dropout_blinds_both_paths():
    """Dropping language must drop it from the text tokens *and* from FiLM, so that a
    dropout-trained policy has practised the same "no language" the ablation sees."""
    torch.manual_seed(0)
    config = make_config(language_dropout=1.0)
    policy = IMPACTPolicy(config)
    policy.train()
    batch = make_batch(config)

    _, mask, pooled = policy.model.encode_language(batch)
    assert not mask.any(), "language dropout left text columns readable"
    assert torch.equal(pooled, torch.zeros_like(pooled)), "language dropout left FiLM driven"


def test_film_ablation_runs_and_differs():
    """``use_film=False`` is the arm that answers 'does the policy need FiLM at all'."""
    torch.manual_seed(0)
    policy = IMPACTPolicy(make_config(use_film=False))
    policy.eval()
    assert not hasattr(policy.model, "film_head")
    with torch.no_grad():
        actions = policy.predict_action_chunk(make_batch(policy.config))
    assert actions.shape == (BATCH, policy.config.chunk_size, ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_frozen_text_tower_stays_out_of_the_checkpoint(policy_and_batch):
    """A frozen encoder is fully determined by its Hub id, so shipping its weights in every
    checkpoint is dead weight - and silently drifting from the Hub copy is worse."""
    policy, _ = policy_and_batch
    keys = policy.state_dict().keys()
    assert not [k for k in keys if ".text.model." in k]
    assert not any(p.requires_grad for p in policy.model.text.parameters())


def test_training_step_produces_gradients():
    torch.manual_seed(0)
    config = make_config()
    policy = IMPACTPolicy(config)
    policy.train()
    loss, loss_dict = policy(make_batch(config))
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict and "kld_loss" in loss_dict
    loss.backward()

    assert policy.model.film_head.net.weight.grad is not None
    assert policy.model.text_input_proj.weight.grad is not None
    assert policy.model.text.model.shared.weight.grad is None, "the frozen tower got gradients"


def test_vae_sees_language_changes_the_latent():
    """Without the instruction the CVAE latent absorbs task identity in training, which
    makes the text pathway redundant in a way inference cannot reveal."""
    torch.manual_seed(0)
    with_lang = IMPACTPolicy(make_config(vae_sees_language=True))
    assert hasattr(with_lang.model, "vae_encoder_language_input_proj")

    torch.manual_seed(0)
    without = IMPACTPolicy(make_config(vae_sees_language=False))
    assert not hasattr(without.model, "vae_encoder_language_input_proj")

    n_with = with_lang.model.vae_encoder_pos_enc.shape[1]
    n_without = without.model.vae_encoder_pos_enc.shape[1]
    assert n_with == n_without + 1, "the language token is missing from the VAE pos encoding"


# ---------------------------------------------------------------------------
# The int8 path, where it touches the policy
# ---------------------------------------------------------------------------


def test_int8_off_is_the_default_and_the_fp32_policy(policy_and_batch):
    policy, _ = policy_and_batch
    assert policy.config.int8_groups == 0
    assert not policy.model.i8


def test_qat_checkpoint_carries_no_quantization_state():
    """The scales are all dynamic, so a QAT checkpoint is an ordinary fp32 checkpoint.

    This is the property that lets an int8-trained model load and export through every
    existing path unchanged, so it is pinned by key equality rather than left as a claim
    in a docstring: the state dict of an int8 policy must match the fp32 one exactly.
    """
    torch.manual_seed(0)
    fp32 = IMPACTPolicy(make_config(int8_groups=0))
    torch.manual_seed(0)
    qat = IMPACTPolicy(make_config(int8_groups=INT8_ALL))

    assert list(fp32.state_dict().keys()) == list(qat.state_dict().keys())
    assert not [k for k in qat.state_dict() if "scale" in k or "zero_point" in k or "observer" in k]


def test_int8_checkpoint_loads_into_an_fp32_policy():
    """`--policy.int8_groups=0` has to recover the float numerics from the same weights."""
    torch.manual_seed(0)
    qat = IMPACTPolicy(make_config(int8_groups=INT8_ALL))
    fp32 = IMPACTPolicy(make_config(int8_groups=0))

    missing, unexpected = fp32.load_state_dict(qat.state_dict(), strict=False)
    assert not unexpected
    assert not [k for k in missing if ".text.model." not in k]


def test_int8_changes_the_actions_but_keeps_them_finite():
    """If selecting every group left the output untouched, nothing is being quantized."""
    torch.manual_seed(0)
    config = make_config(int8_groups=INT8_ALL)
    policy = IMPACTPolicy(config)
    policy.eval()
    batch = make_batch(config)

    # Same weights, only the arithmetic differs - otherwise this would be comparing two
    # random initializations and would pass however the int8 path behaved.
    fp32 = IMPACTPolicy(make_config(int8_groups=0))
    fp32.load_state_dict(policy.state_dict(), strict=False)
    fp32.eval()

    with torch.no_grad():
        quantized = policy.predict_action_chunk(batch)
        floats = fp32.predict_action_chunk(make_batch(config))

    assert torch.isfinite(quantized).all()
    assert quantized.shape == floats.shape
    assert not torch.allclose(quantized, floats)


def test_qat_produces_gradients_through_the_quantized_path():
    """Quantization-aware training is only training if the straight-through estimator
    reaches the weights that the kernel rounds."""
    torch.manual_seed(0)
    config = make_config(int8_groups=INT8_ALL)
    policy = IMPACTPolicy(config)
    policy.train()

    loss, _ = policy(make_batch(config))
    assert torch.isfinite(loss)
    loss.backward()

    assert policy.model.backbone.conv1.weight.grad.abs().sum() > 0
    assert policy.model.encoder.layers[0].linear1.weight.grad.abs().sum() > 0
    assert policy.model.decoder.layers[0].linear1.weight.grad.abs().sum() > 0
