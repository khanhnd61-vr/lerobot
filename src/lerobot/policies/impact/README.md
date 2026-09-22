# IMPACT

**I**nstruction-**M**odulated **P**erception + **ACT**ion chunking.

ACT with a language tower. Two camera frames, a joint state and a natural-language
instruction go in; a 50-step action chunk comes out in one forward pass.

```
"pick up the black bowl"
      │
  T5-small encoder (frozen) ──> (B, L, 512)
      │
      ├─> masked mean-pool ──> FiLM head ──> (γ, β) per ResNet stage ──┐
      └─> text projection ──> L tokens appended to the encoder seq ─┐  │
                                                                    │  │
front ──┐                                                           │  │
        ├─> ResNet-18 + FiLM <─────────────────────────────────────────┘
wrist ──┘        │                                                  │
                 v                                                  v
   encoder sequence [latent | state | cam0 | cam1 | text] + key-padding mask
                 │
       6 × encoder layer ──> 4 × decoder layer, 50 queries ──> (B, 50, action_dim)
```

## Why it exists

ACT has no language tower, so it is a single-task policy: one checkpoint per task. It is
also the cheapest policy in this family to run on a CPU, which is what makes it the right
base to add language to: there is latency headroom to spend.

IMPACT spends a little of it. The added cost is small by construction — the encoder
already carries ~600 visual tokens, so 32 text tokens disappear into the noise, and the
frozen text tower runs once per instruction rather than once per query.

## The instruction enters twice, on purpose

**As tokens** in the transformer encoder sequence. The decoder reads them for free, because
it already cross-attends the whole encoder memory.

**As FiLM** on the ResNet stages — `(1 + γ) ⊙ x + β` per channel, from the pooled
instruction.

The second path is the one that matters. With ~600 visual tokens beside ~32 text tokens,
and manipulation scenes whose visible objects usually disambiguate the task, a policy will
happily learn to ignore the instruction: text tokens are a channel the gradient _may_ use
and never has to. FiLM puts language inside the perceptual path, where routing around it is
not an option.

Set `use_film=False` for the ablation that asks whether that was necessary.

## Configuration deltas from ACT

|                    | ACT | IMPACT                           |
| ------------------ | --- | -------------------------------- |
| `chunk_size`       | 100 | **50**                           |
| `n_encoder_layers` | 4   | **6**                            |
| `n_decoder_layers` | 1   | **4**                            |
| language           | —   | frozen T5-small, 32 tokens, FiLM |
| trainable params   | 34M | **~78M**                         |

`chunk_size` is halved because the point of a policy you can talk to is that you can retask
it, and at 30 Hz a 100-step chunk means a new instruction takes up to 3.33 s to take
effect. 50 halves that, and costs almost nothing: chunk length only touches the decoder's
query count.

`n_encoder_layers` and `n_decoder_layers` go up because ACT's depths were sized for a task
with no language in it — the encoder is where vision and language fuse, and a single
decoder layer is very shallow for reading a fused multimodal memory.

## Four things that are easy to get wrong

Each is pinned by a test in `tests/policies/impact/test_impact.py`, and each is a mistake
that trains, runs, and produces plausible actions.

1. **`(1 + γ) ⊙ x + β`, not `γ ⊙ x + β`.** The FiLM head is zero-initialized, so the `1 +`
   makes it exactly the identity at init and training starts from ordinary ACT. Written as
   a plain product, a zero-init head annihilates the feature map and nothing recovers.
2. **The text tower must see the real padding mask.** T5 is bidirectional: let it attend
   over padded positions and the _real_ tokens' hidden states already carry whatever the
   tokenizer padded with. No downstream mask repairs that. The symptom is that the same
   instruction gives different actions in a differently-padded batch.
3. **The padded rows are masked as keys in two places** — the encoder self-attention and
   the decoder's cross-attention over the memory. They hold genuine T5 outputs, not zeros.
4. **The CVAE style encoder should see the instruction** (`vae_sees_language=True`).
   Otherwise the latent absorbs task identity in training and the language pathway becomes
   redundant — a failure invisible at inference, where the latent is zero.

## The int8 path

`int8_groups` runs selected GEMMs through a simulated W8A8 kernel — in training, which
makes it quantization-aware training, and at inference. The quantizer in
[`quantization.py`](quantization.py) is not a generic utility: it transcribes one specific
deployment kernel, down to the reciprocal-multiply and the round-to-nearest-even, so that
what training optimizes is the arithmetic that actually runs on the target.

Two consequences worth knowing before using it:

- **A QAT checkpoint carries no quantization state.** All scales are dynamic — per output
  row for weights, per token (per image for a convolution) for activations — so the output
  is an ordinary fp32 checkpoint whose weights happen to survive rounding. It loads and
  exports through every existing path unchanged.
- **BatchNorm is folded before the convolution is quantized**, because the exporter folds
  too. Quantizing the raw weight would train the model against rounding error it will never
  meet, and leave it untrained against the error it will.

`int8_groups=0`, the default, is the ordinary fp32 policy and costs nothing.

## What this does not establish

Nothing here is a claim about the policy's behaviour. The tests prove the implementation
does what it says; they say nothing about whether the architecture is any good or whether
a trained policy would attend to language at all.

Answering that needs multi-task data with instruction variation, and an evaluation split
where two tasks **share a scene and differ only by the instruction**. Without that split, a
language-conditioned policy is indistinguishable from a language-shaped ornament — the
`use_film=False` and `language_dropout=1.0` arms exist to make the comparison.
