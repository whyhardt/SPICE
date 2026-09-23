# eckstein2026 model variants

Simplifications of the original model (`spice_eckstein2026.py`), one standalone file per variant
(`spice_eckstein2026_<variant>.py`).

## Why

In the stability runs of the original model (`params/params/`), the RNN reproduced well
(held-out trial likelihood 0.575–0.610 for every ensemble member), but the discovered equations
failed erratically: single members fell to 0.04–0.23, below the 4-action chance level of 0.25.
The cause was the exploration equations. Their state × `dvalue` product terms act as an
input-dependent self-coefficient; with values around +2.8 they amplify the exploration state into
the ±10 clip within a few trials. The fit has no check for that, so whether a member lands on a
stable or unstable solution is chance.

Knock-out analysis on the original RNN (held-out block 3, runs 0/2/8; Δ in held-out trial likelihood):

| Knock-out | Δ |
|---|---|
| `value_reward` removed from the logit | −0.255 |
| exploration removed from the logit | −0.036 (chosen module off: −0.061) |
| exploration input `dvalue` = 0 | −0.024 |
| `value_reward_env` off (acts only via chaining) | −0.024 |
| `bias_attention` removed from the logit | −0.011 |
| both choice modules removed from the logit | −0.019 |
| `value_choice` removed from the logit | −0.006 |
| `value_reward_mean` input = 0 | −0.003 |

Hence: `value_reward_mean` is nearly free to drop; exploration matters but its input can be
simplified; `value_reward_env`, `value_choice` and `bias_attention` have small to moderate effects.

## Dimensions varied

- **Chaining into the reward modules:** `value_reward_mean` (mean reward value, feeds back into the
  reward modules) and `reward_env` (state of `value_reward_env`).
- **Exploration input:** `split` = relu(dvalue), relu(−dvalue) (original); `signed` = dvalue; or no
  exploration module at all.
  reward[t] is deliberately *not* used as an exploration input: the chosen item would then receive
  two updates driven by the same signal that both enter the logit, so only their sum is
  identifiable.
- **Choice history:** `value_choice` and `bias_attention` separate (original), merged into one
  choice module (action[t], action[t−1], is_adjacent, is_opposite), or without `bias_attention`.
- **Exploration library:** all variants drop the state × input product terms in the exploration
  modules (`NO_STATE_PRODUCTS`). This only matters once SINDy is fitted.

## Variants

All variants keep the choice modules as in the original (`value_choice` + `bias_attention`, split):
that split is confirmed useful on dezfouli2019, so the two are only ever ablated as a pair.
Reward-module inputs and exploration are what the ablation steps vary.

| Variant | Reward inputs | Exploration input | Choice modules | Modules | SINDy terms |
|---|---|---|---|---|---|
| original (`spice_eckstein2026.py`) | env + mean | split | both | 7 | 70 |
| `m01_expl_split` | env | split | both | 7 | 57 |
| `m02_expl_signed` | env | signed | both | 7 | 51 |
| `m06_no_exploration` | env | – (no module) | both | 5 | 41 |
| `m13_expl_split_no_env` | none | split | both | 6 | 44 |
| `m08_signed_no_env` | none | signed | both | 6 | 38 |
| `m15_expl_split_chosen_only` | none | split, chosen item only | both | 5 | 36 |
| `m16_expl_signed_chosen_only` | none | signed, chosen item only | both | 5 | 33 |
| `m14_no_exploration_no_env` | none | – (no module) | both | 4 | 28 |
| `m11_no_choice` | env | signed | none | 5 | 32 |
| `m12_reward_exploration_only` | none | signed | none | 4 | 19 |

SINDy terms = candidate terms allowed by the prior mask, summed over modules (polynomial degree 2).

## Ablation protocol

Reward inputs start at their simplest (no env, no mean) and are re-tested in step 2.
Each step trains the configurations rather than knocking modules out of a trained model, since a
retrained model can compensate for what it loses. One run per configuration with `--seed 42` and
`--ensemble 5`; the spread across the 5 members is the noise floor for the decision rule.

1. **Exploration**, as a 2x2 of input form x module structure, with
   `m14_no_exploration_no_env` as the floor:

   | | chosen + not-chosen | chosen only |
   |---|---|---|
   | split dvalue | `m13_expl_split_no_env` | `m15_expl_split_chosen_only` |
   | signed dvalue | `m08_signed_no_env` | `m16_expl_signed_chosen_only` |
2. **Reward inputs** on the step-1 winner: none vs env vs mean vs env + mean.
3. **Module ablation** on the step-2 winner: both choice modules (only as a pair),
   `value_reward_not_chosen`, and the exploration modules if step 1 kept them.
4. **Re-add check:** add each dropped component back to the final model individually, to catch
   components that only help in combination.
5. **Final fit:** full data, SINDy on, refit on, 10 stability runs, checking the per-member
   equation likelihood that motivated this whole comparison.

## Running

Quick RNN-only comparison (no SINDy, no refit, prototyping subset of 50 participants × 100 trials):

```bash
for variant in m01_expl_split m02_expl_signed m06_no_exploration \
               m07_signed_merged m08_signed_no_env m09_signed_no_bias m10_minimal \
               m11_no_choice m12_reward_exploration_only; do
  python weinhardt2026/run.py \
    --module studies.eckstein2026.spice_eckstein2026_${variant} \
    --data weinhardt2026/studies/eckstein2026/data/eckstein2026.csv \
    --model weinhardt2026/studies/eckstein2026/params/variants/spice_eckstein2026_${variant}.pkl \
    --test_blocks 3 --prototyping --sindy_weight 0 --sindy_skip_refit --ensemble 1 --seed 42
done
```

Include the original (`--module studies.eckstein2026.spice_eckstein2026`) with the same flags as
the reference. `--seed` fixes weight initialization and training, so variants are compared on the
same initialization instead of through repeated runs; a run is then exactly reproducible. The quick runs compare module structure, inputs and chaining via the RNN's
validation loss only; the library choice (no state × input products) and equation stability need
full runs with SINDy and the refit.
