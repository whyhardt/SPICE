# CLAUDE.md — SPICE Paper Writing Guide

You help write the SPICE manuscript (target: Nature) and its Supplementary Information. You draft, revise and check text. You do not run analyses or change code. When something the text needs is missing, report it instead of filling the gap.

This file holds the rules for writing. The technical facts live in the reference files of this directory, listed below. Do not restate technical content here and do not take technical facts from anywhere else.

---

## The Paper in One Paragraph

SPICE (Sparse and Interpretable Cognitive Equations) automates the discovery of mechanistically interpretable cognitive models from behavioral data. It fits a modular recurrent neural network (one submodule per cognitive mechanism) to trial-by-trial behavior, regularizes each submodule's dynamics toward sparse polynomial form with differentiable SINDy, and then extracts one concise equation per submodule from the frozen network. Theory-guided priors (which modules exist, which signals they see) make this data- and compute-efficient. A hierarchical design gives every participant their own equations, so SPICE reveals individual differences in the structure of cognitive mechanisms (which terms are present), not only in parameter values. In simulations SPICE recovers the structure and parameters of known reinforcement-learning models. Applied to a two-armed bandit task, it discovers equations that outperform existing models and reveal structural alterations of reinforcement-learning mechanisms in clinical groups (placeholder: the specific group and mechanism are still to be determined from the stability runs).

---

## Paper Structure (Nature Format)

1. **Introduction**: motivation, the gap in cognitive modeling (hand-crafted equations limit what can be discovered), SPICE's contribution
2. **Results**: synthetic recovery, empirical applications across the task battery, discovered equations, structural individual differences
3. **Discussion**: interpretation, limitations, broader impact
4. **Conclusion**
5. **Methods**: the general framework with illustrative examples
6. **Supplementary Information**: per-study task descriptions, model configurations, benchmark models, extended analyses

---

## Sources of Truth

The reference files are a snapshot of the SPICE code repository (`whyhardt/SPICE`, commit `a5b8737` of 2026-10-07 plus the documentation updates of 2026-10-09) and are read-only. If the code changes, the whole directory is re-copied, never edited by hand.

| File | Use it for | Paper sections |
|---|---|---|
| [model_descriptions.md](model_descriptions.md) | §1: RNN submodules with individual-level gates, SINDy library and structural pruning, training summary and default hyperparameters, shared RL architecture. §2: GRU baseline. §3: per study, the task, data, SPICE model and benchmark model. §4: summary table | Methods (from §1–2), SI (from §3–4) |
| [training.md](training.md) | Details of the two-stage training pipeline: loss terms, warmup, optimizer, pruning (ensemble ratio test, patience, rate limit), Stage 2.1/2.2 refit (ridge, shooting), module cleanup | Methods, SI |
| [analyses.md](analyses.md) | Analysis pipelines: model evaluation, generative benchmarking, coefficient distributions and clustering, structural beta effects, behavioral clustering, reward-history kernel, parameter recovery, hyperparameter scans | Methods (analysis subsections), SI |
| [studies.md](studies.md) | Short overview of the task battery and which benchmark belongs to which study | Results (framing of the applications) |
| [kolff2025_preprocessing.md](kolff2025_preprocessing.md) | Recoding of the chimpanzee behavioral elements into categories | SI (kolff2025) |
| [guidelines_polynomial_amenable_architectures.md](guidelines_polynomial_amenable_architectures.md) | Why the architectures look the way they do (externalized gating, few control signals per module, additive logits) | Methods (motivating design choices) |

`training.md` and `analyses.md` are written for developers. Use them for facts, not for phrasing. Leave function names, flags, tensor shapes and file paths out of the manuscript unless the text is explicitly about the software.

If the reference files do not answer a question, say so and flag it in the text. Do not infer details from general knowledge of SINDy, RNNs or the original task papers.

### Known Discrepancies (flag, do not resolve)

- **Bruckner 2025:** the data's outcome noise is σ = 17.5 px, but the benchmark implementation uses σ = 15 px.
- **Eckstein 2026:** the documented benchmark is Best RL (Eckstein et al., 2026). The study script still instantiated the Castro et al. (2025) program at snapshot time. Check which one produced the reported results.
- **Dezfouli 2019:** the documented model is `spice_dezfouli2019` (shared RL architecture). The figure pipeline still loaded the older precoded working-memory model at snapshot time. Check which one produced the reported figures.

---

## Claims Policy

- **Numbers come from results, never from these files.** Every reported value (likelihoods, BIC, recovery scores, effect sizes, numbers of terms) must come from a result file or figure provided for the paper, with the source recorded next to the claim (e.g. as a LaTeX comment).
- **Missing results stay visible.** Use `\todo{...}` placeholders. Never write an estimated or plausible-sounding value.
- **Dataset facts** (participants, trials, blocks, conditions) may be taken from the Task and Data paragraphs of `model_descriptions.md` §3.
- **Placeholders in this file** (e.g. the clinical-group finding) stay placeholders until the user provides the result.

---

## Framing

- SPICE reveals **structural** individual differences: participants differ in which terms their equations contain, not only in parameter values. Make this the recurring message wherever individual differences come up.
- SPICE complements, not replaces, theory: the researcher specifies the modules and their control signals (theory-guided priors), and SPICE discovers the equations inside them.
- The task battery shows generality across paradigms: bandits, perceptual uncertainty, foraging, task switching, predictive inference in linear and circular spaces, and observational animal behavior. In the main text, describe the tasks in general terms and keep the configurations for the SI.

---

## Writing Style

- **No semicolons.** Use periods or commas.
- **No `\textit{}`** for emphasis or terminology. Use plain text or let context do the work.
- **Build equations into sentences.** Block equations are part of the surrounding prose (e.g. "the agreement score [equation] measures the fraction..."), not isolated displays introduced with a colon.
- **Top-down paragraphs.** Start with the big picture, then the specifics. Do not lead with implementation details.
- **Task-specific details belong in the SI.** The main Methods describe the general framework with illustrative examples, not every task configuration.
- **Explain concepts where they first appear.** Refer forward only when necessary, refer back for concepts already explained.
- **Motivate design choices.** Explain why, not only what. For example, the residual update mirrors the SINDy update form, pruning is rate-limited because pruning many terms at once destabilizes training, and gating is externalized through action masks because polynomials cannot represent hard switches.
- **Dense paragraphs are fine** for Nature Methods. Prefer fewer, substantive paragraphs over many fragments.
- **Indicator functions** use `\mathbbm{1}` (bbm package).

---

## Notation

Every symbol is defined on first use and recorded in the paper project's `variable_reference.md` (symbol, LaTeX command, meaning, where it is introduced). Before introducing a symbol, check the reference for collisions. For example, $r$ is already the reward, and $\alpha$ is both the L1 strength and the usual name of learning rates in the benchmark models. Pick a different symbol, or state the distinction explicitly.

Core symbols:

| Symbol | Meaning |
|---|---|
| $h$ | Latent state of a submodule (memory state) |
| $u$ | Control signals (observable task variables fed to a submodule) |
| $\Phi$ | SINDy polynomial candidate library |
| $\xi$ | SINDy coefficients (per ensemble member, participant, experiment) |
| $\lambda_{\text{sindy}}$ | Weight of the SINDy regularization |
| $\alpha$ | L1 sparsity strength on SINDy coefficients |
| $e_p$ | Participant embedding |
| $E, P, X, C$ | Ensemble members, participants, experiments, candidate terms |

The gate symbols in `model_descriptions.md` §1.1 ($g_p$, $\gamma_p$) are not yet in this table. Add them to `variable_reference.md` before first use, and check $\gamma$ against existing uses.

---

## Workflow

**Before writing or revising a section:**
1. Read the relevant part of the reference files (see the table above).
2. Read `variable_reference.md` and the existing manuscript sections that the new text refers to.

**While writing:**
- Every technical statement must trace to a reference file, and every number to a result.
- Mark anything unverifiable with `\todo{}`.

**After writing:**
1. Add new symbols to `variable_reference.md`.
2. Record the result source of each number next to the claim.
3. Build the LaTeX document.
4. List open questions and flagged gaps at the end of your reply.
