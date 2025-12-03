# Momentum-Aware Noise Injection for SINDy Coefficients

## Overview

This feature adds exploration noise to SINDy coefficients during training to help escape local minima and rediscover useful terms that may have been prematurely killed.

## How It Works

The noise injection is **momentum-aware** - it tracks recent gradient history to identify "stagnant" coefficients:

- **Stagnant coefficient**: Small magnitude (< 0.1) AND small recent gradient (< 0.01)
- **Stagnant coefficients receive 5x more noise** than active ones
- Only applies during training (not during evaluation)

### Why This Helps

Your SPICE optimization is non-stationary - the RNN's representations improve during training, which changes which SINDy terms are useful. A coefficient that looks useless at epoch 100 might become important at epoch 500 once the RNN learns better representations.

**Problem**: Once a coefficient is thresholded to zero, it's permanently dead (via `sindy_coefficients_presence` mask).

**Solution**: Noise injection keeps small coefficients "wiggling" around, giving them chances to grow if the behavioral gradient starts favoring them as the RNN improves.

## Usage

### Enable in SpiceEstimator

```python
estimator = SpiceEstimator(
    # ... other parameters ...
    sindy_noise_std=0.01,  # Enable noise injection (0 = disabled)
    # ... other parameters ...
)
```

### Recommended Settings

**For typical use**:
- `sindy_noise_std=0.01`: Moderate noise (1% of typical coefficient scale)
- Combined with patience-based thresholding: `sindy_cutoff_patience=200-300`

**For aggressive exploration** (if coefficients are stuck):
- `sindy_noise_std=0.02-0.05`: Higher noise
- Lower thresholding frequency: `sindy_threshold_frequency=5-10`

**To disable**:
- `sindy_noise_std=0.0` (default)

## Technical Details

### Stagnant Coefficient Detection

```python
coeff_abs = |coefficient|
recent_grad = mean(|grad|) over last 10 steps

is_stagnant = (coeff_abs < 0.1) AND (recent_grad < 0.01)
```

### Noise Scaling

```python
if is_stagnant:
    noise = randn() * noise_std * 5.0  # High noise for exploration
else:
    noise = randn() * noise_std * 0.5  # Low baseline noise
```

### Gradient History Tracking

- Automatically tracks gradient magnitudes over last 10 optimization steps
- No manual intervention needed - updates automatically during training
- Minimal memory overhead (~10 gradient tensors stored per module)

## When to Use

**Use noise injection when**:
- Higher-degree terms are stuck near zero but not dying
- Coefficients seem to be in local minima
- You suspect useful terms were prematurely killed
- You want more exploration before convergence

**Don't use (or use low noise) when**:
- Training is already working well
- You want faster convergence to current local minimum
- In stage 2 (second-stage SINDy fitting) - noise is automatically disabled in eval mode

## Performance Impact

- **Training speed**: Negligible (<1% overhead for gradient history tracking)
- **Convergence**: May slow initial convergence slightly but can find better solutions
- **Memory**: Minimal (~10 KB per module for gradient history)

## Comparison to Alternatives

| Method | Pros | Cons |
|--------|------|------|
| **Noise Injection** | Smooth, continuous, minimal code | Doesn't reactivate truly dead coefficients |
| Dropin | Can revive dead coefficients | Discrete jumps, training instability |
| Higher learning rate | Simple | Affects all parameters, not targeted |
| Grad normalization | Theoretically clean | Complex, interferes with degree weighting |

## Example

```python
from spice.estimator import SpiceEstimator
from spice.precoded import choice

estimator = SpiceEstimator(
    rnn_class=choice.SpiceModel,
    spice_config=choice.CONFIG,
    n_participants=256,
    n_actions=2,

    # Training
    epochs=1000,
    learning_rate=0.01,

    # SINDy
    sindy_weight=0.1,
    sindy_alpha=0.001,
    sindy_threshold=0.05,
    sindy_cutoff_patience=200,

    # Noise injection - ENABLE HERE
    sindy_noise_std=0.01,  # 1% noise

    device=torch.device('cuda'),
)

estimator.fit(dataset_train.xs, dataset_train.ys)
```

## Tuning Guidelines

1. **Start with default**: `sindy_noise_std=0.01`
2. **If coefficients are too stable**: Increase to 0.02-0.05
3. **If training is too noisy**: Decrease to 0.005
4. **If getting NaNs**: Noise is too high, reduce it
5. **Check stagnation**: Print gradient history to see if detection is working

## Disable Noise for Stage 2

Noise is automatically disabled during evaluation (stage 2 SINDy fitting) because the model is set to `.eval()` mode. No manual intervention needed.
