# DVPO (Distributional Value-based Policy Optimization)

This recipe implements DVPO, a distributional reinforcement learning algorithm that learns full return distributions instead of scalar values for improved robustness and generalization.

## Overview

DVPO = PPO + Distributional Critic + Asymmetric Risk Control

**Key Features:**
- **Distributional Critic**: Multi-head quantile ensemble that predicts full return distributions
- **Asymmetric Risk Control**: Different treatment of lower-tail robustness vs upper-tail optimism
- **Research-Friendly**: Built-in ablation switches for all loss components
- **Production-Ready**: Full verl integration with FSDP/Megatron support

## Architecture

```
Policy (Actor)         Critic (Distributional)
───────────────        ─────────────────────────
πθ(a|s)      ───►      N heads × M quantiles
                           │
                           ▼
                 Value distribution Z(s,a)
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      Lower-tail       Central fit     Upper-tail
      robustness       (QR loss)       generalization
```

## Loss Components

DVPO combines 8 specialized loss functions:

1. **Quantile Huber Regression** (Equation 3): Central alignment with stable gradients
2. **Risk-Weighted Quantiles** (Equation 4): Emphasizes lower quantiles for noise suppression
3. **CVaR** (Equation 5): Lower-tail robustness for bad reward spikes
4. **Upper-Tail Gain** (Equation 6): Preserves optimism in upper quantiles
5. **Mean-Shift Penalization** (Equation 7): Prevents excessive pessimism
6. **Tail Shape Regularization** (Equation 8): Asymmetric variance control
7. **Tail Curvature** (Equation 9): Shapes geometry of distribution tails
8. **Multi-Head Consistency** (Equation 10): Stabilizes ensemble predictions

## Quick Start

### Basic Training

```bash
# Set your paths
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="path/to/your/training/data"

# Run training
bash verl-recipe/dvpo/run_dvpo_qwen_7b.sh
```

### Ablation Studies

```bash
# Run ablation tests to see component contributions
bash verl-recipe/dvpo/run_dvpo_ablation_test.sh
```

## Configuration

### Key Parameters

```yaml
algorithm:
  # DVPO-specific parameters
  dvpo_alpha: 0.1      # CVaR/tail shape lower quantile
  dvpo_beta: 0.1       # Upper gain/tail shape upper quantile
  n_quantiles: 200     # Number of quantiles per distribution
  n_heads: 3           # Number of ensemble heads
  critic_hidden_dim: 1024  # Critic network size

  # Loss weights with ablation switches
  dvpo_loss_weights:
    risk: 1.0          # Risk-weighted quantiles weight
    cvar: 1.0          # CVaR weight
    gain: 1.0          # Upper-tail gain weight
    shift: 1.0         # Mean shift penalty weight
    shape: 1.0         # Tail shape regularization weight
    curv: 1.0          # Tail curvature weight
    consist: 1.0       # Multi-head consistency weight

    # Ablation switches (set to true to disable)
    ablate_risk: false
    ablate_cvar: false
    ablate_gain: false    # Most important - expect big performance drop
    ablate_shift: false
    ablate_shape: false
    ablate_curv: false
    ablate_consist: false
```

### Ablation Guidelines

| Component | Ablation Switch | Expected Impact |
|-----------|----------------|-----------------|
| Upper-Tail Gain | `ablate_gain: true` | **Largest impact** - generalization collapse |
| CVaR | `ablate_cvar: true` | Unstable lower tail, bad reward spikes |
| Multi-Head Consistency | `ablate_consist: true` | Head disagreement, unstable training |
| Risk Weighting | `ablate_risk: false` | Less noise robustness |
| Tail Shape | `ablate_shape: false` | Distribution drift |
| Mean Shift | `ablate_shift: false` | Value underestimation |
| Curvature | `ablate_curv: false` | Jagged quantiles |

## Files Structure

```
verl-recipe/dvpo/
├── __init__.py                    # Package init
├── main_dvpo.py                   # Main entry point
├── dvpo_ray_trainer.py           # Modified PPO trainer
├── dvpo_core_algos.py            # All DVPO loss functions
├── dvpo_critic.py                # DataParallel DVPO critic
├── dvpo_fsdp_workers.py          # Custom FSDP workers
├── config/
│   ├── dvpo_trainer.yaml         # FSDP configuration
│   └── dvpo_megatron_trainer.yaml # Megatron configuration
├── run_dvpo_qwen_7b.sh           # Example training script
├── run_dvpo_ablation_test.sh     # Ablation study script
└── README.md                     # This file
```

## Implementation Details

### Distributional Critic

The `DistributionalCritic` class implements:
- Multi-head quantile ensemble for uncertainty quantification
- Fixed quantile levels τ̂_j = j/M for reproducible results
- Shared encoder with independent quantile heads

### Distributional GAE - MATHEMATICAL CORRECTIONS APPLIED

**Previously Incorrect Implementation:**
- ❌ Temporal decoupling: didn't preserve `[T+1, B, M]` structure
- ❌ Missing bootstrap masking: no terminal value masking
- ❌ Invalid GAE computation: wrong temporal dependencies
- ❌ Terminal state confusion: undefined `values[T+1]`

**Fixed Implementation:**
- ✅ **Temporal Alignment**: Values must be `[T+1, B, M]` including terminal state
- ✅ **Bootstrap Masking**: Terminal values set to 0 to prevent invalid bootstrapping
- ✅ **Proper GAE Computation**: Backward pass preserves causal dependencies
- ✅ **Terminal State Handling**: Episodes ending mid-sequence properly masked

**Mathematical Correctness:**
```python
# CORRECT: Terminal masking prevents invalid bootstrapping
next_values_masked = values[1:] * (~dones.unsqueeze(-1))  # Terminal value = 0
delta = rewards.unsqueeze(-1) + gamma * next_values_masked - values[:-1]  # TD error

# Backward GAE pass preserving temporal dependencies
gae = torch.zeros_like(values_t[0])  # [B, M]
for t in reversed(range(T)):
    gae = delta[t] + gamma * lam * (~dones[t].unsqueeze(-1)) * gae
    advantages[t] = gae
    returns[t] = values_t[t] + gae

scalar_adv = advantages.mean(dim=-1)  # For actor compatibility
```

### Ablation Switches

All loss components can be individually disabled via config flags:
```yaml
algorithm.dvpo_loss_weights.ablate_gain: true  # Disable upper-tail gain
```

This enables controlled experiments to measure each component's contribution.

## Performance Expectations

- **In-distribution**: Similar to PPO with slight overhead
- **Out-of-distribution**: Significant improvement due to distributional awareness
- **Robustness**: Better handling of reward spikes and noise
- **Training stability**: Multi-head ensemble reduces variance

## Troubleshooting

### Common Issues

1. **Training instability**: Reduce learning rates or increase `n_heads`
2. **Poor generalization**: Check that `ablate_gain` is `false`
3. **High memory usage**: Reduce `n_quantiles` or increase micro-batch size
4. **Slow convergence**: Increase critic learning rate relative to actor

### Monitoring

Key metrics to track:
- `critic/dvpo_loss`: Overall DVPO loss
- `critic/quantile_mean`: Mean predicted quantile values
- Individual component losses when debugging

## Citation

If you use DVPO, please cite the original paper and this implementation:

```
@article{dvpo2024,
  title={Distributional Value-based Policy Optimization},
  author={...},
  journal={...},
  year={2024}
}
```

## Contributing

- Report issues on the verl repository
- Submit ablation study results
- Propose improvements to loss formulations
