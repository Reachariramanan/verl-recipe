# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
DVPO Core Algorithms - Distributional Value-based Policy Optimization
Implements all loss components and distributional critic functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalCritic(nn.Module):
    """
    Distributional Critic with Multi-Head Quantile Ensemble
    Predicts full return distribution instead of scalar values
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
        n_heads: int = 3,
        n_quantiles: int = 200
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_quantiles = n_quantiles

        # Shared state encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Independent quantile heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_quantiles)
            for _ in range(n_heads)
        ])

        # Fixed quantile levels τ̂_j = j / M
        self.register_buffer(
            "taus",
            torch.linspace(1 / n_quantiles, 1.0, n_quantiles)
        )

    def forward(self, state):
        """
        Returns:
            quantiles: [B, N_heads, M]
        """
        h = self.encoder(state)
        quantiles = torch.stack(
            [head(h) for head in self.heads],
            dim=1
        )
        return quantiles

    def ensemble(self, state):
        """
        Mean over heads → robust baseline
        Returns:
            [B, M]
        """
        return self.forward(state).mean(dim=1)


def quantile_huber_loss(pred, target, taus, delta=1.0):
    """
    Quantile Huber Regression Loss (Equation 3)
    Central alignment of distributions with stable gradients

    Args:
        pred, target: [B, M]
        taus: [M]
        delta: Huber loss threshold
    """
    u = target.unsqueeze(-2) - pred.unsqueeze(-1)
    abs_u = torch.abs(u)

    huber = torch.where(
        abs_u <= delta,
        0.5 * u ** 2,
        delta * (abs_u - 0.5 * delta)
    )

    indicator = (u < 0).float()
    loss = torch.abs(taus.unsqueeze(-1) - indicator) * huber

    return loss.mean()


def risk_weighted_loss(pred, target, taus, gamma):
    """
    Risk-Weighted Quantiles (Equation 4)
    Emphasizes lower quantiles for noise suppression

    Args:
        pred, target: [B, M]
        taus: [M]
        gamma: Risk weighting parameter
    """
    weights = torch.exp(-gamma * (1 - taus))
    return (weights * quantile_huber_loss(pred, target, taus)).mean()


def cvar_loss(pred, target, alpha):
    """
    CVaR (Conditional Value at Risk) Loss (Equation 5)
    Lower-tail robustness for handling bad reward spikes

    Args:
        pred, target: [B, M]
        alpha: CVaR quantile level (e.g., 0.1 for 10% worst cases)
    """
    k = max(1, int(alpha * pred.size(-1)))
    return F.mse_loss(
        pred[:, :k].mean(dim=-1),
        target[:, :k].mean(dim=-1)
    )


def upper_gain_loss(pred, target, beta):
    """
    Upper-Tail Gain Loss (Equation 6)
    Preserves optimism in upper quantiles

    Args:
        pred, target: [B, M]
        beta: Upper tail quantile level (e.g., 0.1 for top 10%)
    """
    k = max(1, int(beta * pred.size(-1)))
    return F.mse_loss(
        pred[:, -k:].mean(dim=-1),
        target[:, -k:].mean(dim=-1)
    )


def mean_shift_loss(pred, target):
    """
    Mean-Shift Penalization (Equation 7)
    Prevents excessive pessimism in value estimates

    Args:
        pred, target: [B, M]
    """
    return F.relu(
        target.mean(dim=-1) - pred.mean(dim=-1)
    ).mean()


def tail_shape_loss(pred, target, alpha, beta):
    """
    Tail Shape Regularization (Equation 8)
    Asymmetric variance control

    Args:
        pred, target: [B, M]
        alpha: Lower tail fraction
        beta: Upper tail fraction
    """
    k_low = max(1, int(alpha * pred.size(-1)))
    k_high = max(1, int(beta * pred.size(-1)))

    low_var_pred = pred[:, :k_low].var(dim=-1)
    low_var_tgt = target[:, :k_low].var(dim=-1)

    high_var_pred = pred[:, -k_high:].var(dim=-1)
    high_var_tgt = target[:, -k_high:].var(dim=-1)

    return (
        F.relu(low_var_pred - low_var_tgt)
        + F.relu(high_var_tgt - high_var_pred)
    ).mean()


def tail_curvature_loss(pred, alpha, beta):
    """
    Tail Curvature Regularization (Equation 9)
    Shapes geometry of distribution tails

    Args:
        pred: [B, M]
        alpha: Lower tail fraction
        beta: Upper tail fraction
    """
    second_diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]

    k_low = max(1, int(alpha * second_diff.size(-1)))
    k_high = max(1, int(beta * second_diff.size(-1)))

    low_curv = second_diff[:, :k_low].mean(dim=-1)
    high_curv = second_diff[:, -k_high:].mean(dim=-1)

    return (
        F.relu(low_curv)
        + F.relu(-high_curv)
    ).mean()


def consistency_loss(quantiles):
    """
    Multi-Head Consistency Loss (Equation 10)
    Stabilizes ensemble predictions

    Args:
        quantiles: [B, N_heads, M]
    """
    B, N, M = quantiles.shape
    loss = 0.0
    pairs = 0

    for i in range(N):
        for j in range(i + 1, N):
            loss += (quantiles[:, i] - quantiles[:, j]).pow(2).sum(dim=-1).mean()
            pairs += 1

    return loss / pairs


def dvpo_critic_loss(
    critic,
    states,
    target_quantiles,
    weights,
    alpha=0.1,
    beta=0.1
):
    """
    Final DVPO Critic Loss (Equation 11)
    Combines all loss components with ablation switches

    Args:
        critic: DistributionalCritic model
        states: [B, state_dim]
        target_quantiles: [B, M] target quantile values
        weights: Dict with loss component weights
        alpha, beta: Tail parameters
    """
    all_q = critic(states)              # [B, N, M]
    q = all_q.mean(dim=1)               # ensemble [B, M]
    taus = critic.taus.to(states.device)

    loss = quantile_huber_loss(q, target_quantiles, taus)

    # Ablation switches for research
    if not weights.get("ablate_risk", False):
        loss += weights.get("risk", 1.0) * risk_weighted_loss(q, target_quantiles, taus, gamma=1.0)

    if not weights.get("ablate_cvar", False):
        loss += weights.get("cvar", 1.0) * cvar_loss(q, target_quantiles, alpha)

    if not weights.get("ablate_gain", False):
        loss += weights.get("gain", 1.0) * upper_gain_loss(q, target_quantiles, beta)

    if not weights.get("ablate_shift", False):
        loss += weights.get("shift", 1.0) * mean_shift_loss(q, target_quantiles)

    if not weights.get("ablate_shape", False):
        loss += weights.get("shape", 1.0) * tail_shape_loss(q, target_quantiles, alpha, beta)

    if not weights.get("ablate_curv", False):
        loss += weights.get("curv", 1.0) * tail_curvature_loss(q, alpha, beta)

    if not weights.get("ablate_consist", False):
        loss += weights.get("consist", 1.0) * consistency_loss(all_q)

    return loss


def distributional_gae(
    rewards,
    values,
    next_values,
    gamma=0.99,
    lam=0.95
):
    """
    Distributional GAE (Equation 2)
    Lifts standard GAE to work with quantile distributions

    Args:
        rewards: [T, B]
        values: [T, B, M] quantile values
        next_values: [T, B, M] next quantile values
        gamma: Discount factor
        lam: GAE lambda

    Returns:
        returns: [T, B, M] target quantile returns
        scalar_adv: [T, B] scalar advantages for actor
    """
    T = rewards.size(0)
    advantages = torch.zeros_like(values)

    gae = torch.zeros_like(values[0])

    for t in reversed(range(T)):
        delta = rewards[t].unsqueeze(-1) \
                + gamma * next_values[t] \
                - values[t]

        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = values + advantages
    scalar_adv = advantages.mean(dim=-1)

    return returns, scalar_adv
