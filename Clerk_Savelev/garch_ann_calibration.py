
# garch_ann_calibration.py
# ANN-based calibration of GARCH(1,1) parameters following
# "A machine learning search for optimal GARCH parameters" (De Clerk & Savel'ev, 2022).
#
# Features:
# - Synthetic dataset generation using analytical moment/autocovariance formulas
# - PyTorch MLP that predicts alpha1 from inputs (moments or autocovariance)
# - Post-processing to recover beta1 and alpha0 from empirical Γ4 and σ^2
# - Early stopping
# - Utilities to compute sample moments and autocovariances from a series
#
# Note: This file avoids any internet access and runs locally with PyTorch installed.

import math
import random
from dataclasses import dataclass
from typing import Tuple, Literal, Optional, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
except Exception as e:
    raise RuntimeError("This module requires PyTorch. Please install torch before use.") from e


# -----------------------------
# Analytical formulas (GARCH(1,1), Gaussian innovations)
# -----------------------------

def sigma2_from_params(alpha0: float, alpha1: float, beta1: float) -> float:
    """Unconditional variance σ^2 = α0 / (1 - α1 - β1)."""
    denom = 1.0 - alpha1 - beta1
    if denom <= 0:
        return np.nan
    return alpha0 / denom


def gamma4_from_params(alpha1: float, beta1: float) -> float:
    """Kurtosis Γ4 = E[x^4]/E[x^2]^2 for GARCH(1,1) with Gaussian innovations.
    Γ4 = 3 + 6 α1^2 / (1 - 3 α1^2 - 2 α1 β1 - β1^2)
    """
    denom = 1.0 - 3.0 * alpha1**2 - 2.0 * alpha1 * beta1 - beta1**2
    if denom <= 0:
        return np.nan
    return 3.0 + 6.0 * alpha1**2 / denom


def gamma6_from_params(alpha1: float, beta1: float) -> float:
    """Sixth standardized moment Γ6 for GARCH(1,1) with Gaussian innovations.
    Expression from the paper (rewritten for code).
    """
    a = alpha1
    b = beta1
    denom4 = 1.0 - 3.0*a*a - 2.0*a*b - b*b
    if denom4 <= 0:
        return np.nan
    num_part = 15.0 * (1.0 - a - b) ** 3
    t1 = 1.0 + 3.0 * (a + b) / (1.0 - a - b)
    t2 = 1.0 + 2.0 * (a + b) / (1.0 - a - b)
    inner = (b*b + 2.0*a*b + 3.0*a*a)
    big_bracket = t1 + 3.0 * t2 * inner / denom4
    denom6 = 1.0 - 15.0*a**3 - 9.0*a*a*b - 3.0*a*b*b - b**3
    if denom6 <= 0:
        return np.nan
    return (num_part * big_bracket) / denom6


def gammahat_from_params(alpha1: float, beta1: float, n: int) -> float:
    """Normalised autocovariance of x_t^2 with lag n (γ̂_n), eq. (6) in paper:
       γ̂_n = [2 α1 (1 - α1 β1 - β1^2) / (1 - 3 α1^2 - 2 α1 β1 - β1^2)] * (α1 + β1)^(n-1)
    """
    a = alpha1
    b = beta1
    denom = 1.0 - 3.0*a*a - 2.0*a*b - b*b
    if denom <= 0:
        return np.nan
    factor = 2.0 * a * (1.0 - a*b - b*b) / denom
    return factor * (a + b) ** (n - 1)


# -----------------------------
# Sampling helpers
# -----------------------------

def sample_garch_params(
    require_gamma6: bool = True,
    max_tries: int = 10000,
) -> Tuple[float, float, float]:
    """Sample (alpha0, alpha1, beta1) satisfying stationarity and (optionally) Γ6 finiteness.
    α0 is drawn log-uniform in [1e-6, 1e-3], α1, β1 in feasible region α1>=0, β1>=0, α1+β1<1,
    plus positivity of denominators used in formulas.
    """
    for _ in range(max_tries):
        a1 = random.random() * 0.3  # keep modest to help denominators stay positive
        b1 = random.random() * (0.999 - a1)
        a0 = 10 ** random.uniform(-6, -3)

        # stationarity
        if a1 + b1 >= 1.0:
            continue
        # denominators
        if 1.0 - 3.0*a1*a1 - 2.0*a1*b1 - b1*b1 <= 0:
            continue
        if require_gamma6:
            if 1.0 - 15.0*a1**3 - 9.0*a1*a1*b1 - 3.0*a1*b1*b1 - b1**3 <= 0:
                continue

        return a0, a1, b1
    raise RuntimeError("Failed to sample feasible GARCH parameters; try relaxing constraints.")


# -----------------------------
# Dataset generation
# -----------------------------

@dataclass
class Sample:
    x: np.ndarray  # inputs
    y: float       # target alpha1


class MomentsDataset(Dataset):
    """Inputs: [σ^2, Γ4, Γ6]  -> target: α1"""
    def __init__(self, n: int):
        xs: List[np.ndarray] = []
        ys: List[float] = []
        for _ in range(n):
            a0, a1, b1 = sample_garch_params(require_gamma6=True)
            s2 = sigma2_from_params(a0, a1, b1)
            g4 = gamma4_from_params(a1, b1)
            g6 = gamma6_from_params(a1, b1)
            if any(map(lambda v: (not np.isfinite(v)) or np.isnan(v), [s2, g4, g6])):
                continue
            xs.append(np.array([s2, g4, g6], dtype=np.float32))
            ys.append(float(a1))
        self.X = torch.tensor(np.vstack(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class AutoCovDataset(Dataset):
    """Inputs: [σ^2, Γ4, γ̂_n]  -> target: α1"""
    def __init__(self, n: int, lag: int = 6):
        xs: List[np.ndarray] = []
        ys: List[float] = []
        for _ in range(n):
            a0, a1, b1 = sample_garch_params(require_gamma6=False)
            s2 = sigma2_from_params(a0, a1, b1)
            g4 = gamma4_from_params(a1, b1)
            ghat = gammahat_from_params(a1, b1, n=lag)
            if any(map(lambda v: (not np.isfinite(v)) or np.isnan(v), [s2, g4, ghat])):
                continue
            xs.append(np.array([s2, g4, ghat], dtype=np.float32))
            ys.append(float(a1))
        self.X = torch.tensor(np.vstack(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# -----------------------------
# Model
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Training / Early stopping
# -----------------------------

@dataclass
class TrainConfig:
    epochs: int = 5000
    batch_size: int = 1024
    lr: float = 1e-2
    patience: int = 50  # early stopping patience


@dataclass
class TrainResult:
    model: nn.Module
    best_val_loss: float
    history: List[tuple]  # (epoch, train_loss, val_loss)


def train_model(
    dataset: Dataset,
    cfg: TrainConfig = TrainConfig(),
    val_split: float = 0.2,
    device: Optional[str] = None,
) -> TrainResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError("Dataset too small for the requested validation split.")
    train_set, val_set = random_split(dataset, [n_train, n_val])

    in_dim = dataset[0][0].shape[-1]
    model = MLP(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append((epoch, train_loss, val_loss))

        # Early stopping
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_val_loss=best_val, history=history)


# -----------------------------
# Post-processing to recover beta1 and alpha0
# -----------------------------

def beta1_from_alpha1_gamma4(alpha1: float, gamma4_emp: float) -> float:
    """Equation (11) from the paper (rearranged):
       β1 = sqrt( 1 - 2 α1^2 - 6 α1^2 / (Γ4_emp - 3) ) - α1
       We clip the inside of the sqrt to [0, 1] range for robustness.
    """
    if gamma4_emp <= 3.0:
        # Kurtosis must exceed 3 under conditional heteroskedasticity; fall back safely.
        return max(0.0, 0.99 - alpha1)
    t = 1.0 - 2.0 * alpha1**2 - (6.0 * alpha1**2) / (gamma4_emp - 3.0)
    t = max(0.0, min(1.0, t))
    return max(0.0, math.sqrt(t) - alpha1)


def alpha0_from_sigma2(alpha1: float, beta1: float, sigma2_emp: float) -> float:
    """α0 = σ^2_emp * (1 - α1 - β1) (eq. 12)."""
    factor = 1.0 - alpha1 - beta1
    return max(0.0, sigma2_emp * factor)


# -----------------------------
# Empirical feature extraction (from a series of returns)
# -----------------------------

def sample_moments_and_gammahat(returns: np.ndarray, lag: int = 6) -> Tuple[float, float, float, float]:
    """Compute empirical σ^2, Γ4, Γ6, γ̂_lag from a return series.
    Γ4 = m4 / m2^2, Γ6 = m6 / m2^3, and γ̂_lag = cov(x^2_t, x^2_{t+lag}) / E[x^2]^2
    """
    x = np.asarray(returns, dtype=np.float64)
    m2 = float(np.mean(x**2))
    if m2 <= 0:
        raise ValueError("Non-positive second moment.")
    m4 = float(np.mean(x**4))
    m6 = float(np.mean(x**6))

    g4 = m4 / (m2**2)
    g6 = m6 / (m2**3)

    # autocovariance of squares
    x2 = x**2
    x2_mean = float(np.mean(x2))
    if lag >= len(x2):
        raise ValueError("Lag too large for the given series length.")
    cov = float(np.mean((x2[:-lag] - x2_mean) * (x2[lag:] - x2_mean)))
    ghat = cov / (m2**2)
    return m2, g4, g6, ghat


# -----------------------------
# Public API
# -----------------------------

Variant = Literal["moments", "acov"]

@dataclass
class CalibratedParams:
    alpha0: float
    alpha1: float
    beta1: float


@dataclass
class Calibrator:
    variant: Variant = "acov"
    lag: int = 6
    model: Optional[nn.Module] = None
    device: Optional[str] = None

    def fit(self, n_samples: int = 150_000, cfg: TrainConfig = TrainConfig()) -> TrainResult:
        """Train the internal ANN on synthetic data mapping features -> α1."""
        if self.variant == "moments":
            dataset = MomentsDataset(n_samples)
        elif self.variant == "acov":
            dataset = AutoCovDataset(n_samples, lag=self.lag)
        else:
            raise ValueError("Unknown variant")

        result = train_model(dataset, cfg=cfg, device=self.device)
        self.model = result.model
        return result

    def features_from_params(self, alpha0: float, alpha1: float, beta1: float) -> np.ndarray:
        """For testing: compute features from true params."""
        if self.variant == "moments":
            s2 = sigma2_from_params(alpha0, alpha1, beta1)
            g4 = gamma4_from_params(alpha1, beta1)
            g6 = gamma6_from_params(alpha1, beta1)
            return np.array([s2, g4, g6], dtype=np.float32)
        else:
            s2 = sigma2_from_params(alpha0, alpha1, beta1)
            g4 = gamma4_from_params(alpha1, beta1)
            ghat = gammahat_from_params(alpha1, beta1, n=self.lag)
            return np.array([s2, g4, ghat], dtype=np.float32)

    def predict_alpha1(self, features: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            pred = self.model.to(dev)(x.to(dev)).cpu().numpy().ravel()[0]
        # Bound into [0,1) for safety
        return float(max(0.0, min(0.999, pred)))

    def calibrate_from_empirical(
        self,
        returns: np.ndarray,
        use_gamma6_if_moments: bool = True,
    ) -> CalibratedParams:
        """Calibrate (α0, α1, β1) from an empirical series of returns.

        Steps:
        - Compute empirical features (σ^2, Γ4, Γ6 or γ̂_lag)
        - Predict α1 via ANN
        - Compute β1 from Γ4 and α1 (eq. 11)
        - Compute α0 from σ^2 (eq. 12)
        """
        s2, g4, g6, ghat = sample_moments_and_gammahat(returns, lag=self.lag)
        if self.variant == "moments":
            feats = np.array([s2, g4, g6 if use_gamma6_if_moments else np.nan], dtype=np.float32)
            # If user wants to use Γ6, feats is [σ^2, Γ4, Γ6] as trained
            feats = np.array([s2, g4, g6], dtype=np.float32)
        else:
            feats = np.array([s2, g4, ghat], dtype=np.float32)

        a1 = self.predict_alpha1(feats)
        b1 = beta1_from_alpha1_gamma4(a1, g4)
        a0 = alpha0_from_sigma2(a1, b1, s2)
        return CalibratedParams(alpha0=a0, alpha1=a1, beta1=b1)


# -----------------------------
# Minimal demo (synthetic run)
# -----------------------------

def _demo_train_and_test():
    """Quick smoke test: trains briefly and checks prediction on synthetic point."""
    print("== Quick demo: training on synthetic data (acov, lag=6) ==")
    calib = Calibrator(variant="acov", lag=6)
    # Small dataset and small epochs for a fast smoke test; increase for real training
    result = calib.fit(n_samples=30_000, cfg=TrainConfig(epochs=500, batch_size=1024, lr=1e-2, patience=20))
    print(f"Best val loss: {result.best_val_loss:.6f} after {len(result.history)} epochs")

    # Test on a random feasible parameter triple
    a0, a1_true, b1_true = sample_garch_params(require_gamma6=False)
    feats = calib.features_from_params(a0, a1_true, b1_true)
    a1_pred = calib.predict_alpha1(feats)
    print(f"True α1={a1_true:.4f} | Pred α1={a1_pred:.4f}")

    # Recover β1 and α0 from synthetic Γ4 and σ^2 (as if 'empirical')
    if calib.variant == "acov":
        s2 = feats[0]
        g4 = feats[1]
        b1_recovered = beta1_from_alpha1_gamma4(a1_pred, g4)
        a0_recovered = alpha0_from_sigma2(a1_pred, b1_recovered, s2)
        print(f"Recovered β1≈{b1_recovered:.4f}, α0≈{a0_recovered:.2e} (True β1={b1_true:.4f}, α0={a0:.2e})")


if __name__ == "__main__":
    _demo_train_and_test()
