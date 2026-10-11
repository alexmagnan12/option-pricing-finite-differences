# -*- coding: utf-8 -*-
"""
================================================================================
  OPTION PRICING VIA FINITE DIFFERENCE METHODS
  Solving the Black-Scholes PDE using Explicit Finite Differences
================================================================================

  Author: Alexandre Magnan
  Context: Numerical Methods Project — Engineering School (ESTP Paris)

  This project demonstrates the resolution of the Black-Scholes PDE for option
  pricing using explicit finite difference schemes. The methodology mirrors
  classical PDE solvers used in geotechnical engineering (consolidation equation),
  adapted here to quantitative finance applications.

  The Black-Scholes PDE:
      ∂V/∂t + 0.5 * σ² * S² * ∂²V/∂S² + r * S * ∂V/∂S - r * V = 0

  After the change of variable x = ln(S), this becomes a convection-diffusion
  equation solved on a uniform grid — structurally identical to the heat equation
  ∂u/∂t = D * ∂²u/∂x² solved in engineering applications.

  Cases studied:
      1. European Call/Put — constant volatility (baseline)
      2. European Call — time-dependent volatility (term structure)
      3. European Call — local volatility surface σ(S, t)
      4. Up-and-Out Barrier Option — sudden change in boundary conditions
      5. Two-asset Basket Option — 2D PDE extension
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICAL BLACK-SCHOLES (for benchmarking)
# ══════════════════════════════════════════════════════════════════════════════

def black_scholes_call(S, K, T, r, sigma):
    """Closed-form Black-Scholes price for a European Call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Closed-form Black-Scholes price for a European Put."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ══════════════════════════════════════════════════════════════════════════════
#  CASE 1: EUROPEAN CALL/PUT — CONSTANT VOLATILITY
#  (Analogous to: homogeneous soil, constant load, constant Cv)
# ══════════════════════════════════════════════════════════════════════════════

def price_european_option_case1(option_type='call'):
    """
    Explicit FDM on the Black-Scholes PDE in log-space.

    After x = ln(S), the BS PDE becomes:
        ∂V/∂τ = 0.5*σ² * ∂²V/∂x² + (r - 0.5*σ²) * ∂V/∂x - r*V

    where τ = T - t (time to maturity, marching forward).

    This is solved with the explicit scheme:
        V_i^{k+1} = V_i^k + β*(V_{i-1}^k - 2*V_i^k + V_{i+1}^k)
                   + α*(V_{i+1}^k - V_{i-1}^k) - r*dτ*V_i^k

    Stability condition: β = σ²*dτ/(2*dx²) ≤ 0.5
    """
    # Market parameters
    S0 = 100.0       # Spot price
    K = 100.0        # Strike
    T = 1.0          # Maturity (years)
    r = 0.05         # Risk-free rate
    sigma = 0.20     # Volatility (constant)

    # Grid parameters
    Nx = 200         # Number of spatial steps
    x_min = np.log(S0) - 4 * sigma * np.sqrt(T)  # ~4σ below
    x_max = np.log(S0) + 4 * sigma * np.sqrt(T)  # ~4σ above
    dx = (x_max - x_min) / Nx

    # Time step from stability condition: β = 0.5*σ²*dτ/dx² ≤ 0.5
    dt = 0.4 * dx**2 / sigma**2  # conservative factor
    Nt = int(T / dt) + 1
    dt = T / Nt

    # Stability check
    beta = 0.5 * sigma**2 * dt / dx**2
    assert beta <= 0.5, f"Stability violated: β = {beta:.4f} > 0.5"

    # Spatial grid
    x = np.linspace(x_min, x_max, Nx + 1)
    S = np.exp(x)

    # Initialize: terminal payoff at τ=0 (i.e., t=T)
    if option_type == 'call':
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # FDM coefficients
    alpha = (r - 0.5 * sigma**2) * dt / (2 * dx)
    # beta already defined

    # Time-stepping (explicit scheme, forward in τ = backward in t)
    V_history = [V.copy()]
    for k in range(Nt):
        V_new = np.zeros_like(V)
        for i in range(1, Nx):
            V_new[i] = V[i] \
                + beta * (V[i-1] - 2*V[i] + V[i+1]) \
                + alpha * (V[i+1] - V[i-1]) \
                - r * dt * V[i]

        # Boundary conditions
        if option_type == 'call':
            V_new[0] = 0.0                                          # Deep OTM
            V_new[Nx] = S[Nx] - K * np.exp(-r * (k+1) * dt)        # Deep ITM
        else:
            V_new[0] = K * np.exp(-r * (k+1) * dt) - S[0]          # Deep ITM
            V_new[Nx] = 0.0                                         # Deep OTM

        V = V_new.copy()
        if k % (Nt // 8) == 0:
            V_history.append(V.copy())

    V_history.append(V.copy())

    # Extract price at S0
    idx_S0 = np.argmin(np.abs(S - S0))
    fdm_price = V[idx_S0]
    if option_type == 'call':
        bs_price = black_scholes_call(S0, K, T, r, sigma)
    else:
        bs_price = black_scholes_put(S0, K, T, r, sigma)

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Option price surface evolution
    for j, Vsnap in enumerate(V_history):
        tau = j * (T / len(V_history))
        axes[0].plot(S, Vsnap, alpha=0.6, label=f'τ={tau:.2f}y')
    axes[0].set_xlim(60, 140)
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Option Value V(S)')
    axes[0].set_title(f'European {option_type.capitalize()} — Price Evolution (FDM)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Right: Final price vs BS analytical
    axes[1].plot(S, V, 'b-', linewidth=2, label='FDM Price')
    if option_type == 'call':
        V_bs = np.array([black_scholes_call(s, K, T, r, sigma) for s in S])
    else:
        V_bs = np.array([black_scholes_put(s, K, T, r, sigma) for s in S])
    axes[1].plot(S, V_bs, 'r--', linewidth=1.5, label='Black-Scholes (analytical)')
    axes[1].set_xlim(60, 140)
    axes[1].set_xlabel('Spot Price S')
    axes[1].set_ylabel('Option Value V(S)')
    axes[1].set_title(f'FDM vs Analytical — {option_type.capitalize()} Option')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'case1_{option_type}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"  CASE 1: European {option_type.capitalize()} — Constant Volatility")
    print(f"{'='*60}")
    print(f"  S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"  Grid: Nx={Nx}, Nt={Nt}, dx={dx:.4f}, dt={dt:.6f}")
    print(f"  Stability: β = {beta:.4f} (must be ≤ 0.5)")
    print(f"  FDM Price at S0:  {fdm_price:.4f}")
    print(f"  BS Analytical:    {bs_price:.4f}")
    print(f"  Error:            {abs(fdm_price - bs_price):.6f}")
    print(f"  Rel. Error:       {abs(fdm_price - bs_price)/bs_price*100:.4f}%")
    print(f"{'='*60}\n")

    return V, S, fdm_price, bs_price


# ══════════════════════════════════════════════════════════════════════════════
#  CASE 2: TIME-DEPENDENT VOLATILITY (TERM STRUCTURE)
#  (Analogous to: constant load, variable Cv at constant rate)
# ══════════════════════════════════════════════════════════════════════════════

def price_european_call_case2():
    """
    European Call with time-dependent volatility σ(t).
    σ decreases linearly from σ_0 to σ_T over the option's life.

    The FDM scheme adapts: at each time step, σ is updated:
        β_k = 0.5 * σ(t_k)² * dτ / dx²

    This mirrors the variable consolidation coefficient C(i,k).
    """
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma_0 = 0.30    # Initial vol
    sigma_T = 0.15     # Terminal vol (vol decreases = vol term structure)

    Nx = 200
    x_min = np.log(S0) - 4 * sigma_0 * np.sqrt(T)
    x_max = np.log(S0) + 4 * sigma_0 * np.sqrt(T)
    dx = (x_max - x_min) / Nx
    dt = 0.4 * dx**2 / sigma_0**2
    Nt = int(T / dt) + 1
    dt = T / Nt

    x = np.linspace(x_min, x_max, Nx + 1)
    S = np.exp(x)
    V = np.maximum(S - K, 0.0)

    sigma_schedule = np.linspace(sigma_T, sigma_0, Nt + 1)  # backward: τ=0→T

    for k in range(Nt):
        sigma_k = sigma_schedule[k]
        beta_k = 0.5 * sigma_k**2 * dt / dx**2
        alpha_k = (r - 0.5 * sigma_k**2) * dt / (2 * dx)

        V_new = np.zeros_like(V)
        for i in range(1, Nx):
            V_new[i] = V[i] \
                + beta_k * (V[i-1] - 2*V[i] + V[i+1]) \
                + alpha_k * (V[i+1] - V[i-1]) \
                - r * dt * V[i]

        V_new[0] = 0.0
        V_new[Nx] = S[Nx] - K * np.exp(-r * (k+1) * dt)
        V = V_new.copy()

    idx_S0 = np.argmin(np.abs(S - S0))
    fdm_price = V[idx_S0]

    # Compare with constant-vol BS using average vol
    sigma_avg = np.sqrt(np.mean(np.linspace(sigma_T, sigma_0, 100)**2))
    bs_approx = black_scholes_call(S0, K, T, r, sigma_avg)

    # Also compute constant-vol prices for comparison
    V_const_high = black_scholes_call(S0, K, T, r, sigma_0)
    V_const_low = black_scholes_call(S0, K, T, r, sigma_T)

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(S, V, 'b-', linewidth=2, label=f'FDM — σ(t): {sigma_0}→{sigma_T}')
    V_bs_high = np.array([black_scholes_call(s, K, T, r, sigma_0) for s in S])
    V_bs_low = np.array([black_scholes_call(s, K, T, r, sigma_T) for s in S])
    axes[0].plot(S, V_bs_high, 'r--', alpha=0.7, label=f'BS σ={sigma_0} (upper bound)')
    axes[0].plot(S, V_bs_low, 'g--', alpha=0.7, label=f'BS σ={sigma_T} (lower bound)')
    axes[0].set_xlim(60, 140)
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Option Value V(S)')
    axes[0].set_title('Case 2: Time-Dependent Volatility')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    t_axis = np.linspace(0, T, Nt + 1)
    axes[1].plot(t_axis, np.linspace(sigma_0, sigma_T, Nt + 1), 'b-', linewidth=2)
    axes[1].set_xlabel('Calendar Time t')
    axes[1].set_ylabel('Volatility σ(t)')
    axes[1].set_title('Volatility Term Structure')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('case2_time_vol.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"  CASE 2: European Call — Time-Dependent Volatility")
    print(f"{'='*60}")
    print(f"  σ(t): {sigma_0} → {sigma_T} (linear decrease)")
    print(f"  FDM Price:           {fdm_price:.4f}")
    print(f"  BS (σ_avg={sigma_avg:.3f}):   {bs_approx:.4f}")
    print(f"  BS (σ={sigma_0}):        {V_const_high:.4f}")
    print(f"  BS (σ={sigma_T}):        {V_const_low:.4f}")
    print(f"{'='*60}\n")

    return V, S, fdm_price


# ══════════════════════════════════════════════════════════════════════════════
#  CASE 3: LOCAL VOLATILITY SURFACE σ(S, t)
#  (Analogous to: heterogeneous soil with different Cv per layer)
# ══════════════════════════════════════════════════════════════════════════════

def price_european_call_case3():
    """
    European Call with local volatility σ(S, t).

    The volatility depends on both the stock price level and time,
    creating a 'volatility surface'. This is analogous to heterogeneous
    soil layers where each layer has different consolidation properties.

    The interface condition in the heterogeneous soil maps to the
    transition between volatility regimes at different strike levels.
    """
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    # Local vol surface: higher vol for low S (skew), decreasing over time
    def local_vol(S_val, t_val):
        skew = 0.20 + 0.10 * np.maximum(1 - S_val / K, 0)  # vol skew
        term = 1.0 - 0.3 * t_val / T                         # vol term decay
        return skew * term

    sigma_max = 0.35
    Nx = 200
    x_min = np.log(S0) - 4 * sigma_max * np.sqrt(T)
    x_max = np.log(S0) + 4 * sigma_max * np.sqrt(T)
    dx = (x_max - x_min) / Nx
    dt = 0.3 * dx**2 / sigma_max**2
    Nt = int(T / dt) + 1
    dt = T / Nt

    x = np.linspace(x_min, x_max, Nx + 1)
    S = np.exp(x)
    V = np.maximum(S - K, 0.0)

    for k in range(Nt):
        tau_k = (k + 0.5) * dt
        t_k = T - tau_k

        V_new = np.zeros_like(V)
        for i in range(1, Nx):
            sig_i = local_vol(S[i], t_k)
            beta_i = 0.5 * sig_i**2 * dt / dx**2
            alpha_i = (r - 0.5 * sig_i**2) * dt / (2 * dx)

            V_new[i] = V[i] \
                + beta_i * (V[i-1] - 2*V[i] + V[i+1]) \
                + alpha_i * (V[i+1] - V[i-1]) \
                - r * dt * V[i]

        V_new[0] = 0.0
        V_new[Nx] = S[Nx] - K * np.exp(-r * (k+1) * dt)
        V = V_new.copy()

    idx_S0 = np.argmin(np.abs(S - S0))
    fdm_price = V[idx_S0]
    bs_flat = black_scholes_call(S0, K, T, r, 0.20)

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Vol surface heatmap
    S_grid = np.linspace(60, 140, 100)
    t_grid = np.linspace(0, T, 50)
    vol_surface = np.zeros((len(S_grid), len(t_grid)))
    for i, s in enumerate(S_grid):
        for j, t_val in enumerate(t_grid):
            vol_surface[i, j] = local_vol(s, t_val)

    im = axes[0].imshow(vol_surface, extent=[0, T, 60, 140], aspect='auto',
                         cmap='viridis', origin='lower')
    plt.colorbar(im, ax=axes[0], label='σ(S,t)')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Spot Price S')
    axes[0].set_title('Local Volatility Surface σ(S, t)')

    # Price comparison
    axes[1].plot(S, V, 'b-', linewidth=2, label='FDM — Local Vol')
    V_bs = np.array([black_scholes_call(s, K, T, r, 0.20) for s in S])
    axes[1].plot(S, V_bs, 'r--', linewidth=1.5, label='BS — Flat Vol (σ=0.20)')
    axes[1].set_xlim(60, 140)
    axes[1].set_xlabel('Spot Price S')
    axes[1].set_ylabel('Option Value V(S)')
    axes[1].set_title('Case 3: Local Volatility Pricing')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('case3_local_vol.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"  CASE 3: European Call — Local Volatility Surface")
    print(f"{'='*60}")
    print(f"  FDM Price (local vol): {fdm_price:.4f}")
    print(f"  BS Price (flat 20%):   {bs_flat:.4f}")
    print(f"  Difference:            {fdm_price - bs_flat:.4f}")
    print(f"{'='*60}\n")

    return V, S, fdm_price


# ══════════════════════════════════════════════════════════════════════════════
#  CASE 4: UP-AND-OUT BARRIER OPTION
#  (Analogous to: sudden load change / abrupt boundary condition)
# ══════════════════════════════════════════════════════════════════════════════

def price_barrier_option_case4():
    """
    Up-and-Out Call: the option becomes worthless if S ever touches
    the barrier B. This introduces an additional boundary condition
    V(B, t) = 0, analogous to the sudden load change in case d)
    of the consolidation problem.

    The barrier acts as an absorbing boundary — similar to the
    impermeable layer boundary condition ∂u/∂z = 0 in consolidation,
    but here it's a Dirichlet condition V = 0.
    """
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    B = 130.0  # Barrier level

    Nx = 200
    x_min = np.log(S0) - 4 * sigma * np.sqrt(T)
    x_max = np.log(B)  # Truncate domain at barrier
    dx = (x_max - x_min) / Nx
    dt = 0.4 * dx**2 / sigma**2
    Nt = int(T / dt) + 1
    dt = T / Nt

    beta = 0.5 * sigma**2 * dt / dx**2
    alpha = (r - 0.5 * sigma**2) * dt / (2 * dx)

    x = np.linspace(x_min, x_max, Nx + 1)
    S = np.exp(x)

    # Terminal payoff (knocked out if S ≥ B)
    V = np.maximum(S - K, 0.0) * (S < B)

    for k in range(Nt):
        V_new = np.zeros_like(V)
        for i in range(1, Nx):
            V_new[i] = V[i] \
                + beta * (V[i-1] - 2*V[i] + V[i+1]) \
                + alpha * (V[i+1] - V[i-1]) \
                - r * dt * V[i]

        V_new[0] = 0.0    # Deep OTM boundary
        V_new[Nx] = 0.0   # Barrier: knocked out (V = 0)
        V = V_new.copy()

    idx_S0 = np.argmin(np.abs(S - S0))
    barrier_price = V[idx_S0]
    vanilla_price = black_scholes_call(S0, K, T, r, sigma)

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(S, V, 'b-', linewidth=2, label='Up-and-Out Call')
    V_vanilla = np.array([black_scholes_call(s, K, T, r, sigma) for s in S])
    axes[0].plot(S, V_vanilla, 'r--', linewidth=1.5, label='Vanilla Call')
    axes[0].axvline(x=B, color='k', linestyle=':', linewidth=2, label=f'Barrier B={B}')
    axes[0].set_xlim(60, 140)
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Option Value V(S)')
    axes[0].set_title('Case 4: Up-and-Out Barrier Call')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Discount visualization
    discount = np.where(V_vanilla > 0.01, V / V_vanilla, np.nan)
    axes[1].plot(S, discount, 'purple', linewidth=2)
    axes[1].axvline(x=B, color='k', linestyle=':', linewidth=2, label=f'Barrier B={B}')
    axes[1].set_xlim(60, 135)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('Spot Price S')
    axes[1].set_ylabel('Barrier / Vanilla Ratio')
    axes[1].set_title('Knock-Out Discount Factor')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('case4_barrier.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"  CASE 4: Up-and-Out Barrier Call")
    print(f"{'='*60}")
    print(f"  Barrier B = {B}")
    print(f"  Barrier Price:  {barrier_price:.4f}")
    print(f"  Vanilla Price:  {vanilla_price:.4f}")
    print(f"  KO Discount:    {barrier_price/vanilla_price*100:.1f}%")
    print(f"{'='*60}\n")

    return V, S, barrier_price


# ══════════════════════════════════════════════════════════════════════════════
#  CASE 5: TWO-ASSET BASKET OPTION — 2D PDE
#  (Analogous to: 2D consolidation with vertical drains)
# ══════════════════════════════════════════════════════════════════════════════

def price_basket_option_case5():
    """
    Two-asset European Call on the basket: payoff = max(S1 + S2 - 2K, 0).

    The 2D Black-Scholes PDE is:
        ∂V/∂τ = 0.5*σ1²*S1²*∂²V/∂S1² + 0.5*σ2²*S2²*∂²V/∂S2²
               + ρ*σ1*σ2*S1*S2*∂²V/∂S1∂S2 + ...

    After log-transform x1 = ln(S1), x2 = ln(S2):
        ∂V/∂τ = D1*∂²V/∂x1² + D2*∂²V/∂x2² + cross terms

    This is structurally analogous to the 2D consolidation equation:
        ∂u/∂t = Cx*∂²u/∂x² + Cz*∂²u/∂z²

    with the vertical drain boundaries mapping to the correlation structure.
    """
    S1_0, S2_0, K = 100.0, 100.0, 100.0
    T, r = 1.0, 0.05
    sigma1, sigma2 = 0.20, 0.25
    rho = 0.5  # Correlation

    # Reduced grid for 2D (computational cost)
    N1 = N2 = 50
    x1_min = np.log(S1_0) - 3 * sigma1 * np.sqrt(T)
    x1_max = np.log(S1_0) + 3 * sigma1 * np.sqrt(T)
    x2_min = np.log(S2_0) - 3 * sigma2 * np.sqrt(T)
    x2_max = np.log(S2_0) + 3 * sigma2 * np.sqrt(T)

    dx1 = (x1_max - x1_min) / N1
    dx2 = (x2_max - x2_min) / N2

    dt = 0.2 * min(dx1**2 / sigma1**2, dx2**2 / sigma2**2)
    Nt = int(T / dt) + 1
    dt = T / Nt

    x1 = np.linspace(x1_min, x1_max, N1 + 1)
    x2 = np.linspace(x2_min, x2_max, N2 + 1)
    S1 = np.exp(x1)
    S2 = np.exp(x2)
    S1_grid, S2_grid = np.meshgrid(S1, S2, indexing='ij')

    # Terminal payoff: basket call
    V = np.maximum(0.5 * (S1_grid + S2_grid) - K, 0.0)

    # Diffusion coefficients
    D1 = 0.5 * sigma1**2
    D2 = 0.5 * sigma2**2
    mu1 = r - 0.5 * sigma1**2
    mu2 = r - 0.5 * sigma2**2
    D12 = rho * sigma1 * sigma2

    print(f"  2D Grid: {N1+1} x {N2+1} = {(N1+1)*(N2+1)} nodes")
    print(f"  Time steps: {Nt}")
    print(f"  Computing...")

    for k in range(Nt):
        V_new = V.copy()
        for i in range(1, N1):
            for j in range(1, N2):
                # Second derivatives
                d2V_dx1 = (V[i+1, j] - 2*V[i, j] + V[i-1, j]) / dx1**2
                d2V_dx2 = (V[i, j+1] - 2*V[i, j] + V[i, j-1]) / dx2**2
                # Cross derivative
                d2V_dx1dx2 = (V[i+1, j+1] - V[i+1, j-1]
                              - V[i-1, j+1] + V[i-1, j-1]) / (4 * dx1 * dx2)
                # First derivatives
                dV_dx1 = (V[i+1, j] - V[i-1, j]) / (2 * dx1)
                dV_dx2 = (V[i, j+1] - V[i, j-1]) / (2 * dx2)

                V_new[i, j] = V[i, j] + dt * (
                    D1 * d2V_dx1 + D2 * d2V_dx2 + D12 * d2V_dx1dx2
                    + mu1 * dV_dx1 + mu2 * dV_dx2
                    - r * V[i, j]
                )

        # Boundary conditions
        V_new[0, :] = 0.0
        V_new[:, 0] = 0.0
        for i in range(N1 + 1):
            V_new[i, N2] = max(0.5 * (S1[i] + S2[N2]) - K * np.exp(-r*(k+1)*dt), 0)
        for j in range(N2 + 1):
            V_new[N1, j] = max(0.5 * (S1[N1] + S2[j]) - K * np.exp(-r*(k+1)*dt), 0)

        V = V_new.copy()

    # Extract price at (S1_0, S2_0)
    i0 = np.argmin(np.abs(S1 - S1_0))
    j0 = np.argmin(np.abs(S2 - S2_0))
    basket_price = V[i0, j0]

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D price surface
    im = axes[0].contourf(S1_grid, S2_grid, V, levels=30, cmap='viridis')
    plt.colorbar(im, ax=axes[0], label='Basket Option Value')
    axes[0].set_xlabel('S₁')
    axes[0].set_ylabel('S₂')
    axes[0].set_title('2D Basket Option Price Surface')
    axes[0].plot(S1_0, S2_0, 'r*', markersize=15, label=f'ATM ({S1_0}, {S2_0})')
    axes[0].legend()

    # Cross-section at S2 = S2_0
    axes[1].plot(S1, V[:, j0], 'b-', linewidth=2, label=f'V(S₁, S₂={S2_0})')
    axes[1].axvline(x=K, color='k', linestyle=':', alpha=0.5, label=f'K={K}')
    axes[1].set_xlabel('S₁')
    axes[1].set_ylabel('Basket Option Value')
    axes[1].set_title(f'Cross-Section at S₂ = {S2_0}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('case5_basket_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"  CASE 5: Two-Asset Basket Option (2D PDE)")
    print(f"{'='*60}")
    print(f"  σ1={sigma1}, σ2={sigma2}, ρ={rho}")
    print(f"  Basket Price at ATM: {basket_price:.4f}")
    print(f"{'='*60}\n")

    return V, S1, S2, basket_price


# ══════════════════════════════════════════════════════════════════════════════
#  GREEKS COMPUTATION (DELTA, GAMMA, THETA, VEGA)
#  (Analogous to: settlement & consolidation degree computation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_greeks():
    """
    Compute option Greeks from the FDM price grid using finite differences.

    This is analogous to computing settlement from pore pressure:
      - Delta (∂V/∂S) ↔ effective stress gradient
      - Gamma (∂²V/∂S²) ↔ curvature of pore pressure profile
      - Theta (∂V/∂t) ↔ rate of consolidation
      - Vega (∂V/∂σ) ↔ sensitivity to Cv changes
    """
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    Nx = 300
    x_min = np.log(S0) - 4 * sigma * np.sqrt(T)
    x_max = np.log(S0) + 4 * sigma * np.sqrt(T)
    dx = (x_max - x_min) / Nx
    dt = 0.4 * dx**2 / sigma**2
    Nt = int(T / dt) + 1
    dt = T / Nt

    beta = 0.5 * sigma**2 * dt / dx**2
    alpha_coeff = (r - 0.5 * sigma**2) * dt / (2 * dx)

    x = np.linspace(x_min, x_max, Nx + 1)
    S = np.exp(x)

    def solve_fdm(sig):
        V = np.maximum(S - K, 0.0)
        b = 0.5 * sig**2 * dt / dx**2
        a = (r - 0.5 * sig**2) * dt / (2 * dx)
        for k in range(Nt):
            V_new = np.zeros_like(V)
            for i in range(1, Nx):
                V_new[i] = V[i] + b*(V[i-1]-2*V[i]+V[i+1]) + a*(V[i+1]-V[i-1]) - r*dt*V[i]
            V_new[0] = 0.0
            V_new[Nx] = S[Nx] - K * np.exp(-r*(k+1)*dt)
            V = V_new.copy()
        return V

    V = solve_fdm(sigma)

    # Delta & Gamma via central differences on S
    delta = np.zeros_like(V)
    gamma = np.zeros_like(V)
    for i in range(1, Nx):
        delta[i] = (V[i+1] - V[i-1]) / (S[i+1] - S[i-1])
        gamma[i] = (V[i+1] - 2*V[i] + V[i-1]) / ((S[i+1] - S[i]) * (S[i] - S[i-1]))

    # Vega via bumping
    dsig = 0.01
    V_up = solve_fdm(sigma + dsig)
    V_down = solve_fdm(sigma - dsig)
    vega = (V_up - V_down) / (2 * dsig)

    # ── Plotting ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mask = (S > 60) & (S < 140)
    axes[0, 0].plot(S[mask], V[mask], 'b-', linewidth=2)
    axes[0, 0].set_title('Option Price V(S)')
    axes[0, 0].set_xlabel('S')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(S[mask], delta[mask], 'r-', linewidth=2)
    axes[0, 1].set_title('Delta ∂V/∂S')
    axes[0, 1].set_xlabel('S')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(S[mask], gamma[mask], 'g-', linewidth=2)
    axes[1, 0].set_title('Gamma ∂²V/∂S²')
    axes[1, 0].set_xlabel('S')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(S[mask], vega[mask], 'purple', linewidth=2)
    axes[1, 1].set_title('Vega ∂V/∂σ')
    axes[1, 1].set_xlabel('S')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Greeks Computed via Finite Differences', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('greeks.png', dpi=150, bbox_inches='tight')
    plt.show()

    idx = np.argmin(np.abs(S - S0))
    print(f"\n{'='*60}")
    print(f"  GREEKS at S = {S0}")
    print(f"{'='*60}")
    print(f"  Price:  {V[idx]:.4f}")
    print(f"  Delta:  {delta[idx]:.4f}")
    print(f"  Gamma:  {gamma[idx]:.6f}")
    print(f"  Vega:   {vega[idx]:.4f}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def convergence_analysis():
    """
    Study the convergence of the FDM scheme as the grid is refined.
    This demonstrates understanding of numerical stability and accuracy.
    """
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_exact = black_scholes_call(S0, K, T, r, sigma)

    grid_sizes = [50, 100, 200, 400, 800]
    errors = []
    computation_times = []

    import time
    for Nx in grid_sizes:
        x_min = np.log(S0) - 4 * sigma * np.sqrt(T)
        x_max = np.log(S0) + 4 * sigma * np.sqrt(T)
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx**2 / sigma**2
        Nt = int(T / dt) + 1
        dt = T / Nt

        x = np.linspace(x_min, x_max, Nx + 1)
        S = np.exp(x)
        V = np.maximum(S - K, 0.0)

        beta = 0.5 * sigma**2 * dt / dx**2
        alpha_c = (r - 0.5 * sigma**2) * dt / (2 * dx)

        t0 = time.time()
        for k in range(Nt):
            V_new = np.zeros_like(V)
            for i in range(1, Nx):
                V_new[i] = V[i] + beta*(V[i-1]-2*V[i]+V[i+1]) \
                           + alpha_c*(V[i+1]-V[i-1]) - r*dt*V[i]
            V_new[0] = 0.0
            V_new[Nx] = S[Nx] - K * np.exp(-r*(k+1)*dt)
            V = V_new.copy()
        elapsed = time.time() - t0

        idx = np.argmin(np.abs(S - S0))
        err = abs(V[idx] - bs_exact)
        errors.append(err)
        computation_times.append(elapsed)
        print(f"  Nx={Nx:4d}, Nt={Nt:6d}, Error={err:.6f}, Time={elapsed:.2f}s")

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(grid_sizes, errors, 'bo-', linewidth=2, markersize=8)
    # Reference O(dx²) line
    ref = errors[0] * (np.array(grid_sizes[0]) / np.array(grid_sizes))**2
    axes[0].loglog(grid_sizes, ref, 'r--', alpha=0.7, label='O(Δx²) reference')
    axes[0].set_xlabel('Number of Spatial Steps Nx')
    axes[0].set_ylabel('Absolute Error |FDM - BS|')
    axes[0].set_title('Convergence: Error vs Grid Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(grid_sizes, computation_times, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Spatial Steps Nx')
    axes[1].set_ylabel('Computation Time (s)')
    axes[1].set_title('Computational Cost vs Grid Size')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═"*70)
    print("  OPTION PRICING VIA EXPLICIT FINITE DIFFERENCES")
    print("═"*70 + "\n")

    # Case 1: Baseline — European Call & Put, constant vol
    print("▸ Case 1: European Call — Constant Volatility")
    V1_call, S1, price1_call, bs1_call = price_european_option_case1('call')
    V1_put, _, price1_put, bs1_put = price_european_option_case1('put')

    # Case 2: Time-dependent volatility
    print("▸ Case 2: European Call — Time-Dependent Volatility")
    V2, S2, price2 = price_european_call_case2()

    # Case 3: Local volatility surface
    print("▸ Case 3: European Call — Local Volatility Surface")
    V3, S3, price3 = price_european_call_case3()

    # Case 4: Barrier option
    print("▸ Case 4: Up-and-Out Barrier Call")
    V4, S4, price4 = price_barrier_option_case4()

    # Case 5: 2D Basket option
    print("▸ Case 5: Two-Asset Basket Option (2D PDE)")
    V5, S5_1, S5_2, price5 = price_basket_option_case5()

    # Greeks
    print("▸ Greeks Computation")
    compute_greeks()

    # Convergence
    print("▸ Convergence Analysis")
    convergence_analysis()

    # Summary table
    print("\n" + "═"*70)
    print("  SUMMARY OF RESULTS")
    print("═"*70)
    print(f"  {'Case':<45} {'Price':>10}")
    print(f"  {'-'*45} {'-'*10}")
    print(f"  {'1. European Call (σ=0.20)':<45} {price1_call:>10.4f}")
    print(f"  {'1. European Put  (σ=0.20)':<45} {price1_put:>10.4f}")
    print(f"  {'2. Call — Time-dep. Vol (0.30→0.15)':<45} {price2:>10.4f}")
    print(f"  {'3. Call — Local Vol Surface':<45} {price3:>10.4f}")
    print(f"  {'4. Up-and-Out Call (B=130)':<45} {price4:>10.4f}")
    print(f"  {'5. Basket Call (2D, ρ=0.5)':<45} {price5:>10.4f}")
    print("═"*70 + "\n")
