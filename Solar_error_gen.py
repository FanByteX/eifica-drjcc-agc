import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from copy import deepcopy
import pandas as pd


def Solar_sce_gen(Solar_no, samples_no, seed=0):
    """
    Generate PV forecast and temporally correlated error scenarios.

    Returns normalized forecast, error, and full-scenario arrays with a
    48-step time axis aligned with the wind scenario generator.
    """
    Solar_forecast = np.load(
        os.path.join(os.getcwd(), 'data', 'processed', 'solar_forecast_prototype.npy')
    )
    Solar_forecast = Solar_forecast.reshape([1, -1, 1])
    Solar_forecast = np.tile(Solar_forecast, (1, 1, Solar_no))

    Solar_error_samples = np.load(
        os.path.join(os.getcwd(), 'data', 'processed', 'solar_temporal_correlated_error_samples.npy')
    )
    Solar_error_samples = Solar_error_samples[:samples_no * Solar_no].reshape(
        samples_no, Solar_no, -1
    )
    Solar_error_samples = np.transpose(Solar_error_samples, (0, 2, 1))

    forecast_std = np.std(Solar_forecast)
    Solar_error_samples = Solar_error_samples * forecast_std
    print(f"  Error rescaling factor (forecast std): {forecast_std:.4f}")

    Solar_samples_full = Solar_forecast + Solar_error_samples
    Solar_samples_full[Solar_samples_full < 0] = 0

    Solar_error_samples = Solar_samples_full - Solar_forecast

    norm_factor = np.max(Solar_samples_full)
    Solar_samples_full = Solar_samples_full / norm_factor
    Solar_error_samples = Solar_error_samples / norm_factor
    Solar_forecast = Solar_forecast / norm_factor

    if not np.allclose(Solar_samples_full, Solar_forecast + Solar_error_samples):
        raise ValueError('PV scenarios are not equal to forecast plus error.')

    if np.min(Solar_samples_full) < 0:
        raise ValueError('PV scenarios contain negative values.')

    print("PV scenario generation completed:")
    print(f"  Forecast shape: {Solar_forecast[0].shape}")
    print(f"  Error sample shape: {Solar_error_samples.shape}")
    print(f"  Full scenario shape: {Solar_samples_full.shape}")
    print(f"  Normalization factor: {norm_factor:.4f}")
    print("  Note: 48 time steps, fully aligned with the wind scenarios")

    return Solar_forecast[0], Solar_error_samples, Solar_samples_full


def plot_paper_figures(save_path='figure'):
    """Generate publication-style PV scenario figures."""
    import seaborn as sns
    from scipy.stats import norm, gaussian_kde

    os.makedirs(save_path, exist_ok=True)

    print("=== Generating reference figures ===")
    Solar_no = 5
    samples_no = 200
    Solar_forecast, Solar_error_samples, Solar_samples_full = Solar_sce_gen(
        Solar_no, samples_no
    )

    T = 24
    Solar_forecast_24h = Solar_forecast[:T, :]
    Solar_error_24h = Solar_error_samples[:, :T, :]
    Solar_samples_24h = Solar_samples_full[:, :T, :]

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    error_flat = Solar_error_24h.flatten()

    ax1.hist(
        error_flat,
        bins=80,
        density=True,
        alpha=0.6,
        color='steelblue',
        edgecolor='white',
        label='Empirical distribution',
    )

    kde = gaussian_kde(error_flat)
    x = np.linspace(error_flat.min(), error_flat.max(), 200)
    ax1.plot(x, kde(x), 'k-', linewidth=2, label='KDE fit')

    mu, std = np.mean(error_flat), np.std(error_flat)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'r--', linewidth=2, label='Gaussian fit')

    ax1.set_xlabel('Normalized forecast error', fontsize=12)
    ax1.set_ylabel('Probability density', fontsize=12)
    ax1.set_title('PV Forecast Error Marginal Distribution', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(
        os.path.join(save_path, 'solar_error_distribution.png'),
        dpi=300,
        bbox_inches='tight',
    )
    fig1.savefig(
        os.path.join(save_path, 'solar_error_distribution.pdf'),
        bbox_inches='tight',
    )
    print(f"  Saved: {save_path}/solar_error_distribution.png/pdf")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    hours = np.arange(T)
    scenarios_data = Solar_samples_24h[:, :, 0]
    forecast_data = Solar_forecast_24h[:, 0]

    p5 = np.percentile(scenarios_data, 5, axis=0)
    p95 = np.percentile(scenarios_data, 95, axis=0)
    scenario_mean = scenarios_data.mean(axis=0)

    ax2.fill_between(
        hours,
        p5,
        p95,
        alpha=0.3,
        color='gold',
        label='90% confidence interval',
    )

    ax2.axvspan(6, 18, alpha=0.1, color='green', label='Dispatch window (6-18h)')

    n_plot = 100
    for i in range(n_plot):
        ax2.plot(hours, scenarios_data[i, :], alpha=0.5, linewidth=0.8, color='royalblue')

    ax2.plot(hours, forecast_data, 'r-', linewidth=2.5, label='Day-ahead forecast')
    ax2.plot(hours, scenario_mean, 'k--', linewidth=2, label='Scenario mean')

    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel('Normalized PV generation', fontsize=12)
    ax2.set_title('PV Generation Scenarios (24-hour dispatch horizon)', fontsize=13)
    ax2.set_xlim([0, T - 1])
    ax2.set_ylim([0, 0.6])
    ax2.set_xticks(np.arange(0, T, 2))
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig2.savefig(
        os.path.join(save_path, 'solar_scenarios_24h.png'),
        dpi=300,
        bbox_inches='tight',
    )
    fig2.savefig(
        os.path.join(save_path, 'solar_scenarios_24h.pdf'),
        bbox_inches='tight',
    )
    print(f"  Saved: {save_path}/solar_scenarios_24h.png/pdf")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 7))

    error_matrix = Solar_error_24h[:, :, 0]
    temp_corr = np.corrcoef(error_matrix, rowvar=False)

    mask = np.triu(np.ones_like(temp_corr, dtype=bool), k=1)
    sns.heatmap(
        temp_corr,
        mask=mask,
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False,
        ax=ax3,
        vmin=-1,
        vmax=1,
    )

    ax3.set_xlabel('Hour', fontsize=12)
    ax3.set_ylabel('Hour', fontsize=12)
    ax3.set_title('Temporal Correlation of PV Forecast Error', fontsize=13)

    tick_positions = np.arange(0, T, 4) + 0.5
    tick_labels = np.arange(0, T, 4)
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(tick_positions)
    ax3.set_yticklabels(tick_labels)

    plt.tight_layout()
    fig3.savefig(
        os.path.join(save_path, 'solar_temporal_correlation.png'),
        dpi=300,
        bbox_inches='tight',
    )
    fig3.savefig(
        os.path.join(save_path, 'solar_temporal_correlation.pdf'),
        bbox_inches='tight',
    )
    print(f"  Saved: {save_path}/solar_temporal_correlation.png/pdf")
    plt.close(fig3)

    print("\n=== Reference figure generation completed ===")
    print(f"Figure output path: {os.path.abspath(save_path)}/")
    print("  - solar_error_distribution.png/pdf  (error marginal distribution)")
    print("  - solar_scenarios_24h.png/pdf       (24-hour scenario curves)")
    print("  - solar_temporal_correlation.png/pdf (temporal correlation heatmap)")


if __name__ == '__main__':
    plot_paper_figures(save_path='figure')
