"""
Solar forecast error generalization script following the wind workflow.

The script filters zero-generation periods, fits marginal error behavior
with KDE, and uses a copula-based method to generate temporally correlated
PV error samples.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, norm, multivariate_normal
from scipy.special import ndtr
from scipy import interpolate


print("=== 1. Data loading and preprocessing ===")

df = pd.read_csv('data/raw/ods032.csv', sep=';')
print(f"Original number of rows: {len(df)}")

df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

actual_values = df['Measured & Upscaled'].values
forecast_values = df['Most recent forecast'].values

mask = (actual_values != 0) | (forecast_values != 0)
actual_daytime = actual_values[mask]
forecast_daytime = forecast_values[mask]

print(f"Rows after removing nighttime periods: {len(actual_daytime)}")

print("\n=== 2. Forecast error calculation ===")

forecast_error = actual_daytime - forecast_daytime
forecast_error_normalized = forecast_error / np.std(actual_daytime)

print("Error statistics:")
print(f"  Mean: {np.mean(forecast_error_normalized):.4f}")
print(f"  Std: {np.std(forecast_error_normalized):.4f}")
print(f"  Min: {np.min(forecast_error_normalized):.4f}")
print(f"  Max: {np.max(forecast_error_normalized):.4f}")

print("\n=== 3. Reshaping into a time-series structure ===")

num_time_points = 48
num_days = len(forecast_error_normalized) // num_time_points

forecast_error_normalized = forecast_error_normalized[:num_days * num_time_points]
forecast_error_matrix = forecast_error_normalized.reshape(num_days, num_time_points)

print(f"Error matrix shape: {forecast_error_matrix.shape}")
print(f"  Number of periods: {num_days}")
print(f"  Time points per period: {num_time_points}")
print("  Note: all 48 points correspond to generation hours only; nighttime zeros are removed")

print("\n=== 4. Temporal correlation analysis ===")

temp_corr = np.corrcoef(forecast_error_matrix, rowvar=False)
print(f"Temporal correlation matrix shape: {temp_corr.shape}")

plt.figure(figsize=(10, 10))
mask_upper = np.triu(np.ones_like(temp_corr, dtype=bool))
sns.heatmap(temp_corr, annot=False, fmt=".2f", mask=mask_upper, cmap='coolwarm')
plt.title('Solar Forecasting Error Temporal Correlation')
plt.tight_layout()
plt.savefig('data/solar_error_temporal_correlation.png', dpi=300)
print("Temporal correlation heatmap saved: data/solar_error_temporal_correlation.png")
plt.close()

print("\n=== 5. Marginal distribution fitting with KDE ===")

marginal_error = forecast_error_matrix.flatten()
kde = gaussian_kde(marginal_error)

x = np.linspace(np.min(marginal_error), np.max(marginal_error), 100)
kde_pdf = kde.pdf(x)

mu, std = np.mean(marginal_error), np.std(marginal_error)
gaussian_pdf = norm.pdf(x, mu, std)

plt.figure(figsize=(14, 5))
plt.hist(marginal_error, bins=100, alpha=0.5, label='Original samples', density=True)
plt.plot(x, kde_pdf, 'k', linewidth=2, label='Gaussian KDE')
plt.plot(x, gaussian_pdf, 'r', linewidth=2, label='Gaussian fit')
plt.xlabel('Normalized forecasting error')
plt.ylabel('Density')
plt.legend()
plt.title('Solar Forecasting Error Marginal Distribution')
plt.tight_layout()
plt.savefig('data/solar_error_marginal_distribution.png', dpi=300)
print("Marginal distribution figure saved: data/solar_error_marginal_distribution.png")
plt.close()

print("\n=== 6. Copula-based generation of temporally correlated samples ===")

n_scenario = 200000
print(f"Number of generated scenarios: {n_scenario}")

mvnorm = multivariate_normal(mean=np.zeros(temp_corr.shape[0]), cov=temp_corr)
samples_gaussian = mvnorm.rvs(size=n_scenario, random_state=0)
print(f"Multivariate Gaussian sample shape: {samples_gaussian.shape}")

norm_dist = norm()
samples_uniform = norm_dist.cdf(samples_gaussian)
print(f"Uniform sample shape: {samples_uniform.shape}")

stdev = np.sqrt(kde.covariance)[0, 0]
xmax = np.max(marginal_error)
xmin = np.min(marginal_error)
minmax_dist = xmax - xmin
xx = np.linspace(xmin - 0.0 * minmax_dist, xmax + 0.0 * minmax_dist, 5000)

print("Computing the KDE CDF in batches to reduce memory usage...")
n_resample = 50000
n = kde.resample(n_resample, seed=0).flatten()

batch_size = 1000
kde_cdf = np.zeros(len(xx))
for i in range(0, len(xx), batch_size):
    batch_end = min(i + batch_size, len(xx))
    xx_batch = xx[i:batch_end]
    kde_cdf[i:batch_end] = ndtr(np.subtract.outer(xx_batch, n) / stdev).mean(axis=1)
    if (i // batch_size) % 5 == 0:
        print(f"  Progress: {batch_end}/{len(xx)}")

print("KDE CDF computation completed")

kde_cdf_inv_func = interpolate.interp1d(
    kde_cdf,
    xx,
    kind='cubic',
    bounds_error=False,
    fill_value='extrapolate',
)

samples_kde_corr = np.vstack(
    [kde_cdf_inv_func(samples_uniform[:, i]) for i in range(temp_corr.shape[0])]
).T
print(f"Final temporally correlated sample shape: {samples_kde_corr.shape}")

print("\n=== 7. Sample validation ===")

plt.figure(figsize=(14, 5))
plt.hist(
    samples_kde_corr.flatten(),
    bins=100,
    alpha=0.5,
    label='Generated samples',
    density=True,
)
plt.hist(
    marginal_error,
    bins=100,
    alpha=0.5,
    label='Original samples',
    density=True,
    color='olive',
)
plt.plot(x, gaussian_pdf, 'r', linewidth=2, label='Gaussian fit')
plt.xlim(np.percentile(marginal_error, 1), np.percentile(marginal_error, 99))
plt.xlabel('Normalized forecasting error')
plt.ylabel('Density')
plt.legend()
plt.title('Comparison of Generated and Original Samples')
plt.tight_layout()
plt.savefig('data/solar_error_comparison.png', dpi=300)
print("Sample comparison figure saved: data/solar_error_comparison.png")
plt.close()

temp_corr_gen = np.corrcoef(samples_kde_corr, rowvar=False)

fig, axs = plt.subplots(1, 2, figsize=(16, 7))
axs[0].set_title('Original samples')
sns.heatmap(temp_corr, annot=False, fmt=".2f", mask=mask_upper, ax=axs[0], cmap='coolwarm')
axs[1].set_title('Generated samples')
sns.heatmap(temp_corr_gen, annot=False, fmt=".2f", mask=mask_upper, ax=axs[1], cmap='coolwarm')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Time')
axs[1].set_xlabel('Time')
plt.tight_layout()
plt.savefig('data/solar_error_temporal_correlation_comparison.png', dpi=300)
print("Temporal correlation comparison figure saved: data/solar_error_temporal_correlation_comparison.png")
plt.close()

print("\n=== 8. Saving results ===")

np.save('data/processed/solar_temporal_correlated_error_samples.npy', samples_kde_corr)
print("Error samples saved: data/processed/solar_temporal_correlated_error_samples.npy")
print(f"  Shape: {samples_kde_corr.shape}")

solar_forecast_prototype = forecast_daytime[:48]
np.save('data/processed/solar_forecast_prototype.npy', solar_forecast_prototype)
print("PV forecast prototype saved: data/processed/solar_forecast_prototype.npy")
print(f"  Shape: {solar_forecast_prototype.shape}")
print("  Note: these 48 points correspond to daytime generation hours")

print("\n=== PV error generalization completed ===")
print(f"Generated {n_scenario} PV error scenarios")
print(f"Each scenario contains {num_time_points} time points, all within daytime generation periods")
print("\n=== Usage notes ===")
print("When constructing a full 24-hour scenario:")
print("  1. Set nighttime periods (19:00-04:00) directly to zero")
print("  2. For daytime periods (05:00-18:00), use forecast + error sample")
print("  3. Ensure the final scenario is nonnegative with clip(0, None)")
