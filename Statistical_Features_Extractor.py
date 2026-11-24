#NEEDED LIBRARIES
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from astropy import units as u
from astropy.cosmology import Planck18
# for FMM+IPM
import gzip
import re
#======================================================================================================================
#LIGHT CURVES LOADING
def load_dataset(filename: str = "glot_microlensing_training_toy.npz"):
    with np.load(filename, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
#=====================================================================================================================

#local derivates
def calculate_local_derivatives(time: np.ndarray,magnitudes: np.ndarray,window_points: int = 5,gap_threshold: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate first and second derivatives using local quadratic polynomial regression.
    Returns
    -------
    first_derivative : ndarray
        Estimated first derivative dm/dt at each usable point.

    second_derivative : ndarray
        Estimated second derivative d²m/dt² at each usable point.

    gap_edge_mask : ndarray of bool
        True for observations immediately before or after detected
        gaps. These points remain usable, but their local fits cannot
        cross the gap.

    Notes
    -----
    Derivatives are estimated using actual observation times, so
    irregular sampling is supported.

    Points belonging to segments with fewer than three observations
    cannot support a quadratic fit and receive NaN derivative values.
    """

    t = np.asarray(time, dtype=np.float64)
    m = np.asarray(magnitudes, dtype=np.float64)
    #Mostly all possible errors from the input data
    if t.ndim != 1 or m.ndim != 1:
        raise ValueError("time and magnitudes must be 1D arrays.")

    if len(t) != len(m):
        raise ValueError("time and magnitudes must have equal lengths.")

    if window_points < 3 or window_points % 2 == 0:
        raise ValueError("window_points must be an odd integer >= 3.")

    if gap_threshold is not None and gap_threshold <= 0:
        raise ValueError("gap_threshold must be positive.")

    if len(t) < 3:
        raise ValueError("At least three observations are needed for quadratic fits.")

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(m)):
        raise ValueError("Pass only finite observations to this function.")

    if np.any(np.diff(t) <= 0):
        raise ValueError("Observation times must be strictly increasing.")

    n = len(t)
    first_derivative = np.full(n, np.nan, dtype=np.float64)
    second_derivative = np.full(n, np.nan, dtype=np.float64)
    gap_edge_mask = np.zeros(n, dtype=bool)
    # ---------------------------------------------------------
    #Detect temporal gaps and identify their edge points
    if gap_threshold is None:
        gap_indices = np.array([], dtype=int)
    else:
        gap_indices = np.flatnonzero(np.diff(t) > gap_threshold)

    # Mark the final observation before each gap and the first observation after it.
    gap_edge_mask[gap_indices] = True
    gap_edge_mask[gap_indices + 1] = True
    # Split the curve immediately after each gap's left endpoint.
    segments = np.split(np.arange(n),gap_indices + 1)

    #Fit local quadratic polynomials within each segment
    for segment in segments:
        n_segment = len(segment)
        # A quadratic polynomial requires at least 3 observations.
        if n_segment < 3:
            continue
        # Short segments use the largest available odd window.
        window = min(window_points, n_segment)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            continue
        half_window = window // 2
        for local_i, global_i in enumerate(segment):
            # Choose a window centered on the current point when
            # possible, or a one-sided window near a segment edge.
            start = local_i - half_window
            start = max(0,min(start, n_segment - window))
            local_indices = segment[start:start + window]
            # Shift time to the evaluation point and scale it to improve numerical conditioning.
            dt = t[local_indices] - t[global_i]
            time_scale = np.max(np.abs(dt))
            if time_scale <= 0:
                continue
            u = dt / time_scale
            coefficients = np.polynomial.polynomial.polyfit(u,m[local_indices],deg=2)
            first_derivative[global_i] = (coefficients[1] / time_scale)
            second_derivative[global_i] = (2.0* coefficients[2]/ time_scale**2)

    return (first_derivative, second_derivative, gap_edge_mask)

#Feature extraction from light curves

def extract_lightcurve_features(magnitudes: np.ndarray,time: np.ndarray | None = None,k_levels: int = 8,level_step: float = 0.1,
    noise_present: bool = False,
    derivative_window_points: int | None = None,
    clean_window_points: int = 5,
    noisy_window_points: int = 7,
    gap_threshold: float | None = None,) -> dict[str, float]:
    """
    Features:
        1. Mean magnitude
        2. Absolute linear slope
        3. Standard deviation of detrended curve
        4. Interquartile range of detrended curve
        5 1st local derivative (mean absolute value)
        6 2nd local derivative (mean absolute value)
        7. Maximum of detrended curve
        8. Minimum of detrended curve
        9. Waviness of detrended curve
    """
    #Prepare light curve

    m = np.asarray(magnitudes, dtype=np.float64).ravel()

    if time is None:
        t = np.linspace(0.0, 1.0, len(m))
    else:
        t = np.asarray(time, dtype=np.float64).ravel()
        if len(t) != len(m):
            raise ValueError("time and magnitudes must have the same length.")

    # Remove invalid observations, if any.
    valid = np.isfinite(t) & np.isfinite(m)
    t = t[valid]
    m = m[valid]

    if len(m) < 2:
        raise ValueError(
            "At least two valid observations are required."
        )

    if np.any(np.diff(t) <= 0):
        raise ValueError(
            "Observation times must be strictly increasing.")

    if k_levels < 1:
        raise ValueError("k_levels must be positive.")

    if level_step <= 0:
        raise ValueError("level_step must be positive.")

    #Mean magnitude
    mean_mag = np.mean(m)

    #Fit a linear trend and detrend
    slope, intercept = np.polyfit(t, m, deg=1)
    abs_slope = abs(slope)
    m_detrended = m - (slope * t + intercept)

    # Center the residuals for the calculation of moments.
    residuals = m_detrended - np.mean(m_detrended)

    # Std of detrended curve
    mu2 = np.mean(residuals**2)
    std_detrended = np.sqrt(mu2)

    # Interquartile range of the detrended light curve.
    q25, q75 = np.percentile(m_detrended,[25, 75])
    iqr_detrended = q75 - q25

    # Maximum and minimum of detrended curve
    max_detrended = np.max(m_detrended)
    min_detrended = np.min(m_detrended)

    # Waviness
    # Construct k levels above the minimum, separated by level_step magnitudes.
    levels = (min_detrended + level_step * np.arange(1, k_levels + 1))

    total_crossings = 0

    for level in levels:
        # A crossing occurs when consecutive points lie on
        # opposite sides of the selected magnitude level.
        below_before = m_detrended[:-1] < level
        below_after = m_detrended[1:] < level
        crossings = np.count_nonzero(below_before != below_after)
        total_crossings += crossings

    # Normalize
    waviness = total_crossings / k_levels
    
    #Configure derivate windows
    if derivative_window_points is None:
        if noise_present:
            derivative_window_points = noisy_window_points
        else:
            derivative_window_points = clean_window_points

    if derivative_window_points < 3:
        raise ValueError("derivative_window_points must be at least 3.")

    if derivative_window_points % 2 == 0:
        raise ValueError("derivative_window_points must be odd.")

    #Calculate local derivatives without crossing gaps

    first_derivative, second_derivative, gap_edge_mask = (
        calculate_local_derivatives(
            time=t,
            magnitudes=m_detrended,
            window_points=derivative_window_points,
            gap_threshold=gap_threshold))

    valid_d1 = np.isfinite(first_derivative)
    valid_d2 = np.isfinite(second_derivative)

    if not np.any(valid_d1) or not np.any(valid_d2):
        raise ValueError("Insufficient observations within continuous segments to estimate both derivatives.")

    mean_abs_first_derivative = np.mean(np.abs(first_derivative[valid_d1]))
    mean_abs_second_derivative = np.mean(np.abs(second_derivative[valid_d2]))

    return {
        "mean_mag": float(mean_mag),
        "abs_slope": float(abs_slope),
        "std_detrended": float(std_detrended),
        "iqr_detrended": float(iqr_detrended),
        "mean_abs_first_derivative": float(mean_abs_first_derivative),
        "mean_abs_second_derivative": float(mean_abs_second_derivative),
        "max_detrended": float(max_detrended),
        "min_detrended": float(min_detrended),
        "waviness": float(waviness)}

def _get_time_matrix(dataset: dict,X: np.ndarray,time: np.ndarray | None = None) -> np.ndarray:
    #time matrix matching the light-curve matrix. Needed to make gaps + subsampling more functional
    if time is None:
        time = dataset.get("time")
    n_curves, n_samples = X.shape #SO we know how many samples we have to generate time for
    # If no times have been provided, assume normalized time.
    if time is None:
        common_time = np.linspace(0.0,1.0,n_samples)
        return np.broadcast_to(common_time,X.shape).copy() #More common case for now

    time = np.asarray(time,dtype=np.float64)

    # Shared time axis.
    if time.ndim == 1:
        if len(time) != n_samples:
            raise ValueError("The 1D time array must match the number of columns in the light-curve matrix.")
        return np.broadcast_to(time,X.shape).copy()

    # Individual time axis for each curve.
    if time.ndim == 2:
        if time.shape != X.shape:
            raise ValueError("The 2D time array must have the same shape as the light-curve matrix.")
        return time.copy()

def parametrize_training_dataset(
    dataset: dict,
    k_levels: int = 8,
    level_step: float = 0.1,
    noise_present: bool = False,
    derivative_window_points: int | None = None,
    clean_window_points: int = 5,
    noisy_window_points: int = 7,
    gap_threshold: float | None = None,
) -> dict:

    light_curves = np.asarray(dataset["X"],dtype=np.float64)

    if light_curves.ndim != 2:
        raise ValueError("dataset['X'] must have shape (N_curves, N_samples).")

    time_matrix = _get_time_matrix(dataset=dataset,X=light_curves,time=dataset["time"])

    # Automatically identify seasonal gaps when the dataset was
    # generated using add_seasonal_gaps().
    if gap_threshold is None:
        gap_duration = dataset.get("seasonal_gap_duration")
        if gap_duration is not None:
            gap_threshold = 0.5 * float(gap_duration)

    feature_rows = []

    for i, light_curve in enumerate(light_curves):
        features = extract_lightcurve_features(
            magnitudes=light_curve,
            time=time_matrix[i],
            k_levels=k_levels,
            level_step=level_step,
            noise_present=noise_present,
            derivative_window_points=derivative_window_points,
            clean_window_points=clean_window_points,
            noisy_window_points=noisy_window_points,
            gap_threshold=gap_threshold)

        feature_rows.append([features[name] for name in FEATURE_NAMES])
    result = dataset.copy()
    result["features"] = np.asarray(feature_rows,dtype=np.float32)
    result["feature_names"] = FEATURE_NAMES
    # Record the choices made for the derivative estimation.
    result["derivative_window_points"] = (derivative_window_points if derivative_window_points is not None
        else (noisy_window_points if noise_present else clean_window_points))

    result["derivative_gap_threshold"] = gap_threshold
    result["derivative_noise_present"] = noise_present

    return result
#====================================================================================================================================
#SAVE
def save_dataset(dataset: dict, filename: str = "glot_microlensing_training_features_toy.npz"):
    np.savez_compressed(filename, **dataset)
#====================================================================================================================================
#MAIN
def main():
    global FEATURE_NAMES
    FEATURE_NAMES = (
        "mean_mag",
        "abs_slope",
        "std_detrended",
        "iqr_detrended",
        "mean_abs_first_derivative",
        "mean_abs_second_derivative",
        "max_detrended",
        "min_detrended",
        "waviness")
    dataset=load_dataset()
    print("LIGHT CURVES LOADED")
    dataset_features = parametrize_training_dataset(dataset,k_levels=8,level_step=0.1,noise_present=dataset["noise"],derivative_window_points=None,clean_window_points=5,noisy_window_points=7,gap_threshold=None)
    print("LIGHT CURVES PARAMETRIZED")
    save_dataset(dataset_features)
    print("FEATURES EXTRACTED")

if __name__ == "__main__":
    main()