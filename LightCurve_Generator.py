#Needed Libraries
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from astropy import units as u
from astropy.cosmology import Planck18
import astropy.cosmology
# for FMM+IPM
import gzip
import re
#=================================================================================================================
# Datatypes needed
@dataclass
class SimulationConfig:
    # Input data
    map_file: str = "map.bin"
    metadata_file: str = "mapmeta.dat"
    # Map handling
    # GERLUMPH maps are commonly 10000 x 10000 and 25 R_E wide.
    crop_size: None | int = 4096
    #---------------------------------------------------------------------
    # Physical assumptions
    # Mean microlens mass used only for converting R_E to physical units.
    use_physical_units : bool = False
    z_lens : float = 0.5
    z_source : float = 2.0
    microlens_mass_msun: float = 1.0
    # Effective velocity in km/s.
    # NOT important for dimensionless toy experiment, but needed to convert the trajectory into time.
    velocity_kms: float = 600.0
    # Light-curve geometry
    # Length of each trajectory in Einstein radii / (V&T,2019) used 5 R_E.
    track_length_re: float = 1.0 #Due to cropping, can be changed
    # Number of samples along every trajectory.
    n_samples: int = 1000
    # Random trajectory angle range - Full [0, 2*pi] = random orientation.
    angle_min: float = 0.0
    angle_max: float = 2.0 * np.pi
    # Source sizes: Half-light radii in units of R_E.-  (V&T,2019) used 0.02 - 0.20 R_E.
    source_sizes_re: tuple[float, ...] = (0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20)
    # Number of trajectories
    trajectories_per_size: int = 100
    # -------------------------------------------------------------------------
    # Selection
    # V&T 2019 selected curves with maximum magnification >= 1.5.
    # Set to None to keep all trajectories.
    max_magnification_threshold: float | None = 1.5
    # Maximum attempts when rejection sampling is used.
    max_attempts_multiplier: int = 20
    # -------------------------------------------------------------------------
    # Noise
    add_noise: bool = False
    # Gaussian photometric noise in magnitudes.
    noise_sigma_mag: float = 0.02
    #-------------------------------------------------------------------------
    add_subsampling: bool = False
    sampling_factor: float = 70.0
    add_seasonal_gaps: bool = False
    gap_period: float = 0.25
    gap_duration: float = 0.08
    #gap_start: float = 0.0
    # -------------------------------------------------------------------------
    # Random seed
    random_seed: int = 17092002

@dataclass
class GLOTMap:
    magnification: np.ndarray
    # Final magnification-map properties
    n_pixels: int
    map_size_re: float
    pixel_size_re: float
    # Lens parameters
    kappa: float
    smooth_fraction: float
    gamma: float
    # Mass-unit convention
    mass_unit_msun: float
    # Auxiliary values appearing in param.dat. NO idea what they mean for now...
    auxiliary_size: float | None = None
    auxiliary_n: int | None = None
    @property
    def mean_magnification(self) -> float:
        return float(np.mean(self.magnification))
    @property
    def map_shape(self) -> tuple[int, int]:
        return self.magnification.shape
#===============================================================================================================
#Data Reading
# for FMM+IPM dat files
def open_text_file(path: str | Path):
    path = Path(path)
    return path.open("r")

def read_glot_param_file(param_file: str | Path) -> dict:
    """
    line 1: half_map_size_re   n_pixel
    line 2: auxiliary_size    auxiliary_n (No idea for now, sadly)
    line 3:kappa             smooth_fraction   gamma
    line 4:mass_unit_msun
    """
    param_file = Path(param_file)
    with open_text_file(param_file) as f:
        # Read all non-empty, non-comment lines.
        lines = []
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            # Remove inline comments if present.
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(line)

    if len(lines) < 4:
        raise ValueError( #for incomplete parameters maybe
            f"{param_file} contains only {len(lines)} "
            "usable lines. Expected at least 4.")
    # LINE 1
    line1 = lines[0].split()

    if len(line1) < 2:
        raise ValueError("First line of param.dat should contain. Half-map-size and number of pixels.")

    half_map_size_re = float(line1[0])
    n_pixels = int(round(float(line1[1])))
    # LINE 2
    line2 = lines[1].split()
    if len(line2) >= 2:
        auxiliary_size = float(line2[0])
        auxiliary_n = int(round(float(line2[1])))
    else:
        auxiliary_size = None
        auxiliary_n = None
    # LINE 3
    line3 = lines[2].split()
    if len(line3) < 3:
        raise ValueError("Third line of param.dat should contain: kappa, smooth matter fraction, and gamma.")

    kappa = float(line3[0])
    smooth_fraction = float(line3[1])
    gamma = float(line3[2])
    # LINE 4
    line4 = lines[3].split()
    if len(line4) < 1:
        raise ValueError("Fourth line of param.dat should contain the mass unit.")
    mass_unit_msun = float(line4[0])
    # Construct final map dimensions.
    map_size_re = 2.0 * half_map_size_re
    pixel_size_re = map_size_re / n_pixels

    return {
        "half_map_size_re": half_map_size_re,
        "map_size_re": map_size_re,
        "n_pixels": n_pixels,
        "pixel_size_re": pixel_size_re,
        "auxiliary_size": auxiliary_size,
        "auxiliary_n": auxiliary_n,
        "kappa": kappa,
        "smooth_fraction": smooth_fraction,
        "gamma": gamma,
        "mass_unit_msun": mass_unit_msun}

#Now lets load the FMM+IPM mag map ono the datatype
def load_glot_map(map_file: str | Path, param_file: str | Path) -> GLOTMap:
    # Read metadata
    params = read_glot_param_file(param_file)
    n_pixels = params["n_pixels"]
    map_size_re = params["map_size_re"]
    pixel_size_re = params["pixel_size_re"]
    expected_values = n_pixels * n_pixels

    # Read magnification map
    print("\nREADING MAGNIFICATION MAP")
    values = []

    with open_text_file(map_file) as f:
        for line_number, raw_line in enumerate(f,start=1,):

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            line = line.split("#",1)[0].strip()

            if not line:
                continue

            for token in line.split():
                try:
                    values.append(float(token))

                except ValueError as exc:
                    raise ValueError(
                        f"Could not interpret '{token}' "
                        f"as a magnification in "
                        f"{map_file}, line "
                        f"{line_number}."
                    ) from exc

    magnification = np.asarray(values,dtype=np.float32)

    if magnification.size != expected_values:

        raise ValueError(
            "\nThe number of magnification values does not "
            "match param.dat.\n\n"
            f"Read:     {magnification.size:,}\n"
            f"Expected: {n_pixels:,} x {n_pixels:,} "
            f"= {expected_values:,}\n\n"
            "Check that magmap.dat and param.dat belong "
            "to the same GLOT simulation.")

    magnification = magnification.reshape(n_pixels,n_pixels) #Just in case
 
    return GLOTMap(
        magnification=magnification,
        n_pixels=n_pixels,
        map_size_re=map_size_re,
        pixel_size_re=pixel_size_re,
        kappa=params["kappa"],
        smooth_fraction=params["smooth_fraction"],
        gamma=params["gamma"],
        mass_unit_msun=params["mass_unit_msun"],
        auxiliary_size=params["auxiliary_size"],
        auxiliary_n=params["auxiliary_n"])
#===================================================================================================================
#Define the source profile and convolve it
def gaussian_sigma_from_half_light_radius(r_half_re: float) -> float:
    #Convert Gaussian half-light radius to Gaussian sigma.
    #For a circular 2-D Gaussian:
    # R_half = sqrt(2 ln 2) * sigma ~ 1.1774100225 * sigma   
    return r_half_re / np.sqrt(2* np.log(2))

def convolve_map_gaussian(magnification_map: np.ndarray,r_half_re: float,pixel_size_re: float) -> np.ndarray:
    #Convolve a magnification map with a circular Gaussian source of the quasar.
    sigma_re = gaussian_sigma_from_half_light_radius(r_half_re)
    sigma_pix = sigma_re / pixel_size_re

    if sigma_pix < 0.5:
        raise ValueError(f"Source size corresponds to only {sigma_pix:.2f} pixels.")

    # gaussian_filter approximates the full Gaussian rather than explicitly truncating at exactly 2 R_half.
    convolved = gaussian_filter(magnification_map,sigma=sigma_pix,mode="nearest")
    return convolved.astype(np.float32)
#==================================================================================================================
#Physical Units Translation
def angular_diameter_distances_cm(z_lens: float,z_source: float,cosmology=Planck18) -> tuple[float, float, float]:
    """
    Calculate angular-diameter distances for a gravitational lens, from redshifts.
    Returns:
    D_S_cm, D_LS_cm, D_L_cm : float
        Angular-diameter distances in cm:
        D_S  : observer to source
        D_LS : lens to source
        D_L  : observer to lens
    """

    if z_lens < 0 or z_source <= z_lens:
        raise ValueError("Require 0 <= z_lens < z_source.")

    # Observer -> lens
    D_L_cm = cosmology.angular_diameter_distance(z_lens).to_value(u.cm)
    # Observer -> source
    D_S_cm = cosmology.angular_diameter_distance(z_source).to_value(u.cm)
    # Lens -> source
    D_LS_cm = cosmology.angular_diameter_distance_z1z2(z_lens,z_source).to_value(u.cm)

    return D_S_cm, D_LS_cm, D_L_cm

# TIME CONVERSION
def einstein_radius_source_cm(microlens_mass_msun: float,D_S_cm: float,D_LS_cm: float,D_L_cm: float) -> float:
    """
    Physical source-plane Einstein radius.
        R_E = sqrt[(4 G M / c^2)*(D_S D_LS / D_L)]
    Distances must be supplied in cm.
    """
    G = 6.67430e-8 #cgs units
    c = 2.99792458e10
    M_sun = 1.98847e33
    M = microlens_mass_msun * M_sun
    return np.sqrt((4.0 * G * M / c**2)*(D_S_cm * D_LS_cm / D_L_cm))

def trajectory_time_axis_days(track_length_re: float,velocity_kms: float,R_E_cm: float,n_samples: int) -> np.ndarray:
    #Convert a trajectory length in R_E into a time axis.
    velocity_cm_s = velocity_kms * 1.0e5
    total_distance_cm = track_length_re * R_E_cm
    total_time_seconds = total_distance_cm / velocity_cm_s
    total_time_days = total_time_seconds / 86400.0
    return np.linspace(0.0,total_time_days,n_samples)

#=====================================================================================================================
#DATESET GENERATION
#Random Trjectories
def generate_random_trajectory(map_shape: tuple[int, int],pixel_size_re: float,track_length_re: float,n_samples: int,rng: np.random.Generator,angle: float | None = None,margin_pixels: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    ny, nx = map_shape
    length_pix = track_length_re / pixel_size_re

    if length_pix >= min(nx, ny) - 2 * margin_pixels:
        raise ValueError("Trajectory is too long for the selected map/crop.")

    if angle is None:
        angle = rng.uniform(0.0, 2.0 * np.pi)

    dx = length_pix * np.cos(angle)
    dy = length_pix * np.sin(angle)
    # Coordinates of the trajectory midpoint.
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    # Keep the complete trajectory inside the map.
    xmin = margin_pixels + abs(half_dx)
    xmax = nx - 1 - margin_pixels - abs(half_dx)
    ymin = margin_pixels + abs(half_dy)
    ymax = ny - 1 - margin_pixels - abs(half_dy)

    if xmin >= xmax or ymin >= ymax: #Error handling for bad positions
        raise ValueError("No valid starting position exists with the requested trajectory length and margin.")

    x_center = rng.uniform(xmin, xmax)
    y_center = rng.uniform(ymin, ymax)
    u = np.linspace(-0.5,0.5,n_samples)
    x = x_center + u * dx
    y = y_center + u * dy

    return x, y, angle


def sample_map_along_trajectory(magnification_map: np.ndarray,x_pixels: np.ndarray,y_pixels: np.ndarray,order: int = 1) -> np.ndarray:
    coords = np.vstack([y_pixels,x_pixels])
    magnification = map_coordinates(magnification_map,coords,order=order,mode="nearest")
    return magnification.astype(np.float32)
#======================================================================================================================================================
# LIGHT CURVE GENERATION
#Add photometric noise
def magnification_to_delta_mag(magnification: np.ndarray) -> np.ndarray:
    #Delta m = -2.5 log10(mu)
    # Minimal mag setting fixated, works to a avoid 0
    magnification = np.maximum(magnification,1e-12)
    return (-2.5* np.log10(magnification/np.mean(magnification))).astype(np.float32)
def add_photometric_noise(magnitudes: np.ndarray,sigma_mag: float,rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    #Gaussian photometric noise + uncertanties
    uncertainty = np.full_like(magnitudes,fill_value=sigma_mag,dtype=np.float32)
    noise = rng.normal(loc=0.0,scale=sigma_mag,size=magnitudes.shape)
    noisy = (magnitudes + noise).astype(np.float32)
    return noisy, uncertainty
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# ONE LIGHT CURVE
def generate_single_light_curve(convolved_map: np.ndarray,config: SimulationConfig,r_half_re: float,rng: np.random.Generator,R_E_cm: float | None = None,) -> dict:
    sigma_re = gaussian_sigma_from_half_light_radius(r_half_re)
    #source profile may extend beyond the trajectory by roughl a few sigma, so leave conservative margin
    margin_re = max(0.05,5.0 * sigma_re)
    margin_pixels = int(np.ceil(margin_re / config.track_length_re ))
    # Generate trajectory
    x, y, angle = generate_random_trajectory(
        map_shape=convolved_map.shape,
        pixel_size_re=GLOBAL_MAP_PIXEL_SIZE_RE,
        track_length_re=config.track_length_re,
        n_samples=config.n_samples,
        rng=rng,
        margin_pixels=margin_pixels)
    # Sample the map
    mu = sample_map_along_trajectory(convolved_map,x,y,order=1)
    # Convert to magnitude variation.
    delta_mag = magnification_to_delta_mag(mu)
    # observational noise.
    if config.add_noise:
        delta_mag_obs, sigma_mag = add_photometric_noise(delta_mag,config.noise_sigma_mag,rng)
    else:
        delta_mag_obs = delta_mag
        sigma_mag = np.zeros_like(delta_mag)

    # Time axis.
    if R_E_cm is not None:
        time_days = trajectory_time_axis_days(config.track_length_re,config.velocity_kms,R_E_cm,config.n_samples)
        time_unit = "days"
    else:
        # Dimensionless toy time.
        # tau=0 beginning of trajectory,tau=1 end
        time_days = np.linspace(0.0,1.0,config.n_samples)
        time_unit = "normalized"

    return {
        "time": time_days.astype(np.float32),
        "time_unit": time_unit,
        "magnification": mu,
        "delta_mag": delta_mag_obs,
        "sigma_mag": sigma_mag,
        "x_pixels": x.astype(np.float32),
        "y_pixels": y.astype(np.float32),
        "angle": np.float32(angle),
        "r_half_re": np.float32(r_half_re), #In Einstein Radius Units
        "track_length_re": np.float32(config.track_length_re),
        "velocity_kms": np.float32(config.velocity_kms)}

#group of light curves
def generate_training_dataset(convolved_maps: dict[float, np.ndarray],config: SimulationConfig,map_pixel_size_re: float, R_E_cm: float | None = None,) -> dict:
    rng = np.random.default_rng(config.random_seed)
    curves = []
    labels = []
    metadata = []
    common_time = None
    for r_half in config.source_sizes_re:
        convolved_map = convolved_maps[r_half]
        accepted = 0
        attempts = 0
        max_attempts = (config.trajectories_per_size * config.max_attempts_multiplier)

        while (accepted < config.trajectories_per_size and attempts < max_attempts):
            attempts += 1
            curve = generate_single_light_curve(convolved_map=convolved_map,config=config,r_half_re=r_half,rng=rng,R_E_cm=R_E_cm)
            # -------------------------------------------------------------
            # Optional selection criterion.
            # (V&T, 2019) discarded "flat" tracks with maximum magnification below 1.5.
            if (config.max_magnification_threshold is not None):
                if (np.max(curve["magnification"])<config.max_magnification_threshold):
                    continue
            curves.append(curve["delta_mag"])
            labels.append(r_half)
            metadata.append({
                "source_size_re": r_half,
                "angle_rad": float(curve["angle"]),
                "track_length_re": float(curve["track_length_re"]),
                "velocity_kms": float(curve["velocity_kms"])})

            if common_time is None:
                common_time = curve["time"].copy()
            accepted += 1

        if accepted < config.trajectories_per_size:
            print("WARNING: could not obtain the requested number of accepted trajectories.")

    # Convert to arrays.
    X = np.asarray(curves,dtype=np.float32)
    Y = np.asarray(labels,dtype=np.float32)

    return {
        "X": X,
        "Y": Y,
        "time": common_time,
        "time_unit": ("days" if R_E_cm is not None else "normalized"),
        "metadata": metadata,
        "noise": config.add_noise}

#=================================================================================================================================
#ADD SEASONAL GAPS
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


def _pad_lightcurves(curves: list[np.ndarray],times: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    #Convert light curves of different lengths into NaN-padded matrices.
    if len(curves) == 0:
        raise ValueError("No light curves were provided.")

    lengths = np.array([len(curve) for curve in curves],dtype=int)

    if np.any(lengths < 2):
        raise ValueError("Every light curve must retain at least two observations.") #If for some reason there is excessive subsampling

    max_length = int(lengths.max())
    X_padded = np.full((len(curves), max_length),np.nan,dtype=np.float32)
    time_padded = np.full((len(curves), max_length),np.nan,dtype=np.float64)

    for i, (curve, t) in enumerate(zip(curves, times)):
        n = len(curve)
        X_padded[i, :n] = curve
        time_padded[i, :n] = t
    return X_padded, time_padded # So that we have no problem adding gaps on irregularly sampled light curves


def add_seasonal_gaps(dataset: dict,gap_period: float,gap_duration: float,gap_start: float | None = None,time: np.ndarray | None = None,random_phase: bool = False,seed: int = 24073) -> dict:
    """
    dataset : dict
        Dataset containing:
            dataset["X"]: (N_curves, N_samples)
            dataset["y"]: physical labels
        Existing NaN padding from previous sampling operations is supported
    """
    X = np.asarray(dataset["X"],dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("dataset['X'] must be a 2D matrix.")

    gap_period= np.abs(gap_period)
    gap_duration= np.abs(gap_duration)

    if gap_duration >= gap_period:
        raise ValueError("gap_duration must be smaller than gap_period.")

    #time coordinates
    time_matrix = _get_time_matrix(dataset=dataset,X=X,time=time)
    if gap_start is None:
        gap_start = gap_period - gap_duration #Lets try not to start the LC in a gap

    rng = np.random.default_rng(seed)
    #Apply recurring gaps to each LC
    retained_curves = []
    retained_times = []
    observation_counts = []

    for i in range(X.shape[0]): # 1 by 1 O(n)
        curve = X[i]
        t = time_matrix[i]
        # Remove existing padding or invalid observations
        valid = (np.isfinite(curve)& np.isfinite(t))
        curve = curve[valid]
        t = t[valid]
        order = np.argsort(t) #observations are in chronological order
        curve = curve[order]
        t = t[order]

        if len(t) < 2: #Again, it shouldnt happen, but just in case
            raise ValueError(f"Curve {i} has fewer than two valid observations before seasonal gaps are applied.")

        if np.any(np.diff(t) <= 0):
            raise ValueError(f"Curve {i} has duplicate observation times.")

        # Optional random phase so each LC has a different gap start time. maybe useful for data augmentation
        phase_offset = gap_start
        if random_phase:
            phase_offset += rng.uniform(0.0,gap_period)

        # Determine observation if falls in a gap
        # phase = 0 is beginning of a gap
        phase = np.mod(t - phase_offset,gap_period)
        inside_gap = phase < gap_duration
        keep = ~inside_gap
        curve_kept = curve[keep]
        time_kept = t[keep]

        if len(curve_kept) < 2: #... May happen under extreme input, but happened
            raise ValueError(f"Seasonal gaps leave fewer than two observations for curve {i}. Reduce gap_duration or increase the observation baseline.")

        retained_curves.append(curve_kept)
        retained_times.append(time_kept)
        observation_counts.append(len(curve_kept))

    #Pad curves to a common matrix shape
    X_processed, time_processed = _pad_lightcurves(retained_curves,retained_times)
    # ---------------------------------------------------------
    #Preserve labels and update metadata
    result = dataset.copy()
    result["X"] = X_processed
    result["time"] = time_processed
    result["n_observations_per_curve"] = np.asarray(observation_counts,dtype=int)
    result["seasonal_gap_period"] = gap_period
    result["seasonal_gap_duration"] = gap_duration
    result["seasonal_gap_start"] = gap_start
    result["seasonal_gap_random_phase"] = random_phase
    
    return result

#=================================================================================================================================================================0
#IRREGULAR SAMPLING
def randomly_subsample_lightcurves(dataset: dict,keep_percentage: float,time: np.ndarray | None = None,seed: int = 170902) -> dict:
    """
    Supports NaN-padded datasets with variable numbers of observations.
    The time coordinates are preserved.
    """
    X = np.asarray(dataset["X"],dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("dataset['X'] must be a 2D matrix.")

    if not 0 < keep_percentage <= 100:
        raise ValueError("keep_percentage must be in (0, 100].")

    time_matrix = _get_time_matrix(dataset=dataset,X=X,time=time)
    rng = np.random.default_rng(seed)
    retained_curves = []
    retained_times = []
    retained_indices = []

    for i in range(X.shape[0]):
        curve = X[i]
        t = time_matrix[i]
        valid = (np.isfinite(curve) & np.isfinite(t))
        valid_indices = np.where(valid)[0]
        curve = curve[valid]
        t = t[valid]
        n_valid = len(curve)

        if n_valid < 2: #better safe than sorry I guess
            raise ValueError(f"Curve {i} has fewer than two valid observations before subsampling.")

        n_keep = int(round(n_valid * keep_percentage / 100.0))
        n_keep = max(2,min(n_valid, n_keep))
        chosen = rng.choice(n_valid,size=n_keep,replace=False) #IMPORTANT NOT REPLACE
        chosen.sort()
        retained_curves.append(curve[chosen])
        retained_times.append(t[chosen])
        # Indices refer to columns in the input dataset.
        retained_indices.append(valid_indices[chosen])

    X_processed, time_processed = _pad_lightcurves(retained_curves,retained_times) #Padding missing data
    result = dataset.copy()
    result["X"] = X_processed
    result["time"] = time_processed
    result["sampling_indices"] = retained_indices
    result["sampling_keep_percentage"] = keep_percentage
    result["n_observations_per_curve"] = np.asarray([len(c) for c in retained_curves],dtype=int)
    return result

#===================================================================================================================================================
#SAVING FILE
def save_dataset(dataset: dict, filename: str = "glot_microlensing_training_toy.npz"):
    np.savez_compressed(filename, **dataset)
#===============================================================================================================================================
#MAIN
def main(): #for FMM+IPM
    global GLOBAL_MAP_PIXEL_SIZE_RE
    
    config = SimulationConfig(
        map_file="magmap.dat (1)",
        metadata_file="param.dat (1)",
        #map_file="vernardos_mag_map.dat",
        #metadata_file="param_vernardos.dat",
        #map_file="jimenez_vicente_magmap.dat",
        #metadata_file="jimenez_vicente_param.dat",
        #map_file="bates2_magmap.dat",
        #metadata_file="bates2_param.dat",
        crop_size=None,
        source_sizes_re=(0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20),
        trajectories_per_size=3,
        track_length_re=5.0,
        n_samples=1000,
        max_magnification_threshold=1.5,
        #Physicial units
        use_physical_units= True,
        z_lens= 0.5,
        z_source= 2.0,
        velocity_kms= 600.0,
        add_noise=True,
        noise_sigma_mag=0.02,
        add_subsampling=True,
        sampling_factor=70,
        add_seasonal_gaps=True,
        gap_period=0.25,
        gap_duration=0.08,
        random_seed=240874)

    glot = load_glot_map(map_file=config.map_file,param_file=config.metadata_file)
    GLOBAL_MAP_PIXEL_SIZE_RE = (glot.pixel_size_re)
    

    #Determine if using physical units
    if config.use_physical_units== True:
        D_S_cm, D_LS_cm, D_L_cm = (angular_diameter_distances_cm(z_lens=config.z_lens,z_source=config.z_source))
        # Use the mass normalization associated with the GLOT map.
        R_E_cm = einstein_radius_source_cm(glot.mass_unit_msun,D_S_cm,D_LS_cm,D_L_cm)
        # Convert the Einstein radius into light-days for reporting.
        track_duration = trajectory_time_axis_days(track_length_re=config.track_length_re,velocity_kms=config.velocity_kms,R_E_cm=R_E_cm,n_samples=config.n_samples)
        config.gap_period= config.gap_period*float(track_duration[-1])
        config.gap_duration= config.gap_duration*float(track_duration[-1])
    else:
        R_E_cm = None
    
    convolved_maps = {}

    print("APPLYING CONVOLUTION")

    for r_half in config.source_sizes_re:
        convolved_maps[r_half] = (convolve_map_gaussian(magnification_map=(glot.magnification),r_half_re=r_half,pixel_size_re=(glot.pixel_size_re)))


    # GENERATE LIGHT CURVES

    print("GENERATING SYNTHETIC LIGHT CURVES")
    dataset = generate_training_dataset(convolved_maps=convolved_maps,config=config,map_pixel_size_re=GLOBAL_MAP_PIXEL_SIZE_RE,R_E_cm=R_E_cm)

  
    print("ADDING SUPSAMPLING AND GAPS TO TRAINING DATASET")

    #Subsampling for irregular sampling
    if config.add_subsampling==True and config.add_seasonal_gaps==False:
        dataset_subsampled = randomly_subsample_lightcurves(dataset,keep_percentage=config.sampling_factor,seed=config.random_seed)
    #Seasonal gaps
    elif config.add_subsampling==False and config.add_seasonal_gaps==True:
        dataset_subsampled = add_seasonal_gaps(dataset,gap_period=config.gap_period,gap_duration=config.gap_duration,seed=config.random_seed)
    elif config.add_subsampling==True and config.add_seasonal_gaps==True:
        dataset_subsampled = randomly_subsample_lightcurves(dataset,keep_percentage=config.sampling_factor,seed=config.random_seed)
        dataset_subsampled = add_seasonal_gaps(dataset_subsampled,gap_period=config.gap_period,gap_duration=config.gap_duration,seed=config.random_seed)
    else:
        dataset_subsampled=dataset.copy()

    # SAVE
    save_dataset(dataset_subsampled,"glot_microlensing_training_toy.npz")
    print("LIGHT CURVES SAVED")

if __name__ == "__main__":
    main()
