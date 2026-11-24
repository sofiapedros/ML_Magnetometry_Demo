# Diameter of the nanoparticle in nanometers
diameter_nm: float = 10.0

# Magnetization saturation
Ms: float = 4.80e5

# Constant for the exchange energy term. Exchange stiffness
A: float = 2e-11

# Uniaxial anisotropy constant [J/m^3].
K1: float = 2e4
K1_2: float = 1.3e4

# Mean anisotropy values used as an explicit dataset dimension.
K1_VALUES_J_PER_M3: list[float] = [K1, K1_2]

# Mesh cell size
cell_nm: float = 2.5

# Maximum field in Oersted. The loop goes +Hmax -> -Hmax -> +Hmax
Hmax_Oe: float = 5000.0

# Number of linear intervals in EACH half of the loop.
# 100 gives ~201 stored field points in the complete loop because the shared
# endpoint between the two ranges is not duplicated.
field_steps: int = 100

# Limit on the number of successful evolver steps allowed per stage
stage_iteration_limit: int = 500

## Stopping criterion for TimeDriver [degrees/ns]. 0.0 to not apply it
stopping_dm_dt: float = 0.0


# Temperature used by UHH_ThetaEvolve
temperature_K: float = 300.0

# Fixed physical timestep of the thermal evolver
fixed_timestep_s: float = 1e-13

# Physical relaxation time applied before the hysteresis field sweep.
# The corresponding number of solver steps is derived only inside the MIF
# and is also stored in generated metadata; relax_steps is derived in the MIF.
relax_time_s: float = 1e-12

# Gilbert damping
alpha: float = 0.5

# Random seed for the stochastic thermal field
thermal_seed: int = 12345

# Seed for the per-cell random initial magnetization. Kept separate from
# thermal_seed so the frozen initial texture and thermal noise are reproducible
# independently.
m0_seed: int = 111


# ----------------------------------------------------------------------
# Magnetic-state classification (SPM / SD / MD)
# ----------------------------------------------------------------------

# A run is labelled "SPM" when, in addition to behaving as
# a single spatially-uniform macrospin (no domain wall ever appears),
# the loop's coercivity and remanence both fall below these thresholds.
spm_hc_threshold_mT: float = 5.0
spm_mr_threshold: float = 0.05

# Diameter-only reference labels. These remain separate from the ensemble
# classification and act only as an additional fixed-size suggestion.
suggested_spm_sd_diameter_nm: float = 20.0
suggested_sd_md_diameter_nm: float = 80.0

# ----------------------------------------------------------------------
# Configuration averaging (ensemble over K1, easy-axis orientation,
# and thermal-noise realizations for the same nominal particle)
# ----------------------------------------------------------------------
# Number of ensemble members simulated per particle (diameter) AT EACH
# temperature. One "sample" of the dataset (fixed diameter + fixed
# timescale) sweeps the whole TEMPERATURES_K grid below, and every
# point of that sweep is itself an ensemble average over this many
# microscopic realisations.
ensemble_size: int = 20

# Relative standard deviation used to sample K1 per member:
#     K1_i ~ Normal(K1, K1_relative_sigma * |K1|)e.
K1_relative_sigma: float = 0.10

# How each ensemble member's uniaxial EASY AXIS is chosen. Uniaxial
# anisotropy needs one axis only, so there is no axis2 anywhere any more.
#
# "aligned":    every member keeps axis1=(1,0,0), i.e. a single-crystal-
#               like particle with its easy axis locked to the field
# "isotropic":  every member gets an independent, uniformly-random easy
#               axis on the sphere, drawn with Marsaglia's rejection
#               method -- the random-texture / powder average
orientation_mode: str = "isotropic"

# Master seed controlling how K1/orientation/thermal_seed are SAMPLED
# across ensemble members (reproducibility of the ensemble composition
# itself; each member still gets its own distinct thermal_seed for the
# stochastic field inside the simulation).
ensemble_master_seed: int = 2024

# ----------------------------------------------------------------------
# Temperature sweep (one dataset "sample" = fixed diameter + fixed
# timescale, swept across this whole grid).
# ----------------------------------------------------------------------
TEMPERATURES_K: list[float] = [5.0, 250.0, 300.0]

# ----------------------------------------------------------------------
# ZFC/FC magnetization-vs-temperature curves
# ----------------------------------------------------------------------
# Small constant probe field (Oe) used to build the ZFC/FC M(T) points
# that accompany every hysteresis-derived sample. Large enough to give a
# measurable signal, small enough not to itself drive the SPM/SD/MD
# transition being probed.
zfc_fc_probe_Oe: float = 50.0

# ----------------------------------------------------------------------
# Manual-labeling review
# ----------------------------------------------------------------------
# A temperature point is flagged for mandatory manual review (regardless
# of ensemble consensus) whenever the ensemble's majority state changes
# relative to the previous, colder temperature point in the same sample
# -- i.e. exactly at the SPM<->SD and SD<->MD crossings, where the
# automatic per-member classification is least reliable.
transition_ambiguous_margin: float = 0.25
