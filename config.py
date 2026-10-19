"""
Editable parameters for the coupled human + ankle-exoskeleton gait model.

Every value below is an assumption unless you replace it with a measurement.
Units are stated next to each constant.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------
# The kinematic equations are implemented exactly as specified when every
# sign below is +1.
#
#   y  anterior  (m)     positive forward
#   z  superior  (m)     positive upward
#   angle          positive counterclockwise in the y-z plane
#                  (positive hip angle swings the knee forward)
#
# Set a sign to -1 to flip that joint before it enters the kinematics.
# SIGN_ANKLE = -1 makes an increase in the ankle data lift the toe.
SIGN_HIP = 1.0
SIGN_KNEE = 1.0
SIGN_ANKLE = 1.0

# Raw ankle Fourier series is shifted by this many degrees before use.
# The project statement requires a +20 degree correction.
ANKLE_ZERO_OFFSET_DEG = 20.0

# ---------------------------------------------------------------------------
# Gait timing
# ---------------------------------------------------------------------------
T_STRIDE = 1.1          # s, duration of one gait cycle (user-adjustable)
DT = 0.001              # s, sample time. Small enough for np.gradient.

# Stance is the first part of the cycle. Swing is the rest.
# Fractions of the gait cycle, not seconds.
STANCE_END_FRACTION = 0.60

# ---------------------------------------------------------------------------
# Subject anthropometry  (ASSUMPTIONS, except the lengths you specified)
# ---------------------------------------------------------------------------
BODY_MASS = 75.0        # kg
BODY_HEIGHT = 1.80      # m
GRAVITY = 9.81          # m/s^2

L_THIGH = 0.46          # m, hip joint to knee joint
L_SHANK = 0.45          # m, knee joint to ankle joint
FOOT_LENGTH_FRACTION = 0.15   # L_foot = this * BODY_HEIGHT
D5 = 0.094              # m, ankle-to-heel offset along the sole normal

# Segment mass as a fraction of body mass. One limb. Approximate.
# Typical source: Winter / Dempster-style fractions, not a measured subject.
MASS_FRAC_THIGH = 0.10
MASS_FRAC_SHANK = 0.0465
MASS_FRAC_FOOT = 0.0145

# COM location as a fraction of segment length, measured from the
# proximal end (hip, knee, or heel). Approximate.
COM_FRAC_THIGH = 0.433  # from the hip toward the knee
COM_FRAC_SHANK = 0.433  # from the knee toward the ankle
COM_FRAC_FOOT = 0.50    # from the heel toward the toe

# Sagittal radius of gyration about the segment COM, as a fraction of
# segment length. I = m * (k * L)^2. Approximate.
K_GYR_THIGH = 0.323
K_GYR_SHANK = 0.302
K_GYR_FOOT = 0.475

# ---------------------------------------------------------------------------
# Synthetic ground reaction force  (ASSUMPTION until measured GRF is supplied)
# ---------------------------------------------------------------------------
# Peak |anterior/posterior GRF| as a fraction of body weight.
GRF_AP_PEAK_FRACTION = 0.20
# Double-hump shape parameters on the normalized stance coordinate s in [0, 1].
# Centers and widths are fractions of stance, not of the whole stride.
# These values give the usual walking shape: two peaks near 1.1 body
# weights and a mid-stance valley near 0.7 body weights, after the
# impulse is scaled to half a body weight over the stride.
GRF_HUMP1_CENTER = 0.22
GRF_HUMP2_CENTER = 0.75
GRF_HUMP1_WEIGHT = 1.00
GRF_HUMP2_WEIGHT = 1.05
GRF_HUMP_SIGMA = 0.18

# ---------------------------------------------------------------------------
# Exoskeleton actuator
# ---------------------------------------------------------------------------
# "assistance" : tau_exo = ASSISTANCE_RATIO * tau_ankle_required
# "pd"         : tau_exo = Kp*(theta_des - theta) + Kd*(omega_des - omega)
# With prescribed kinematics and zero offset, PD torque is approximately zero.
# Assistance mode is the default because it answers the design question:
# how much ankle torque remains for the human.
CONTROLLER_MODE = "assistance"
ASSISTANCE_RATIO = 0.30     # 0 to 1. 0.30 means the motor takes 30%.

KP = 80.0                   # N*m/rad, PD proportional gain on ankle angle
KD = 4.0                    # N*m/(rad/s), PD derivative gain
TAU_MAX = 120.0             # N*m, joint-side torque limit after gearing
GEAR_RATIO = 100.0          # joint torque / motor torque, dimensionless
MOTOR_TORQUE_CONSTANT = 0.05  # N*m/A, motor Kt
# Desired ankle angle = measured ankle angle + this offset (PD mode only).
THETA_DES_OFFSET_DEG = 0.0

# Effective moment arm used to turn actuator torque into an interface force.
# F = tau_exo / R_EXO. This is a simplified model, not a contact solution.
R_EXO = 0.05                # m

# Cuff sits this fraction of the way from the ankle up toward the knee.
CUFF_FRACTION_FROM_ANKLE = 0.62
# Drawing offsets so the brace is visible next to the limb. Meters.
EXO_UPRIGHT_OFFSET = 0.025  # m, anterior offset of the upright from the shank
EXO_PLATE_OFFSET = 0.012    # m, soleward offset of the foot plate

# ---------------------------------------------------------------------------
# Inversion / eversion brace  (placeholder motion, mechanical stop)
# ---------------------------------------------------------------------------
# The sagittal model does not produce this angle. The waveform below is a
# stand-in so the brace constraint can be integrated and plotted.
IE_AMP1_DEG = 4.0           # deg, once-per-cycle component
IE_AMP2_DEG = 2.5           # deg, twice-per-cycle component
THETA_IE_LIMIT_DEG = 5.0    # deg, safe range is +/- this value
K_IE = 20.0                 # N*m/rad, soft brace stiffness inside the range
C_IE = 0.4                  # N*m/(rad/s), brace damping
K_IE_STOP = 600.0           # N*m/rad, extra stiffness beyond the safe range

# ---------------------------------------------------------------------------
# Derived lengths and masses (do not edit; change the inputs above)
# ---------------------------------------------------------------------------
L_FOOT = FOOT_LENGTH_FRACTION * BODY_HEIGHT          # m
M_THIGH = MASS_FRAC_THIGH * BODY_MASS                # kg
M_SHANK = MASS_FRAC_SHANK * BODY_MASS                # kg
M_FOOT = MASS_FRAC_FOOT * BODY_MASS                  # kg
I_THIGH = M_THIGH * (K_GYR_THIGH * L_THIGH) ** 2     # kg*m^2
I_SHANK = M_SHANK * (K_GYR_SHANK * L_SHANK) ** 2     # kg*m^2
I_FOOT = M_FOOT * (K_GYR_FOOT * L_FOOT) ** 2         # kg*m^2
BODY_WEIGHT = BODY_MASS * GRAVITY                    # N

# Animation
ANIM_FRAMES = 100
ANIM_FPS = 20
ANIM_DPI = 90
