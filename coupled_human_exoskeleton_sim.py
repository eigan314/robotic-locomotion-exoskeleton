"""
Coupled human lower-limb + ankle exoskeleton gait simulation.

MODELING NOTES
--------------
This is a simplified research and engineering model. It is not a clinical
model and it must not be used to size a device for a patient.

Segment masses, centers of mass, radii of gyration, and the ground reaction
force are approximate unless you replace them with measured data.

The human limb and the exoskeleton are coupled but distinct systems.
The rotary actuator assists only ankle dorsiflexion/plantarflexion.
Inversion/eversion is not actuated. A passive brace restricts it.
The result of interest is how actuator torque changes the ankle torque
that the human must still produce.

Hip position is fixed. Joint angles are prescribed by Fourier fits, so this
is an inverse-dynamics simulation, not a forward simulation of walking.

SIGN CONVENTION
---------------
y  anterior, meters, positive forward
z  superior, meters, positive upward
Segment angles are positive counterclockwise in the y-z plane.
A positive moment on a segment increases that segment's absolute angle.

  H     hip flexion angle. H = 0 is a vertical thigh. Positive H moves
        the knee forward.
  K     knee flexion angle, relative. Positive K bends the knee.
  A     ankle angle from the supplied fit, after the +20 degree correction.
        With SIGN_ANKLE = +1 the kinematic map is phi = H - K - A, so an
        increase in A points the toe down.
  beta  absolute shank angle from the downward vertical.
  phi   absolute foot angle. phi = 0 is a horizontal foot, toe forward.
        Increasing phi lifts the toe.

Flip SIGN_HIP, SIGN_KNEE, or SIGN_ANKLE in config.py to change a sign
without editing the equation blocks below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import trapezoid

import config as cfg


# Project root is the folder that contains this script.
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def gait_angles(t, T_stride):
    """
    Hip, knee, and ankle angles from the supplied Fourier fits.

    Parameters
    ----------
    t : ndarray, s
        Time from the start of the stride.
    T_stride : float, s
        Stride period.

    Returns
    -------
    dict of ndarrays
        g              gait-cycle percentage, 0 to 100
        H, K, A        joint angles, rad, after the sign block and the
                       ankle zero correction
        H_deg, K_deg, A_deg
                       the same angles in degrees, for plots
    """
    # g is the percent of the gait cycle. It is dimensionless in the sense
    # "percent", and the fitted frequencies already have units of rad/percent.
    g = 100.0 * t / T_stride

    H_deg = _fourier(
        g,
        a0=25.28,
        w=0.03928,
        cos_coeff=(8.343, 4.463, -4.964, 2.062, 0.1402),
        sin_coeff=(-25.6, 3.979, 4.285, 1.46, 0.4413),
    )
    K_deg = _fourier(
        g,
        a0=24.61,
        w=0.06336,
        cos_coeff=(-0.6681, -13.39, -1.235, -0.8609, -0.148),
        sin_coeff=(-18.81, 8.829, 4.033, 0.5253, 0.9367),
    )
    # A_raw is the fitted series in degrees. The constant term is a0 = -0.69.
    A_raw_deg = _fourier(
        g,
        a0=-0.69,
        w=0.06122,
        cos_coeff=(1.087, -3.397, -1.707, 1.568, -0.4787, 0.7571, -0.09973, 0.3602),
        sin_coeff=(6.182, -6.986, 3.019, -1.766, -0.03357, -0.4707, -0.4509, -0.07662),
    )
    # Requested zero correction, degrees. Applied to the whole ankle signal.
    A_deg = A_raw_deg + cfg.ANKLE_ZERO_OFFSET_DEG

    # Convert to radians here, once. Every later trig call uses radians.
    H = cfg.SIGN_HIP * np.deg2rad(H_deg)
    K = cfg.SIGN_KNEE * np.deg2rad(K_deg)
    A = cfg.SIGN_ANKLE * np.deg2rad(A_deg)

    return {
        "g": g,
        "H": H,
        "K": K,
        "A": A,
        "H_deg": cfg.SIGN_HIP * H_deg,
        "K_deg": cfg.SIGN_KNEE * K_deg,
        "A_deg": cfg.SIGN_ANKLE * A_deg,
        "A_raw_deg": A_raw_deg,
    }


def _fourier(g, a0, w, cos_coeff, sin_coeff):
    """
    H(g), K(g), or A(g).

    a0            constant term, degrees
    w             fundamental frequency, rad per percent of gait
    cos_coeff[n]  cosine coefficient of harmonic n+1, degrees
    sin_coeff[n]  sine coefficient of harmonic n+1, degrees
    g             gait percentage, 0 to 100

    The argument of each trig function is (n * w * g) and is in radians.
    The returned angle is in degrees.
    """
    angle = np.full_like(g, a0, dtype=float)
    for n, (c_n, s_n) in enumerate(zip(cos_coeff, sin_coeff), start=1):
        angle = angle + c_n * np.cos(n * w * g) + s_n * np.sin(n * w * g)
    return angle


def forward_kinematics(H, K, A):
    """
    Joint and foot landmark positions with the hip fixed at the origin.

    H, K, A are in radians.

    beta = H - K
        absolute shank angle, rad. Knee flexion reduces beta.
    phi = H - K - A
        absolute foot angle, rad. With the default sign, increasing A
        decreases phi and points the toe down.

    y_*  anterior coordinate, m
    z_*  vertical coordinate, m
    """
    L_thigh = cfg.L_THIGH
    L_shank = cfg.L_SHANK
    L_foot = cfg.L_FOOT
    D5 = cfg.D5

    # Hip is the base. No pelvic translation is included.
    y_hip = np.zeros_like(H)
    z_hip = np.zeros_like(H)

    # Thigh, from the hip. At H = 0 the knee is directly below the hip.
    y_knee = L_thigh * np.sin(H)
    z_knee = -L_thigh * np.cos(H)

    # Shank absolute angle and ankle position.
    beta = H - K
    y_ankle = y_knee + L_shank * np.sin(beta)
    z_ankle = z_knee - L_shank * np.cos(beta)

    # Foot. D5 is the ankle height above the heel along the sole normal.
    phi = H - K - A
    y_heel = y_ankle - D5 * np.sin(phi)
    z_heel = z_ankle - D5 * np.cos(phi)
    y_toe = y_heel + L_foot * np.cos(phi)
    z_toe = z_heel + L_foot * np.sin(phi)

    return {
        "y_hip": y_hip,
        "z_hip": z_hip,
        "y_knee": y_knee,
        "z_knee": z_knee,
        "y_ankle": y_ankle,
        "z_ankle": z_ankle,
        "y_heel": y_heel,
        "z_heel": z_heel,
        "y_toe": y_toe,
        "z_toe": z_toe,
        "beta": beta,
        "phi": phi,
    }


def segment_com_kinematics(fk):
    """
    Segment center-of-mass positions.

    Thigh COM is COM_FRAC_THIGH of the way from the hip to the knee.
    Shank COM is COM_FRAC_SHANK of the way from the knee to the ankle.
    Foot COM is COM_FRAC_FOOT of the way from the heel to the toe.

    These fractions are assumptions. See config.py.
    """
    f_th = cfg.COM_FRAC_THIGH
    f_sh = cfg.COM_FRAC_SHANK
    f_ft = cfg.COM_FRAC_FOOT

    # The knee lies on the thigh axis, so scaling that vector places the COM.
    # This stays consistent with the hip sign without a second copy of the
    # hip equation.
    y_thigh = f_th * fk["y_knee"]
    z_thigh = f_th * fk["z_knee"]

    y_shank = fk["y_knee"] + f_sh * (fk["y_ankle"] - fk["y_knee"])
    z_shank = fk["z_knee"] + f_sh * (fk["z_ankle"] - fk["z_knee"])

    y_foot = fk["y_heel"] + f_ft * (fk["y_toe"] - fk["y_heel"])
    z_foot = fk["z_heel"] + f_ft * (fk["z_toe"] - fk["z_heel"])

    return {
        "y_thigh": y_thigh,
        "z_thigh": z_thigh,
        "y_shank": y_shank,
        "z_shank": z_shank,
        "y_foot": y_foot,
        "z_foot": z_foot,
    }


def compute_vel_acc(t, angles, fk, com):
    """
    Angular and linear velocity and acceleration.

    numpy.gradient is a central difference in the interior and a one-sided
    difference at the first and last samples. The fitted angles are not
    forced to be periodic, so the ends are not wrapped together.

    theta_dot   rad/s
    theta_ddot  rad/s^2
    v           m/s
    a           m/s^2
    """
    dt = float(t[1] - t[0])

    def grad(x):
        # Central difference in the interior, one-sided at the two endpoints.
        # The Fourier fits are not exactly periodic, so the first sample is
        # not used as the neighbor of the last sample. Wrapping them would
        # turn a degree or two of closure error into a false acceleration spike.
        return np.gradient(x, dt)

    H_dot, K_dot, A_dot = grad(angles["H"]), grad(angles["K"]), grad(angles["A"])
    H_ddot, K_ddot, A_ddot = grad(H_dot), grad(K_dot), grad(A_dot)

    # Absolute segment angles, consistent with forward_kinematics.
    # beta = H - K, so beta_dot = H_dot - K_dot.
    # phi = H - K - A, so phi_dot = H_dot - K_dot - A_dot.
    beta_dot = H_dot - K_dot
    beta_ddot = H_ddot - K_ddot
    phi_dot = H_dot - K_dot - A_dot
    phi_ddot = H_ddot - K_ddot - A_ddot

    linear = {}
    for name in ("thigh", "shank", "foot"):
        y = com[f"y_{name}"]
        z = com[f"z_{name}"]
        linear[f"vy_{name}"] = grad(y)
        linear[f"vz_{name}"] = grad(z)
        linear[f"ay_{name}"] = grad(linear[f"vy_{name}"])
        linear[f"az_{name}"] = grad(linear[f"vz_{name}"])

    out = {
        "H_dot": H_dot,
        "K_dot": K_dot,
        "A_dot": A_dot,
        "H_ddot": H_ddot,
        "K_ddot": K_ddot,
        "A_ddot": A_ddot,
        "beta_dot": beta_dot,
        "beta_ddot": beta_ddot,
        "phi_dot": phi_dot,
        "phi_ddot": phi_ddot,
    }
    out.update(linear)
    return out


def generate_grf(t, g, fk, grf_y=None, grf_z=None, cop_fraction=None):
    """
    Sagittal ground reaction force and center of pressure.

    Two sources are supported.

    A) Synthetic, used when grf_y and grf_z are None.
       Stance is g <= 60 percent. Swing force is exactly zero.
       Vertical force is a smooth double hump, scaled so that the mean
       vertical force over the stride equals half of body weight
       (one leg of a symmetric walker).
       Anterior/posterior force brakes in early stance and propels in
       late stance. Its peak is GRF_AP_PEAK_FRACTION * body weight.
       COP moves from the heel to the toe across stance.

    B) User arrays. Pass grf_y, grf_z, and optionally cop_fraction.
       cop_fraction is 0 at the heel and 1 at the toe. If it is omitted,
       the synthetic heel-to-toe ramp is used during stance.

    Fy  anterior force of the ground on the foot, N
    Fz  vertical force of the ground on the foot, N
    """
    stance = g <= (100.0 * cfg.STANCE_END_FRACTION)
    # s = 0 at heel contact and s = 1 at the stance/swing boundary.
    s = np.clip(g / (100.0 * cfg.STANCE_END_FRACTION), 0.0, 1.0)

    if cop_fraction is None:
        cop_fraction = s.copy()
    cop_fraction = np.where(stance, cop_fraction, np.nan)

    if grf_y is None or grf_z is None:
        grf_y, grf_z = _synthetic_grf(t, s, stance)

    # COP lies on the heel-toe segment. During swing the force is zero, so
    # the application point does not affect the moment. Park it on the heel
    # so the arrays stay finite for the inverse-dynamics products.
    cop_for_dynamics = np.where(np.isnan(cop_fraction), 0.0, cop_fraction)
    y_cop = fk["y_heel"] + cop_for_dynamics * (fk["y_toe"] - fk["y_heel"])
    z_cop = fk["z_heel"] + cop_for_dynamics * (fk["z_toe"] - fk["z_heel"])

    return {
        "GRF_y": np.asarray(grf_y, dtype=float),
        "GRF_z": np.asarray(grf_z, dtype=float),
        "y_cop": y_cop,
        "z_cop": z_cop,
        "cop_fraction": cop_fraction,
        "stance": stance,
    }


def _synthetic_grf(t, s, stance):
    """Double-hump vertical GRF and braking/propulsion shear. See generate_grf."""
    # Rises from 0 at heel contact and returns to 0 at toe-off, but stays
    # near 1 through the middle of stance. A full-stance sine window spends
    # too much time near zero, and the impulse constraint then forces an
    # unrealistically tall mid-stance peak.
    edge = 0.10
    window = np.ones_like(s)
    rise = s < edge
    fall = s > (1.0 - edge)
    window[rise] = 0.5 - 0.5 * np.cos(np.pi * s[rise] / edge)
    window[fall] = 0.5 - 0.5 * np.cos(np.pi * (1.0 - s[fall]) / edge)

    hump = (
        cfg.GRF_HUMP1_WEIGHT * np.exp(-0.5 * ((s - cfg.GRF_HUMP1_CENTER) / cfg.GRF_HUMP_SIGMA) ** 2)
        + cfg.GRF_HUMP2_WEIGHT * np.exp(-0.5 * ((s - cfg.GRF_HUMP2_CENTER) / cfg.GRF_HUMP_SIGMA) ** 2)
    )
    fz_shape = hump * window
    fz_shape = np.where(stance, fz_shape, 0.0)

    # Target: (1 / T) * integral(Fz dt) = body_weight / 2.
    # trapezoid returns the integral over the stride, N*s.
    impulse = trapezoid(fz_shape, t)
    target_impulse = 0.5 * cfg.BODY_WEIGHT * (t[-1] - t[0] + (t[1] - t[0]))
    Fz = fz_shape * (target_impulse / impulse)

    # -sin(2*pi*s) is negative in early stance (braking, force on the foot
    # directed posterior) and positive in late stance (propulsion).
    fy_shape = -np.sin(2.0 * np.pi * s) * window
    fy_shape = np.where(stance, fy_shape, 0.0)
    peak = np.max(np.abs(fy_shape))
    Fy = fy_shape * (cfg.GRF_AP_PEAK_FRACTION * cfg.BODY_WEIGHT / peak)
    return Fy, Fz


def human_inverse_dynamics(fk, com, kin, grf):
    """
    Planar Newton-Euler inverse dynamics, solved distal to proximal.

    For every segment the unknowns are the proximal joint force and the
    proximal joint moment. Distal loads are already known.

    Force balance, written on the segment:
        sum Fy = m * a_y
        sum Fz = m * a_z
    with weight (0, -m*g) included in the sum.

    Moment balance about the segment COM:
        sum M = I_com * alpha
    Positive M increases the segment absolute angle
    (H for the thigh, beta for the shank, phi for the foot).

    The 2-D moment of a force about the COM is
        M = r_y * F_z - r_z * F_y
    where r is the vector from the COM to the force, in meters,
    and F is the force on the segment, in newtons. M is in N*m.

    tau_ankle_required
        moment of the shank on the foot, N*m. This is the net ankle
        moment required by the prescribed motion. It will be split
        later into human muscle moment and exoskeleton moment.
        Positive tau_ankle_required increases phi (toes up).
    tau_knee_required
        moment of the thigh on the shank, N*m.
        Positive increases beta. With the hip held, that is knee extension,
        because beta = H - K.
    tau_hip_required
        moment of the pelvis on the thigh, N*m.
        Positive increases H (hip flexion).

    Exoskeleton interface forces are not inside this balance. They are
    estimated afterward from tau_exo. See interface_forces().
    """
    g_acc = cfg.GRAVITY
    m_f, m_s, m_t = cfg.M_FOOT, cfg.M_SHANK, cfg.M_THIGH
    I_f, I_s, I_t = cfg.I_FOOT, cfg.I_SHANK, cfg.I_THIGH

    GRF_y = grf["GRF_y"]
    GRF_z = grf["GRF_z"]

    # ------------------------------------------------------------------
    # Foot. Distal load is the ground reaction. Proximal joint is the ankle.
    # ------------------------------------------------------------------
    # F_ankle is the force of the shank on the foot.
    #   F_ankle_y + GRF_y                 = m_f * a_y
    #   F_ankle_z + GRF_z - m_f * g_acc   = m_f * a_z
    F_ankle_y = m_f * kin["ay_foot"] - GRF_y
    F_ankle_z = m_f * kin["az_foot"] - GRF_z + m_f * g_acc

    # r_ankle: COM -> ankle. r_cop: COM -> center of pressure.
    r_ank_y = fk["y_ankle"] - com["y_foot"]
    r_ank_z = fk["z_ankle"] - com["z_foot"]
    r_cop_y = grf["y_cop"] - com["y_foot"]
    r_cop_z = grf["z_cop"] - com["z_foot"]

    M_ankle_force = r_ank_y * F_ankle_z - r_ank_z * F_ankle_y
    M_grf = r_cop_y * GRF_z - r_cop_z * GRF_y

    # tau_ankle + M_ankle_force + M_grf = I_f * alpha_foot
    # Weight acts at the COM, so it has no moment about the COM.
    tau_ankle = I_f * kin["phi_ddot"] - M_ankle_force - M_grf

    # ------------------------------------------------------------------
    # Shank. Distal load is minus the ankle load. Proximal joint is the knee.
    # ------------------------------------------------------------------
    #   F_knee_y - F_ankle_y               = m_s * a_y
    #   F_knee_z - F_ankle_z - m_s * g_acc = m_s * a_z
    F_knee_y = m_s * kin["ay_shank"] + F_ankle_y
    F_knee_z = m_s * kin["az_shank"] + F_ankle_z + m_s * g_acc

    r_knee_y = fk["y_knee"] - com["y_shank"]
    r_knee_z = fk["z_knee"] - com["z_shank"]
    r_as_y = fk["y_ankle"] - com["y_shank"]
    r_as_z = fk["z_ankle"] - com["z_shank"]

    # Force of the foot on the shank is -F_ankle.
    # Moment of the foot on the shank is -tau_ankle.
    M_knee_force = r_knee_y * F_knee_z - r_knee_z * F_knee_y
    M_ankle_on_shank = r_as_y * (-F_ankle_z) - r_as_z * (-F_ankle_y)

    # tau_knee - tau_ankle + M_knee_force + M_ankle_on_shank = I_s * alpha_shank
    tau_knee = I_s * kin["beta_ddot"] + tau_ankle - M_knee_force - M_ankle_on_shank

    # ------------------------------------------------------------------
    # Thigh. Distal load is minus the knee load. Proximal joint is the hip.
    # ------------------------------------------------------------------
    #   F_hip_y - F_knee_y               = m_t * a_y
    #   F_hip_z - F_knee_z - m_t * g_acc = m_t * a_z
    F_hip_y = m_t * kin["ay_thigh"] + F_knee_y
    F_hip_z = m_t * kin["az_thigh"] + F_knee_z + m_t * g_acc

    r_hip_y = fk["y_hip"] - com["y_thigh"]
    r_hip_z = fk["z_hip"] - com["z_thigh"]
    r_kt_y = fk["y_knee"] - com["y_thigh"]
    r_kt_z = fk["z_knee"] - com["z_thigh"]

    # Force of the shank on the thigh is -F_knee.
    # Moment of the shank on the thigh is -tau_knee.
    M_hip_force = r_hip_y * F_hip_z - r_hip_z * F_hip_y
    M_knee_on_thigh = r_kt_y * (-F_knee_z) - r_kt_z * (-F_knee_y)

    # tau_hip - tau_knee + M_hip_force + M_knee_on_thigh = I_t * alpha_thigh
    tau_hip = I_t * kin["H_ddot"] + tau_knee - M_hip_force - M_knee_on_thigh

    return {
        "tau_ankle_required": tau_ankle,
        "tau_knee_required": tau_knee,
        "tau_hip_required": tau_hip,
        "F_ankle_y": F_ankle_y,
        "F_ankle_z": F_ankle_z,
        "F_knee_y": F_knee_y,
        "F_knee_z": F_knee_z,
        "F_hip_y": F_hip_y,
        "F_hip_z": F_hip_z,
    }


def exo_controller(theta_ankle, theta_dot_ankle, tau_ankle_required):
    """
    Ankle rotary actuator.

    theta_exo is locked to the anatomical ankle angle:
        theta_exo = theta_ankle = A
    because the brace is strapped across the ankle and this simulation
    prescribes the human kinematics.

    Assistance mode
        tau_exo = assistance_ratio * tau_ankle_required
        assistance_ratio = 0.30 means the motor produces 30 percent of the
        net ankle moment. The human produces the rest.

    PD mode
        tau_exo = Kp * (theta_des - theta_ankle)
                + Kd * (theta_dot_des - theta_dot_ankle)
        theta_des = theta_ankle_reference + offset.
        If the offset is zero and the limb is already on the reference,
        this torque is approximately zero. That is expected: a position
        controller has nothing to do when the angle is prescribed.
        Use assistance mode to study torque sharing.

    Both modes are then limited to +/- TAU_MAX.

    The motor torque and current follow the gear model
        tau_exo = gear_ratio * Kt * i_motor
        tau_motor = tau_exo / gear_ratio
        i_motor = tau_motor / Kt
    gear_ratio is dimensionless, Kt is N*m/A, i_motor is A.

    Human moment after assistance, which is a primary output:
        tau_human_ankle = tau_ankle_required - tau_exo
    """
    if cfg.CONTROLLER_MODE == "assistance":
        tau_cmd = cfg.ASSISTANCE_RATIO * tau_ankle_required
    elif cfg.CONTROLLER_MODE == "pd":
        theta_des = theta_ankle + np.deg2rad(cfg.THETA_DES_OFFSET_DEG)
        # The reference is the prescribed ankle angle plus a constant, so
        # its derivative equals the ankle angular velocity.
        theta_dot_des = theta_dot_ankle
        tau_cmd = cfg.KP * (theta_des - theta_ankle) + cfg.KD * (theta_dot_des - theta_dot_ankle)
    else:
        raise ValueError("CONTROLLER_MODE must be 'assistance' or 'pd'.")

    tau_exo = np.clip(tau_cmd, -cfg.TAU_MAX, cfg.TAU_MAX)
    tau_human = tau_ankle_required - tau_exo

    # Actuator effort on the motor side of the gearbox.
    tau_motor = tau_exo / cfg.GEAR_RATIO
    i_motor = tau_motor / cfg.MOTOR_TORQUE_CONSTANT

    return {
        "tau_exo": tau_exo,
        "tau_human_ankle": tau_human,
        "tau_motor": tau_motor,
        "i_motor": i_motor,
        "saturated": np.abs(tau_cmd) > cfg.TAU_MAX + 1e-9,
    }


def inversion_eversion_constraint(t, T_stride):
    """
    Passive inversion/eversion brace. This is not a second actuator.

    theta_IE is a placeholder waveform, because the sagittal-plane gait
    fit does not contain a frontal-plane measurement. Replace it with
    measured inversion/eversion before treating the torque as real.

    Soft constraint inside the safe range, and a much stiffer stop outside:
        tau_IE = -k_IE * theta_IE - c_IE * theta_IE_dot
                 - k_stop * sign(theta_IE) * excess
        excess = max(|theta_IE| - theta_limit, 0)

    theta_IE      rad, inversion positive by declaration of this placeholder
    theta_IE_dot  rad/s
    tau_IE        N*m, moment of the brace on the human. The negative
                  stiffness term restores the angle toward zero.
    theta_limit   +/- 5 degrees by default.
    """
    g = 100.0 * t / T_stride
    # Placeholder frontal-plane angle. Amplitudes are in degrees in config.
    theta_deg = (
        cfg.IE_AMP1_DEG * np.sin(2.0 * np.pi * g / 100.0)
        + cfg.IE_AMP2_DEG * np.sin(4.0 * np.pi * g / 100.0)
    )
    theta = np.deg2rad(theta_deg)

    dt = float(t[1] - t[0])
    theta_dot = np.gradient(theta, dt)

    limit = np.deg2rad(cfg.THETA_IE_LIMIT_DEG)
    excess = np.maximum(np.abs(theta) - limit, 0.0)

    # Restoring spring-damper, plus the hard-range term that is zero
    # while |theta_IE| stays inside the safe range.
    tau = (
        -cfg.K_IE * theta
        - cfg.C_IE * theta_dot
        - cfg.K_IE_STOP * np.sign(theta) * excess
    )
    return {
        "theta_IE": theta,
        "theta_IE_deg": theta_deg,
        "theta_IE_dot": theta_dot,
        "tau_IE": tau,
        "excess": excess,
    }


def interface_forces(tau_exo, beta):
    """
    Simplified human-exoskeleton contact loads.

    This is not a contact solution and it is not fed back into the inverse
    dynamics. The actuator moment is converted to a force pair with one
    equivalent moment arm:

        F = tau_exo / r_exo

    r_exo is in meters, tau_exo is in N*m, F is in newtons.

    The force on the foot plate is perpendicular to the shank (the upright),
    and the calf cuff receives the equal and opposite force:

        F_cuff = - F_foot

    The human is assigned +F at each interface in the sense above, and the
    exoskeleton is assigned the opposite. Because the pair is equal and
    opposite, it produces no net force on the combined human+device system.
    It produces a pure couple equal to tau_exo only if the geometric
    separation of the two forces equals r_exo. Here r_exo is an effective
    arm (default 5 cm), not the cuff-to-ankle distance, so the couple of
    these two arrows is a visualization of the load path rather than a
    second calculation of the actuator moment.
    """
    r = cfg.R_EXO
    # Unit vector perpendicular to the shank. At beta = 0 (vertical shank)
    # this points anterior, +y.
    perp_y = np.cos(beta)
    perp_z = np.sin(beta)

    F_foot_y = (tau_exo / r) * perp_y
    F_foot_z = (tau_exo / r) * perp_z
    F_cuff_y = -F_foot_y
    F_cuff_z = -F_foot_z
    F_mag = np.hypot(F_foot_y, F_foot_z)

    return {
        "F_foot_y": F_foot_y,
        "F_foot_z": F_foot_z,
        "F_cuff_y": F_cuff_y,
        "F_cuff_z": F_cuff_z,
        "F_interface": F_mag,
    }


def exoskeleton_geometry(fk):
    """
    Cuff, upright, and foot-plate landmarks used for plotting.

    The upright is drawn a few centimeters anterior to the shank so it
    does not sit on top of the bone. The foot plate is drawn slightly
    soleward of the heel-toe line. Those offsets are graphical.
    """
    frac = cfg.CUFF_FRACTION_FROM_ANKLE
    # Point on the shank, frac of the way from the ankle toward the knee.
    y_cuff_bone = fk["y_ankle"] + frac * (fk["y_knee"] - fk["y_ankle"])
    z_cuff_bone = fk["z_ankle"] + frac * (fk["z_knee"] - fk["z_ankle"])

    # Anterior offset of the upright. perp is the same unit vector used
    # for the interface force at beta, but the offset is a constant length.
    perp_y = np.cos(fk["beta"])
    perp_z = np.sin(fk["beta"])
    y_cuff = y_cuff_bone + cfg.EXO_UPRIGHT_OFFSET * perp_y
    z_cuff = z_cuff_bone + cfg.EXO_UPRIGHT_OFFSET * perp_z
    y_act = fk["y_ankle"] + cfg.EXO_UPRIGHT_OFFSET * perp_y
    z_act = fk["z_ankle"] + cfg.EXO_UPRIGHT_OFFSET * perp_z

    # Soleward unit vector. At phi = 0 it is (0, -1), straight down.
    sole_y = -np.sin(fk["phi"])
    sole_z = -np.cos(fk["phi"])
    y_plate_heel = fk["y_heel"] + cfg.EXO_PLATE_OFFSET * sole_y
    z_plate_heel = fk["z_heel"] + cfg.EXO_PLATE_OFFSET * sole_z
    y_plate_toe = fk["y_toe"] + cfg.EXO_PLATE_OFFSET * sole_y
    z_plate_toe = fk["z_toe"] + cfg.EXO_PLATE_OFFSET * sole_z

    return {
        "y_cuff": y_cuff,
        "z_cuff": z_cuff,
        "y_actuator": y_act,
        "z_actuator": z_act,
        "y_plate_heel": y_plate_heel,
        "z_plate_heel": z_plate_heel,
        "y_plate_toe": y_plate_toe,
        "z_plate_toe": z_plate_toe,
    }


def plot_results(t, angles, fk, kin, grf, dyn, exo, ie, forces):
    """Write the twelve requested figures into outputs/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    toe_off = cfg.STANCE_END_FRACTION * cfg.T_STRIDE

    def mark_toe_off(ax):
        ax.axvline(toe_off, color="0.6", lw=0.8, ls="--", label="toe-off")

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / name, dpi=120)
        plt.close(fig)

    # 1. Joint angles.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, angles["H_deg"], label="hip")
    ax.plot(t, angles["K_deg"], label="knee")
    ax.plot(t, angles["A_deg"], label="ankle (after +20 deg)")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Hip, knee, and ankle angle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "01_joint_angles.png")

    # 2. Angular velocity. Stored in rad/s, plotted in deg/s.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, np.rad2deg(kin["H_dot"]), label="hip")
    ax.plot(t, np.rad2deg(kin["K_dot"]), label="knee")
    ax.plot(t, np.rad2deg(kin["A_dot"]), label="ankle")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular velocity (deg/s)")
    ax.set_title("Hip, knee, and ankle angular velocity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "02_angular_velocity.png")

    # 3. Angular acceleration.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, np.rad2deg(kin["H_ddot"]), label="hip")
    ax.plot(t, np.rad2deg(kin["K_ddot"]), label="knee")
    ax.plot(t, np.rad2deg(kin["A_ddot"]), label="ankle")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular acceleration (deg/s^2)")
    ax.set_title("Hip, knee, and ankle angular acceleration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "03_angular_acceleration.png")

    # 4. Heel and toe paths.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fk["y_heel"], fk["z_heel"], label="heel")
    ax.plot(fk["y_toe"], fk["z_toe"], label="toe")
    ax.plot(fk["y_ankle"], fk["z_ankle"], label="ankle", alpha=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("anterior y (m)")
    ax.set_ylabel("vertical z (m)")
    ax.set_title("Heel and toe trajectories (hip fixed at origin)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "04_heel_toe_trajectories.png")

    # 5. Ground reaction force.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, grf["GRF_z"], label="Fz vertical")
    ax.plot(t, grf["GRF_y"], label="Fy anterior/posterior")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force (N)")
    ax.set_title("Ground reaction force on the foot")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "05_ground_reaction_force.png")

    # 6. Required biological joint moments from inverse dynamics.
    # These are the net moments. At the ankle the exoskeleton has not yet
    # been subtracted; that split is in figures 7-9.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, dyn["tau_hip_required"], label="hip")
    ax.plot(t, dyn["tau_knee_required"], label="knee")
    ax.plot(t, dyn["tau_ankle_required"], label="ankle")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("moment (N*m)")
    ax.set_title("Required joint moments (positive increases segment angle)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "06_required_joint_torques.png")

    # 7. Actuator torque.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, exo["tau_exo"], color="C0", label="exoskeleton")
    ax.axhline(cfg.TAU_MAX, color="0.5", lw=0.7, ls=":")
    ax.axhline(-cfg.TAU_MAX, color="0.5", lw=0.7, ls=":")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("moment (N*m)")
    ax.set_title(f"Exoskeleton ankle torque ({cfg.CONTROLLER_MODE} mode)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "07_exoskeleton_torque.png")

    # 8. Human ankle torque remaining after assistance.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, exo["tau_human_ankle"], color="C3", label="human ankle")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("moment (N*m)")
    ax.set_title("Human ankle torque after exoskeleton assistance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "08_human_ankle_torque.png")

    # 9. Comparison of the three ankle moments.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, dyn["tau_ankle_required"], color="k", label="required ankle")
    ax.plot(t, exo["tau_exo"], color="C0", label="exoskeleton")
    ax.plot(t, exo["tau_human_ankle"], color="C3", label="remaining human")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("moment (N*m)")
    ax.set_title("Ankle torque sharing")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "09_ankle_torque_comparison.png")

    # 10. Interface force components and magnitude.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, forces["F_cuff_y"], label="cuff Fy")
    ax.plot(t, forces["F_cuff_z"], label="cuff Fz")
    ax.plot(t, forces["F_foot_y"], label="foot plate Fy")
    ax.plot(t, forces["F_foot_z"], label="foot plate Fz")
    ax.plot(t, forces["F_interface"], color="k", lw=1.5, label="magnitude")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force (N)")
    ax.set_title("Human-exoskeleton interface force (simplified tau / r)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    finish(fig, "10_interface_force.png")

    # 11. Inversion/eversion angle and brace torque.
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(t, ie["theta_IE_deg"], color="C0", label="IE angle")
    ax1.axhline(cfg.THETA_IE_LIMIT_DEG, color="C0", lw=0.7, ls="--")
    ax1.axhline(-cfg.THETA_IE_LIMIT_DEG, color="C0", lw=0.7, ls="--")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("inversion/eversion angle (deg)", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(t, ie["tau_IE"], color="C3", label="brace torque")
    ax2.set_ylabel("brace torque (N*m)", color="C3")
    ax1.set_title("Inversion/eversion angle and passive brace torque")
    ax1.grid(True, alpha=0.3)
    finish(fig, "11_inversion_eversion.png")

    # 12. Center-of-pressure progression along the foot.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, grf["cop_fraction"] * cfg.L_FOOT, color="C2", label="COP from heel")
    mark_toe_off(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("COP distance from heel (m)")
    ax.set_title("Foot center-of-pressure progression during stance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, "12_center_of_pressure.png")


def animate_system(t, angles, fk, grf, dyn, exo, forces, geo):
    """
    Sagittal animation of the limb, the brace, and the force arrows.

    Saved as coupled_human_exoskeleton.gif in the project folder.
    """
    n = len(t)
    frame_idx = np.linspace(0, n - 1, cfg.ANIM_FRAMES, dtype=int)

    # Fixed view so the limb does not jump around between frames.
    ys = np.concatenate([
        fk["y_hip"], fk["y_knee"], fk["y_ankle"], fk["y_heel"], fk["y_toe"],
        geo["y_cuff"], geo["y_plate_toe"],
    ])
    zs = np.concatenate([
        fk["z_hip"], fk["z_knee"], fk["z_ankle"], fk["z_heel"], fk["z_toe"],
        geo["z_cuff"], geo["z_plate_heel"],
    ])
    pad = 0.25
    xlim = (np.min(ys) - pad, np.max(ys) + pad)
    ylim = (np.min(zs) - pad, np.max(zs) + 0.15)

    # Arrow lengths in meters. Chosen so a body-weight force is visible
    # and does not cover the whole leg.
    force_scale = 0.00035  # m/N

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    def update(i):
        ax.clear()
        k = frame_idx[i]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("anterior y (m)")
        ax.set_ylabel("vertical z (m)")
        ax.set_title("Coupled human limb and ankle exoskeleton")
        ax.grid(True, alpha=0.25)

        # Reference line at the lowest sole point. The foot is not
        # constrained to this line. The hip is fixed.
        z_ref = np.min(np.minimum(fk["z_heel"], fk["z_toe"]))
        ax.axhline(z_ref, color="0.75", lw=1.0, ls="--")

        # Human segments.
        ax.plot(
            [fk["y_hip"][k], fk["y_knee"][k]],
            [fk["z_hip"][k], fk["z_knee"][k]],
            color="k", lw=3, solid_capstyle="round", label="thigh",
        )
        ax.plot(
            [fk["y_knee"][k], fk["y_ankle"][k]],
            [fk["z_knee"][k], fk["z_ankle"][k]],
            color="0.25", lw=3, solid_capstyle="round", label="shank",
        )
        ax.plot(
            [fk["y_heel"][k], fk["y_toe"][k]],
            [fk["z_heel"][k], fk["z_toe"][k]],
            color="0.45", lw=3, solid_capstyle="round", label="foot",
        )

        # Exoskeleton upright, cuff, and foot plate.
        ax.plot(
            [geo["y_actuator"][k], geo["y_cuff"][k]],
            [geo["z_actuator"][k], geo["z_cuff"][k]],
            color="C0", lw=4, solid_capstyle="round", alpha=0.9, label="upright",
        )
        ax.plot(
            [geo["y_plate_heel"][k], geo["y_plate_toe"][k]],
            [geo["z_plate_heel"][k], geo["z_plate_toe"][k]],
            color="C0", lw=4, solid_capstyle="round", alpha=0.9, label="foot plate",
        )
        ax.plot(geo["y_cuff"][k], geo["z_cuff"][k], "s", color="C0", ms=8, label="cuff")

        # Joints and foot landmarks.
        ax.plot(fk["y_hip"][k], fk["z_hip"][k], "o", color="k", ms=7, label="hip")
        ax.plot(fk["y_knee"][k], fk["z_knee"][k], "o", color="C1", ms=6, label="knee")
        ax.plot(fk["y_ankle"][k], fk["z_ankle"][k], "o", color="C3", ms=6, label="ankle")
        ax.plot(fk["y_heel"][k], fk["z_heel"][k], "o", color="C2", ms=4, label="heel")
        ax.plot(fk["y_toe"][k], fk["z_toe"][k], "o", color="C4", ms=4, label="toe")

        _force_arrow(
            ax,
            grf["y_cop"][k], grf["z_cop"][k],
            grf["GRF_y"][k], grf["GRF_z"][k],
            force_scale, "C2",
        )
        _force_arrow(
            ax,
            fk["y_ankle"][k], fk["z_ankle"][k],
            dyn["F_ankle_y"][k], dyn["F_ankle_z"][k],
            force_scale, "C1",
        )
        _torque_arc(ax, geo["y_actuator"][k], geo["z_actuator"][k], exo["tau_exo"][k])

        # Interface arrows are drawn smaller because tau/r is a large
        # equivalent force from a short moment arm.
        _force_arrow(
            ax, geo["y_cuff"][k], geo["z_cuff"][k],
            forces["F_cuff_y"][k], forces["F_cuff_z"][k],
            force_scale * 0.35, "C0",
        )

        text = (
            f"time  {t[k]:5.3f} s\n"
            f"gait  {angles['g'][k]:5.1f} %\n"
            f"hip   {angles['H_deg'][k]:6.1f} deg\n"
            f"knee  {angles['K_deg'][k]:6.1f} deg\n"
            f"ankle {angles['A_deg'][k]:6.1f} deg\n"
            f"tau required  {dyn['tau_ankle_required'][k]:7.1f} N·m\n"
            f"tau exo       {exo['tau_exo'][k]:7.1f} N·m\n"
            f"tau human     {exo['tau_human_ankle'][k]:7.1f} N·m"
        )
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
    gif_path = ROOT / "coupled_human_exoskeleton.gif"
    anim.save(gif_path, writer=PillowWriter(fps=cfg.ANIM_FPS), dpi=cfg.ANIM_DPI)
    plt.close(fig)
    return gif_path


def _force_arrow(ax, y, z, Fy, Fz, scale, color):
    """Draw a force as an arrow. scale has units m/N."""
    if np.hypot(Fy, Fz) < 1.0:
        return
    ax.annotate(
        "",
        xy=(y + scale * Fy, z + scale * Fz),
        xytext=(y, z),
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6},
    )


def _torque_arc(ax, y, z, tau):
    """
    Small arc at the actuator. Arc length grows with |tau| and the
    direction follows the sign. tau is in N*m. The arc is a drawing,
    not a length-scaled force.
    """
    span = np.clip(abs(tau) / 80.0, 0.15, 1.1)
    direction = 1.0 if tau >= 0.0 else -1.0
    start = -0.4
    psi = np.linspace(start, start + direction * span, 30)
    radius = 0.07  # m, graphical
    ax.plot(y + radius * np.cos(psi), z + radius * np.sin(psi), color="C3", lw=1.8)
    ax.annotate(
        "",
        xy=(y + radius * np.cos(psi[-1]), z + radius * np.sin(psi[-1])),
        xytext=(y + radius * np.cos(psi[-3]), z + radius * np.sin(psi[-3])),
        arrowprops={"arrowstyle": "->", "color": "C3", "lw": 1.6},
    )


def run_static_sign_check():
    """
    Three checks that do not use the gait fit.

    1) A static leg loaded by GRF equal to the leg weight has zero hip
       force. The ground carries the leg, and the pelvis does not have to.
    2) If that same GRF passes through the foot COM, the ankle moment is
       the moment needed to support the proximal segments, not a COM offset
       of the foot weight. The dedicated foot check is separate.
    3) Foot alone, GRF equal to foot weight, COP 0.10 m anterior to the
       foot COM: the ground pushes up in front of the COM and tends to
       lift the toe. The required ankle moment is therefore negative
       (a plantarflexion moment in this sign convention) and equals
       -0.10 * m_foot * g.
    """
    n = 8
    z = np.zeros(n)
    H = z.copy()
    K = z.copy()
    A = z.copy()
    fk = forward_kinematics(H, K, A)
    # forward_kinematics does not store H. segment_com uses the knee vector,
    # which is correct. Inverse dynamics needs COM positions only.
    fk_for_com = dict(fk)
    com = segment_com_kinematics(fk_for_com)

    kin = {key: z.copy() for key in (
        "ay_foot", "az_foot", "ay_shank", "az_shank", "ay_thigh", "az_thigh",
        "phi_ddot", "beta_ddot", "H_ddot",
    )}

    m_leg = cfg.M_FOOT + cfg.M_SHANK + cfg.M_THIGH
    grf = {
        "GRF_y": z.copy(),
        "GRF_z": np.full(n, m_leg * cfg.GRAVITY),
        "y_cop": com["y_foot"].copy(),
        "z_cop": com["z_foot"].copy(),
    }
    dyn = human_inverse_dynamics(fk, com, kin, grf)
    hip_force = np.max(np.abs(dyn["F_hip_y"])) + np.max(np.abs(dyn["F_hip_z"]))
    if hip_force > 1e-6:
        raise RuntimeError(f"Static hip-force check failed: residual {hip_force:.3e} N")

    # Knee must carry the thigh weight downward onto the shank: F_knee_z = -m_t*g.
    knee_err = np.max(np.abs(dyn["F_knee_z"] + cfg.M_THIGH * cfg.GRAVITY))
    if knee_err > 1e-6:
        raise RuntimeError(f"Static knee-force check failed: residual {knee_err:.3e} N")

    # Foot-only moment check. COP is 0.10 m anterior to the foot COM.
    grf_foot = {
        "GRF_y": z.copy(),
        "GRF_z": np.full(n, cfg.M_FOOT * cfg.GRAVITY),
        "y_cop": com["y_foot"] + 0.10,
        "z_cop": com["z_foot"].copy(),
    }
    dyn_foot = human_inverse_dynamics(fk, com, kin, grf_foot)
    expected = -0.10 * cfg.M_FOOT * cfg.GRAVITY
    moment_err = np.max(np.abs(dyn_foot["tau_ankle_required"] - expected))
    if moment_err > 1e-6:
        raise RuntimeError(
            f"Static ankle-moment check failed: got {dyn_foot['tau_ankle_required'][0]:.6f}, "
            f"expected {expected:.6f}"
        )
    return "passed"


def _peak_abs(x):
    """Largest absolute value, and the sample where it occurs."""
    k = int(np.argmax(np.abs(x)))
    return float(x[k]), k


def print_summary(t, angles, fk, kin, grf, dyn, exo, ie, forces, sign_check):
    """Console report requested for the coupled simulation."""
    tau_h = dyn["tau_hip_required"]
    tau_k = dyn["tau_knee_required"]
    tau_a = dyn["tau_ankle_required"]
    tau_e = exo["tau_exo"]
    tau_human = exo["tau_human_ankle"]

    hip_max, _ = _peak_abs(tau_h)
    knee_max, _ = _peak_abs(tau_k)
    ankle_max, _ = _peak_abs(tau_a)
    exo_max, _ = _peak_abs(tau_e)
    human_max, _ = _peak_abs(tau_human)
    iface_max = float(np.max(forces["F_interface"]))
    ie_max = float(np.max(np.abs(ie["theta_IE"])))
    brace_max, _ = _peak_abs(ie["tau_IE"])

    reduction = 100.0 * (1.0 - human_max / ankle_max)

    # Mean vertical GRF should be near half body weight for the synthetic model.
    mean_fz = float(np.mean(grf["GRF_z"]))
    half_weight = 0.5 * cfg.BODY_WEIGHT

    # End-to-start jump of the fitted angles. The series are not constrained
    # to be periodic, so a jump here is a property of the fit.
    jump_H = float(angles["H_deg"][0] - angles["H_deg"][-1])
    jump_K = float(angles["K_deg"][0] - angles["K_deg"][-1])
    jump_A = float(angles["A_deg"][0] - angles["A_deg"][-1])

    print("Coupled human + ankle exoskeleton simulation")
    print(f"  controller                 {cfg.CONTROLLER_MODE}")
    print(f"  stride duration            {cfg.T_STRIDE:.2f} s")
    print(f"  static sign check          {sign_check}")
    print()
    print(f"  maximum hip torque                         {abs(hip_max):8.2f} N*m")
    print(f"  maximum knee torque                        {abs(knee_max):8.2f} N*m")
    print(f"  maximum ankle torque required              {abs(ankle_max):8.2f} N*m")
    print(f"  maximum exoskeleton torque                 {abs(exo_max):8.2f} N*m")
    print(f"  maximum remaining human ankle torque       {abs(human_max):8.2f} N*m")
    print(f"  maximum exoskeleton interface force        {iface_max:8.2f} N")
    print(f"  maximum inversion angle                    {np.rad2deg(ie_max):8.2f} deg")
    print(f"  maximum brace inversion/eversion torque    {abs(brace_max):8.2f} N*m")
    print()
    print(f"  reduction_percent                          {reduction:8.2f} %")
    print()
    print("  signed ranges (min, max)")
    print(f"    hip torque     {tau_h.min():8.2f} to {tau_h.max():8.2f} N*m")
    print(f"    knee torque    {tau_k.min():8.2f} to {tau_k.max():8.2f} N*m")
    print(f"    ankle required {tau_a.min():8.2f} to {tau_a.max():8.2f} N*m")
    print(f"    exo torque     {tau_e.min():8.2f} to {tau_e.max():8.2f} N*m")
    print(f"    human ankle    {tau_human.min():8.2f} to {tau_human.max():8.2f} N*m")
    print()
    print("  diagnostics")
    print(f"    hip angle                  {angles['H_deg'].min():6.1f} to {angles['H_deg'].max():6.1f} deg")
    print(f"    knee angle                 {angles['K_deg'].min():6.1f} to {angles['K_deg'].max():6.1f} deg")
    print(f"    ankle angle                {angles['A_deg'].min():6.1f} to {angles['A_deg'].max():6.1f} deg")
    print(f"    mean Fz                    {mean_fz:8.1f} N   (target {half_weight:.1f} N)")
    print(f"    peak Fz                    {grf['GRF_z'].max():8.1f} N   ({grf['GRF_z'].max() / cfg.BODY_WEIGHT:.2f} body weight)")
    print(f"    angle jump at cycle wrap   hip {jump_H:.2f} deg, knee {jump_K:.2f} deg, ankle {jump_A:.2f} deg")
    print(f"    actuator samples saturated {int(np.sum(exo['saturated']))}")
    print(f"    peak motor current         {np.max(np.abs(exo['i_motor'])):.2f} A")
    # Quasi-static ankle moment: moment of GRF about the ankle, N*m.
    # Inertial terms and foot weight are left out, so this is a check on
    # the size of the ground-reaction couple, not a second inverse dynamics.
    r_y = grf["y_cop"] - fk["y_ankle"]
    r_z = grf["z_cop"] - fk["z_ankle"]
    m_grf_ankle = r_y * grf["GRF_z"] - r_z * grf["GRF_y"]

    print(f"    peak ankle torque          {abs(ankle_max) / cfg.BODY_MASS:.2f} N*m/kg")
    print(f"    foot angle phi             {np.rad2deg(fk['phi'].min()):6.1f} to {np.rad2deg(fk['phi'].max()):6.1f} deg")
    print(f"    max |COM acceleration|     foot {np.max(np.hypot(kin['ay_foot'], kin['az_foot'])):6.1f} m/s^2")
    print(f"    GRF moment about ankle     {np.min(m_grf_ankle):8.1f} to {np.max(m_grf_ankle):8.1f} N*m")
    print(f"    toe anterior position      {fk['y_toe'].min():6.2f} to {fk['y_toe'].max():6.2f} m")


def _crop(result, sl):
    """Slice every time-series array. Scalars are left alone."""
    return {key: (value[sl] if isinstance(value, np.ndarray) else value) for key, value in result.items()}


def main():
    sign_check = run_static_sign_check()

    n = int(round(cfg.T_STRIDE / cfg.DT))
    dt = cfg.DT
    # Two extra samples on each side. numpy.gradient is one-sided at the
    # ends of whatever array it is given, and a second derivative needs two
    # samples of padding before every reported sample is a central difference.
    # The Fourier formulas are evaluated at those extra times. They are not
    # copies of the opposite end of the cycle.
    t_core = np.arange(n) * dt
    t = np.concatenate((t_core[0] + dt * np.array([-2.0, -1.0]), t_core, t_core[-1] + dt * np.array([1.0, 2.0])))

    angles = gait_angles(t, cfg.T_STRIDE)
    fk = forward_kinematics(angles["H"], angles["K"], angles["A"])
    # Store H on the kinematics dict so later geometry and checks can use it.
    fk["H"] = angles["H"]
    com = segment_com_kinematics(fk)
    kin = compute_vel_acc(t, angles, fk, com)

    sl = slice(2, -2)
    t = t[sl]
    angles = _crop(angles, sl)
    fk = _crop(fk, sl)
    com = _crop(com, sl)
    kin = _crop(kin, sl)

    grf = generate_grf(t, angles["g"], fk)
    dyn = human_inverse_dynamics(fk, com, kin, grf)
    exo = exo_controller(angles["A"], kin["A_dot"], dyn["tau_ankle_required"])
    ie = inversion_eversion_constraint(t, cfg.T_STRIDE)
    forces = interface_forces(exo["tau_exo"], fk["beta"])
    geo = exoskeleton_geometry(fk)

    # Torque split identity: human + exo = required, including when the
    # motor torque limit is active (the human makes up the difference).
    split_err = np.max(np.abs(exo["tau_human_ankle"] + exo["tau_exo"] - dyn["tau_ankle_required"]))
    if split_err > 1e-8:
        raise RuntimeError(f"Ankle torque split is inconsistent: {split_err:.3e} N*m")

    plot_results(t, angles, fk, kin, grf, dyn, exo, ie, forces)
    gif_path = animate_system(t, angles, fk, grf, dyn, exo, forces, geo)
    print_summary(t, angles, fk, kin, grf, dyn, exo, ie, forces, sign_check)
    print()
    print(f"  plots written to {OUTPUT_DIR}")
    print(f"  animation written to {gif_path}")


if __name__ == "__main__":
    main()
