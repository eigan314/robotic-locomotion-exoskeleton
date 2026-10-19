# Dynamics: Human Lower Limb and Ankle Exoskeleton

**ME139 / ME239 style notes for this project**

Source style: Lecture Notes on Dynamics (Lagrange form $`M\ddot{Q}+V+G=T`$).

Implementation: Newton–Euler inverse dynamics in `coupled_human_exoskeleton_sim.py`, parameters in `config.py`.

Kinematics prerequisite: `KINEMATICS.md`.

---

## 1. What dynamics answers

Sample questions for this model:

- What hip, knee, and ankle moments are required to produce the prescribed gait?
- How much of the ankle moment can the exoskeleton take?
- What ankle moment remains for the human?
- What joint reaction forces act at the ankle, knee, and hip?
- What interface forces appear at the cuff and foot plate?

Kinematics gives positions and rates from the joint angles. Dynamics gives the **forces and torques** consistent with those rates (and with the ground reaction).

This project is an **inverse-dynamics** problem:

```math
\text{given } Q(t),\ \dot{Q}(t),\ \ddot{Q}(t),\ \mathrm{GRF}(t)
\quad\Longrightarrow\quad
\text{find } T(t)
```

It is not a forward simulation that integrates $`T`$ to get $`Q`$.

---

## 2. Lagrange equation (lecture form)

```math
\frac{d}{dt}\left(\frac{\partial K}{\partial \dot{Q}_i}\right)
-
\frac{\partial K}{\partial Q_i}
+
\frac{\partial P}{\partial Q_i}
=
T_i
```

| Symbol | Meaning |
| --- | --- |
| $`Q`$ | vector of generalized coordinates |
| $`\dot{Q}`$ | generalized velocities |
| $`K`$ | total kinetic energy |
| $`P`$ | total potential energy |
| $`T_i`$ | generalized force/torque that does work on $`Q_i`$ |

For this leg:

```math
Q
=
\begin{bmatrix} Q_1 \\ Q_2 \\ Q_3 \end{bmatrix}
=
\begin{bmatrix} H \\ K \\ A \end{bmatrix}
```

with $`H`$ hip flexion, $`K`$ knee flexion, and $`A`$ ankle angle (radians), as in `KINEMATICS.md`.

$`T_i`$ is the net generalized torque associated with that coordinate. Ground reaction and (if included) interface loads contribute to $`T`$ through the virtual work they do on $`Q`$.

---

## 3. Kinetic energy of each link

For link $`i`$ (thigh, shank, or foot):

```math
K_i
=
\frac{1}{2}\, m_i\, v_{C_i}^{\top} v_{C_i}
+
\frac{1}{2}\, \omega_i^{\top} I_{C_i}\, \omega_i
```

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $`m_i`$ | mass of link $`i`$ | kg |
| $`v_{C_i}`$ | absolute linear velocity of the COM in the base frame | m/s |
| $`\omega_i`$ | absolute angular velocity of the link in the base frame | rad/s |
| $`I_{C_i}`$ | inertia about the COM | kg·m² |

Total kinetic energy:

```math
K = \sum_i K_i
```

In the sagittal plane each link has one angular speed. With absolute segment angles:

```math
\omega_{\mathrm{thigh}} = \dot{H}
```

```math
\omega_{\mathrm{shank}} = \dot{\beta} = \dot{H}-\dot{K}
```

```math
\omega_{\mathrm{foot}} = \dot{\phi} = \dot{H}-\dot{K}-\dot{A}
```

and the code uses the scalar planar inertias $`I_{\mathrm{thigh}}`$, $`I_{\mathrm{shank}}`$, $`I_{\mathrm{foot}}`$ about each COM (out-of-plane axis).

---

## 4. Mass matrix form of the kinetic energy

The lecture writes:

```math
K = \frac{1}{2}\, \dot{Q}^{\top} M(Q)\, \dot{Q}
```

using the Jacobians that map $`\dot{Q}`$ into COM velocity and angular velocity:

```math
v_{C_i} = J_{v,i}(Q)\, \dot{Q}
```

```math
\omega_i = J_{\omega,i}(Q)\, \dot{Q}
```

Then:

```math
M(Q)
=
\sum_i
\left(
m_i\, J_{v,i}^{\top} J_{v,i}
+
J_{\omega,i}^{\top} I_{C_i}\, J_{\omega,i}
\right)
```

$`M`$ is the **inertia matrix**. It depends on configuration $`Q`$, not on $`\dot{Q}`$.

In this project the COM positions come from the kinematics, and $`v_{C_i}`$, $`\omega_i`$, accelerations are obtained by numerical differentiation of those trajectories (`compute_vel_acc`). The mass matrix is not assembled explicitly; Newton–Euler evaluates the same physics segment by segment.

---

## 5. Compact equation of motion

From the Lagrange equation, the lecture form is:

```math
M(Q)\,\ddot{Q} + V(Q,\dot{Q}) + G(Q) = T
```

| Symbol | Meaning |
| --- | --- |
| $`M\ddot{Q}`$ | inertial (acceleration-dependent) torque |
| $`V(Q,\dot{Q})`$ | Coriolis and centrifugal vector |
| $`G(Q)`$ | gravity vector |
| $`T`$ | generalized external torques that do work on $`Q`$ |

### 5.1 Coriolis / centrifugal vector

```math
V(Q,\dot{Q})
=
\dot{M}\,\dot{Q}
-
\frac{1}{2}
\begin{bmatrix}
\dot{Q}^{\top} \dfrac{\partial M}{\partial Q_1}\, \dot{Q} \\[6pt]
\dot{Q}^{\top} \dfrac{\partial M}{\partial Q_2}\, \dot{Q} \\[6pt]
\vdots \\[6pt]
\dot{Q}^{\top} \dfrac{\partial M}{\partial Q_n}\, \dot{Q}
\end{bmatrix}
```

### 5.2 Gravity vector

Potential energy, with $`h_i`$ the height of each COM above an arbitrary level and $`g = 9.81\,\mathrm{m/s}^2`$:

```math
P = \sum_i m_i g\, h_i
```

```math
G_i = \frac{\partial P}{\partial Q_i}
```

In this project $`z`$ is upward, so the potential of a COM is $`P_i = m_i g\, z_{C_i}`$ (plus a constant). Weight appears in the Newton–Euler force balance as $`(0,\ -m_i g)`$.

### 5.3 External torque vector $`T`$

$`T`$ contains every generalized force that does work on $`Q`$: joint muscle/actuator moments, and the contribution of the ground reaction (and, if folded into the balance, interface forces).

For inverse dynamics with prescribed gait:

```math
T = M\ddot{Q} + V + G
```

is what the code reports as the **required** joint moments (before the human/exo split at the ankle).

---

## 6. Segment properties used in the code

Defaults for a $`75\,\mathrm{kg}`$, $`1.80\,\mathrm{m}`$ subject (one limb). Marked as assumptions in `config.py`.

| Segment | Mass fraction | Mass $`m`$ | COM from proximal end | $`k`$ (radius of gyration / $`L`$) | $`I = m(kL)^2`$ |
| --- | --- | --- | --- | --- | --- |
| thigh | $`0.10`$ | $`7.50\,\mathrm{kg}`$ | $`0.433\,L_{\mathrm{thigh}}`$ from hip | $`0.323`$ | about COM |
| shank | $`0.0465`$ | $`3.4875\,\mathrm{kg}`$ | $`0.433\,L_{\mathrm{shank}}`$ from knee | $`0.302`$ | about COM |
| foot | $`0.0145`$ | $`1.0875\,\mathrm{kg}`$ | $`0.50\,L_{\mathrm{foot}}`$ from heel | $`0.475`$ | about COM |

---

## 7. This project's method: Newton–Euler, distal → proximal

The lecture builds $`M\ddot{Q}+V+G=T`$ from energy. This simulation instead writes Newton’s and Euler’s laws on each rigid segment and solves for the unknown proximal joint force and moment. For the same $`Q(t)`$ and external loads, both methods give the same joint torques.

Order:

```text
foot  →  shank  →  thigh
```

On every segment:

```math
\sum F_y = m\, a_{C,y}
```

```math
\sum F_z = m\, a_{C,z}
```

```math
\sum M_C = I_C\, \alpha
```

Moment of a force about the COM (planar, N·m):

```math
M = r_y F_z - r_z F_y
```

where $`r`$ is the vector from the COM to the force application point, in meters, and $`F`$ is in newtons.

Sign convention (same as kinematics): a positive moment on a segment increases that segment’s absolute angle ($`H`$ for the thigh, $`\beta`$ for the shank, $`\phi`$ for the foot).

---

## 8. Ground reaction (external load on the foot)

Stance ≈ 0–60% of the gait cycle. Swing force is exactly zero.

```math
\mathrm{GRF}
=
\begin{bmatrix} F_y \\ F_z \end{bmatrix}
\quad\text{(N), force of the ground on the foot}
```

- $`F_z`$: vertical, synthetic double hump, mean over the stride ≈ half body weight for one leg
- $`F_y`$: anterior/posterior; braking then propulsion
- Center of pressure moves from heel ($`u=0`$) to toe ($`u=1`$) during stance:

```math
p_{\mathrm{COP}} = p_{\mathrm{heel}} + u\,(p_{\mathrm{toe}}-p_{\mathrm{heel}})
```

In the Lagrange picture, GRF contributes to $`T`$ through the virtual work

```math
\delta W = F_y\,\delta y_{\mathrm{COP}} + F_z\,\delta z_{\mathrm{COP}}
```

In Newton–Euler it appears directly on the foot free-body diagram.

---

## 9. Free-body equations (as coded)

### 9.1 Foot

Unknowns: force of the shank on the foot $`F_{\mathrm{ankle}}`$, moment of the shank on the foot $`\tau_{\mathrm{ankle}}`$.

```math
\begin{aligned}
F_{\mathrm{ankle},y} + F_y &= m_f a_{f,y} \\
F_{\mathrm{ankle},z} + F_z - m_f g &= m_f a_{f,z} \\
\tau_{\mathrm{ankle}} + M(F_{\mathrm{ankle}}) + M(\mathrm{GRF}) &= I_f\, \ddot{\phi}
\end{aligned}
```

Solved as:

```math
\begin{aligned}
F_{\mathrm{ankle},y} &= m_f a_{f,y} - F_y \\
F_{\mathrm{ankle},z} &= m_f a_{f,z} - F_z + m_f g \\
\tau_{\mathrm{ankle}} &= I_f\ddot{\phi} - M(F_{\mathrm{ankle}}) - M(\mathrm{GRF})
\end{aligned}
```

Weight has no moment about the COM.

$`\tau_{\mathrm{ankle}}`$ is **`tau_ankle_required`**: the net ankle moment needed by the motion. Positive $`\tau_{\mathrm{ankle}}`$ increases $`\phi`$ (lifts the toe).

### 9.2 Shank

Distal load is equal and opposite to the ankle load on the foot. Unknowns: $`F_{\mathrm{knee}}`$, $`\tau_{\mathrm{knee}}`$.

```math
\begin{aligned}
F_{\mathrm{knee},y} - F_{\mathrm{ankle},y} &= m_s a_{s,y} \\
F_{\mathrm{knee},z} - F_{\mathrm{ankle},z} - m_s g &= m_s a_{s,z} \\
\tau_{\mathrm{knee}} - \tau_{\mathrm{ankle}} + M(F_{\mathrm{knee}}) + M(-F_{\mathrm{ankle}}) &= I_s\, \ddot{\beta}
\end{aligned}
```

```math
\tau_{\mathrm{knee}}
=
I_s\ddot{\beta} + \tau_{\mathrm{ankle}} - M(F_{\mathrm{knee}}) - M(-F_{\mathrm{ankle}})
```

Positive $`\tau_{\mathrm{knee}}`$ increases $`\beta`$. With the hip held, that is knee extension because $`\beta = H - K`$.

### 9.3 Thigh

Distal load is equal and opposite to the knee load on the shank. Unknowns: $`F_{\mathrm{hip}}`$, $`\tau_{\mathrm{hip}}`$.

```math
\begin{aligned}
F_{\mathrm{hip},y} - F_{\mathrm{knee},y} &= m_t a_{t,y} \\
F_{\mathrm{hip},z} - F_{\mathrm{knee},z} - m_t g &= m_t a_{t,z} \\
\tau_{\mathrm{hip}} - \tau_{\mathrm{knee}} + M(F_{\mathrm{hip}}) + M(-F_{\mathrm{knee}}) &= I_t\, \ddot{H}
\end{aligned}
```

```math
\tau_{\mathrm{hip}}
=
I_t\ddot{H} + \tau_{\mathrm{knee}} - M(F_{\mathrm{hip}}) - M(-F_{\mathrm{knee}})
```

Positive $`\tau_{\mathrm{hip}}`$ increases $`H`$ (hip flexion).

### 9.4 Link to $`M\ddot{Q}+V+G=T`$

The triple

```math
T_{\mathrm{required}}
=
\begin{bmatrix}
\tau_{\mathrm{hip}} \\
\tau_{\mathrm{knee}} \\
\tau_{\mathrm{ankle}}
\end{bmatrix}
```

is the inverse-dynamics answer for the prescribed $`Q(t)`$ and GRF. It is the same $`T`$ that would come from evaluating $`M\ddot{Q}+V+G`$ (plus the GRF contribution) at that instant. The code never builds $`M`$, $`V`$, and $`G`$ as separate arrays; the free-body path already includes inertia, velocity products, and gravity.

---

## 10. Exoskeleton dynamics at the ankle

The rotary actuator assists only dorsiflexion/plantarflexion. Its angle is locked to the human ankle:

```math
\theta_{\mathrm{exo}} = A
```

### 10.1 Assistance mode (default)

```math
\tau_{\mathrm{exo}} = \rho\, \tau_{\mathrm{ankle,required}},\qquad 0 \le \rho \le 1
```

Default $`\rho = 0.30`$. Then clip to $`\pm \tau_{\max}`$ ($`\tau_{\max} = 120\,\mathrm{N\cdot m}`$).

Human moment remaining:

```math
\tau_{\mathrm{human,ankle}}
=
\tau_{\mathrm{ankle,required}} - \tau_{\mathrm{exo}}
```

With unsaturated assistance, $`\tau_{\mathrm{human}} = (1-\rho)\,\tau_{\mathrm{ankle,required}}`$. That is the main coupled-system output.

### 10.2 PD mode (optional)

```math
\tau_{\mathrm{exo}}
=
K_p(\theta_{\mathrm{des}}-A) + K_d(\dot{\theta}_{\mathrm{des}}-\dot{A})
```

With a prescribed trajectory and zero angle offset, this torque is approximately zero.

### 10.3 Motor side

```math
\tau_{\mathrm{motor}} = \frac{\tau_{\mathrm{exo}}}{N_{\mathrm{gear}}}
```

```math
i_{\mathrm{motor}} = \frac{\tau_{\mathrm{motor}}}{K_t}
```

Defaults: $`N_{\mathrm{gear}} = 100`$, $`K_t = 0.05\,\mathrm{N\cdot m/A}`$.

### 10.4 Where the exo sits in $`M\ddot{Q}+V+G=T`$

At the ankle coordinate, the required generalized torque is shared:

```math
T_{\mathrm{ankle,required}} = \tau_{\mathrm{human,ankle}} + \tau_{\mathrm{exo}}
```

The inverse-dynamics free-body solve uses the **net** ankle moment. The split into human and exo is applied afterward.

---

## 11. Simplified interface forces

Not a contact solution and **not** fed back into the inverse-dynamics balance. Effective moment arm $`r_{\mathrm{exo}} = 0.05\,\mathrm{m}`$:

```math
F = \frac{\tau_{\mathrm{exo}}}{r_{\mathrm{exo}}}
```

Direction: unit normal to the shank $`\hat{n} = (\cos\beta,\ \sin\beta)`$.

```math
F_{\mathrm{foot}} = F\,\hat{n}
```

```math
F_{\mathrm{cuff}} = -F_{\mathrm{foot}}
```

Equal-and-opposite action/reaction: force on the exoskeleton at each pad is the negation of the force on the human. Units: N.

---

## 12. Inversion / eversion brace (passive)

Not an actuator. Soft constraint plus a hard stop:

```math
\tau_{IE}
=
-k_{IE}\,\theta_{IE}
-c_{IE}\,\dot{\theta}_{IE}
-k_{\mathrm{stop}}\,\mathrm{sign}(\theta_{IE})\,\mathrm{excess}
```

```math
\mathrm{excess} = \max\bigl(|\theta_{IE}| - \theta_{\mathrm{limit}},\ 0\bigr),\qquad
\theta_{\mathrm{limit}} = 5^{\circ}
```

Defaults: $`k_{IE}=20\,\mathrm{N\cdot m/rad}`$, $`c_{IE}=0.4\,\mathrm{N\cdot m/(rad/s)}`$, $`k_{\mathrm{stop}}=600\,\mathrm{N\cdot m/rad}`$.

$`\theta_{IE}`$ is a placeholder waveform unless measured data replaces it. It does not change the sagittal Newton–Euler solve.

---

## 13. Units checklist

| Quantity | Unit |
| --- | --- |
| angles $`Q`$ | rad (fits in degrees, converted once) |
| $`\dot{Q}`$, $`\omega`$ | rad/s |
| $`\ddot{Q}`$, $`\alpha`$ | rad/s² |
| positions, $`r`$ | m |
| $`v`$, $`a`$ | m/s, m/s² |
| mass $`m`$ | kg |
| inertia $`I`$ | kg·m² |
| force $`F`$, GRF | N |
| moment $`\tau`$, $`T`$ | N·m |
| current $`i_{\mathrm{motor}}`$ | A |

---

## 14. Map to the code

| Lecture / note object | Code |
| --- | --- |
| $`Q,\dot{Q},\ddot{Q}`$ and COM rates | `gait_angles`, `forward_kinematics`, `segment_com_kinematics`, `compute_vel_acc` |
| GRF and COP | `generate_grf` |
| Foot → shank → thigh free bodies | `human_inverse_dynamics` |
| $`\tau_{\mathrm{hip}},\tau_{\mathrm{knee}},\tau_{\mathrm{ankle}}`$ | `tau_hip_required`, `tau_knee_required`, `tau_ankle_required` |
| $`\tau_{\mathrm{exo}}`$, $`\tau_{\mathrm{human}}`$ | `exo_controller` |
| cuff / plate forces | `interface_forces` |
| $`\tau_{IE}`$ | `inversion_eversion_constraint` |
| masses, $`I`$, $`\rho`$, gains | `config.py` |

---

## 15. One-line summary

Lecture form:

```math
M\ddot{Q} + V + G = T_{\mathrm{required}}
```

This project’s ankle result after assistance:

```math
\tau_{\mathrm{human,ankle}}
=
\tau_{\mathrm{ankle,required}} - \tau_{\mathrm{exo}}
```

```math
\tau_{\mathrm{exo}} = \rho\, \tau_{\mathrm{ankle,required}}
```

with $`T_{\mathrm{required}}`$ obtained by Newton–Euler from the foot to the thigh using the prescribed gait and the ground reaction.
