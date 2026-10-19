# Forward Kinematics: Human Lower Limb and Ankle Exoskeleton

**ME139 / ME239 style notes for this project**

Source style: Kazerooni forward-kinematics lecture (homogeneous transforms, humanoid-leg example).

Implementation: `coupled_human_exoskeleton_sim.py`, parameters in `config.py`.

---

## 1. What kinematics answers

Sample questions for this model:

- Where is the knee relative to the hip?
- Where is the ankle relative to the hip?
- Where are the heel and toe relative to the hip?
- Where is the exoskeleton cuff relative to the shank?
- Where is the foot plate relative to the ankle?

Kinematics describes the **location and orientation** of every humanoid and exoskeleton segment from the joint angles. It does not compute forces or torques.

---

## 2. Coordinate frames and the basic transform

### 2.1 Project base frame

The hip is the origin. It does not translate.

| Axis | Direction | Unit |
| --- | --- | --- |
| $`y_0`$ | anterior (forward) | m |
| $`z_0`$ | superior (upward) | m |
| $`x_0`$ | lateral (out of the sagittal page) | m |

The walking plane is the $`y_0`$–$`z_0`$ plane. Hip, knee, and ankle dorsiflexion/plantarflexion joints rotate about axes parallel to $`x_0`$.

Angles are positive **counterclockwise** when looking along $`+x_0`$.

### 2.2 Rotation of a point (lecture form)

If frame 1 is rotated from frame 0 by $`\theta_0`$ about the shared origin, and $`P_1 = (y_1, z_1)`$ are the coordinates of a point in frame 1, then in frame 0:

```math
\begin{bmatrix} y_0 \\ z_0 \end{bmatrix}
=
\begin{bmatrix}
\cos\theta_0 & -\sin\theta_0 \\
\sin\theta_0 & \cos\theta_0
\end{bmatrix}
\begin{bmatrix} y_1 \\ z_1 \end{bmatrix}
= R_0^{1}\, P_1
```

### 2.3 Translation and orientation

If frame 1 is also displaced by a vector $`D`$ from the origin of frame 0:

```math
P_0 = R_0^{1}\, P_1 + D
```

### 2.4 Homogeneous transformation

The same map as one matrix multiply:

```math
\begin{bmatrix} P_0 \\ 1 \end{bmatrix}
=
\begin{bmatrix}
R_0^{1} & D \\
0 & 1
\end{bmatrix}
\begin{bmatrix} P_1 \\ 1 \end{bmatrix}
= T_0^{1}
\begin{bmatrix} P_1 \\ 1 \end{bmatrix}
```

More explicitly, with a $`3\times 3`$ rotation and a $`3\times 1`$ displacement:

```math
\begin{bmatrix} P_0 \\ 1 \end{bmatrix}
=
\begin{bmatrix}
R_0^{1} & | & D \\
\hline
0\ 0\ 0 & | & 1
\end{bmatrix}
\begin{bmatrix} P_1 \\ 1 \end{bmatrix}
```

The $`4\times 4`$ matrix $`T_0^{1}`$ is the **homogeneous transformation** from frame 1 into frame 0.

Notation used below (same as the lecture):

- $`C_i = \cos(\cdot)`$, $`S_i = \sin(\cdot)`$
- $`T_{i}^{i+1}`$: transform that takes a point written in frame $`i+1`$ into frame $`i`$

---

## 3. Example: a humanoid leg (this project)

### 3.1 Link lengths

| Symbol in code | Lecture-style name | Meaning | Default |
| --- | --- | --- | --- |
| `L_thigh` | $`D_1`$ | hip → knee | $`0.46\,\mathrm{m}`$ |
| `L_shank` | $`D_2`$ | knee → ankle | $`0.45\,\mathrm{m}`$ |
| `L_foot` | $`L_f`$ | heel → toe | $`0.15`$ × body height $`= 0.27\,\mathrm{m}`$ |
| `D5` | $`D_5`$ | ankle → heel along the sole normal | $`0.094\,\mathrm{m}`$ |

### 3.2 Joint angles

Gait percentage:

```math
g = 100\,\frac{t}{T_{\mathrm{stride}}}
```

Hip $`H`$, knee $`K`$, and ankle $`A`$ come from Fourier fits in degrees, then:

```math
H = H_{\mathrm{deg}}\frac{\pi}{180},\quad
K = K_{\mathrm{deg}}\frac{\pi}{180},\quad
A = \bigl(A_{\mathrm{raw}}+20^{\circ}\bigr)\frac{\pi}{180}
```

(with optional sign flips `SIGN_HIP`, `SIGN_KNEE`, `SIGN_ANKLE`).

| Angle | Meaning | Zero | Positive |
| --- | --- | --- | --- |
| $`H`$ | hip flexion (absolute thigh) | thigh vertical | knee forward |
| $`K`$ | knee flexion (relative) | knee fully extended | knee bends |
| $`A`$ | ankle angle after $`+20^{\circ}`$ | see $`\phi`$ | toe points down |

Absolute segment angles used in the transforms:

```math
\beta = H - K
```

```math
\phi = H - K - A = \beta - A
```

- $`\beta`$: absolute shank angle from the downward vertical
- $`\phi`$: absolute foot angle; $`\phi = 0`$ is a horizontal foot, toe forward; increasing $`\phi`$ lifts the toe

### 3.3 Frame assignment (sagittal chain)

Same idea as the lecture “Humanoid Leg” example: one frame per joint, $`X`$ of each frame along the next link.

```text
Frame 0  hip / trunk          origin at hip
Frame 1  thigh                X1 along thigh, hip → knee
Frame 2  shank                X2 along shank, knee → ankle
Frame 3  foot                 X3 along sole, heel → toe
```

**Frame 0.** Origin at the hip. $`y_0`$ anterior, $`z_0`$ superior.

**Frame 1.** Origin at the hip. Axis $`X_1`$ points from the hip toward the knee.
When $`H = 0`$, $`X_1`$ points straight down ($`-z_0`$).

**Frame 2.** Origin at the knee. Axis $`X_2`$ points from the knee toward the ankle.
When $`\beta = 0`$, $`X_2`$ points straight down.

**Frame 3.** Origin at the heel. Axis $`X_3`$ points from the heel toward the toe.
When $`\phi = 0`$, $`X_3`$ is horizontal and anterior ($`+y_0`$).

The ankle joint itself is offset from the heel by $`D_5`$ along the sole normal (see Section 4.3).

### 3.4 Diagram (lecture-style stick figure)

```text
            y0 (anterior)
               →
              O0  hip          z0 ↑
             /  \
            / H  \
           /      \
          O1 knee
           \
            \ β
             \
              O2 ankle
             / \
        heel ·——· toe     φ along the sole
              X3 →
```

---

## 4. Homogeneous transforms for the leg

A point written in frame $`k`$ is written in frame $`0`$ by the product of the intervening transforms, exactly as in the lecture:

```math
p_0 = T_0^{1}\, T_1^{2}\, T_2^{3}\, p_3
```

### 4.1 Hip: frame 0 → frame 1

Pure rotation. No offset between hip and the start of the thigh.

The unit vector along $`X_1`$ (thigh) in frame 0 is $`(S_H,\ -C_H)`$ in $`(y,z)`$.
The homogeneous transform that maps a point in the thigh plane into frame 0 is:

```math
T_0^{1}
=
\begin{bmatrix}
S_H & C_H & 0 & 0 \\
-C_H & S_H & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
,\qquad
S_H=\sin H,\quad C_H=\cos H
```

A point on the thigh axis at distance $`D_1`$ from the hip is $`p_1 = [D_1,\ 0,\ 0,\ 1]^{\top}`$, so the knee in frame 0 is:

```math
\begin{bmatrix} y_{\mathrm{knee}} \\ z_{\mathrm{knee}} \\ 0 \\ 1 \end{bmatrix}
=
T_0^{1}
\begin{bmatrix} D_1 \\ 0 \\ 0 \\ 1 \end{bmatrix}
=
\begin{bmatrix} D_1 S_H \\ -D_1 C_H \\ 0 \\ 1 \end{bmatrix}
```

which matches the code:

```math
y_{\mathrm{knee}} = L_{\mathrm{thigh}}\sin H
```

```math
z_{\mathrm{knee}} = -L_{\mathrm{thigh}}\cos H
```

### 4.2 Knee: frame 1 → frame 2

Translate along the thigh by $`D_1`$, then rotate by the relative knee angle so the shank absolute angle becomes $`\beta = H - K`$.

Equivalently, place frame 2 at the knee with absolute orientation $`\beta`$:

```math
T_0^{2}
=
\begin{bmatrix}
S_{\beta} & C_{\beta} & 0 & y_{\mathrm{knee}} \\
-C_{\beta} & S_{\beta} & 0 & z_{\mathrm{knee}} \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
,\qquad
S_{\beta}=\sin\beta,\quad C_{\beta}=\cos\beta
```

A point on the shank axis at distance $`D_2`$ from the knee is $`p_2 = [D_2,\ 0,\ 0,\ 1]^{\top}`$:

```math
\begin{bmatrix} y_{\mathrm{ankle}} \\ z_{\mathrm{ankle}} \\ 0 \\ 1 \end{bmatrix}
=
T_0^{2}
\begin{bmatrix} D_2 \\ 0 \\ 0 \\ 1 \end{bmatrix}
=
\begin{bmatrix}
y_{\mathrm{knee}} + D_2 S_{\beta} \\
z_{\mathrm{knee}} - D_2 C_{\beta} \\
0 \\ 1
\end{bmatrix}
```

which matches the code:

```math
y_{\mathrm{ankle}} = y_{\mathrm{knee}} + L_{\mathrm{shank}}\sin\beta
```

```math
z_{\mathrm{ankle}} = z_{\mathrm{knee}} - L_{\mathrm{shank}}\cos\beta
```

### 4.3 Ankle / foot: frame 2 → frame 3

The foot absolute angle is $`\phi = H - K - A`$.
The heel is not at the ankle: it is offset by $`D_5`$ along the sole normal. With $`\phi = 0`$ (level foot), that normal is straight down:

```math
\begin{bmatrix} y_{\mathrm{heel}} \\ z_{\mathrm{heel}} \end{bmatrix}
=
\begin{bmatrix} y_{\mathrm{ankle}} \\ z_{\mathrm{ankle}} \end{bmatrix}
+
\begin{bmatrix} -D_5\sin\phi \\ -D_5\cos\phi \end{bmatrix}
```

Frame 3 sits at the heel with $`X_3`$ along the sole (heel → toe):

```math
T_0^{3}
=
\begin{bmatrix}
C_{\phi} & -S_{\phi} & 0 & y_{\mathrm{heel}} \\
S_{\phi} & C_{\phi} & 0 & z_{\mathrm{heel}} \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
,\qquad
C_{\phi}=\cos\phi,\quad S_{\phi}=\sin\phi
```

The toe is the point $`p_3 = [L_f,\ 0,\ 0,\ 1]^{\top}`$ in frame 3:

```math
\begin{bmatrix} y_{\mathrm{toe}} \\ z_{\mathrm{toe}} \\ 0 \\ 1 \end{bmatrix}
=
T_0^{3}
\begin{bmatrix} L_f \\ 0 \\ 0 \\ 1 \end{bmatrix}
=
\begin{bmatrix}
y_{\mathrm{heel}} + L_f\cos\phi \\
z_{\mathrm{heel}} + L_f\sin\phi \\
0 \\ 1
\end{bmatrix}
```

### 4.4 Closed form (matches `forward_kinematics`)

```math
\begin{aligned}
\beta &= H - K \\
\phi &= H - K - A \\
y_{\mathrm{knee}} &= L_{\mathrm{thigh}}\sin H \\
z_{\mathrm{knee}} &= -L_{\mathrm{thigh}}\cos H \\
y_{\mathrm{ankle}} &= y_{\mathrm{knee}} + L_{\mathrm{shank}}\sin\beta \\
z_{\mathrm{ankle}} &= z_{\mathrm{knee}} - L_{\mathrm{shank}}\cos\beta \\
y_{\mathrm{heel}} &= y_{\mathrm{ankle}} - D_5\sin\phi \\
z_{\mathrm{heel}} &= z_{\mathrm{ankle}} - D_5\cos\phi \\
y_{\mathrm{toe}} &= y_{\mathrm{heel}} + L_{\mathrm{foot}}\cos\phi \\
z_{\mathrm{toe}} &= z_{\mathrm{heel}} + L_{\mathrm{foot}}\sin\phi
\end{aligned}
```

### 4.5 Neutral-pose check ($`H = K = A = 0`$)

Then $`\beta = 0`$, $`\phi = 0`$:

| Point | Position $`(y,z)`$ |
| --- | --- |
| knee | $`(0,\ -L_{\mathrm{thigh}})`$ |
| ankle | $`(0,\ -(L_{\mathrm{thigh}}+L_{\mathrm{shank}}))`$ |
| heel | $`(0,\ z_{\mathrm{ankle}}-D_5)`$ |
| toe | $`(L_{\mathrm{foot}},\ z_{\mathrm{heel}})`$ |

---

## 5. Chaining like the lecture

For any point written in the foot frame as $`\bar{p}_3 = [\bar{y}_3,\ \bar{z}_3,\ 0,\ 1]^{\top}`$:

```math
p_0^{3}
=
T_0^{1}\, T_1^{2}\, T_2^{3}\, \bar{p}_3
```

Special cases used in the simulation:

| Point | Local vector | Result in frame 0 |
| --- | --- | --- |
| knee | $`[D_1,0,0,1]^{\top}`$ in frame 1 | $`T_0^{1}\,p_1`$ |
| ankle | $`[D_2,0,0,1]^{\top}`$ in frame 2 | $`T_0^{2}\,p_2`$ |
| heel | origin of frame 3 | translation column of $`T_0^{3}`$ |
| toe | $`[L_f,0,0,1]^{\top}`$ in frame 3 | $`T_0^{3}\,p_3`$ |

This is the same product written on the lecture slide for the humanoid leg:

```math
p_0^{3} = T_0^{1}\, T_1^{2}\, T_2^{3}
\begin{bmatrix} \bar{x}_3 \\ \bar{y}_3 \\ 0 \\ 1 \end{bmatrix}
```

with our $`(y,z)`$ axes in place of the lecture’s $`(X,Y)`$ drawing axes.

---

## 6. Segment centers of mass

Points on the same transforms, at a fraction of each link:

```math
\begin{aligned}
p_{\mathrm{thigh}} &= 0.433\, p_{\mathrm{knee}} \\
p_{\mathrm{shank}} &= p_{\mathrm{knee}} + 0.433\,(p_{\mathrm{ankle}}-p_{\mathrm{knee}}) \\
p_{\mathrm{foot}} &= p_{\mathrm{heel}} + 0.50\,(p_{\mathrm{toe}}-p_{\mathrm{heel}})
\end{aligned}
```

(Fractions are modeling assumptions in `config.py`.)

---

## 7. Exoskeleton kinematics

The device is a second chain strapped to the limb:

```text
calf cuff  →  rigid upright  →  rotary ankle actuator  →  foot plate
```

### 7.1 Actuator angle locked to the human ankle

```math
\theta_{\mathrm{exo}} = A
```

There is no independent exoskeleton joint angle in this version. The upright follows the shank orientation $`\beta`$. The foot plate follows the foot orientation $`\phi`$.

### 7.2 Frame on the upright (cuff)

Let $`f_c = 0.62`$ be the fraction of the way from the ankle toward the knee, and let $`d_u = 0.025\,\mathrm{m}`$ be the anterior drawing offset of the upright.

Unit vector along the shank (frame-2 $`X`$-axis in base coordinates):

```math
\hat{u}_{\mathrm{shank}}
=
\begin{bmatrix} \sin\beta \\ -\cos\beta \end{bmatrix}
```

Unit vector perpendicular to the shank (anterior when $`\beta = 0`$):

```math
\hat{n}
=
\begin{bmatrix} \cos\beta \\ \sin\beta \end{bmatrix}
```

Cuff and actuator landmarks in frame 0:

```math
\begin{aligned}
p_{\mathrm{cuff,bone}}
&=
p_{\mathrm{ankle}} + f_c\,(p_{\mathrm{knee}}-p_{\mathrm{ankle}}) \\
p_{\mathrm{cuff}}
&=
p_{\mathrm{cuff,bone}} + d_u\,\hat{n} \\
p_{\mathrm{actuator}}
&=
p_{\mathrm{ankle}} + d_u\,\hat{n}
\end{aligned}
```

### 7.3 Frame on the foot plate

Soleward unit vector (straight down when $`\phi = 0`$):

```math
\hat{s}
=
\begin{bmatrix} -\sin\phi \\ -\cos\phi \end{bmatrix}
```

With plate offset $`d_p = 0.012\,\mathrm{m}`$:

```math
\begin{aligned}
p_{\mathrm{plate,heel}} &= p_{\mathrm{heel}} + d_p\,\hat{s} \\
p_{\mathrm{plate,toe}}  &= p_{\mathrm{toe}}  + d_p\,\hat{s}
\end{aligned}
```

The offsets $`d_u`$ and $`d_p`$ are graphical so the brace does not sit on top of the bone in the animation. They do not change the human joint positions in Section 4.

### 7.4 Interface force direction (geometry only)

The simplified interface model uses an effective moment arm $`r_{\mathrm{exo}} = 0.05\,\mathrm{m}`$ and the shank normal $`\hat{n}`$. That is a load estimate, not an extra kinematic degree of freedom.

---

## 8. Inversion / eversion (not in the sagittal chain)

Inversion/eversion is a separate scalar angle $`\theta_{IE}`$. It is **not** a fourth sagittal transform. It does not move $`y`$ or $`z`$ of the hip–knee–ankle–heel–toe chain.

A placeholder waveform is used because the gait fit is sagittal only:

```math
\theta_{IE}(g)
=
4^{\circ}\sin\!\left(2\pi\frac{g}{100}\right)
+
2.5^{\circ}\sin\!\left(4\pi\frac{g}{100}\right)
```

The brace allows about $`\pm 5^{\circ}`$. Outside that range a stiffer stop engages.

---

## 9. Velocities and accelerations

Differentiate the angle definitions:

```math
\begin{aligned}
\dot{\beta} &= \dot{H} - \dot{K} \\
\ddot{\beta} &= \ddot{H} - \ddot{K} \\
\dot{\phi} &= \dot{H} - \dot{K} - \dot{A} \\
\ddot{\phi} &= \ddot{H} - \ddot{K} - \ddot{A}
\end{aligned}
```

Joint rates and center-of-mass rates are numerical derivatives (`numpy.gradient`) with $`\Delta t = 0.001\,\mathrm{s}`$. Units: rad/s, rad/s², m/s, m/s².

---

## 10. What this kinematic model leaves out

- Pelvic translation and tilt (hip origin fixed)
- Ground contact as a kinematic constraint
- A heel that projects behind the ankle when the foot is level
- Independent exoskeleton motion relative to the limb ($`\theta_{\mathrm{exo}}=A`$)
- Frontal-plane position from $`\theta_{IE}`$

---

## 11. Map to the code

| Lecture object | Code |
| --- | --- |
| $`H, K, A`$ | `gait_angles()` → `H`, `K`, `A` (radians) |
| $`\beta, \phi`$ and landmarks | `forward_kinematics()` |
| COM points | `segment_com_kinematics()` |
| rates and accelerations | `compute_vel_acc()` |
| cuff / upright / plate | `exoskeleton_geometry()` |
| lengths and offsets | `config.py` |

---

## 12. One-line summary

```math
p_{\mathrm{toe}}
=
T_0^{1}(H)\,
T_1^{2}(H,K)\,
T_2^{3}(H,K,A)\,
\begin{bmatrix} L_{\mathrm{foot}} \\ 0 \\ 0 \\ 1 \end{bmatrix}
```

with $`\theta_{\mathrm{exo}} = A`$ for the ankle actuator, and the cuff/plate obtained from the same $`\beta`$ and $`\phi`$ by the offsets in Section 7.
