"""
Question 3 - C-space, workspace and C-obstacles of the RP manipulator.

Robot model (from Figure 1):
  joint 1: revolute about base Z, theta in [-pi, pi]
  joint 2: prismatic along Z, q2 in [0, d]
  vertical column on the Z axis, horizontal arm of length l at height
  h = a/2 + q2, end effector at its tip.
Links are modelled as zero-thickness line segments; touching = collision.

Numerical values (not given in the question, so assumed; change freely):
  a = 1 m (cube side), l = 4 m (arm length), d = 2 m (prismatic stroke),
  k = l/2 (from Figure 2).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

A, L, D = 1.0, 4.0, 2.0
K = L / 2
OUT = "/home/soumoroy/Downloads/RDC_A4_Soumo_Roy/"

# cube geometry in its own frame
U = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
              for sz in (-1, 1)]) * A / 2
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.isclose(np.abs(U[i] - U[j]).sum(), A)]
FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
         [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]

def fk(theta, q2):
    """End-effector position."""
    return np.array([L * np.cos(theta), L * np.sin(theta), A / 2 + q2])


def cube_vertices(center, R):
    return U @ R.T + center


# ----------------------------------------------------------------------
# C-obstacle by slicing: at arm height h, the arm hits the cube iff the
# ray at angle theta hits the horizontal cross-section P(h) of the cube.
# P(h) is convex and does not contain the Z axis, so the colliding
# thetas form one interval [theta_min(h), theta_max(h)].
# ----------------------------------------------------------------------
def cobs_slices(center, R, n=600):
    V = cube_vertices(center, R)
    phi_c = np.arctan2(center[1], center[0])
    lo = max(V[:, 2].min(), A / 2)          # arm can only be in [a/2, a/2+d]
    hi = min(V[:, 2].max(), A / 2 + D)
    if lo >= hi:
        return None
    hs = lo + (np.arange(n) + 0.5) * (hi - lo) / n     # midpoint samples
    i, j = np.array(EDGES).T
    z1, z2 = V[i, 2][None, :], V[j, 2][None, :]          # (1, 12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (hs[:, None] - z1) / (z2 - z1)               # (n, 12)
    ok = (t >= -1e-9) & (t <= 1 + 1e-9) & np.isfinite(t)
    t = np.where(ok, np.clip(t, 0, 1), 0.0)
    P = V[i][None] + t[..., None] * (V[j] - V[i])[None]  # (n, 12, 3)
    # vertices lying (numerically) on the cutting plane, e.g. flat top/bottom faces
    onv = np.abs(hs[:, None] - V[None, :, 2]) < 1e-9          # (n, 8)
    P = np.concatenate([P, np.broadcast_to(V, (n, 8, 3))], axis=1)
    ok = np.concatenate([ok, onv], axis=1)
    if np.any(np.hypot(P[..., 0], P[..., 1])[ok] > L):
        raise ValueError("cube partly out of reach: also clip by r <= l")
    rel = np.angle(np.exp(1j * (np.arctan2(P[..., 1], P[..., 0]) - phi_c)))
    tmin = phi_c + np.where(ok, rel, np.inf).min(axis=1)
    tmax = phi_c + np.where(ok, rel, -np.inf).max(axis=1)
    empty = ~ok.any(axis=1)
    tmin[empty], tmax[empty] = np.nan, np.nan
    return hs - A / 2, tmin, tmax           # convert height h to q2


def cobs_area(center, R, n=600):
    s = cobs_slices(center, R, n)
    if s is None:
        return 0.0
    q2, tmin, tmax = s
    dq = (q2[-1] - q2[0]) / (len(q2) - 1)
    return np.sum(np.nan_to_num(tmax - tmin)) * dq     # midpoint rule


def yaw(psi):
    return Rot.from_euler("z", psi).as_matrix()


def cspace_axes(ax, title):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, D)
    ax.set_xticks(np.linspace(-np.pi, np.pi, 9))
    ax.set_xticklabels([r"$-\pi$", r"$-3\pi/4$", r"$-\pi/2$", r"$-\pi/4$", "0",
                        r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"$\theta_1$ (rad)  [revolute]")
    ax.set_ylabel(r"$q_2$ (m)  [prismatic]")
    ax.set_title(title)
    ax.axhspan(0, D, color="#cfe8cf", zorder=0)


def draw_cobs(ax, slices, color="#d62728", label=None):
    q2, tmin, tmax = slices
    for shift in (-2 * np.pi, 0, 2 * np.pi):     # handle theta wrap-around
        ax.fill_betweenx(q2, tmin + shift, tmax + shift, color=color,
                         alpha=0.85, lw=0, label=label if shift == 0 else None)


def draw_cube(ax3, center, R, color="#d62728"):
    V = cube_vertices(center, R)
    ax3.add_collection3d(Poly3DCollection([V[f] for f in FACES], facecolor=color,
                                          edgecolor="k", alpha=0.35))


def draw_robot(ax3, theta, q2, color="k"):
    h = A / 2 + q2
    e = fk(theta, q2)
    ax3.plot([0, 0], [0, 0], [0, h], color=color, lw=3)
    ax3.plot([0, e[0]], [0, e[1]], [h, h], color=color, lw=3)
    ax3.scatter(*e, color="limegreen", s=40, zorder=5)


def draw_workspace(ax3):
    th = np.linspace(-np.pi, np.pi, 80)
    z = np.linspace(A / 2, A / 2 + D, 2)
    TH, Z = np.meshgrid(th, z)
    ax3.plot_surface(L * np.cos(TH), L * np.sin(TH), Z, color="tab:blue",
                     alpha=0.15, lw=0)
    for zz in (A / 2, A / 2 + D):
        ax3.plot(L * np.cos(th), L * np.sin(th), zz, "b--", lw=1)


def set_equal_3d(ax3, r=L, zmax=A / 2 + D + 0.3):
    ax3.set_xlim(-r, r); ax3.set_ylim(-r, r); ax3.set_zlim(0, zmax)
    ax3.set_box_aspect((2 * r, 2 * r, zmax))
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("z (m)")


# ======================================================================
# (a) C-space and workspace
# ======================================================================
fig = plt.figure(figsize=(14, 4.6))
ax = fig.add_subplot(1, 3, 1)
cspace_axes(ax, r"(a) C-space  $\mathcal{C}=S^1\times[0,d]$")
ax.plot([-np.pi, -np.pi], [0, D], "k", lw=2)
ax.plot([np.pi, np.pi], [0, D], "k", lw=2)
ax.annotate("", xy=(np.pi - 0.05, D * 0.5), xytext=(-np.pi + 0.05, D * 0.5),
            arrowprops=dict(arrowstyle="<->", ls="--", color="gray"))
ax.text(0, D * 0.55, r"edges $\theta_1=-\pi$ and $\theta_1=\pi$ are identified",
        ha="center", color="gray", fontsize=9)
ax.text(-1.8, 0.06, r"$q_2=0$ (lower limit)", ha="center", fontsize=9)
ax.text(0, D - 0.15, r"$q_2=d$ (upper limit)", ha="center", fontsize=9)
ax.scatter([0], [0], color="limegreen", zorder=5)
ax.annotate("home (0,0)", (0, 0), (0.6, 0.35), arrowprops=dict(arrowstyle="->"))

ax3 = fig.add_subplot(1, 3, 2, projection="3d")
draw_workspace(ax3)
draw_robot(ax3, np.radians(30), 0.8)
set_equal_3d(ax3)
ax3.set_title("Workspace: cylindrical surface\n"
              r"$x^2+y^2=l^2,\ a/2\leq z\leq a/2+d$")

ax = fig.add_subplot(1, 3, 3)
th = np.linspace(-np.pi, np.pi, 300)
ax.plot(L * np.cos(th), L * np.sin(th), "b", lw=2, label="workspace (top view)")
ax.plot([0, L * np.cos(0.5)], [0, L * np.sin(0.5)], "k", lw=2)
ax.scatter(0, 0, color="k")
ax.annotate(r"$l$", (L * np.cos(0.5) / 2, L * np.sin(0.5) / 2 + 0.2))
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(r"Top view: circle of radius $l$" "\n"
             r"(at every height $z\in[a/2,\,a/2+d]$)")
fig.tight_layout()
fig.savefig(OUT + "q3a_cspace_workspace.png", dpi=170)
plt.close(fig)

# ======================================================================
# (b) axis-aligned cube centred at (-k, 0, a/2)
# ======================================================================
cb = np.array([-K, 0, A / 2])
beta = np.arctan(A / (2 * K - A))
sl_b = cobs_slices(cb, np.eye(3))
area_b_num = cobs_area(cb, np.eye(3))
area_b_exact = 2 * beta * min(A / 2, D)

fig = plt.figure(figsize=(13, 5))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
draw_workspace(ax3)
draw_cube(ax3, cb, np.eye(3))
draw_robot(ax3, np.radians(110), 1.0, color="k")
set_equal_3d(ax3)
ax3.set_title("(b) Scene: cube centred at $(-k,0,a/2)$, $k=l/2$")

ax = fig.add_subplot(1, 2, 2)
cspace_axes(ax, r"(b) C-space with $\mathcal{C}_{obs}$ and $\mathcal{C}_{free}$")
draw_cobs(ax, sl_b)
ax.axvline(np.pi - beta, color="k", ls=":", lw=1)
ax.axvline(-np.pi + beta, color="k", ls=":", lw=1)
ax.axhline(A / 2, color="k", ls=":", lw=1)
ax.text(np.pi - beta - 0.05, A / 2 + 0.18, r"$\pi-\beta$", ha="right")
ax.text(-np.pi + beta + 0.05, A / 2 + 0.18, r"$-\pi+\beta$")
ax.text(0, A / 2 + 0.04, r"$q_2=a/2$  (arm at cube top, $z=a$)", ha="center")
ax.legend(handles=[Patch(color="#d62728", label=r"$\mathcal{C}_{obs}$"),
                   Patch(color="#cfe8cf", label=r"$\mathcal{C}_{free}$")],
          loc="upper left")
fig.tight_layout()
fig.savefig(OUT + "q3b_cobs.png", dpi=170)
plt.close(fig)

# ======================================================================
# (c) cube at (-k+1, 1, a/2+1) with variable orientation
# ======================================================================
cc = np.array([-K + 1, 1, A / 2 + 1])
rho = np.hypot(cc[0], cc[1])
phi_c = np.arctan2(cc[1], cc[0])
gamma_star = np.arcsin(A / (2 * rho))

# 1-D scan over yaw (cube stays upright)
psis = np.radians(np.linspace(0, 90, 1801))
A_yaw = np.array([cobs_area(cc, yaw(p), 200) for p in psis])
psi_opt_scan = psis[np.argmin(A_yaw)]
psi_star = np.mod(phi_c - gamma_star, np.pi / 2)        # analytical optimum

# global search over SO(3)
f = lambda r: cobs_area(cc, Rot.from_rotvec(r).as_matrix(), 250)
best = min((minimize(f, Rot.random(random_state=s).as_rotvec(),
                     method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-7, maxiter=3000))
            for s in range(12)), key=lambda r: r.fun)
R_best = Rot.from_rotvec(best.x).as_matrix()
A_best = cobs_area(cc, R_best, 2000)            # re-evaluate accurately

# random orientations: area versus tilt (angle between vertical and the
# closest cube axis)
Rs = Rot.random(800, random_state=1).as_matrix()
tilt = np.degrees(np.arccos(np.max(np.abs(Rs[:, 2, :]), axis=1)))
A_rand = np.array([cobs_area(cc, R, 150) for R in Rs])

cases = [(r"Axis-aligned ($\psi=0^\circ$)", np.eye(3)),
         (r"Yaw $\psi=45^\circ$ about Z", yaw(np.pi / 4)),
         (rf"Optimal $R^*$: yaw $\psi^*={np.degrees(psi_star):.1f}^\circ$",
          yaw(psi_star)),
         (r"Tilted: $45^\circ$ about X, then $30^\circ$ about Z",
          Rot.from_euler("xz", [45, 30], degrees=True).as_matrix())]

fig, axs = plt.subplots(2, 4, figsize=(18, 8.2),
                        gridspec_kw=dict(height_ratios=[1, 1.15]))
for col, (name, R) in enumerate(cases):
    # top view of the cube and the two tangent rays
    ax = axs[0, col]
    V = cube_vertices(cc, R)
    s = cobs_slices(cc, R)
    for fidx in FACES:
        ax.fill(V[fidx, 0], V[fidx, 1], color="#d62728", alpha=0.12, ec="k", lw=0.6)
    tlo, thi = np.nanmin(s[1]), np.nanmax(s[2])
    for t in (tlo, thi):
        ax.plot([0, 3 * np.cos(t)], [0, 3 * np.sin(t)], "b--", lw=1)
    ax.scatter(0, 0, color="k", zorder=5)
    ax.set_xlim(-2.6, 0.6); ax.set_ylim(-0.4, 2.6)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("Top view: " + name, fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axs[1, col]
    cspace_axes(ax, rf"$\mathcal{{C}}_{{obs}}$,  $A(R)={cobs_area(cc, R):.4f}$ rad$\cdot$m")
    draw_cobs(ax, s)
    ax.set_xlim(np.pi / 2 - 0.1, np.pi)          # zoom on the obstacle
    ax.set_xticks(np.linspace(np.pi / 2, np.pi, 5))
    ax.set_xticklabels([r"$\pi/2$", r"$5\pi/8$", r"$3\pi/4$", r"$7\pi/8$", r"$\pi$"])
    ax.legend(handles=[Patch(color="#d62728", label=r"$\mathcal{C}_{obs}$"),
                       Patch(color="#cfe8cf", label=r"$\mathcal{C}_{free}$")],
              loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "q3c_cobs_orientations.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axs[0]
ax.plot(np.degrees(psis), A_yaw, lw=2)
for p, lab in [(0, r"axis-aligned"), (45, r"$45^\circ$ (face-on)")]:
    ax.scatter(p, cobs_area(cc, yaw(np.radians(p))), zorder=5)
    ax.annotate(lab, (p, cobs_area(cc, yaw(np.radians(p)))), (p + 3, 0.99))
for p in (psi_star, np.pi / 2 - psi_star):
    ax.axvline(np.degrees(p), color="g", ls="--")
ax.text(np.degrees(psi_star) + 1, 0.885,
        rf"$\psi^*={np.degrees(psi_star):.2f}^\circ$", color="g")
ax.set_xlabel(r"yaw $\psi$ about Z (deg), cube kept upright")
ax.set_ylabel(r"$A(\psi)$ (rad$\cdot$m)")
ax.set_title(r"C-obstacle area vs yaw (period $90^\circ$)")
ax.grid(alpha=0.3)

ax = axs[1]
ax.scatter(tilt, A_rand, s=6, alpha=0.5, label="random $R\\in SO(3)$")
ax.axhline(A_best, color="g", ls="--", label=f"global min = {A_best:.4f}")
ax.set_xlabel("tilt: angle between vertical and nearest cube axis (deg)")
ax.set_ylabel(r"$A(R)$ (rad$\cdot$m)")
ax.set_title("Any tilt increases the C-obstacle area")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "q3c_area_vs_orientation.png", dpi=170)
plt.close(fig)


print(f"(b) beta = {np.degrees(beta):.3f} deg, "
      f"C_obs: theta in [pi-beta, pi] U [-pi, -pi+beta], q2 in [0, {min(A/2, D)}]")
print(f"(b) area exact = {area_b_exact:.5f}, numeric = {area_b_num:.5f}")
print(f"(c) rho = {rho:.4f}, phi_c = {np.degrees(phi_c):.2f} deg, "
      f"gamma* = asin(a/2rho) = {np.degrees(gamma_star):.3f} deg")
print(f"(c) psi* analytic = {np.degrees(psi_star):.3f} deg, "
      f"scan = {np.degrees(psi_opt_scan):.3f} deg")
for name, R in cases:
    print(f"    {name:55s} A = {cobs_area(cc, R, 2000):.5f}")
print("(c) global SO(3) optimum area =", round(A_best, 5))
print("    R_best =\n", np.round(R_best, 4))
print("    angle between R_best's nearest axis and vertical (deg):",
      round(np.degrees(np.arccos(np.max(np.abs(R_best[2])))), 4))
