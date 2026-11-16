"""
Question 2 - Jacobian of the 3-link planar (RRR) manipulator in frame {3}.

Link data (Craig, modified DH):
    i | alpha_{i-1} | a_{i-1} | d_i | theta_i
    1 |      0      |    0    |  0  | theta_1
    2 |      0      |   l1    |  0  | theta_2
    3 |      0      |   l2    |  0  | theta_3

The Jacobian maps joint rates to [ 3v_x, 3v_y, 3w_z ] of the origin of
frame {3}, expressed in frame {3}:   [vx, vy, wz]^T = 3J(theta) * thetadot
"""

import numpy as np

L1 = 0.5  # metres
L2 = 0.5  # metres


def rot_z(q):
    """Rotation matrix i-1_i R for a rotation q about Z."""
    c, s = np.cos(q), np.sin(q)
    rot_vec = np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])
    return rot_vec


def jacobian(theta, l1=L1, l2=L2):
    """
    Procedure Jacobian (VAR theta: vec3; VAR Jac: mat33)
    Closed-form 3J(theta) obtained from velocity propagation.
    theta: angles [theta1, theta2, theta3] in radians.
    Returns the 3x3 Jacobian expressed in frame {3}.
    """
    _, t2, t3 = theta
    s3, c3 = np.sin(t3), np.cos(t3)
    s23, c23 = np.sin(t2 + t3), np.cos(t2 + t3)

    jac = np.array([[l1 * s23 + l2 * s3, l2 * s3, 0.0],
                    [l1 * c23 + l2 * c3, l2 * c3, 0.0],
                    [1.0,                1.0,     1.0]])
    return jac


def jacobian_by_propagation(theta, l1=L1, l2=L2):
    """
    Same Jacobian, built numerically by running the outward
    velocity-propagation recursion once per joint with a unit rate:
        w_{i+1} = R^T w_i + thetadot_{i+1} Z
        v_{i+1} = R^T (v_i + w_i x P_{i+1})
    Column k = frame-{3} velocity when thetadot = e_k.
    """
    P = [np.zeros(3), np.array([l1, 0.0, 0.0]), np.array([l2, 0.0, 0.0])]
    Z = np.array([0.0, 0.0, 1.0])
    jac = np.zeros((3, 3))

    for k in range(3):
        thetadot = np.zeros(3)
        thetadot[k] = 1.0
        w, v = np.zeros(3), np.zeros(3)          # base is stationary
        for i in range(3):
            R_T = rot_z(theta[i]).T              # i_{i-1} R
            v = R_T @ (v + np.cross(w, P[i]))
            w = R_T @ w + thetadot[i] * Z
        jac[:, k] = [v[0], v[1], w[2]]
    return jac


def forward_kinematics(theta, l1=L1, l2=L2):
    """Position of frame {3} origin and orientation 0_3 R."""
    t1, t2, t3 = theta
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return np.array([x, y]), rot_z(t1 + t2 + t3)


def jacobian_numeric_check(theta, h=1e-6):
    """Finite-difference 0J, rotated into frame {3} for comparison."""
    J0 = np.zeros((3, 3))
    for k in range(3):
        dq = np.zeros(3)
        dq[k] = h
        p_plus, _ = forward_kinematics(theta + dq)
        p_minus, _ = forward_kinematics(theta - dq)
        J0[:2, k] = (p_plus - p_minus) / (2 * h)
        J0[2, k] = 1.0                           # all joints rotate about Z
    _, R03 = forward_kinematics(theta)
    R = np.eye(3)
    R[:2, :2] = R03[:2, :2]
    return R.T @ J0                              # 3J = 3_0R * 0J


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    tests = [np.radians([0, 90, 0]),
             np.radians([30, 45, -60]),
             np.radians([10, 20, 30]),
             np.radians([0, 0, 0])]               # singular: theta2 = 0

    for th in tests:
        J = jacobian(th)
        print(f"theta (deg) = {np.degrees(th)}")
        print("3J =\n", J)
        print("det(3J) =", round(np.linalg.det(J), 6),
              "  expected l1*l2*sin(theta2) =",
              round(L1 * L2 * np.sin(th[1]), 6))
        print("matches propagation:", np.allclose(J, jacobian_by_propagation(th)),
              "| matches finite difference:",
              np.allclose(J, jacobian_numeric_check(th), atol=1e-6))
        print("-" * 60)
