import numpy as np
 
def rotation_to_angle_axis(R):
    """
    Input : R       - 3x3 rotation matrix
    Output: theta,k - angle in radians (between 0 and pi), unit vector along the rotation axis
    """
    tol = 1e-6  # any number smaller than this is treated as zero
 
    # find the angle
    # trace(R) = 1 + 2*cos(theta), so cos(theta) = (trace(R) - 1) / 2
    cos_theta = (np.trace(R) - 1) / 2
 
    # Round-off can give values like 1.0000000001, which arccos cannot accept so keep cos_theta between -1 and 1.
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
 
    theta = np.arccos(cos_theta)
 
    # special case theta = 0
    # R is the identity matrix: nothing rotates, so every axis is valid.
    # return the z axis as a default choice.
    if theta < tol:
        K = np.array([0.0, 0.0, 1.0])
        return 0.0, K
 
    # special case theta = 180 degrees 
    # here sin(theta) = 0, so equation (1) would give 0/0.
    # instead we use  K*K^T = (R + I)/2.
    if abs(theta - np.pi) < tol:
        M = (R + np.eye(3)) / 2
 
        # the diagonal of M holds k1^2, k2^2, k3^2.
        # pick the largest one so we never divide by a number close to zero.
        i = np.argmax(np.diag(M))
 
        # take k_i as the positive square root.
        k_i = np.sqrt(M[i, i])
 
        # column i of M is (k1*k_i, k2*k_i, k3*k_i).
        # dividing it by k_i gives the full axis (k1, k2, k3).
        K = M[:, i] / k_i
        return np.pi, K
 
    # general case, equation (1)
    K = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
                  
    return theta, K
 
 
def angle_axis_to_rotation(theta, K):
    """
    Rebuild R from theta and K using Rodrigues' formula:
        R = I + sin(theta)*[K] + (1 - cos(theta))*[K]^2
    Used only to check our answers.
    """
    K_skew = np.array([[0,     -K[2],  K[1]],
                       [K[2],   0,    -K[0]],
                       [-K[1],  K[0],  0   ]])
    final_rotation_matrix = np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)
    return final_rotation_matrix
 
if __name__ == "__main__":
 
    examples = {
        "Identity (theta = 0)":
            np.eye(3),
 
        "90 deg about z":
            np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]]),
 
        "120 deg about (1,1,1)":
            np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]]),
 
        "180 deg about x":
            np.array([[1,  0,  0],
                      [0, -1,  0],
                      [0,  0, -1]]),
 
        "180 deg about z":
            np.array([[-1,  0, 0],
                      [ 0, -1, 0],
                      [ 0,  0, 1]]),
 
        "180 deg about (2,-1,2)/3":
            np.array([[-1, -4,  8],
                      [-4, -7, -4],
                      [ 8, -4, -1]]) / 9,
    }
 
    for name, R in examples.items():
        theta, K = rotation_to_angle_axis(R)
 
        # rebuilding R from theta and K should give the original R
        R_check = angle_axis_to_rotation(theta, K)
        result = "correct" if np.allclose(R, R_check) else "WRONG"
 
        print(name)
        print("   theta =", round(np.degrees(theta), 4), "degrees")
        print("   axis  =", np.round(K, 4))
        print("   check :", result)
        print()
