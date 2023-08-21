"""
This code was provided by;

https://github.com/astra-vision/rain-rendering

@article{tremblay2020rain,
  title={Rain Rendering for Evaluating and Improving Robustness to Bad Weather},
  author={Tremblay, Maxime and Halder, Shirsendu S. and de Charette, Raoul and Lalonde, Jean-François},
  journal={International Journal of Computer Vision},
  year={2020}
}

"""

import numpy as np

np.random.seed(0)

def fastfloor(x):
    return np.array(np.floor(x), dtype=np.int32)


p = np.random.randint(256, size=256)  # random 256 values from 0-255
# To remove the need for index wrapping, double the permutation table length
perm = np.arange(512, dtype='i2')
perm = p[perm & 255]
permMod12 = perm % 12

class SimplexNoise(object):
    grad3 = np.array([[1, 1, 0],
                      [-1, 1, 0],
                      [1, -1, 0],
                      [-1, -1, 0],
                      [1, 0, 1],
                      [-1, 0, 1],
                      [1, 0, -1],
                      [-1, 0, -1],
                      [0, 1, 1],
                      [0, -1, 1],
                      [0, 1, -1],
                      [0, -1, -1]])

    # Skewing and unskewing factors
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0

    def __init__(self):
        self.p = np.random.randint(255, size=256)  # random 256 values from 0-255
        # To remove the need for index wrapping, double the permutation table length
        self.perm = np.arange(512, dtype='i2')
        self.perm = p[perm & 255]
        self.permMod12 = perm % 12

        self.n0, self.n1, self.n2, self.n3 = None, None, None, None

    def setup(self, matTpl):
        self.n0 = np.zeros_like(matTpl)
        self.n1 = np.zeros_like(matTpl)
        self.n2 = np.zeros_like(matTpl)
        self.n3 = np.zeros_like(matTpl)  # Noise contributions from the four corners

    def noise3d(self, xin, yin, zin):
        # Skew the input space to determine which simplex cell we're in
        s = (xin + yin + zin) * self.F3
        i = fastfloor(xin + s)
        j = fastfloor(yin + s)
        k = fastfloor(zin + s)
        t = (i + j + k) * self.G3

        # Unskew the cell origin back to (x,y,z) space
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        # The x,y,z distances from the cell origin
        x0 = xin - X0
        y0 = yin - Y0
        z0 = zin - Z0

        # For the 3D case, the simplex shape is a slightly irregular tetrahedron.
        # Determine which simplex we are in.
        #
        xy = x0 >= y0
        yz = y0 >= z0
        xz = x0 >= z0
        i1 = xy & (yz | xz)
        i2 = xy | ~xy & yz & xz
        j1 = ~xy & yz
        j2 = xy & yz | ~xy
        k1 = xy & ~yz & ~xz | ~xy & ~yz
        k2 = xy & ~yz | ~xy & (~yz | ~xz)

        #
        # A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
        # a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
        # a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where c = 1/6.

        x1 = x0 - i1 + self.G3  # OfFsets for second corner in (x,y,z) coords
        y1 = y0 - j1 + self.G3
        z1 = z0 - k1 + self.G3
        x2 = x0 - i2 + 2.0 * self.G3  # Offsets for third corner in (x,y,z) coords
        y2 = y0 - j2 + 2.0 * self.G3
        z2 = z0 - k2 + 2.0 * self.G3
        x3 = x0 - 1.0 + 3.0 * self.G3  # Offsets for last corner in (x,y,z) coords
        y3 = y0 - 1.0 + 3.0 * self.G3
        z3 = z0 - 1.0 + 3.0 * self.G3

        # Work out the hashed gradient indices of the four simplex corners
        ii = i & 255
        jj = j & 255
        kk = k & 255

        gi0 = self.permMod12[ii + self.perm[jj + self.perm[kk]]]
        gi1 = self.permMod12[ii + i1 + self.perm[jj + j1 + self.perm[kk + k1]]]
        gi2 = self.permMod12[ii + i2 + self.perm[jj + j2 + self.perm[kk + k2]]]
        gi3 = self.permMod12[ii + 1 + self.perm[jj + 1 + self.perm[kk + 1]]]

        # Calculate the contribution from the four corners
        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        m = t0 >= 0
        t0[m] *= t0[m]
        self.n0[m] = t0[m] * t0[m] * (
        self.grad3[gi0[m], 0] * x0[m] + self.grad3[gi0[m], 1] * y0[m] + self.grad3[gi0[m], 2] * z0[m])

        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        m = t1 >= 0
        t1[m] *= t1[m]
        self.n1[m] = t1[m] * t1[m] * (
        self.grad3[gi1[m], 0] * x1[m] + self.grad3[gi1[m], 1] * y1[m] + self.grad3[gi1[m], 2] * z1[m])

        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        m = t2 >= 0
        t2[m] *= t2[m]
        self.n2[m] = t2[m] * t2[m] * (
        self.grad3[gi2[m], 0] * x2[m] + self.grad3[gi2[m], 1] * y2[m] + self.grad3[gi2[m], 2] * z2[m])

        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        m = t3 >= 0
        t3[m] *= t3[m]
        self.n3[m] = t3[m] * t3[m] * (
        self.grad3[gi3[m], 0] * x3[m] + self.grad3[gi3[m], 1] * y3[m] + self.grad3[gi3[m], 2] * z3[m])

        # Add contributions from each corner to get the final noise value.
        # The result is scaled to stay just inside [-1,1]
        return 32.0 * (self.n0 + self.n1 + self.n2 + self.n3)
