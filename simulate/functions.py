# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:06:07 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import abc

import numpy as np

from utils import TOLERANCE
from utils import RandomUniform
from utils import norm2

__all__ = ["Function", "L1", "SmoothedL1", "L2", "L2Squared",
           "NesterovFunction", "TotalVariation", "GroupLasso",
           "SmoothedTotalVariation", "SmoothedGroupLasso",
           "SmoothedGroupTotalVariation"]


class Function(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, l, **kwargs):

        self.l = float(l)

        for k in kwargs:
            setattr(self, k, kwargs[k])

    @abc.abstractmethod
    def grad(self, x):
        raise NotImplementedError("Abstract method 'grad' must be "
                                  "specialised!")


class L1(Function):

    def __init__(self, l, rng=RandomUniform(-1, 1)):

        super(L1, self).__init__(l, rng=rng)

    def grad(self, x):
        """Sub-gradient of the function

            f(x) = l * |x|_1,

        where |x|_1 is the L1-norm.
        """
        grad = np.zeros((x.shape[0], 1))
        grad[x >= TOLERANCE] = 1.0
        grad[x <= -TOLERANCE] = -1.0
        between = (x > -TOLERANCE) & (x < TOLERANCE)
        grad[between] = self.rng(between.sum())

        return self.l * grad


#def grad_l1(beta, rng=RandomUniform(-1, 1)):
#    """Sub-gradient of the function
#
#        f(x) = |x|_1,
#
#    where |x|_1 is the L1-norm.
#    """
#    grad = np.zeros((beta.shape[0], 1))
#    grad[beta >= TOLERANCE] = 1.0
#    grad[beta <= -TOLERANCE] = -1.0
#    between = (beta > -TOLERANCE) & (beta < TOLERANCE)
#    grad[between] = rng(between.sum())
#
#    return grad


class SmoothedL1(Function):

    def __init__(self, l, mu=TOLERANCE):

        super(SmoothedL1, self).__init__(l, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = l * L1(mu, x),

        where L1(mu, x) is the Nesterov smoothed L1-norm.
        """
        alpha = (1.0 / self.mu) * x
        asnorm = np.abs(alpha)
        i = asnorm > 1.0
        alpha[i] = np.divide(alpha[i], asnorm[i])

        return self.l * alpha


#def grad_l1mu(beta, mu):
#    """Gradient of the function
#
#        f(x) = L1(mu, x),
#
#    where L1(mu, x) is the Nesterov smoothed L1-norm.
#    """
#    alpha = (1.0 / mu) * beta
#    asnorm = np.abs(alpha)
#    i = asnorm > 1.0
#    alpha[i] = np.divide(alpha[i], asnorm[i])
#
#    return alpha


class L2(Function):

    def __init__(self, l, rng=RandomUniform(0, 1)):

        super(L2, self).__init__(l, rng=rng)

    def grad(self, x):
        """Sub-gradient of the function

            f(x) = l * |x|_2,

        where |x|_2 is the L2-norm.
        """
        norm_beta = norm2(x)
        if norm_beta > TOLERANCE:
            return x / norm_beta
        else:
            D = x.shape[0]
            u = (self.rng(D, 1) * 2.0) - 1.0  # [-1, 1]^D
            norm_u = norm2(u)
            a = self.rng()  # [0, 1]

            return (self.l * (a / norm_u)) * u


#def grad_l2(beta, rng=RandomUniform(0, 1)):
#    """Sub-gradient of the function
#
#        f(x) = |x|_2,
#
#    where |x|_2 is the L2-norm.
#    """
#    norm_beta = norm2(beta)
#    if norm_beta > TOLERANCE:
#
#        return beta / norm_beta
#    else:
#        D = beta.shape[0]
#        u = (rng(D, 1) * 2.0) - 1.0  # [-1, 1]^D
#        norm_u = norm2(u)
#        a = rng()  # [0, 1]
#
#        return u * (a / norm_u)


class L2Squared(Function):

    def __init__(self, l):

        super(L2Squared, self).__init__(l)

    def grad(self, x):
        """Gradient of the function

            f(x) = (l / 2) * |x|²_2,

        where |x|²_2 is the squared L2-norm.
        """
        return self.l * x


#def grad_l2_squared(beta, rng=None):
#    """Gradient of the function
#
#        f(x) = (1 / 2) * |x|²_2,
#
#    where |x|²_2 is the squared L2-norm.
#    """
#    return beta


class NesterovFunction(Function):

    __metaclass__ = abc.ABCMeta

    def __init__(self, l, A, mu=TOLERANCE, rng=RandomUniform(-1, 1),
                 norm=L2.grad, **kwargs):

        super(NesterovFunction, self).__init__(l, rng=rng, norm=norm, **kwargs)

        self.A = A
        self.mu = mu

    def grad(self, x):

        grad_Ab = 0
        for i in xrange(len(self.A)):
            Ai = self.A[i]
            Ab = Ai.dot(x)
            grad_Ab += Ai.T.dot(self.norm(Ab, self.rng))

        return self.l * grad_Ab

    def smoothed_grad(self, x):

        alpha = self.alpha(x)

        Aa = self.A[0].T.dot(alpha[0])
        for i in xrange(1, len(self.A)):
            Aa += self.A[i].T.dot(alpha[i])

        return self.l * Aa

    def alpha(self, x):
        """ Dual variable of the Nesterov function.
        """
        alpha = [0] * len(self.A)
        for i in xrange(len(self.A)):
            alpha[i] = self.A[i].dot(x) / self.mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def project(self, alpha):

        for i in xrange(len(alpha)):
            astar = alpha[i]
            normas = np.sqrt(np.sum(astar ** 2.0))
            if normas > 1.0:
                astar /= normas
            alpha[i] = astar

        return alpha


class TotalVariation(Function):

    def __init__(self, l, A, rng=RandomUniform(0, 1), **kwargs):

        super(TotalVariation, self).__init__(l, A=A, rng=rng, **kwargs)

    def grad(self, x):
        """Gradient of the function

            f(x) = TotalVariation(x),

        where TotalVariation(x) is the total variation function.
        """
        beta_flat = x.ravel()
        Ab = np.vstack([Ai.dot(beta_flat) for Ai in self.A]).T
        Ab_norm2 = np.sqrt(np.sum(Ab ** 2.0, axis=1))

        upper = Ab_norm2 > TOLERANCE
        grad_Ab_norm2 = Ab
        grad_Ab_norm2[upper] = (Ab[upper].T / Ab_norm2[upper]).T

        lower = Ab_norm2 <= TOLERANCE
        n_lower = lower.sum()

        if n_lower:
            D = len(self.A)
            vec_rnd = (self.rng(n_lower, D) * 2.0) - 1.0
            norm_vec = np.sqrt(np.sum(vec_rnd ** 2.0, axis=1))
            a = self.rng(n_lower)
            grad_Ab_norm2[lower] = (vec_rnd.T * (a / norm_vec)).T

        grad = np.vstack([self.A[i].T.dot(grad_Ab_norm2[:, i]) \
                          for i in xrange(len(self.A))])
        grad = grad.sum(axis=0)

        return self.l * grad.reshape(x.shape)

    @staticmethod
    def A_from_shape(shape):
        """Generates the linear operator for the total variation Nesterov
        function from the shape of a 3D image.

        Parameters
        ----------
        shape : List or tuple with 1, 2 or 3 integers. The shape of the 1D, 2D
                or 3D image. shape has the form (X,), (Y, X) or (Z, Y, X),
                where Z is the number of "layers", Y is the number of rows and
                X is the number of columns. The shape does not involve any
                intercept variables.
        """
        import scipy.sparse as sparse

        if not isinstance(shape, (list, tuple)):
            shape = [shape]
        while len(shape) < 3:
            shape = tuple([1] + list(shape))
        nz = shape[0]
        ny = shape[1]
        nx = shape[2]
        p = nx * ny * nz
        ind = np.arange(p).reshape((nz, ny, nx))
        if nx > 1:
            Ax = sparse.eye(p, p, 1, format='csr') - \
                 sparse.eye(p, p)
            zind = ind[:, :, -1].ravel()
            for i in zind:
                Ax.data[Ax.indptr[i]: \
                        Ax.indptr[i + 1]] = 0
            Ax.eliminate_zeros()
        else:
            Ax = sparse.csc_matrix((p, p), dtype=float)
        if ny > 1:
            Ay = sparse.eye(p, p, nx, format='csr') - \
                 sparse.eye(p, p)
            yind = ind[:, -1, :].ravel()
            for i in yind:
                Ay.data[Ay.indptr[i]: \
                        Ay.indptr[i + 1]] = 0
            Ay.eliminate_zeros()
        else:
            Ay = sparse.csc_matrix((p, p), dtype=float)
        if nz > 1:
            Az = (sparse.eye(p, p, ny * nx, format='csr') - \
                  sparse.eye(p, p))
            xind = ind[-1, :, :].ravel()
            for i in xind:
                Az.data[Az.indptr[i]: \
                        Az.indptr[i + 1]] = 0
            Az.eliminate_zeros()
        else:
            Az = sparse.csc_matrix((p, p), dtype=float)

        return [Ax, Ay, Az]

    @staticmethod
    def A_from_subset_mask(mask):
        """Generates the linear operator for the total variation Nesterov
        function from a mask for a 3D image.

        The binary mask marks the variables that are supposed to be smoothed.
        The mask has the same size as the input and output image.

        Parameters
        ----------
        mask : Numpy array. The mask. The mask does not involve any intercept
                variables.
        """
        import scipy.sparse as sparse

        while len(mask.shape) < 3:
            mask = mask[np.newaxis, :]

        nz, ny, nx = mask.shape
        mask = mask.astype(bool)
        zyx_mask = np.where(mask)
        Ax_i = list()
        Ax_j = list()
        Ax_v = list()
        Ay_i = list()
        Ay_j = list()
        Ay_v = list()
        Az_i = list()
        Az_j = list()
        Az_v = list()
    #    p = np.sum(mask)

        # Mapping from image coordinate to flat masked array.
        def im2flat(sub, dims):
            return sub[0] * dims[2] * dims[1] + \
                   sub[1] * dims[2] + \
                   sub[2]
    #    im2flat = np.zeros(mask.shape, dtype=int)
    #    im2flat[:] = -1
    #    im2flat[mask] = np.arange(p)
    #    im2flat[np.arange(p)] = np.arange(p)

        for pt in xrange(len(zyx_mask[0])):

            z, y, x = zyx_mask[0][pt], zyx_mask[1][pt], zyx_mask[2][pt]
            i_pt = im2flat((z, y, x), mask.shape)

            if z + 1 < nz and mask[z + 1, y, x]:
                Az_i += [i_pt, i_pt]
                Az_j += [i_pt, im2flat((z + 1, y, x), mask.shape)]
                Az_v += [-1., 1.]
            if y + 1 < ny and mask[z, y + 1, x]:
                Ay_i += [i_pt, i_pt]
                Ay_j += [i_pt, im2flat((z, y + 1, x), mask.shape)]
                Ay_v += [-1., 1.]
            if x + 1 < nx and mask[z, y, x + 1]:
                Ax_i += [i_pt, i_pt]
                Ax_j += [i_pt, im2flat((z, y, x + 1), mask.shape)]
                Ax_v += [-1., 1.]

        p = np.prod(mask.shape)
        Az = sparse.csr_matrix((Az_v, (Az_i, Az_j)), shape=(p, p))
        Ay = sparse.csr_matrix((Ay_v, (Ay_i, Ay_j)), shape=(p, p))
        Ax = sparse.csr_matrix((Ax_v, (Ax_i, Ax_j)), shape=(p, p))

        return [Ax, Ay, Az]


#def grad_tv(beta, A, rng=RandomUniform(0, 1)):
#    beta_flat = beta.ravel()
#    Ab = np.vstack([Ai.dot(beta_flat) for Ai in A]).T
#    Ab_norm2 = np.sqrt(np.sum(Ab ** 2.0, axis=1))
#
#    upper = Ab_norm2 > TOLERANCE
#    grad_Ab_norm2 = Ab
#    grad_Ab_norm2[upper] = (Ab[upper].T / Ab_norm2[upper]).T
#
#    lower = Ab_norm2 <= TOLERANCE
#    n_lower = lower.sum()
#
#    if n_lower:
#        D = len(A)
#        vec_rnd = (rng(n_lower, D) * 2.0) - 1.0
#        norm_vec = np.sqrt(np.sum(vec_rnd ** 2.0, axis=1))
#        a = rng(n_lower)
#        grad_Ab_norm2[lower] = (vec_rnd.T * (a / norm_vec)).T
#
#    grad = np.vstack([A[i].T.dot(grad_Ab_norm2[:, i]) for i in xrange(len(A))])
#    grad = grad.sum(axis=0)
#
#    return grad.reshape(beta.shape)


class GroupLasso(Function):

    def __init__(self, l, A, rng=RandomUniform(-1, 1), **kwargs):

        super(GroupLasso, self).__init__(l, A, rng=rng, **kwargs)

    def grad(self, x):
        """Gradient of the function

            f(x) = GroupLasso(x),

        where GroupLasso(x) is the group lasso (l1-l2) function.
        """
        grad_Ab = 0
        for i in xrange(len(self.A)):
            Ai = self.A[i]
            Ab = Ai.dot(x)
            grad_Ab += Ai.T.dot(L2.grad(Ab, self.rng))

        return self.l * grad_Ab

    @staticmethod
    def A_from_groups(num_variables, groups, weights=None, penalty_start=0):
        """Generates the linear operator for the group lasso Nesterov function
        from the groups of variables.

        Parameters:
        ----------
        num_variables : Integer. The total number of variables, including the
                intercept variable(s).

        groups : A list of lists. The outer list represents the groups and the
                inner lists represent the variables in the groups. E.g.
                [[1, 2], [2, 3]] contains two groups ([1, 2] and [2, 3]) with
                variable 1 and 2 in the first group and variables 2 and 3 in
                the second group.

        weights : List. Weights put on the groups. Default is weight 1 for each
                group.

        penalty_start : Non-negative integer. The number of variables to exempt
                from penalisation. Equivalently, the first index to be
                penalised. Default is 0, all variables are included.
        """
        import scipy.sparse as sparse

        if weights is None:
            weights = [1.0] * len(groups)

        A = list()
        for g in xrange(len(groups)):
            Gi = groups[g]
            lenGi = len(Gi)
            Ag = sparse.lil_matrix((lenGi, num_variables - penalty_start))
            for i in xrange(lenGi):
                w = weights[g]
                Ag[i, Gi[i] - penalty_start] = w

            # Matrix operations are a lot faster when the sparse matrix is csr
            A.append(Ag.tocsr())

        return A


#def grad_gl(beta, A, rng=RandomUniform(-1, 1)):
#
#    return _Nesterov_grad(beta, A, rng, grad_l2)


class SmoothedTotalVariation(TotalVariation, NesterovFunction):

    def __init__(self, l, A, mu=TOLERANCE):

        super(SmoothedTotalVariation, self).__init__(l, A, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = TotalVariation(mu, x),

        where TotalVariation(mu, x) is the Nesterov smoothed total variation
        function.
        """
        return self.smoothed_grad(x)

    def project(self, alpha):
        """ Projection onto the compact space of the smoothed TV function.
        """
        ax = alpha[0]
        ay = alpha[1]
        az = alpha[2]
        anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i = anorm > 1.0

        anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
        ax[i] = np.divide(ax[i], anorm_i)
        ay[i] = np.divide(ay[i], anorm_i)
        az[i] = np.divide(az[i], anorm_i)

        return [ax, ay, az]


#def grad_tvmu(beta, A, mu):
#
#    alpha = _Nestetov_alpha(beta, A, mu, _Nesterov_TV_project)
#
#    return _Nesterov_grad_smoothed(A, alpha)


class SmoothedGroupLasso(GroupLasso, NesterovFunction):

    def __init__(self, l, A, mu=TOLERANCE):

        super(SmoothedGroupLasso, self).__init__(l, A, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = GroupLasso(mu, x),

        where GroupLasso(mu, x) is the Nesterov smoothed group lasso function.
        """
        return self.smoothed_grad(x)


#def grad_glmu(beta, A, mu):
#
#    alpha = _Nestetov_alpha(beta, A, mu, _Nesterov_project)
#
#    return _Nesterov_grad_smoothed(A, alpha)


class SmoothedGroupTotalVariation(NesterovFunction):

    def __init__(self, l, A, mu=TOLERANCE):

        super(SmoothedGroupTotalVariation, self).__init__(l, A, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = GroupTotalVariation(mu, x),

        where GroupTotalVariation(mu, x) is the Nesterov smoothed group total
        variation function.
        """
        return self.smoothed_grad(x)

    def project(self, a):
        """ Projection onto the compact space of the smoothed Group TV
        function.
        """
        for g in xrange(0, len(a), 3):

            ax = a[g + 0]
            ay = a[g + 1]
            az = a[g + 2]
            anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
            i = anorm > 1.0

            anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
            ax[i] = np.divide(ax[i], anorm_i)
            ay[i] = np.divide(ay[i], anorm_i)
            az[i] = np.divide(az[i], anorm_i)

            a[g + 0] = ax
            a[g + 1] = ay
            a[g + 2] = az

        return a

    @staticmethod
    def A_from_subset_masks(masks, weights=None):
        """Generates the linear operator for the group total variation
        Nesterov function from a mask for a 3D image.

        Parameters
        ----------
        masks : List of numpy arrays. The mask for each group. Each mask is an
                integer (0 or 1) or boolean numpy array or the same shape as
                the actual data. The mask does not involve any intercept
                variables.

        weights : List of floats. The weights account for different group
                sizes, or incorporates some prior knowledge about the
                importance of the groups. Default value is the square roots of
                the group sizes.
        """
        import parsimony.functions.nesterov.tv as tv

        if isinstance(masks, tuple):
            masks = list(masks)

        A = []

        G = len(masks)
        for g in xrange(G):
            mask = masks[g]

            if weights is None:
                weight = np.sqrt(np.sum(mask))
            else:
                weight = weights[g]

            # Compute group A matrix
            Ag, _ = tv.A_from_subset_mask(mask)

            # Include the weights
            if weight != 1.0 and weight != 1:
                for A_ in Ag:
                    A_ *= weight

            A += Ag

        return A

    @staticmethod
    def A_from_rects(rects, shape, weights=None):
        """Generates the linear operator for the group total variation Nesterov
        function from the rectange of a 3D image.

        Parameters
        ----------
        rects : List of lists or tuples with 2-, 4- or 6-tuple elements. The
                shape of the patch of the 1D, 2D or 3D image to smooth. The
                elements of rects has the form ((x1, x2),), ((y1, y2),
                (x1, x2)) or ((z1, z2), (y1, y2), (x1, x2)), where z is the
                "layers", y rows and x is the columns and x1 means the first
                column to include, x2 is one beyond the last column to include,
                and similarly for y and z. The rect does not involve any
                intercept variables.

        shape : List or tuple with 1, 2 or 3 integers. The shape of the 1D, 2D
                or 3D image. shape has the form (X,), (Y, X) or (Z, Y, X),
                where Z is the number of "layers", Y is the number of rows and
                X is the number of columns. The shape does not involve any
                intercept variables.

        weights : List of floats. The weights account for different group
                sizes, or incorporates some prior knowledge about the
                importance of the groups. Default value is the square roots of
                the group sizes.
        """
        import parsimony.functions.nesterov.tv as tv

        A = []
        G = len(rects)
        for g in xrange(G):
            rect = rects[g]
            if len(rect) == 1:
                rect = [(0, 1), (0, 1), rect[0]]
            elif len(rect) == 2:
                rect = [(0, 1), rect[0], rect[1]]

            while len(shape) < 3:
                shape = tuple([1] + list(shape))

            mask = np.zeros(shape, dtype=bool)
            z1 = rect[0][0]
            z2 = rect[0][1]
            y1 = rect[1][0]
            y2 = rect[1][1]
            x1 = rect[2][0]
            x2 = rect[2][1]
            mask[z1:z2, y1:y2, x1:x2] = True

            if weights is None:
                weight = np.sqrt(np.sum(mask))
            else:
                weight = weights[g]

            # Compute group A matrix
            Ag, _ = tv.A_from_subset_mask(mask)

            # Include the weights
            if weight != 1.0 and weight != 1:
                for A_ in Ag:
                    A_ *= weight

            A += Ag

        return A


#def grad_grouptvmu(beta, A, mu):
#
#    alpha = _Nestetov_alpha(beta, A, mu, _Nesterov_GroupTV_project)
#
#    return _Nesterov_grad_smoothed(A, alpha)


#def _Nesterov_GroupTV_project(a):
#    """ Projection onto the compact space of the smoothed Group TV function.
#    """
#    for g in xrange(0, len(a), 3):
#
#        ax = a[g + 0]
#        ay = a[g + 1]
#        az = a[g + 2]
#        anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
#        i = anorm > 1.0
#
#        anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
#        ax[i] = np.divide(ax[i], anorm_i)
#        ay[i] = np.divide(ay[i], anorm_i)
#        az[i] = np.divide(az[i], anorm_i)
#
#        a[g + 0] = ax
#        a[g + 1] = ay
#        a[g + 2] = az
#
#    return a


#def _Nesterov_grad(beta, A, rng=RandomUniform(-1, 1), grad_norm=grad_l2):
#
#    grad_Ab = 0
#    for i in xrange(len(A)):
#        Ai = A[i]
#        Ab = Ai.dot(beta)
#        grad_Ab += Ai.T.dot(grad_norm(Ab, rng))
#
#    return grad_Ab


#def _Nesterov_grad_smoothed(A, alpha):
#
#    Aa = A[0].T.dot(alpha[0])
#    for i in xrange(1, len(A)):
#        Aa += A[i].T.dot(alpha[i])
#
#    return Aa


#def _Nestetov_alpha(beta, A, mu, proj):
#    """ Dual variable of the Nesterov function.
#    """
#    alpha = [0] * len(A)
#    for i in xrange(len(A)):
#        alpha[i] = A[i].dot(beta) / mu
#
#    # Apply projection
#    alpha = proj(alpha)
#
#    return alpha


#def _Nesterov_project(alpha):
#
#    for i in xrange(len(alpha)):
#        astar = alpha[i]
#        normas = np.sqrt(np.sum(astar ** 2.0))
#        if normas > 1.0:
#            astar /= normas
#        alpha[i] = astar
#
#    return alpha


#def _Nesterov_TV_project(alpha):
#    """ Projection onto the compact space of the smoothed TV function.
#    """
#    ax = alpha[0]
#    ay = alpha[1]
#    az = alpha[2]
#    anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
#    i = anorm > 1.0
#
#    anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
#    ax[i] = np.divide(ax[i], anorm_i)
#    ay[i] = np.divide(ay[i], anorm_i)
#    az[i] = np.divide(az[i], anorm_i)
#
#    return [ax, ay, az]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
