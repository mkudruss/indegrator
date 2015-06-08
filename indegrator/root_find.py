# -*- coding: utf-8 -*-
"""
Efficient root finding algorithm and function evaluation using Aitkin-Neville
interpolation formulae.
"""

import numpy


class NevilleInterpolationPolynomial(object):

    """
    Polynomial interpolations using Atkin-Neville algorithm.

    Through any N points y[i] = f(x[i]), there is a unique polynomial P order
    N-1.  Neville's algorithm is used for finding interpolates of this unique
    polynomial at any point x.
    """

    def __init__(self, xs, ys):
        """
        Setup interpolant values.

        Parameters
        ----------
        x : array-like (n,)
            time nodes for evaluated values
        y : array-like (nxd, n)
            function values on evaluated nodes x
        """

        # time nodes for evaluated values
        self._Xs = numpy.asarray(xs)
        self._Ys = numpy.mat(ys)

    def __call__(self, x0, y0=[]):
        """
        Evaluate interpolation polynomial using Atkin-Neville algorithm.

        Parameters
        ----------
        x0 : array-like (nts)
            time nodes for evaluation

        Returns
        -------
        y0 : array-like (nxd, nts)
            evaluated function values on nodes x0
        """
        # cast to numpy
        x0 = numpy.asarray(x0)
        y0 = numpy.mat(y0)

        # rename for convenience
        Xs = self._Xs
        Ys = self._Ys

        # get dimensions of
        print Ys.shape
        nxd, n = Ys.shape
        if y0:
            nyd, nts = y0.shape
        else:
            nyd = nxd
            nts = x0.shape[-1]

        # add some asserts on dimensions
        assert nyd == nxd
        assert n   == Xs.shape[0]
        assert nts == x0.shape[0]

        # evaluate interpolation polynomial
        p = numpy.zeros((nyd, nts, n))
        for k in range(n):
            for j in range(n-k):
                if k == 0:
                    p[:, :, j] = numpy.tile(Ys[:, j], (1, nts))
                else:
                    p[:, :, j] = (
                        (x0[:]-Xs[j+k])*p[:, :, j]+(Xs[j]-x0[:])*p[:, :, j+1]
                    ) / (Xs[j]-Xs[j+k])
        else:
            # return values
            if not y0:
                return p[:, :, 0]
            else:
                    y0[:, :] = p[:, :, 0]


def root_find(s, t, t0, t1, func, tol=1e-08, eps=1e-16):
    """
    Root-finding algorithm due to Brent & Decker [1].

    Searches for root ts of function func in the interval [t0, t0] from a given
    initial guess t up to a given tolerance tol using inverse quadratic
    interpolation safeguarded by *regula falsi* and a bisection strategy.

    Parameters
    ----------
    s : double
        value of at expected root

    t : double
        initial guess of root of function

    t0 : double
        left border of search interval

    t1 : double
        right border of search interval

    func : univariate function with interface f(t)
        function analyze for roots

    tol : double
        root finding accuracy tolerance up to what a root is accepted,
        defaults to single precision 1e-08

    eps : double
        machine precision, defaults to double precision 1e-16

    Returns
    -------
    root : double
        accurate estimate of actual root up to a given tolerance

    value : double
        accurate estimate of actual root up to a given tolerance

    """
    # a,b,c: abscissae fa,fb,fc: corresponding function values
    a = t0
    b = t1
    c = a

    fa = func(a)
    fb = func(b)
    fc = fa

    # Main iteration loop
    n_iter = 0
    while True:
        n_iter += 1
        prev_step = b-a  # Distance from the last but one to the last approx.
        tol_act   = 0.0   # Actual tolerance
        new_step  = 0.0   # Step at this iteration

        # Interpolation step is calculated in the form p/q
        # division operations is delayed until the last moment
        p = 0.0
        q = 0.0

        if numpy.abs(fc) < numpy.abs(fb):
            # Swap data for b to be the best approximation
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol_act  = 2.0*eps*numpy.abs(b) + tol/2.0
        new_step = (c - b)/2.0

        # Acceptable approximation found ?
        if numpy.abs(new_step) <= tol_act or fb == 0.0:
            root = b
            value = fb
            print 'finished after {} iterations.'.format(n_iter)
            return (root, value)

        # Interpolation may be tried if prev_step was large enough and in true direction
        if numpy.abs(prev_step) >= tol_act and numpy.abs(fa) > numpy.abs(fb):
            cb = c-b

            if a == c:
                # If we have only two distinct points, linear interpolation can only be applied
                t1 = fb / fa
                p  = cb * t1
                q  = 1.0 - t1
            else:
                # Inverse quadratic interpolation
                q  = fa/fc
                t1 = fb/fc
                t2 = fb/fa
                p  = t2 * (cb*q*(q - t1) - (b - a)*(t1 - 1.0))
                q  = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)

            # p was calculated with the opposite sign make p positive and assign possible minus to q
            if p > 0.0:
                q = -q
            else:
                p = -p

            # If b+p/q falls in [b,c] and isn't too large, it is accepted
            # If p/q is too large then the bisection procedure can reduce [b,c] range to a larger extent
            if (p < 0.75*cb*q - numpy.abs(tol_act*q)/2.0
            and p < numpy.abs(prev_step*q/2.0)):
                new_step = p/q

        # Adjust the step to be not less than tolerance
        if numpy.abs(new_step) < tol_act:
            if new_step > 0.0:
                new_step = tol_act
            else:
                new_step = -tol_act

        # Save the previous approximate
        a  = b
        fa = fb

        # Do step to a new approximation
        b += new_step
        fb = func(b)

        # Adjust c for it to have a sign opposite to that of b
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
