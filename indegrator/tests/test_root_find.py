"""Test suite for the root finding algorithms."""

import os
import numpy
from numpy.testing import (TestCase,)
from numpy.testing import (assert_array_almost_equal, assert_allclose)

from root_find import (root_find, NevilleInterpolationPolynomial)

case1_roots = (-1.5,  0.0,  0.5)
case2_roots = (-1.5,  0.0,  0.5)
case3_roots = (-1.0, -0.5,  0.0,  2.5)
case4_roots = (-2.2,  0.0,  1.0,  1.7)
case5_roots = (-2.2, -1.7, -0.5,  0.3)
case6_roots = (-0.2,  0.0,  0.7,  1.3, 2.3)
case7_roots = (-1.3, -0.7, -0.2, -0.2, 2.0)


def get_polynomial_from_roots(roots):
    """Create polynomial evaluating functor for given roots."""
    class Poly(object):
        roots = ()

        def __call__(self, t):
            # evaluate a multivariate polynomial at different nodes
            # with given roots
            # NOTE: use newaxis to calculate root_ij - tk then prod over all j
            t = numpy.asarray(t)
            roots = numpy.asarray(self.roots)
            return (roots[:, :, numpy.newaxis] - t).prod(axis=1)

    poly = Poly()
    roots = numpy.mat(roots)
    poly.roots = roots

    return poly


def neville_vs_polynomial(roots):
    """Evaluate reference and Neville polynomial and return values."""
    # get polynomial
    roots = numpy.mat(roots)
    polynomial = get_polynomial_from_roots(roots)

    # define evaluation interval
    t0 = -10.0
    tf =  10.0

    # get evaluation nodes for unique determination of polynomial
    ts = numpy.random.uniform(t0, tf, roots.shape[1]+1)
    ys = polynomial(ts)

    # setup Neville evaluation
    neville = NevilleInterpolationPolynomial(ts, ys)

    # get values at evaluation grid
    ts = numpy.random.uniform(t0, tf, 100)

    yr = polynomial(ts)
    yn = neville(ts)

    return yr, yn


def root_finding_test(roots, tol, eps):
    """Evaluate reference and Neville polynomial and return values."""
    # get polynomial
    roots = numpy.mat(roots)
    polynomial = get_polynomial_from_roots(roots)

    # define evaluation interval
    t0 = -10.0
    tf =  10.0

    # get root guesses
    ts = numpy.linspace(t0, tf, 1000, True)
    ys = polynomial(ts)

    # get dimensions
    nts = ts.shape[0]
    nxd = ys.shape[0]

    # solutions of root find
    rs = []
    vs = []

    # add any intermediate value
    for its in range(nts-1):
        for ixd in range(nxd):
            if numpy.sign(ys[ixd, its]*ys[ixd, its + 1]) < 0:
                t0 = ts[its]
                tf = ts[its + 1]
                t  = numpy.random.uniform(t0, tf)
                y  = polynomial(t)
                r, v = root_find(y, t, t0, tf, polynomial, tol, eps)
                rs.append(r)
                vs.append(v)

    rs = numpy.asarray(rs)
    vs = numpy.asarray(vs)
    return numpy.asarray((rs[:, 0, 0], vs[:, 0, 0]))


class TestNevilleInterpolationPolynomial(TestCase):

    """Testing the Aitkin-Neville interpolation."""

    def test_polynomial_evaluation_case1(self):
        yr, yn = neville_vs_polynomial(case1_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_case2(self):
        yr, yn = neville_vs_polynomial(case2_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_case3(self):
        yr, yn = neville_vs_polynomial(case3_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_case4(self):
        yr, yn = neville_vs_polynomial(case4_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_case5(self):
        yr, yn = neville_vs_polynomial(case5_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_case6(self):
        yr, yn = neville_vs_polynomial(case6_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_case7(self):
        yr, yn = neville_vs_polynomial(case7_roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_multivariate1(self):
        roots = numpy.mat((case1_roots, case2_roots))
        yr, yn = neville_vs_polynomial(roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_multivariate2(self):
        roots = numpy.mat((case3_roots, case4_roots, case5_roots))
        yr, yn = neville_vs_polynomial(roots)
        assert_array_almost_equal(yr, yn)

    def test_polynomial_evaluation_multivariate3(self):
        roots = numpy.mat((case6_roots, case7_roots))
        yr, yn = neville_vs_polynomial(roots)
        assert_array_almost_equal(yr, yn)


class TestRootFind(TestCase):

    """Testing root finding algorithm."""

    TOL = 1e-10
    EPS = 1e-16
    DEC = 7

    def test_root_find_case1(self):
        roots = case1_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case2(self):
        roots = case2_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case2(self):
        roots = case2_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case3(self):
        roots = case3_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case4(self):
        roots = case4_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case5(self):
        roots = case5_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case6(self):
        roots = case6_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)

    def test_root_find_case7(self):
        roots = case7_roots
        res = root_finding_test(roots, tol=self.TOL, eps=self.EPS)
        rs = res[0, :]
        vs = res[1, :]
        nulls = numpy.zeros(vs.shape)
        assert_array_almost_equal(rs, roots, self.DEC)
        assert_array_almost_equal(vs, nulls, self.DEC)


if __name__ == "__main__":
    try:
        import nose
        nose.runmodule()
    except ImportError:
        print 'Please install nosetests for unit testing!'
