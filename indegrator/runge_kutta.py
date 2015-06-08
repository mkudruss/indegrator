# -*- coding: utf-8 -*-
"""Collection of Runge-Kutta methods."""

import numpy

class RK4(object):

    def __init__(self, backend):

        self.printlevel = 0

        self.NY  = 0           # number of differential variables y
        self.NZ  = 0           # number of algebraic variables z
        self.NX  = 0           # number of variables x = (y,z)
        self.NP  = 0           # number of parameters
        self.NU  = 0           # number of control functions
        self.NQI = 0           # number of q in one control interval

        self.backend = backend

    def zo_check(self, ts, x0, p, q):

        # set dimeions
        self.M   = ts.size              # number of time steps
        self.NQ  = self.NU*self.M*2     # number of control variables

        # assert that the dimensions match

        self.NX = x0.size
        self.NP = p.size

        self.NU = q.shape[0]
        self.M  = q.shape[1]

        # assign variables
        self.ts = ts
        self.x0 = x0
        self.p  = p
        self.q  = q

        # allocate memory
        self.xs  = numpy.zeros((self.M, self.NX))
        self.f   = numpy.zeros(self.NX)
        self.u   = numpy.zeros(self.NU)

    def fo_check(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        self.zo_check(ts, x0, p, q)

        self.P = x0_dot.shape[1]

        assert self.NP == p_dot.shape[0]

        assert self.P == p_dot.shape[1]
        assert self.P == q_dot.shape[3]

        # assign variables
        self.x0_dot = x0_dot
        self.p_dot  = p_dot
        self.q_dot  = q_dot

        # allocate memory
        self.xs_dot = numpy.zeros((self.M, self.NX, self.P))
        self.f_dot  = numpy.zeros((self.NX, self.P))
        self.u_dot  = numpy.zeros((self.NU, self.P))


    def so_check(self, ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p, p_dot2, p_dot1, p_ddot,
                 q, q_dot2, q_dot1, q_ddot):
        self.zo_check(ts, x0, p, q)

        self.P1 = x0_dot1.shape[1]
        self.P2 = x0_dot2.shape[1]

        assert self.NP == p_dot1.shape[0]
        assert self.NP == p_dot2.shape[0]
        assert self.NP == p_ddot.shape[0]

        assert self.P1 == p_dot1.shape[1]
        assert self.P1 == q_dot1.shape[3]

        assert self.P2 == p_dot2.shape[1]
        assert self.P2 == q_dot2.shape[3]

        assert self.P1 == x0_ddot.shape[1]
        assert self.P1 == p_ddot.shape[1]
        assert self.P1 == q_ddot.shape[3]

        assert self.P2 == x0_ddot.shape[2]
        assert self.P2 == p_ddot.shape[2]
        assert self.P2 == q_ddot.shape[4]


        # assign variables
        self.x0_dot1 = x0_dot1
        self.p_dot1  = p_dot1
        self.q_dot1  = q_dot1

        self.x0_dot2 = x0_dot2
        self.p_dot2  = p_dot2
        self.q_dot2  = q_dot2

        self.x0_ddot = x0_ddot
        self.p_ddot  = p_ddot
        self.q_ddot  = q_ddot

        # allocate memory
        self.xs_dot1 = numpy.zeros((self.M, self.NX, self.P1))
        self.xs_dot2 = numpy.zeros((self.M, self.NX, self.P2))
        self.xs_ddot = numpy.zeros((self.M, self.NX, self.P1, self.P2))

        self.f_dot1  = numpy.zeros((self.NX, self.P1))
        self.f_dot2  = numpy.zeros((self.NX, self.P2))
        self.f_ddot  = numpy.zeros((self.NX, self.P1, self.P2))

        self.u_dot1  = numpy.zeros((self.NU, self.P1))
        self.u_dot2  = numpy.zeros((self.NU, self.P2))
        self.u_ddot  = numpy.zeros((self.NU, self.P1, self.P2))


    def zo_forward(self, ts, x0, p, q):
        self.zo_check(ts, x0, p, q)

        self.xs[0, :] = x0

        t = numpy.zeros(1)
        K1    = numpy.zeros(self.f.shape)
        K2    = numpy.zeros(self.f.shape)
        K3    = numpy.zeros(self.f.shape)
        K4    = numpy.zeros(self.f.shape)
        y     = numpy.zeros(self.f.shape)

        for i in range(self.M-1):
            self.update_u(i)

            h     = self.ts[i+1] - self.ts[i]
            h2    = h/2.0

            # K1    = h*f(t, y, p, u)
            t[0]  = self.ts[i]
            y[:]  = self.xs[i, :]
            self.backend.ffcn(t, y, K1, self.p, self.u )
            K1   *= h

            # K2    = h*f(t + h2, y + 0.5*K1, p, u)
            t[0]  = self.ts[i] + h2
            y[:]  = self.xs[i, :] + 0.5*K1
            self.backend.ffcn(t, y, K2, self.p, self.u )
            K2   *= h

            # K3    = h*f(t + h2, y + 0.5*K2, p, u)
            t[0]  = self.ts[i] + h2
            y[:]  = self.xs[i, :] + 0.5*K2
            self.backend.ffcn(t, y, K3, self.p, self.u )
            K3   *= h

            # K4    = h*f(t + h, y + K3, p, u)
            t[0]  = self.ts[i] + h
            y[:]  = self.xs[i, :] + K3
            self.backend.ffcn(t, y, K4, self.p, self.u )
            K4   *= h

            self.xs[i + 1, :]  = self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4)

    def fo_forward_xpu(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        self.fo_check(ts, x0, x0_dot, p, p_dot, q, q_dot)

        self.xs[0, :]         = x0
        self.xs_dot[0, :, :]  = x0_dot

        t        = numpy.zeros(1)
        K1       = numpy.zeros(self.f.shape)
        K2       = numpy.zeros(self.f.shape)
        K3       = numpy.zeros(self.f.shape)
        K4       = numpy.zeros(self.f.shape)
        y        = numpy.zeros(self.f.shape)

        K1_dot   = numpy.zeros(self.f.shape + (self.P,))
        K2_dot   = numpy.zeros(self.f.shape + (self.P,))
        K3_dot   = numpy.zeros(self.f.shape + (self.P,))
        K4_dot   = numpy.zeros(self.f.shape + (self.P,))
        y_dot    = numpy.zeros(self.f.shape + (self.P,))

        for i in range(self.M-1):
            self.update_u_dot(i)
            h     = self.ts[i+1] - self.ts[i]
            h2    = h/2.0

            # K1      = h*f(t, y, p, u)
            t[0]      = self.ts[i]
            y[:]      = self.xs[i, :]
            y_dot[:]  = self.xs_dot[i, :]
            self.backend.ffcn_dot(t, y, y_dot,
                                K1, K1_dot,
                                self.p, self.p_dot,
                                self.u, self.u_dot )
            K1       *= h
            K1_dot   *= h

            # K2    = h*f(t + h2, y + 0.5*K1, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K1
            y_dot[:]  = self.xs_dot[i, :] + 0.5*K1_dot
            self.backend.ffcn_dot(t, y, y_dot,
                            K2, K2_dot,
                            self.p, self.p_dot,
                            self.u, self.u_dot )
            K2       *= h
            K2_dot   *= h

            # K3    = h*f(t + h2, y + 0.5*K2, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K2
            y_dot[:]  = self.xs_dot[i, :] + 0.5*K2_dot
            self.backend.ffcn_dot(t, y, y_dot,
                            K3, K3_dot,
                            self.p, self.p_dot,
                            self.u, self.u_dot )
            K3       *= h
            K3_dot   *= h


            # K4    = h*f(t + h, y + K3, p, u)
            t[0]      = self.ts[i] + h
            y[:]      = self.xs[i, :] + K3
            y_dot[:]  = self.xs_dot[i, :] + K3_dot
            self.backend.ffcn_dot(t, y, y_dot,
                            K4, K4_dot,
                            self.p, self.p_dot,
                            self.u, self.u_dot )
            K4       *= h
            K4_dot   *= h

            self.xs[i + 1, :]      = self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4)
            self.xs_dot[i + 1, :]  = self.xs_dot[i,:] +  (1./6.0)*(K1_dot + 2*K2_dot + 2*K3_dot + K4_dot)


    def so_forward_xpu_xpu(self, ts, x0, x0_dot2, x0_dot1, x0_ddot,
                                     p,   p_dot2,  p_dot1, p_ddot,
                                     q,   q_dot2,  q_dot1, q_ddot):

        self.so_check(ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p,   p_dot2,  p_dot1, p_ddot,
                 q,   q_dot2,  q_dot1, q_ddot)


        self.xs[0, :]          = x0
        self.xs_dot1[0, :, :]  = x0_dot1
        self.xs_dot2[0, :, :]  = x0_dot2
        self.xs_ddot[0, :, :]  = x0_ddot

        t        = numpy.zeros(1)
        K1       = numpy.zeros(self.f.shape)
        K2       = numpy.zeros(self.f.shape)
        K3       = numpy.zeros(self.f.shape)
        K4       = numpy.zeros(self.f.shape)
        y        = numpy.zeros(self.f.shape)

        K1_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        K2_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        K3_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        K4_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        y_dot1    = numpy.zeros(self.f.shape + (self.P1,))

        K1_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        K2_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        K3_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        K4_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        y_dot2    = numpy.zeros(self.f.shape + (self.P2,))

        K1_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        K2_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        K3_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        K4_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        y_ddot    = numpy.zeros(self.f.shape + (self.P1, self.P2))


        for i in range(self.M-1):
            self.update_u_ddot(i)
            h     = self.ts[i+1] - self.ts[i]
            h2    = h/2.0

            # K1      = h*f(t, y, p, u)
            t[0]      = self.ts[i]
            y[:]      = self.xs[i, :]
            y_dot1[:]  = self.xs_dot1[i, :]
            y_dot2[:]  = self.xs_dot2[i, :]
            y_ddot[:]  = self.xs_ddot[i, :]
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K1, K1_dot2, K1_dot1, K1_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K1        *= h
            K1_dot1   *= h
            K1_dot2   *= h
            K1_ddot   *= h

            # K2    = h*f(t + h2, y + 0.5*K1, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K1
            y_dot1[:]  = self.xs_dot1[i, :] + 0.5 * K1_dot1
            y_dot2[:]  = self.xs_dot2[i, :] + 0.5 * K1_dot2
            y_ddot[:]  = self.xs_ddot[i, :] + 0.5 * K1_ddot
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K2, K2_dot2, K2_dot1, K2_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K2        *= h
            K2_dot1   *= h
            K2_dot2   *= h
            K2_ddot   *= h

            # K3    = h*f(t + h2, y + 0.5*K2, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K2
            y_dot1[:]  = self.xs_dot1[i, :] + 0.5 * K2_dot1
            y_dot2[:]  = self.xs_dot2[i, :] + 0.5 * K2_dot2
            y_ddot[:]  = self.xs_ddot[i, :] + 0.5 * K2_ddot
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K3, K3_dot2, K3_dot1, K3_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K3        *= h
            K3_dot1   *= h
            K3_dot2   *= h
            K3_ddot   *= h


            # K4    = h*f(t + h, y + K3, p, u)
            t[0]      = self.ts[i] + h
            y[:]      = self.xs[i, :] + K3
            y_dot1[:]  = self.xs_dot1[i, :] + K3_dot1
            y_dot2[:]  = self.xs_dot2[i, :] + K3_dot2
            y_ddot[:]  = self.xs_ddot[i, :] + K3_ddot
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K4, K4_dot2, K4_dot1, K4_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K4        *= h
            K4_dot1   *= h
            K4_dot2   *= h
            K4_ddot   *= h

            self.xs[i + 1, :]      = self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4)
            self.xs_dot1[i + 1, :]  = self.xs_dot1[i,:] +  (1./6.0)*(K1_dot1 + 2*K2_dot1 + 2*K3_dot1 + K4_dot1)
            self.xs_dot2[i + 1, :]  = self.xs_dot2[i,:] +  (1./6.0)*(K1_dot2 + 2*K2_dot2 + 2*K3_dot2 + K4_dot2)
            self.xs_ddot[i + 1, :]  = self.xs_ddot[i,:] +  (1./6.0)*(K1_ddot + 2*K2_ddot + 2*K3_ddot + K4_ddot)



    def fo_reverse(self, xs_bar):

        numpy.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=200, suppress=None, nanstr=None, infstr=None, formatter=None)
        self.xs_bar = xs_bar.copy()

        t      = numpy.zeros(1)
        K1     = numpy.zeros(self.f.shape)
        K2     = numpy.zeros(self.f.shape)
        K3     = numpy.zeros(self.f.shape)
        K4     = numpy.zeros(self.f.shape)
        y      = numpy.zeros(self.f.shape)

        K1_bar = numpy.zeros(self.f.shape)
        K2_bar = numpy.zeros(self.f.shape)
        K3_bar = numpy.zeros(self.f.shape)
        K4_bar = numpy.zeros(self.f.shape)
        y_bar  = numpy.zeros(self.f.shape)

        self.x0_bar = numpy.zeros(self.x0.shape)
        self.f_bar  = numpy.zeros(self.f.shape)
        self.p_bar  = numpy.zeros(self.p.shape)
        self.q_bar  = numpy.zeros(self.q.shape)
        self.u_bar  = numpy.zeros(self.u.shape)

        ts = self.ts
        p  = self.p
        u  = self.u
        xs = self.xs

        p_bar = self.p_bar
        u_bar = self.u_bar
        xs_bar = self.xs_bar

        for i in range(self.M-1)[::-1]:
            self.update_u(i)
            h = ts[i+1] - ts[i]
            h2 = h/2.0

            # forward K1 = h*f[t, y, p, u]
            t[0]  = ts[i]
            y[:]  = xs[i, :]
            self.backend.ffcn(t, y, K1, p, u)
            K1   *= h

            # forward K2 = h*f[t + h2, y + 0.5*K1, p, u]
            t[0]  = ts[i] + h2
            y[:]  = xs[i, :] + 0.5*K1
            self.backend.ffcn(t, y, K2, p, u)
            K2   *= h

            # forward K3 = h*f[t + h2, y + 0.5*K2, p, u]
            t[0]  = ts[i] + h2
            y[:]  = xs[i, :] + 0.5*K2
            self.backend.ffcn(t, y, K3, p, u)
            K3   *= h

            # foward K4    = h*f(t + h, y + K3, p, u)
            t[0]  = self.ts[i] + h
            y[:]  = self.xs[i, :] + K3
            self.backend.ffcn(t, y, K4, self.p, self.u )
            K4   *= h

            # forward accumulation
            # from numpy.testing import assert_almost_equal
            # assert_almost_equal(self.xs[i + 1, :],
            #                     self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4))

            # reverse accumulation
            y_bar[:]               = 0.
            u_bar[:]               = 0.

            self.xs_bar[i, :]     += self.xs_bar[i + 1, :]
            K1_bar[:]              = (1./6.) * self.xs_bar[i + 1, :]
            K2_bar[:]              = (2./6.) * self.xs_bar[i + 1, :]
            K3_bar[:]              = (2./6.) * self.xs_bar[i + 1, :]
            K4_bar[:]              = (1./6.) * self.xs_bar[i + 1, :]
            xs_bar[i+1, :]         = 0.

            # reverse K4
            t[0]                   = ts[i] + h
            K4_bar                *= h
            y[:]                   = self.xs[i, :] + K3
            self.backend.ffcn_bar(t, y, y_bar, K4, K4_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            K3_bar                += y_bar
            y_bar[:]               = 0.

            # reverse K3
            t[0]                   = ts[i] + h2
            K3_bar                *= h
            y[:]                   = self.xs[i, :] + 0.5*K2
            self.backend.ffcn_bar(t, y, y_bar, K3, K3_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            K2_bar                += 0.5*y_bar
            y_bar[:]               = 0.

            # reverse K2
            t[0]                   = ts[i] + h2
            K2_bar                *= h
            y[:]  = xs[i, :] + 0.5*K1
            self.backend.ffcn_bar(t, y, y_bar, K2, K2_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            K1_bar                += 0.5*y_bar
            y_bar[:]               = 0.

            # reverse K1
            t[0]                   = ts[i]
            K1_bar                *= h
            y[:]                   = self.xs[i, :]
            self.backend.ffcn_bar(t, y, y_bar, K1, K1_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            y_bar[:]               = 0.

            self.update_u_bar(i)

        self.x0_bar[:] += self.xs_bar[0, :]
        self.xs_bar[0, :] = 0.


    def update_u(self, i):
        self.u[:] = self.q[:, i, 0]

    def update_u_dot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot[:, :] = self.q_dot[:, i, 0, :]

    def update_u_bar(self, i):
        self.q_bar[:, i, 0] += self.u_bar[:]
        self.u_bar[:] = 0.

    def update_u_ddot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot1[:, :] = self.q_dot1[:, i, 0, :]
        self.u_dot2[:, :] = self.q_dot2[:, i, 0, :]
        self.u_ddot[:, :] = self.q_ddot[:, i, 0, :, :]


class RKF45SW(object):

    """
    A continuous Runge-Kutta-Fehlberg integrator.

    Features
    --------
    * continuous least-squares function support,
    * automatic detection of implicit switches in the states & right-hand side,
    * treats variational differential equations (not varied trajectories)

    based on C implementation
    (c) Christian Kirches, 2006

    based on rkfXXadj.cpp
    (c) Leonard Wirsching, 2004

    partially based on FORTRAN implementation rkf45s.F
    (c) Andreas Schaefer, 1996

    References
    ----------
    *  Fehlberg, E.: Klassische Runge-Kutta-Formeln vierter
       und niedrigerer Ordnung. Computing 6, 1970, pp. 61-70
    *  Stoer, J., and R. Bulirsch: Introduction to Numerical
       Analysis. Springer, New York, 1993.
       (formula for step size estimate)
    *  Bock, H.-G.: Numerical treatment of inverse problems in
       chemical reaction kinetics. In K.H. Ebert, P. Deuflhard, and
       W. Jaeger (eds.): Modelling of Chemical Reaction Systems
       (Springer Series in Chemical Physics 18). Springer, Heidelberg, 1981.
       (IND principle)
    *  Shampine, L. F.: Numerical Solution of Ordinary Differential
       Equations. Chapman & Hall, New York, 1994.
       (initial step size selection)
    *  Mombaur, K. D.: Stability Optimization of Open Loop Controlled
       Walking Robots, PhD thesis, University of Heidelberg, 2001.
       (1st and 2nd order sensitivity updates for sensitivity matrices)
    *  Enright, W., Jackson, K.R., Norsett, S.P., Thomsen, P.G.: Interpolants
       for Runge-Kutta Formulas. ACM Transactions on Mathematical Software
       (TOMS), Vol. 12, No. 3, September 1986, pp. 193-218 (interpolants)
    """

    # orders
    _orders = numpy.array([4, 5])
    # stage counts
    _stages = numpy.array([5, 6])

    # butcher-tableau of the Runge-Kutta method
    #  0 |
    # a0 | b00
    # a1 | b10 b11
    # a2 | b20 b21 b22
    # .. | ... ... ...
    # as | bs0 bs1 bs2 ... bss
    # ------------------------
    #    | c00 c01 c02 ... c0s
    #    | c10 c11 c12 ... c1s

    # coefficients alpha (times)
    _As = numpy.array([
        0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0
    ])
    # triangular coefficient matrix beta
    _Bs = numpy.array([
        [       1.0/4.0,            0.0,            0.0,           0.0,        0.0],
        [      3.0/32.0,       9.0/32.0,            0.0,           0.0,        0.0],
        [ 1932.0/2197.0, -7200.0/2197.0,  7296.0/2197.0,           0.0,        0.0],
        [   439.0/216.0,           -8.0,   3680.0/513.0, -845.0/4104.0,        0.0],
        [     -8.0/27.0,            2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0],
    ])
    # coefficients c of lower-order-method
    # coefficients c of higher-order method
    _Cs = numpy.array([
        [25.0/216.0, 0.0, 1408.0/2565.0,  2197.0/4104.0,  -1.0/5.0,       0.0],
        [-1.0/360.0, 0.0,  128.0/4275.0, 2197.0/75240.0, -1.0/50.0, -2.0/55.0],
    ])

    # coefficients for Horn's locally sixth-order approximation in tau=0.6
    _ws = numpy.array([
              1559.0/  12500.0,
                           0.0,
            153856.0/ 296875.0,
             68107.0/2612500.0,
              -243.0/  31250.0,
             -2106.0/  34375.0
    ])

    def _get_interpolation_polynomial (self, x0, x1, ks, h, res):
        """
        Get interpolation polynomial on intervall [x0, x1].

        Polynomial interpolations using Atkin-Neville algorithm. Callable
        evaluates 6th order Hermite-Birkhoff polynomial recycling evaluation of
        the embedded, continuous Runge-Kutta-Fehlberg method.

        Parameters
        ----------
        x0 : array-like of shape (NX,)
            Solution y(t)
        x1 : array-like of shape (NX,)
            Solution y(t+h)
        ks : array-like of shape (7, NX)
            Matrix of approximate derivatives, we need y'(t), y'(t+h)
        h : scalar
            Step size h

        Returns
        -------
        res : Functor
            Cubic Hermite polynomial evaluation using Atkin-Neville's algorithm
        """
        # check dimensions of derivatives
        assert ks.shape == (7, self.NX)

        # allocate memory
        Xs = numpy.zeros(8, )
        Ys = numpy.zeros(8, self.NX)

        # assign function evaluations from integration
        Xs[:-1,]   = 0.0
        Ys[:-1, :] = ks[:,:]

        # compute Horn's locally sixth-order approximation in tau=0.6
        Ys[:, -1] = x0 + h*(
            w[0]*ks[:, 0] + w[2]*ks[:, 2] + w[3]*ks[:, 3] +
            w[4]*ks[:, 4] + w[5]*ks[:, 5]
        )

        # return interpolation polynomial
        return NevilleInterpolationPolynomial(Xs, Ys)

    def __init__(self, backend):
        """Initialize integrator."""
        self.printlevel = 0

        self.NY  = 0           # number of differential variables y
        self.NZ  = 0           # number of algebraic variables z
        self.NX  = 0           # number of variables x = (y,z)
        self.NP  = 0           # number of parameters
        self.NU  = 0           # number of control functions
        self.NQI = 0           # number of q in one control interval

        self.backend = backend

    def zo_check(self, ts, x0, p, q):
        """Check dimensions and allocate memory."""
        # set dimensions
        self.M   = ts.size              # number of time steps
        self.NQ  = self.NU*self.M*2     # number of control variables

        # assert that the dimensions match
        self.NX = x0.size
        self.NP = p.size

        self.NU = q.shape[0]
        self.M  = q.shape[1]

        # assign variables
        self.ts = ts
        self.x0 = x0
        self.p  = p
        self.q  = q

        # allocate memory
        self.xs   = numpy.zeros((self.M, self.NX))
        self._x1s = numpy.zeros((self.M, self.NX)) # 2nd procedure
        self._Ks  = numpy.zeros((self._stages[-1], self.NX))
        self.f    = numpy.zeros((self.NX,))
        self.u    = numpy.zeros((self.NU,))

    def fo_check(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        """Check dimensions and allocate memory."""
        self.zo_check(ts, x0, p, q)

        self.P = x0_dot.shape[1]

        assert self.NP == p_dot.shape[0]

        assert self.P == p_dot.shape[1]
        assert self.P == q_dot.shape[3]

        # assign variables
        self.x0_dot = x0_dot
        self.p_dot  = p_dot
        self.q_dot  = q_dot

        # allocate memory
        self.xs_dot = numpy.zeros((self.M, self.NX, self.P))
        self.f_dot  = numpy.zeros((self.NX, self.P))
        self.u_dot  = numpy.zeros((self.NU, self.P))

    def so_check(
        self, ts,
        x0, x0_dot2, x0_dot1, x0_ddot,
        p, p_dot2, p_dot1, p_ddot,
        q, q_dot2, q_dot1, q_ddot
    ):
        self.zo_check(ts, x0, p, q)

        self.P1 = x0_dot1.shape[1]
        self.P2 = x0_dot2.shape[1]

        assert self.NP == p_dot1.shape[0]
        assert self.NP == p_dot2.shape[0]
        assert self.NP == p_ddot.shape[0]

        assert self.P1 == p_dot1.shape[1]
        assert self.P1 == q_dot1.shape[3]

        assert self.P2 == p_dot2.shape[1]
        assert self.P2 == q_dot2.shape[3]

        assert self.P1 == x0_ddot.shape[1]
        assert self.P1 == p_ddot.shape[1]
        assert self.P1 == q_ddot.shape[3]

        assert self.P2 == x0_ddot.shape[2]
        assert self.P2 == p_ddot.shape[2]
        assert self.P2 == q_ddot.shape[4]

        # assign variables
        self.x0_dot1 = x0_dot1
        self.p_dot1  = p_dot1
        self.q_dot1  = q_dot1

        self.x0_dot2 = x0_dot2
        self.p_dot2  = p_dot2
        self.q_dot2  = q_dot2

        self.x0_ddot = x0_ddot
        self.p_ddot  = p_ddot
        self.q_ddot  = q_ddot

        # allocate memory
        self.xs_dot1 = numpy.zeros((self.M, self.NX, self.P1))
        self.xs_dot2 = numpy.zeros((self.M, self.NX, self.P2))
        self.xs_ddot = numpy.zeros((self.M, self.NX, self.P1, self.P2))

        self.f_dot1  = numpy.zeros((self.NX, self.P1))
        self.f_dot2  = numpy.zeros((self.NX, self.P2))
        self.f_ddot  = numpy.zeros((self.NX, self.P1, self.P2))

        self.u_dot1  = numpy.zeros((self.NU, self.P1))
        self.u_dot2  = numpy.zeros((self.NU, self.P2))
        self.u_ddot  = numpy.zeros((self.NU, self.P1, self.P2))

    def zo_forward(self, ts, x0, p, q):
        # check if dimensions are correct
        self.zo_check(ts, x0, p, q)

        # set initial value
        self.xs[0, :]   = x0
        self._x1s[0, :] = x0 # 2nd procedure

        # initialize intermediate values
        t = numpy.zeros(1)
        y = numpy.zeros(self.f.shape)

        # k_i = f(t + a_i h, y + h * sum_i^s B_ij * k_j)
        for i in range(self.M-1):
            self.update_u(i)

            # get time step
            h = self.ts[i+1] - self.ts[i]

            # evaluate first
            if i == 0:

            #TODO: copy last evaluation to be first of new iterate

            # evaluate Runge-Kutta scheme (1)
            for j in range(self._stages[0]):
                # tau = t + a_i * h
                t[0] = self.ts[i] + self._As[j]*h
                y[:] = self.xs[i, :]
                # eta = y + h*SUM_j=0^s B_ij*k_j
                nonzero = self._Bs[j, :].nonzero()
                for k, coeff in enumerate(self._Bs[j, :][nonzero]):
                    y[:] += h*coeff*self._Ks[k]
                # k_i = f(t + a_i*h, y + h*SUM_j=0^s B_ij*k_j)
                self.backend.ffcn(t, y, self.f, self.p, self.u)
                self._Ks[j, :] = self.f[:]
            # evaluate Runge-Kutta scheme (2)
            else:

            # y(t + h) = y(t) + h * phi
            # phi = SUM_i^s c_i * k_i
            self.xs  [i + 1, :] = self.  xs[i, :] + h*self._Cs[0, :].dot(self._Ks)
            self._x1s[i + 1, :] = self._x1s[i, :] + h*self._Cs[1, :].dot(self._Ks)

    def fo_forward_xpu(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        self.fo_check(ts, x0, x0_dot, p, p_dot, q, q_dot)

        self.xs[0, :]         = x0
        self.xs_dot[0, :, :]  = x0_dot

        t        = numpy.zeros(1)
        K1       = numpy.zeros(self.f.shape)
        K2       = numpy.zeros(self.f.shape)
        K3       = numpy.zeros(self.f.shape)
        K4       = numpy.zeros(self.f.shape)
        y        = numpy.zeros(self.f.shape)

        K1_dot   = numpy.zeros(self.f.shape + (self.P,))
        K2_dot   = numpy.zeros(self.f.shape + (self.P,))
        K3_dot   = numpy.zeros(self.f.shape + (self.P,))
        K4_dot   = numpy.zeros(self.f.shape + (self.P,))
        y_dot    = numpy.zeros(self.f.shape + (self.P,))

        for i in range(self.M-1):
            self.update_u_dot(i)
            h     = self.ts[i+1] - self.ts[i]
            h2    = h/2.0

            # K1      = h*f(t, y, p, u)
            t[0]      = self.ts[i]
            y[:]      = self.xs[i, :]
            y_dot[:]  = self.xs_dot[i, :]
            self.backend.ffcn_dot(t, y, y_dot,
                                K1, K1_dot,
                                self.p, self.p_dot,
                                self.u, self.u_dot )
            K1       *= h
            K1_dot   *= h

            # K2    = h*f(t + h2, y + 0.5*K1, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K1
            y_dot[:]  = self.xs_dot[i, :] + 0.5*K1_dot
            self.backend.ffcn_dot(t, y, y_dot,
                            K2, K2_dot,
                            self.p, self.p_dot,
                            self.u, self.u_dot )
            K2       *= h
            K2_dot   *= h

            # K3    = h*f(t + h2, y + 0.5*K2, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K2
            y_dot[:]  = self.xs_dot[i, :] + 0.5*K2_dot
            self.backend.ffcn_dot(t, y, y_dot,
                            K3, K3_dot,
                            self.p, self.p_dot,
                            self.u, self.u_dot )
            K3       *= h
            K3_dot   *= h


            # K4    = h*f(t + h, y + K3, p, u)
            t[0]      = self.ts[i] + h
            y[:]      = self.xs[i, :] + K3
            y_dot[:]  = self.xs_dot[i, :] + K3_dot
            self.backend.ffcn_dot(t, y, y_dot,
                            K4, K4_dot,
                            self.p, self.p_dot,
                            self.u, self.u_dot )
            K4       *= h
            K4_dot   *= h

            self.xs[i + 1, :]      = self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4)
            self.xs_dot[i + 1, :]  = self.xs_dot[i,:] +  (1./6.0)*(K1_dot + 2*K2_dot + 2*K3_dot + K4_dot)


    def so_forward_xpu_xpu(self, ts, x0, x0_dot2, x0_dot1, x0_ddot,
                                     p,   p_dot2,  p_dot1, p_ddot,
                                     q,   q_dot2,  q_dot1, q_ddot):

        self.so_check(ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p,   p_dot2,  p_dot1, p_ddot,
                 q,   q_dot2,  q_dot1, q_ddot)


        self.xs[0, :]          = x0
        self.xs_dot1[0, :, :]  = x0_dot1
        self.xs_dot2[0, :, :]  = x0_dot2
        self.xs_ddot[0, :, :]  = x0_ddot

        t        = numpy.zeros(1)
        K1       = numpy.zeros(self.f.shape)
        K2       = numpy.zeros(self.f.shape)
        K3       = numpy.zeros(self.f.shape)
        K4       = numpy.zeros(self.f.shape)
        y        = numpy.zeros(self.f.shape)

        K1_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        K2_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        K3_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        K4_dot1   = numpy.zeros(self.f.shape + (self.P1,))
        y_dot1    = numpy.zeros(self.f.shape + (self.P1,))

        K1_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        K2_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        K3_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        K4_dot2   = numpy.zeros(self.f.shape + (self.P2,))
        y_dot2    = numpy.zeros(self.f.shape + (self.P2,))

        K1_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        K2_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        K3_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        K4_ddot   = numpy.zeros(self.f.shape + (self.P1, self.P2))
        y_ddot    = numpy.zeros(self.f.shape + (self.P1, self.P2))


        for i in range(self.M-1):
            self.update_u_ddot(i)
            h     = self.ts[i+1] - self.ts[i]
            h2    = h/2.0

            # K1      = h*f(t, y, p, u)
            t[0]      = self.ts[i]
            y[:]      = self.xs[i, :]
            y_dot1[:]  = self.xs_dot1[i, :]
            y_dot2[:]  = self.xs_dot2[i, :]
            y_ddot[:]  = self.xs_ddot[i, :]
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K1, K1_dot2, K1_dot1, K1_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K1        *= h
            K1_dot1   *= h
            K1_dot2   *= h
            K1_ddot   *= h

            # K2    = h*f(t + h2, y + 0.5*K1, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K1
            y_dot1[:]  = self.xs_dot1[i, :] + 0.5 * K1_dot1
            y_dot2[:]  = self.xs_dot2[i, :] + 0.5 * K1_dot2
            y_ddot[:]  = self.xs_ddot[i, :] + 0.5 * K1_ddot
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K2, K2_dot2, K2_dot1, K2_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K2        *= h
            K2_dot1   *= h
            K2_dot2   *= h
            K2_ddot   *= h

            # K3    = h*f(t + h2, y + 0.5*K2, p, u)
            t[0]      = self.ts[i] + h2
            y[:]      = self.xs[i, :] + 0.5*K2
            y_dot1[:]  = self.xs_dot1[i, :] + 0.5 * K2_dot1
            y_dot2[:]  = self.xs_dot2[i, :] + 0.5 * K2_dot2
            y_ddot[:]  = self.xs_ddot[i, :] + 0.5 * K2_ddot
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K3, K3_dot2, K3_dot1, K3_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K3        *= h
            K3_dot1   *= h
            K3_dot2   *= h
            K3_ddot   *= h


            # K4    = h*f(t + h, y + K3, p, u)
            t[0]      = self.ts[i] + h
            y[:]      = self.xs[i, :] + K3
            y_dot1[:]  = self.xs_dot1[i, :] + K3_dot1
            y_dot2[:]  = self.xs_dot2[i, :] + K3_dot2
            y_ddot[:]  = self.xs_ddot[i, :] + K3_ddot
            self.backend.ffcn_ddot(t,
                              y, y_dot2, y_dot1, y_ddot,
                              K4, K4_dot2, K4_dot1, K4_ddot,
                              self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                              self.u, self.u_dot2, self.u_dot1, self.u_ddot)
            K4        *= h
            K4_dot1   *= h
            K4_dot2   *= h
            K4_ddot   *= h

            self.xs[i + 1, :]      = self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4)
            self.xs_dot1[i + 1, :]  = self.xs_dot1[i,:] +  (1./6.0)*(K1_dot1 + 2*K2_dot1 + 2*K3_dot1 + K4_dot1)
            self.xs_dot2[i + 1, :]  = self.xs_dot2[i,:] +  (1./6.0)*(K1_dot2 + 2*K2_dot2 + 2*K3_dot2 + K4_dot2)
            self.xs_ddot[i + 1, :]  = self.xs_ddot[i,:] +  (1./6.0)*(K1_ddot + 2*K2_ddot + 2*K3_ddot + K4_ddot)



    def fo_reverse(self, xs_bar):

        numpy.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=200, suppress=None, nanstr=None, infstr=None, formatter=None)
        self.xs_bar = xs_bar.copy()

        t      = numpy.zeros(1)
        K1     = numpy.zeros(self.f.shape)
        K2     = numpy.zeros(self.f.shape)
        K3     = numpy.zeros(self.f.shape)
        K4     = numpy.zeros(self.f.shape)
        y      = numpy.zeros(self.f.shape)

        K1_bar = numpy.zeros(self.f.shape)
        K2_bar = numpy.zeros(self.f.shape)
        K3_bar = numpy.zeros(self.f.shape)
        K4_bar = numpy.zeros(self.f.shape)
        y_bar  = numpy.zeros(self.f.shape)

        self.x0_bar = numpy.zeros(self.x0.shape)
        self.f_bar  = numpy.zeros(self.f.shape)
        self.p_bar  = numpy.zeros(self.p.shape)
        self.q_bar  = numpy.zeros(self.q.shape)
        self.u_bar  = numpy.zeros(self.u.shape)

        ts = self.ts
        p  = self.p
        u  = self.u
        xs = self.xs

        p_bar = self.p_bar
        u_bar = self.u_bar
        xs_bar = self.xs_bar

        for i in range(self.M-1)[::-1]:
            self.update_u(i)
            h = ts[i+1] - ts[i]
            h2 = h/2.0

            # forward K1 = h*f[t, y, p, u]
            t[0]  = ts[i]
            y[:]  = xs[i, :]
            self.backend.ffcn(t, y, K1, p, u)
            K1   *= h

            # forward K2 = h*f[t + h2, y + 0.5*K1, p, u]
            t[0]  = ts[i] + h2
            y[:]  = xs[i, :] + 0.5*K1
            self.backend.ffcn(t, y, K2, p, u)
            K2   *= h

            # forward K3 = h*f[t + h2, y + 0.5*K2, p, u]
            t[0]  = ts[i] + h2
            y[:]  = xs[i, :] + 0.5*K2
            self.backend.ffcn(t, y, K3, p, u)
            K3   *= h

            # foward K4    = h*f(t + h, y + K3, p, u)
            t[0]  = self.ts[i] + h
            y[:]  = self.xs[i, :] + K3
            self.backend.ffcn(t, y, K4, self.p, self.u )
            K4   *= h

            # forward accumulation
            # from numpy.testing import assert_almost_equal
            # assert_almost_equal(self.xs[i + 1, :],
            #                     self.xs[i,:] +  (1./6.0)*(K1 + 2*K2 + 2*K3 + K4))

            # reverse accumulation
            y_bar[:]               = 0.
            u_bar[:]               = 0.

            self.xs_bar[i, :]     += self.xs_bar[i + 1, :]
            K1_bar[:]              = (1./6.) * self.xs_bar[i + 1, :]
            K2_bar[:]              = (2./6.) * self.xs_bar[i + 1, :]
            K3_bar[:]              = (2./6.) * self.xs_bar[i + 1, :]
            K4_bar[:]              = (1./6.) * self.xs_bar[i + 1, :]
            xs_bar[i+1, :]         = 0.

            # reverse K4
            t[0]                   = ts[i] + h
            K4_bar                *= h
            y[:]                   = self.xs[i, :] + K3
            self.backend.ffcn_bar(t, y, y_bar, K4, K4_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            K3_bar                += y_bar
            y_bar[:]               = 0.

            # reverse K3
            t[0]                   = ts[i] + h2
            K3_bar                *= h
            y[:]                   = self.xs[i, :] + 0.5*K2
            self.backend.ffcn_bar(t, y, y_bar, K3, K3_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            K2_bar                += 0.5*y_bar
            y_bar[:]               = 0.

            # reverse K2
            t[0]                   = ts[i] + h2
            K2_bar                *= h
            y[:]  = xs[i, :] + 0.5*K1
            self.backend.ffcn_bar(t, y, y_bar, K2, K2_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            K1_bar                += 0.5*y_bar
            y_bar[:]               = 0.

            # reverse K1
            t[0]                   = ts[i]
            K1_bar                *= h
            y[:]                   = self.xs[i, :]
            self.backend.ffcn_bar(t, y, y_bar, K1, K1_bar, p, p_bar, u, u_bar)
            xs_bar[i, :]          += y_bar
            y_bar[:]               = 0.

            self.update_u_bar(i)

        self.x0_bar[:] += self.xs_bar[0, :]
        self.xs_bar[0, :] = 0.


    def update_u(self, i):
        self.u[:] = self.q[:, i, 0]

    def update_u_dot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot[:, :] = self.q_dot[:, i, 0, :]

    def update_u_bar(self, i):
        self.q_bar[:, i, 0] += self.u_bar[:]
        self.u_bar[:] = 0.

    def update_u_ddot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot1[:, :] = self.q_dot1[:, i, 0, :]
        self.u_dot2[:, :] = self.q_dot2[:, i, 0, :]
        self.u_ddot[:, :] = self.q_ddot[:, i, 0, :, :]

'''
        class InterpolationPolynomial(functor):

            """Callable 6th order Hermite-Birkhoff polynomial."""
            _NX = NX
            _h = h

            def __call__(self, tau, res):
                """Return evaluation of polynomial at given time tau."""
                # check if tau is in unit interval
                err_str = 'tau is not in unit interval, 0 <= tau = {} <= 1.0?'.format(tau)
                assert 0 <= tau <= 1.0, err_str

                # check if results array has right dimension
                err_str = 'wrong dimension of res: NX = {} != {} = res.shape'.format(self._NX, res.shape)
                assert res.shape[0] == self._NX

                # rename for convenience
                h = self._h

                # zeroize results
                res[...] = 0.0

                # evaluate polynomial
                ww0 = 1.0 + (-82.0/9.0 + (128.0/9.0 - 55.0/9.0*tau)*tau)*tau*tau
                ww1 = h * (1.0 + (-11.0/3.0 + ( 13.0/3.0 -  5.0/3.0*tau)*tau)*tau)*tau
                ww2 =     (-33.0/4.0 + (41.0/2.0 - 45.0/4.0*tau)*tau)*tau*tau
                ww3 = h * (3.0/2.0 + (-4.0 + 5.0/2.0*tau)*tau)*tau*tau
                ww4 =     (625.0/36.0 + (-625.0/18.0 + 625.0/36.0*tau)*tau)*tau*tau
                res[...] = ww0*x0 + ww1*k[0] + ww2*x1 + ww3*k[6] + ww4*eta
'''
