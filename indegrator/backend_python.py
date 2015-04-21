"""Pure Python back-end of indegrator."""

import os
import sys
import numpy


class BackendPython(object):

    """
    Pure Python back-end of indegrator.

    Requires the manual implementation of all wanted derivative, i.e:

    zo_forward: ffcn
    fo_forward: ffcn, ffcn_dot
    fo_reverse: ffcn, ffcn_bar
    so_reverse: ffcn, ffcn_dot, ffcn_ddot
    """

    _methods = (
        'ffcn',
        'ffcn_dot',
        'ffcn_bar',
        'ffcn_ddot',
    )

    def __init__(self, ffcn):
        """
        Load the provided pure Python back-end.

        Where ffcn is either

        1) a class or an object
        2) or a path to a directory containing a python file '*.py'

        providing methods of the form
        * ``def ffcn(self, t, x, f, p, u)''
        * ``def ffcn_dot(self, t, x, x_dot, f, f_dot, p, p_dot, u, u_dot)''
        * ``def ffcn_bar(self, t, x, x_bar, f, f_bar, p, p_bar, u, u_bar)''
        * ``def ffcn_ddot(self, t,''
        * ``              x, x_dot2, x_dot1, x_ddot,''
        * ``              f, f_dot2, f_dot1, f_ddot,''
        * ``              p, p_dot2, p_dot1, p_ddot,''
        * ``              u, u_dot2, u_dot1, u_ddot)''
        """

        # check if object has respective members
        check_members = [hasattr(ffcn, method) for method in self._methods]
        check_members = numpy.asarray(check_members, dtype=bool)

        if check_members.any():
            # avoid rechecking by iterating over check_members
            for i, check in enumerate(check_members):
                if check:
                    setattr(self,
                            self._methods[i],
                            getattr(ffcn, self._methods[i],))
                else:
                    # implemented abstract methods are taken
                    err_str = (
                        'could not register method {method}, dummy is taken instead'
                        .format(method=self._methods[i])
                        )
                    sys.stdout.write(err_str + '\n')
                    pass
        else:
            self.path = os.path.abspath(ffcn)
            self.dir  = os.path.dirname(self.path)

            sys.path.insert(0, self.dir)
            import ffcn

            for method in self._methods:
                # check if module has respective members
                if hasattr(ffcn, method):
                    setattr(self, method, getattr(ffcn, method))
                else:
                    # implemented abstract methods are taken
                    err_str = (
                        'WARNING: could not register "{method}", dummy is taken instead'
                        .format(method=method)
                        )
                    sys.stdout.write(err_str + '\n')
                    pass

    def ffcn(self, t, x, f, p, u):
        """dummy right-hand side function."""
        raise NotImplementedError

    def ffcn_dot(self, t, x, x_dot, f, f_dot, p, p_dot, u, u_dot):
        """dummy forward derivative of right-hand side function."""
        raise NotImplementedError

    def ffcn_bar(self, t, x, x_bar, f, f_bar, p, p_bar, u, u_bar):
        """dummy reverse derivative of right-hand side function."""
        raise NotImplementedError

    def ffcn_ddot(self, t,
                  x, x_dot2, x_dot1, x_ddot,
                  f, f_dot2, f_dot1, f_ddot,
                  p, p_dot2, p_dot1, p_ddot,
                  u, u_dot2, u_dot1, u_ddot):
        """
        dummy second order forward derivative of right-hand side function.
        """
        raise NotImplementedError
