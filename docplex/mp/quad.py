# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
from six import iteritems

from docplex.mp.compat23 import unitext, izip
from docplex.mp.linear import Expr, AbstractLinearExpr, Var, ZeroExpr, LinearExpr
from docplex.mp.utils import *

from docplex.mp.xcounter import FastOrderedDict


def _compare_vars(v1, v2):
    v1i = v1._index
    v2i = v2._index
    return (v1i > v2i) - (v2i > v1i)


class VarPair(object):
    __slots__ = ("first", "second", "_cached_hash")

    def __init__(self, v1, v2=None):
        if v2 is None:
            self.first = v1
            self.second = v1
        elif _compare_vars(v1, v2) <= 0:
            self.first = v1
            self.second = v2
        else:
            self.first = v2
            self.second = v1
        self._cached_hash = self._hash_pair()

    def is_square(self):
        return self.first is self.second

    def __eq__(self, other):
        # INTERNAL: necessary for use as dict keys
        # VarPair ensures variables are sorted by indices
        return isinstance(other, VarPair) and (self.first is other.first) and (self.second is other.second)

    def _hash_pair(self):
        if self.is_square():
            return hash(self.first)
        else:
            f = hash(self.first)
            s = hash(self.second)
            # cantor encoding. must cast to int() for py3
            self_hash = int(((f + s) * (s + f + 1) / 2) + s)
            if self_hash == -1:
                # value -1 is reserved for errors
                self_hash = -2
            return self_hash

    def __hash__(self):
        return self._cached_hash

    def __repr__(self):
        return "docplex.mp.quad.VarPair(first={0!s},second={1!s})".format(self.first, self.second)

    def __str__(self):
        return "VarPair({0!s}, {1!s})".format(self.first, self.second)

    def __getitem__(self, item):
        if 0 == item:
            return self.first
        elif 1 == item:
            return self.second
        else:
            raise StopIteration


class QuadExpr(Expr):
    """QuadExpr()

    This class models quadratic expressions.
    This class is not intended to be instantiated. Quadratic expressions are built
    either by using operators or by using :func:`docplex.mp.model.Model.quad_expr`.

    """
    _qterms_dict_type = FastOrderedDict

    def copy(self, target_model, var_mapping):
        copied_quads = self._qterms_dict_type()
        for qv1, qv2, qk in self.iter_quad_triplets():
            new_v1 = var_mapping[qv1]
            new_v2 = var_mapping[qv2]
            copied_quads[VarPair(new_v1, new_v2)] = qk

        copied_linear = self._linexpr.copy(target_model, var_mapping)
        return QuadExpr(model=target_model,
                        quads=copied_quads,
                        linexpr=copied_linear,
                        name=self.name,
                        safe=True)

    def is_quad_expr(self):
        return True

    def has_quadratic_term(self):
        """ Returns true if there is at least one quadratic term in the expression.
        """
        return len(self._quadterms) > 0

    def square(self):
        if self.has_quadratic_term():
            self.fatal("Cannot take the square of a quadratic term: {0!s}".format(self))
        else:
            return self._linexpr.square()

    def _get_solution_value(self):
        # INTERNAL
        quad_value = 0
        for qv0, qv1, qk in self.iter_quad_triplets():
            quad_value += qk * (qv0.unchecked_solution_value * qv1.unchecked_solution_value)
        lin_value = self._linexpr._get_solution_value()
        return quad_value + lin_value

    __slots__ = ('_quadterms', '_linexpr')

    def __init__(self, model, quads=None, linexpr=None, name=None, safe=False, _qterm_type=_qterms_dict_type):
        Expr.__init__(self, model, name)
        model._quadexpr_instance_counter += 1
        if quads is None:
            self._quadterms = _qterm_type()
        elif isinstance(quads, dict):
            if safe:
                self._quadterms = quads
            else:
                # check
                for qv, qk in iteritems(quads):
                    self.model.typecheck_num(qk)
                    if not isinstance(qv, VarPair):
                        self.fatal("Expecting variable-pair, got: {0!s}", qv)
                self._quadterms = _qterm_type(quads)

        elif isinstance(quads, tuple):
            tuplesize = len(quads)
            if 2 == tuplesize:
                vp = quads[0]
                qk = quads[1]
                if not safe:
                    assert isinstance(vp, VarPair)
                    model.typecheck_num(qk)

                self._quadterms = _qterm_type()
                self._quadterms[vp] = qk

            elif 3 == tuplesize:
                v1 = quads[0]
                v2 = quads[1]
                qk = quads[2]
                if not safe:
                    model.typecheck_var(v1)
                    model.typecheck_var(v2)
                    model.typecheck_num(qk, 'QuadExpr')
                self._quadterms = _qterm_type()
                self._quadterms[VarPair(v1, v2)] = qk
            else:
                self.fatal("only tuples of len 2,3 are valid for QuadExpr, got: {0!r}", quads)

        elif is_iterable(quads):
            self._quadterms = _qterm_type(quads)
        else:
            self.fatal("unexpected argument for QuadExpr: {0!r}", quads)

        if linexpr is None:
            self._linexpr = self.model._linear_expr()
        else:
            self._linexpr = self.model._to_linear_expr(linexpr)


    def clone(self):
        """ Makes a copy of the quadratic expression and returns it.

        Returns:
            A quadratic expression.
        """
        self._model._quadexpr_clone_counter += 1
        cloned_linear = self._linexpr.clone()
        self_name = self.name
        cloned_name = self_name if self_name is None else self_name[:]
        new_quad = QuadExpr(self.model, quads=self._quadterms.copy(),
                            linexpr=cloned_linear,
                            name=cloned_name, safe=True)
        return new_quad

    def is_discrete(self):
        for qv0, qv1, qk in self.iter_quad_triplets():
            if not qv0.is_discrete or not qv1.is_discrete() or not is_int(qk):
                return False
        else:
            return self._linexpr.is_discrete()

    def generate_quad_triplets(self):
        # INTERNAL
        # a generator that returns triplets (i.e. tuples of len 3)
        # with the variable pair and the coefficient
        for qvp, qk in iteritems(self._quadterms):
            yield qvp[0], qvp[1], qk

    def iter_quads(self):
        return iteritems(self._quadterms)

    def iter_opposite_quads(self):
        # INTERNAL
        for qv, qk in self.iter_quads():
            yield qv, -qk

    def iter_quad_triplets(self):
        """ Iterates over quadratic terms.

        This iterator returns triplets of the form `v1,v2,k`, where `v1` and `v2` are decision
        variables and `k` is a number.

        Returns:
            An iterator object.
        """
        return self.generate_quad_triplets()

    def iter_terms(self):
        return self._linexpr.iter_terms()

    @property
    def number_of_quadratic_terms(self):
        """ This property returns the number of quadratic terms.

        Counts both the square and product terms.

        Examples:
        
        .. code-block:: python

           q1 = x**2
           q1.number_of_quadratic_terms
           >>> 1
           q2 = (x+y+1)**2
           q2.number_of_quadratic_terms
           >>> 3
        """
        return len(self._quadterms)

    def is_separable(self):
        """ Checks if all quadratic terms are separable.

        Returns:
            True if all quadratic terms are separable.
        """
        for qv, _ in self.iter_quads():
            if not qv.is_square():
                return False
        else:
            return True


    def compute_separable_convexity(self):
        # INTERNAL
        # returns 1 if separable, convex
        # returns -1 if separable non convex
        # return s0 if non separable

        justifier = None
        for qv, qk in self.iter_quads():
            if not qv.is_square():
                return 0, None  # non separable: fast exit
            elif qk < 0:
                if justifier is None:
                    justifier = (qk, qv[0])  # separable, non convex, kept
        else:
            return justifier or (1, None)  # (1, None) is for separable, convex

    def get_quadratic_coefficient(self, var1, var2=None):
        ''' Returns the coefficient of a quadratic term in the expression.

        Returns the coefficient of the quadratic term `var1*var2` in the expression, if any.
        If the product is not present in the expression, returns 0.

        Args:
            var1: The first variable of the product (an instance of class Var)
            var2: the second variable of the product. If passed None, returns the coefficient
                of the square of `var1` in the expression.

        Example:
            Assuming `x` and `y` are decision variables and `q` is the expression `2*x**2 + 3*x*y + 5*y**2`, then

            `q.get_quadratic_coefficient(x)` returns 2

            `q.get_quadratic_coefficient(x, y)` returns 3

            `q.get_quadratic_coefficient(y)` returns 5

        Returns:
            The coefficient of one quadratic product term in the expression.
        '''
        self.model.typecheck_var(var1)
        if var2 is None:
            var2 = var1
        else:
            self.model.typecheck_var(var2)
        return self._get_quadratic_coefficient(var1, var2)


    def _get_quadratic_coefficient(self, var1, var2):
        # INTERNAL, no checks
        vp = VarPair(var1, var2 or var1)
        return self._get_quadratic_coefficient_from_var_pair(vp)

    def _get_quadratic_coefficient_from_var_pair(self, vp):
        # INTERNAL
        return self._quadterms.get(vp, 0)

    # ---
    def equals(self, other):
        if not isinstance(other, QuadExpr):
            return False
        if self.number_of_quadratic_terms != other.number_of_quadratic_terms:
            return False

        for qvp, qk in self.iter_quads():
            if other._get_quadratic_coefficient_from_var_pair(qvp) != qk:
                return False
        return self._linexpr.equals_expr(other._linexpr)

    def is_constant(self):
        return not self.has_quadratic_term() and self._linexpr.is_constant()

    @property
    def constant(self):
        return self.get_constant()

    def get_constant(self):
        return self._linexpr.constant

    @property
    def linear_part(self):
        return self.get_linear_part()

    def get_linear_part(self):
        return self._linexpr

    def iter_variables(self):
        for qvp, _ in self.iter_quads():
            if qvp.is_square():
                yield qvp[0]
            else:
                yield qvp[0]
                yield qvp[1]
        for lv in self._linexpr.iter_variables():
            yield lv

    def contains_var(self, dvar):
        # required by tests...
        for qv in self.iter_variables():
            if qv is dvar:
                return True
        else:
            return False

    def contains_quad(self, qv):
        # INTERNAL
        return qv in self._quadterms

    def __repr__(self):
        return "docplex.mp.quad.QuadExpr(%s)" % self.truncated_str()

    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        q = 0
        # noinspection PyPep8Naming
        SP = u' '
        for qv1, qv2, qk in self.iter_quad_triplets():
            if 0 == qk:
                continue  # pragma: no cover

            # ---
            # sign is printed if  non-first OR negative
            # at the end of this block coeff is positive
            wrote_sign = False
            if qk < 0 or q > 0:
                oss.write(u'-' if qk < 0 else u'+')
                wrote_sign = True
                if qk < 0:
                    qk = -qk

            # write coeff if <> 1
            varname1 = var_namer(qv1)
            if use_space:
                if wrote_sign and q > 0 or varname1[0].isdigit():
                    oss.write(SP)

            if 1 != qk:
                self._num_to_stringio(oss, num=qk, ndigits=nb_digits)
                if prod_symbol:
                    oss.write(unitext(prod_symbol))

            oss.write(unitext(varname1))
            if qv1 is qv2:
                oss.write(u"^2")
            else:
                if prod_symbol:
                    oss.write(unitext(prod_symbol))
                oss.write(unitext(var_namer(qv2)))
            q += 1
        # problem for linexpr: force '+' ssi c>0
        linexpr = self._linexpr
        if linexpr.is_constant():
            k = linexpr.constant
            if k == 0 and q > 0:
                pass
            else:
                sign = u'-' if k < 0 else u'+'
                if use_space:
                    oss.write(u' ')
                if k < 0 or q > 0:
                    oss.write(sign)
                    if use_space:
                        oss.write(SP)
                self._num_to_stringio(oss, num=abs(k), ndigits=nb_digits)
        else:
            # linear part is NOT a constant
            print_plus_sign = False
            if q > 0:
                try:
                    fv = next(linexpr.iter_variables())
                except StopIteration:
                    fv = None
                if fv is not None:
                    if linexpr[fv] > 0:
                        print_plus_sign = True
                elif linexpr.constant > 0:
                    print_plus_sign = True
            # ---
            if use_space:
                oss.write(u' ')
            if print_plus_sign:
                oss.write(u"+")
                if use_space:
                    oss.write(SP)
            self._linexpr.to_stringio(oss, nb_digits, prod_symbol, use_space, var_namer)


    def plus(self, other):
        cloned = self.clone()
        cloned.add(other)
        return cloned

    def minus(self, other):
        cloned = self.clone()
        cloned.subtract(other)
        return cloned

    def rminus(self, other):
        # other - self
        cloned = self.clone()
        cloned.negate()
        cloned.add(other)
        return cloned

    def times(self, other):
        if is_number(other) and 0 == other:
            return self.zero_expr()

        elif isinstance(other, ZeroExpr):
            return other

        elif self.is_constant():
            k = self.constant
            if 0 == k:
                return self.zero_expr()
            elif 1 == k:
                return self._model._to_linear_expr(other)
            else:
                return other * k
        else:
            cloned = self.clone()
            cloned.multiply(other)
            return cloned

    def __add__(self, other):
        return self.plus(other)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __radd__(self, other):
        return self.plus(other)

    def __sub__(self, other):
        return self.minus(other)

    def __rsub__(self, other):
        # e - self
        return self.rsubtract(other)

    def __isub__(self, other):
        self.subtract(other)
        return self

    def __mul__(self, other):
        return self.times(other)

    def __rmul__(self, other):
        return self.times(other)

    def __imul__(self, other):
        # self is modified
        return self.multiply(other)

    def __div__(self, e):
        return self.quotient(e)

    def __idiv__(self, other):
        self._divide(other, check=True)
        return self

    def __truediv__(self, e):
        return self.quotient(e)  # pragma: no cover

    def __neg__(self):
        cloned = self.clone()
        cloned.negate()
        return cloned

    def add(self, other):
        # increment the QuadExpr with some other argument
        if isinstance(other, QuadExpr):
            self._add_quad(other)
        else:
            self_linexpr = self._linexpr
            if isinstance(self_linexpr, ZeroExpr):
                self._linexpr = self.model._to_linear_expr(other)
            else:
                self._linexpr.add(other)

    def rsubtract(self, other):
        # to compute (other - self) we copy self, negate the copy and add other
        # result is always cloned even if other is zero (optimization possible here)
        cloned = self.clone()
        cloned.negate()
        cloned.add(other)
        return cloned

    def subtract(self, other):
        if isinstance(other, QuadExpr):
            return self._subtract_quad(other)
        else:
            self._linexpr.subtract(other)

    def negate(self):
        # INTERNAL: negate sall coefficients, modify self
        qterms = self._quadterms
        for qvp, qk in iteritems(qterms):
            qterms[qvp] = -qk
        self._linexpr.negate()
        return self

    def multiply(self, other):
        if is_number(other):
            self._scale(other)

        elif self.is_constant():
            this_constant = self._linexpr.get_constant()
            if 0 == this_constant:
                # do nothing
                pass
            else:
                self._assign_scaled(other, this_constant)

        elif self.is_quad_expr():
            self.fatal(
                "Cannot multiply {0!s} by {1!s}, exponent would be greater than 2. A variable's exponent can be at most 2",
                self, other)
        else:
            self._linexpr.multiply(other)
        return self

    def quotient(self, e):
        self.model.typecheck_as_denominator(e, self)
        cloned = self.clone()
        cloned._divide(e, check=False)
        return cloned

    def _divide(self, other, check=True):
        if check:
            self.model.typecheck_as_denominator(other, self)  # only a nonzero number is allowed...
        inverse = 1.0 / other
        self._scale(inverse)

    def _scale(self, factor):
        # INTERNAL: scales a quad expr from a numeric constant.
        # no checks done!
        # this method modifies self.
        if 0 == factor:
            self.clear()
        elif 1 == factor:
            # nothing to do
            pass
        else:
            # scale quads
            self_quadterms = self._quadterms
            for qv, qk in self.iter_quads():
                self_quadterms[qv] = factor * qk
            # scale linear part
            self._linexpr._scale(factor)

    def _assign_scaled(self, other, factor):
        # INTERNAL
        if isinstance(other, (AbstractLinearExpr, Var)):
            scaled = self._model._to_linear_expr(other, force_clone=True)
            scaled *= factor
            self._linexpr = scaled
        elif isinstance(other, QuadExpr):
            for qv, qk in other.iter_quads():
                self._add_one_quad_term(qv, qk * factor)
            self._assign_scaled(other.linear_part, factor)
        else:
            pass

    def _add_one_quad_term(self, qv, qk):
        if qk != 0:
            qterms = self._quadterms
            if qv in qterms:
                new_qk = qterms[qv] + qk
                if 0 != new_qk:
                    qterms[qv] = new_qk
                else:
                    del qterms[qv]
            else:
                qterms[qv] = qk

    def _add_one_quad_triplet(self, qv1, qv2, qk):
        self._add_one_quad_term(VarPair(qv1, qv2), qk)

    def _add_linear_term(self, v, k):
        if k:
            self._linexpr._add_term(v, k)

    def normalize(self):
        # INTERNAL
        if self._quadterms:
            return self
        elif not self._linexpr.is_constant():
            return self._linexpr
        else:
            k = self.get_constant()
            if 0 == k:
                return self._model._get_zero_expr()
            else:
                return self._linexpr


    def clear(self):
        self._quadterms = self._qterms_dict_type()  # clear quads
        self._linexpr = self.zero_expr()

    # quad-specific
    def _add_quad(self, other_quad):
        # add quad part
        for oqv, oqk in other_quad.iter_quads():
            self._add_one_quad_term(oqv, oqk)
        # add linear part
        self._linexpr.add(other_quad._linexpr)

    def _subtract_quad(self, other_quad):
        # subtract quad
        for oqv, oqk in other_quad.iter_quads():
            self._add_one_quad_term(oqv, -oqk)
        # subtract linear part
        self._linexpr.subtract(other_quad._linexpr)


    def to_linear_expr(self):
        if self.has_quadratic_term():
            raise DOCPlexQuadraticArithException(
                "quadratic expression [{0!s}] cannot be converted to a linear expression", self)
        else:
            return self._linexpr.clone()

    def _is_normal_form(self):
        # INTERNAL
        for _, qk in self.iter_quads():
            if 0 == qk:
                return False  # pragma: no cover
        return True

    # --- relational operators
    def __eq__(self, other):
        return self._model._qfactory.new_eq_constraint(self, other)

    def __le__(self, other):
        return self._model._qfactory.new_le_constraint(self, other)

    def __ge__(self, other):
        return self._model._qfactory.new_ge_constraint(self, other)
