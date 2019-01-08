# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.model import Model
from docplex.mp.aggregator import ModelAggregator
from docplex.mp.quad import VarPair
from docplex.mp.xcounter import FastOrderedDict, FastOrderedCounter
from docplex.mp.utils import is_number, is_iterable, generate_constant
import sys
from docplex.mp.compat23 import izip, fast_range
from docplex.mp.linear import LinearExpr


class AdvAggregator(ModelAggregator):

    def __init__(self, linear_factory, quad_factory):
        ModelAggregator.__init__(self, linear_factory, quad_factory)

    def _scal_prod_vars(self, terms, coefs):
        # INTERNAL
        checker = self._checker
        total_num = 0
        lcc = FastOrderedCounter()

        for item, coef in izip(terms, coefs):
            if 0 == coef:
                continue

            safe_coef = checker.to_valid_number(coef, context_msg=lambda: "Model.dot({0!s}..)".format(item))
            lcc.__setitem__(item, safe_coef)

        #sorted_terms = LinearExpr._sort_terms_if_needed(self._model, counter=lcc)
        linear_expr = LinearExpr(self._model, e=lcc, safe=True)
        return linear_expr

    def scal_prod_triple(self, left_terms, right_terms, coefs):
        used_coefs = None

        if is_iterable(coefs):
            used_coefs = coefs
        elif is_number(coefs):
            if 0 == coefs:
                return self.zero_expr
            else:
                used_coefs = generate_constant(coefs, count_max=sys.maxsize)
        else:
            self._model.fatal("scalprod_triple expects iterable or number, got: {0!s}", coefs)

        if is_iterable(left_terms):
            used_left = left_terms
        else:
            used_left = generate_constant(left_terms, count_max=sys.maxsize)

        if is_iterable(right_terms):
            used_right = right_terms
        else:
            used_right = generate_constant(right_terms, count_max=sys.maxsize)

        if used_coefs is not coefs and used_left is not left_terms and used_right is not right_terms:
            return left_terms * right_terms * coefs
        return self._scal_prod_triple(coefs=used_coefs, left_terms=used_left, right_terms=used_right)


    def _scal_prod_triple(self, coefs, left_terms, right_terms):
        # INTERNAL
        accumulated_ct = 0
        checker = self._checker
        quad = self._quad_factory.new_quad(quad_args=None, linexpr=None, safe=True)
        for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
            safe_coef = checker.to_valid_number(coef, context_msg=lambda: "Model.scalprodq({0!s}..)".format(coef))
            if 0 == safe_coef:
                continue

            checker.typecheck_operand(lterm, accept_numbers=False)
            checker.typecheck_operand(rterm, accept_numbers=False)
            lcst = lterm.get_constant()
            rcst = rterm.get_constant()
            accumulated_ct += safe_coef * lcst * rcst
            for lv, lk in lterm.iter_terms():
                for rv, rk in rterm.iter_terms():
                    coef3 = safe_coef * lk * rk
                    quad._add_one_quad_triplet(qv1=lv, qv2=rv, qk=coef3)
            if rcst:
                for lv, lk in lterm.iter_terms():
                    quad._add_linear_term(v=lv, k=safe_coef * lk * rcst)
            if lcst:
                for rv, rk in rterm.iter_terms():
                    quad._add_linear_term(v=rv, k=safe_coef * rk * lcst)

        quad += accumulated_ct
        return quad

    def _scal_prod_vars_triple(self, coefs, left_terms, right_terms):
        dcc = self._quad_factory._qterm_type
        terms = dcc()
        for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
            if coef != 0:
                terms[VarPair(lterm ,rterm)] = coef
        quad = self._quad_factory.new_quad(quad_args=terms, linexpr=None, safe=True)
        return quad

    # new
    def quad_matrix_sum(self, matrix, dvars):
        # assume matrix is a NxN matrix
        # vars is a N-vector of variables
        n = len(dvars)
        dcc = self._quad_factory._qterm_type
        terms = dcc()

        for i in fast_range(n):
            for j in fast_range(i + 1):
                terms[VarPair(dvars[i], dvars[j])] = matrix[i][j] + matrix[j][i] if j != i else matrix[i][i]
        return self._quad_factory.new_quad(quad_args=terms, linexpr=None, safe=True)


# noinspection PyProtectedMember
class AdvModel(Model):
    """ Experimental.
    This class is experimental and may change, be renamed or disappear in a next version.
    It is a specialized version of docplex.mp.model.Model class with useful non standard modeling functions.
    """
    def __init__(self, name, context=None, **kwargs):
        Model.__init__(self, name=name, context=context, **kwargs)
        self._aggregator = AdvAggregator(self._lfactory, self._qfactory)

    def scal_prod_vars(self, terms, coefs):
        """
        Creates a linear expression equal to the scalar product of a list of decision variables and a sequence of coefficients.

        The variable sequence is a list or an iterator of variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.

        :param terms: A list or an iterator on variables only, with no duplicate (no check is done to verify this assertion).
        :param coefs: A list or an iterator on numbers, or a number.

        Note:
           If either list or iterator is empty, this method returns zero.

        :return: A linear expression or 0.
        """
        self._check_ordered(arg=terms, header='Model.scal_prod() requires a list of expressions/variables')
        return self._aggregator._scal_prod_vars(terms, coefs)

    def quad_matrix_sum(self, matrix, dvars):
        """
        Experimental
        :param matrix: A 2 dimensional list
        :param dvars: A list or an iterator on variables.
        :return: An instance of :class:`docplex.mp.quad.QuadExpr` or 0
        """
        return self._aggregator.quad_matrix_sum(matrix, dvars)

    def scal_prod_triple(self, left_terms, right_terms, coefs):
        """ Experimental.
        Creates a quadratic expression from two lists of linear expressions and a sequence of coefficients.


        This method is experimental and may change, be renamed or disappear in a next version.

        This method sums all quadratic terms built by multiplying the i_th coefficient by the product of the i_th
        expression in `left_terms` and the i_th expression in `right_terms`

        This method accepts different types of input for its arguments. The expression sequences can be either lists
        or iterators of objects that can be converted to linear expressions, that is, variables or linear expressions
        (but no quadratic expressions).
        The most usual case is variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.

        Example:
            `Model.scalprod_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`

        :param left_terms: A list or an iterator on variables or expressions.
        :param right_terms: A list or an iterator on variables or expressions.
        :param coefs: A list or an iterator on numbers or a number.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           If either list or iterator is empty, this method returns zero.
        """
        return self._aggregator.scal_prod_triple(left_terms=left_terms, right_terms=right_terms, coefs=coefs)


    def scal_prod_vars_triple(self, left_terms, right_terms, coefs):
        """ Experimental.
        Creates a quadratic expression from two lists of variables and a sequence of coefficients.

        This method is experimental and may change, be renamed or disappear in a next version.

        This method sums all quadratic terms built by multiplying the i_th coefficient by the product of the i_th
        expression in `left_terms` and the i_th expression in `right_terms`

        Example:
            `Model.scalprod_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`

        :param left_terms: A list or an iterator on variables.
        :param right_terms: A list or an iterator on variables.
        :param coefs: A list or an iterator on numbers or a number.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           If either list or iterator is empty, this method returns zero.
        """
        return self._aggregator._scal_prod_vars_triple(left_terms=left_terms, right_terms=right_terms, coefs=coefs)
