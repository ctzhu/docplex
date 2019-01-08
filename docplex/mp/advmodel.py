# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------


from docplex.mp.model import Model
from docplex.mp.aggregator import ModelAggregator
from docplex.mp.quad import VarPair
from docplex.mp.utils import is_number, is_iterable, generate_constant

from docplex.mp.compat23 import izip, fast_range


class AdvAggregator(ModelAggregator):
    def __init__(self, linear_factory, quad_factory):
        ModelAggregator.__init__(self, linear_factory, quad_factory)

    def _scal_prod_vars_all_different(self, terms, coefs):
        checker = self._checker
        if not coefs:
            return self.zero_expr
        elif not is_iterable(coefs, accept_string=False):
            checker.typecheck_num(coefs)
            return coefs * self._sum_vars_all_different(terms)
        else:
            # coefs is iterable
            lcc_type = self.counter_type
            lcc = lcc_type()
            lcc_setitem = lcc_type.__setitem__
            number_validation_fn = checker.get_number_validation_fn()
            for dvar, coef in izip(terms, coefs):
                if coef:
                    safe_coef = number_validation_fn(coef) if number_validation_fn else coef
                    lcc_setitem(lcc, dvar, safe_coef)

            return self._to_expr(qcc=None, lcc=lcc)

    def scal_prod_triple(self, left_terms, right_terms, coefs):
        used_coefs = None

        if is_iterable(coefs, accept_string=False):
            used_coefs = coefs
        elif is_number(coefs):
            if coefs:
                used_coefs = generate_constant(coefs, count_max=None)
            else:
                return self.zero_expr
        else:
            self._model.fatal("scal_prod_triple expects iterable or number as coefficients, got: {0!r}", coefs)

        if is_iterable(left_terms):
            used_left = left_terms
        else:
            used_left = generate_constant(left_terms, count_max=None)

        if is_iterable(right_terms):
            used_right = right_terms
        else:
            used_right = generate_constant(right_terms, count_max=None)

        if used_coefs is not coefs and used_left is not left_terms and used_right is not right_terms:
            return left_terms * right_terms * coefs

        return self._scal_prod_triple(coefs=used_coefs, left_terms=used_left, right_terms=used_right)

    def _scal_prod_triple(self, coefs, left_terms, right_terms):
        # INTERNAL
        accumulated_ct = 0
        qcc = self.counter_type()
        lcc = self.counter_type()
        number_validation_fn = self._checker.get_number_validation_fn()
        for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
            if coef:
                safe_coef = number_validation_fn(coef) if number_validation_fn else coef
                lcst = lterm.get_constant()
                rcst = rterm.get_constant()
                accumulated_ct += safe_coef * lcst * rcst
                for lv, lk in lterm.iter_terms():
                    for rv, rk in rterm.iter_terms():
                        coef3 = safe_coef * lk * rk
                        qcc.update_from_item_value(VarPair(lv, rv), coef3)
                if rcst:
                    for lv, lk in lterm.iter_terms():
                        lcc.update_from_item_value(lv, safe_coef * lk * rcst)
                if lcst:
                    for rv, rk in rterm.iter_terms():
                        lcc.update_from_item_value(rv, safe_coef * rk * lcst)

        return self._to_expr(qcc, lcc, constant=accumulated_ct)

    def _scal_prod_triple_vars(self, coefs, left_terms, right_terms):
        # INTERNAL
        # assuming all arguments are iterable.
        dcc = self.counter_type
        qcc = dcc()
        number_validation_fn = self._checker.get_number_validation_fn()
        for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
            if coef:
                safe_coef = number_validation_fn(coef) if number_validation_fn else coef
                qcc.update_from_item_value(item=VarPair(lterm, rterm), value=safe_coef)
        return self._to_expr(qcc=qcc)

    def _sumsq_vars_all_different(self, dvars):
        dcc = self.counter_type
        qcc = dcc()
        qcc_setitem = dcc.__setitem__
        for t in dvars:
            qcc_setitem(qcc, VarPair(t), 1)
        return self._to_expr(qcc=qcc)

    def _sumsq_vars(self, dvars):
        qcc = self._quad_factory.term_dict_type()
        for v in dvars:
            qcc.update_from_item_value(VarPair(v), 1)
        return self._to_expr(qcc=qcc)

    def _sum_vars_all_different(self, dvars):
        lcc = self.counter_type()

        setitem_fn = lcc.__setitem__

        for v in dvars:
            setitem_fn(v, 1)
        return self._to_expr(qcc=None, lcc=lcc)

    def quad_matrix_sum(self, matrix, dvars, symmetric=False):
        # assume matrix is a NxN matrix
        # vars is a N-vector of variables
        n = len(dvars)
        dcc = self._quad_factory.term_dict_type
        qterms = dcc()

        for i in fast_range(n):
            for j in fast_range(i + 1):
                if i == j:
                    qterms[VarPair(dvars[i])] = matrix[i][i]
                elif symmetric:
                    qterms[VarPair(dvars[i], dvars[j])] = 2 * matrix[i][j]
                else:
                    qterms[VarPair(dvars[i], dvars[j])] = matrix[i][j] + matrix[j][i]
        return self._to_expr(qcc=qterms)


# noinspection PyProtectedMember
class AdvModel(Model):
    """
    This class is a specialized version of the :class:`docplex.mp.model.Model` class with useful non-standard modeling
    functions.
    """

    def __init__(self, name=None, context=None, **kwargs):
        Model.__init__(self, name=name, context=context, **kwargs)
        self._aggregator = AdvAggregator(self._lfactory, self._qfactory)

    def sum_vars(self, terms):
        return self._aggregator._sum_vars(terms)

    def sum_vars_all_different(self, terms):
        """
        Creates a linear expression equal to sum of a list of decision variables.
        The variable sequence is a list or an iterator of variables.

        This method is faster than the standard generic summation method due to the fact that it takes only
        variables and does not take expressions as arguments.
        By default, check for no duplicates will be done. So, best performances may be obtained only when using deployment modes (no checker).

        :param terms: A list or an iterator on variables only, with no duplicates.

        :return: A linear expression or 0.

        Note:
           If the list or iterator is empty, this method returns zero.

        Note:
            To improve performance, the check for duplicates can be turned off by setting
            `checker='off'` in the `kwargs` of the :class:`docplex.mp.model.Model` object. As this argument
            turns off checking everywhere, it should be used with extreme caution.
        """
        var_seq = self._checker.typecheck_var_seq_all_different(terms)
        return self._aggregator._sum_vars_all_different(var_seq)

    def sumsq_vars(self, terms):
        return self._aggregator._sumsq_vars(terms)

    def sumsq_vars_all_different(self, terms):
        """
        Creates a quadratic expression by summing squares over a sequence.

        The variable sequence is a list or an iterator of variables.

        This method is faster than the standard summation of squares method due to the fact that it takes only variables and does not take expressions as arguments.
        By default, check for no duplicates will be done. So, best performances may be obtained only when using deployment modes (no checker).

        :param terms: A list or an iterator on variables only, with no duplicates.

        :return: A quadratic expression or 0.

        Note:
           If the list or iterator is empty, this method returns zero.

        Note:
            To improve performance, the check for duplicates can be turned off by setting
            `checker='off'` in the `kwargs` of the :class:`docplex.mp.model.Model` object. As this argument
            turns off checking everywhere, it should be used with extreme caution.
        """
        var_seq = self._checker.typecheck_var_seq_all_different(terms)
        return self._aggregator._sumsq_vars_all_different(var_seq)

    def scal_prod_vars_all_different(self, terms, coefs):
        """
        Creates a linear expression equal to the scalar product of a list of decision variables and a sequence of coefficients.

        The variable sequence is a list or an iterator of variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.

        This method is faster than the standard generic scalar product method due to the fact that it takes only variables and does not take expressions as arguments.
        By default, check for no duplicates will be done. So, best performances may be obtained only when using deployment modes (no checker).

        :param terms: A list or an iterator on variables only, with no duplicates.
        :param coefs: A list or an iterator on numbers, or a number.

        :return: A linear expression or 0.

        Note:
           If either list or iterator is empty, this method returns zero.

        Note:
            To improve performance, the check for duplicates can be turned off by setting
            `checker='off'` in the `kwargs` of the :class:`docplex.mp.model.Model` object. As this argument
            turns off checking everywhere, it should be used with extreme caution.
        """
        self._checker.check_ordered_sequence(arg=terms,
                                             header='Model.scal_prod() requires a list of expressions/variables')
        var_seq = self._checker.typecheck_var_seq_all_different(terms)
        return self._aggregator._scal_prod_vars_all_different(var_seq, coefs)

    def quad_matrix_sum(self, matrix, dvars, symmetric=False):
        """
        Creates a quadratic expression equal to the quadratic form of a list of decision variables and
        a matrix of coefficients.

        This method sums all quadratic terms built by multiplying the [i,j]th coefficient in the matrix
        by the product of the i_th and j_th variables in `dvars`; in mathematical terms, the expression formed
        by x'Qx.

        :param matrix: A 2-dimensional list.
        :param dvars: A list or an iterator on variables.
        :param symmetric: A boolean indicating whether the matrix is symmetric or not (default is False).
            No check is done.

        :return: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           The matrix must be square but not necessarily symmetric. The number of rows of the matrix must be equal
           to the size of the variable sequence.
        """
        return self._aggregator.quad_matrix_sum(matrix, dvars, symmetric=symmetric)

    def scal_prod_triple(self, left_terms, right_terms, coefs):
        """
        Creates a quadratic expression from two lists of linear expressions and a sequence of coefficients.

        This method sums all quadratic terms built by multiplying the i_th coefficient by the product of the i_th
        expression in `left_terms` and the i_th expression in `right_terms`

        This method accepts different types of input for its arguments. The expression sequences can be either lists
        or iterators of objects that can be converted to linear expressions, that is, variables or linear expressions
        (but no quadratic expressions).
        The most usual case is variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.

        Example:
            `Model.scal_prod_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`.

        :param left_terms: A list or an iterator on variables or expressions.
        :param right_terms: A list or an iterator on variables or expressions.
        :param coefs: A list or an iterator on numbers or a number.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           If either list or iterator is empty, this method returns zero.
        """
        return self._aggregator.scal_prod_triple(left_terms=left_terms, right_terms=right_terms, coefs=coefs)

    def scal_prod_triple_vars(self, left_terms, right_terms, coefs):
        """
        Creates a quadratic expression from two lists of variables and a sequence of coefficients.

        This method sums all quadratic terms built by multiplying the i_th coefficient by the product of the i_th
        expression in `left_terms` and the i_th expression in `right_terms`

        This method is faster than the standard generic scalar quadratic product method due to the fact that it takes only variables and does not take expressions as arguments.
        By default, check for no duplicates will be done. So, best performances may be obtained only when using deployment modes (no checker).

        Example:
            `Model.scal_prod_vars_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`.

        :param left_terms: A list or an iterator on variables.
        :param right_terms: A list or an iterator on variables.
        :param coefs: A list or an iterator on numbers or a number.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           If either list or iterator is empty, this method returns zero.
        """
        used_coefs = None
        checker = self._checker
        nb_non_iterables = 0

        if is_iterable(coefs, accept_string=False):
            used_coefs = coefs
        elif is_number(coefs):
            if coefs:
                used_coefs = generate_constant(coefs, count_max=None)
                nb_non_iterables += 1
            else:
                return self._aggregator.zero_expr
        else:
            self.fatal("scal_prod_triple expects iterable or number as coefficients, got: {0!r}", coefs)

        if is_iterable(left_terms):
            used_left = checker.typecheck_var_seq(left_terms)
        else:
            nb_non_iterables += 1
            checker.typecheck_var(left_terms)
            used_left = generate_constant(left_terms, count_max=None)

        if is_iterable(right_terms):
            used_right = checker.typecheck_var_seq(right_terms)
        else:
            nb_non_iterables += 1
            checker.typecheck_var(right_terms)
            used_right = generate_constant(right_terms, count_max=None)

        if nb_non_iterables >= 3:
            return left_terms * right_terms * coefs
        else:
            return self._aggregator._scal_prod_triple_vars(left_terms=used_left,
                                                           right_terms=used_right, coefs=used_coefs)
