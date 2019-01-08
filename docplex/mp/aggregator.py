# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from six import itervalues
from docplex.mp.compat23 import izip

from docplex.mp.xcounter import FastOrderedCounter, ExprCounter

from docplex.mp.utils import is_number, is_iterable, is_iterator, is_pandas_series, \
    is_numpy_ndarray, is_pandas_dataframe, is_numpy_matrix
from docplex.mp.linear import Var, MonomialExpr, AbstractLinearExpr, LinearExpr, ZeroExpr
from docplex.mp.functional import _IAdvancedExpr
from docplex.mp.quad import QuadExpr, VarPair


class ModelAggregator(object):
    # what type to use for merging dicts

    def __init__(self, linear_factory, quad_factory, ordered=True):
        self._linear_factory = linear_factory
        self._checker = linear_factory._checker
        self._quad_factory = quad_factory
        self._model = linear_factory._model
        self.set_ordering(ordered)
        self._generate_transients = True

    def new_zero_expr(self):
        return ZeroExpr(model=self._model)

    def _to_expr(self, qcc, lcc=None, constant=0):
        # no need to sort here, sort is done by str() on the fly.
        if qcc:
            linear_expr = LinearExpr(self._model, e=lcc, constant=constant, safe=True)
            quad = self._quad_factory.new_quad(quads=qcc, linexpr=linear_expr, safe=True)
            quad._transient = self._generate_transients
            return quad
        elif lcc or constant:
            linear_expr = LinearExpr(self._model, e=lcc, constant=constant, safe=True)
            linear_expr._transient = self._generate_transients
            return linear_expr
        else:
            return self.new_zero_expr()

    def set_ordering(self, ordered):
        self._ordered = ordered
        self.counter_type = FastOrderedCounter if ordered else ExprCounter

    def scal_prod(self, terms, coefs=1.0):
        # Testing anumpy array for its logical value will not work:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # we would have to trap the test for ValueError then call any()
        #
        if is_iterable(coefs):
            pass  # ok
        elif is_number(coefs):
            if 0 == coefs:
                return self.new_zero_expr()
            else:
                sum_expr = self.sum(terms)
                return sum_expr * coefs
        else:
            self._model.fatal("scal_prod expects iterable or number, gort: {0!s}", coefs)

        # model has checked terms is an ordered sequence
        return self._scal_prod(terms, coefs)

    def _scal_prod(self, terms, coefs):
        # INTERNAL
        checker = self._checker
        total_num = 0
        lcc = self.counter_type()
        qcc = None

        number_validation_fn = checker.get_number_validation_fn()

        for item, coef in izip(terms, coefs):
            if not coef:
                continue

            safe_coef = number_validation_fn(coef) if number_validation_fn else coef
            if isinstance(item, Var):
                lcc.update_from_item_value(item, safe_coef)

            elif isinstance(item, AbstractLinearExpr):
                total_num += safe_coef * item.get_constant()
                for lv, lk in item.iter_terms():
                    lcc.update_from_item_value(lv, lk * safe_coef)

            elif isinstance(item, QuadExpr):
                if qcc is None:
                    qcc = self.counter_type()
                for qv, qk in item.iter_quads():
                    qcc.update_from_item_value(qv, qk * safe_coef)
                qlin = item.get_linear_part()
                for v, k in qlin.iter_terms():
                    lcc.update_from_item_value(v, k * safe_coef)

                total_num += safe_coef * qlin.constant

            # --- all is lost ---
            else:
                self._model.fatal("scal_prod accepts variables, expressions, numbers, not: {0!s}", item)

        return self._to_expr(qcc, lcc, total_num)

    def sum(self, sum_args):
        if is_iterator(sum_args):
            return self._sum_with_iter(sum_args)

        elif isinstance(sum_args, dict):
            # handle dict: sum all values
            return self._sum_with_iter(itervalues(sum_args))

        if is_numpy_ndarray(sum_args):
            return self._sum_with_iter(sum_args.flat)
        elif is_pandas_series(sum_args):
            return self.sum(sum_args.values)
        elif is_iterable(sum_args):
            return self._sum_with_seq(sum_args)

        elif is_number(sum_args):
            return sum_args
        else:
            return self._linear_factory._to_linear_expr(sum_args)

    def _sum_with_iter(self, args):
        sum_of_nums = 0
        lcc = self.counter_type()
        checker = self._checker
        qcc = None
        number_validation_fn = checker.get_number_validation_fn()
        for item in args:
            if isinstance(item, Var):
                lcc.update_from_item_value(item)
            elif isinstance(item, MonomialExpr):
                lcc.update_from_item_value(item._dvar, item._coef)
            elif isinstance(item, LinearExpr):
                lcc.update(item._get_terms_dict())
                sum_of_nums += item.constant
            elif isinstance(item, _IAdvancedExpr):
                lcc.update_from_item_value(item.functional_var)
            elif isinstance(item, ZeroExpr):
                pass
            elif is_number(item):
                sum_of_nums += number_validation_fn(item) if number_validation_fn else item
            elif isinstance(item, QuadExpr):
                for v, k in item.linear_part.iter_terms():
                    lcc.update_from_item_value(v, k)
                if qcc is None:
                    qcc = self.counter_type()
                for qvp, qk in item.iter_quads():
                    qcc.update_from_item_value(qvp, qk)
                sum_of_nums += item.constant

            else:
                try:
                    le = item.to_linear_expr()
                    lcc.update(le._get_terms_dict())
                    sum_of_nums += le.constant
                except AttributeError:
                    self._model.fatal("Model.sum() expects numbers/variables/expressions, got: {0!s}", item)

        return self._to_expr(qcc, lcc, sum_of_nums)

    def _sum_vars(self, dvars):
        sumvars_terms = self._varlist_to_terms(dvars)
        return self._to_expr(qcc=None, lcc=sumvars_terms)

    def _varlist_to_terms(self, var_list):
        # INTERNAL: converts a sum of vars to a dict, sorting if needed.
        linear_term_dict_type = self._linear_factory.term_dict_type
        if len(var_list) == len(set(var_list)):
            varsum_terms = linear_term_dict_type()
            linear_terms_setitem = linear_term_dict_type.__setitem__
            for v in var_list:
                linear_terms_setitem(varsum_terms, v, 1)
        else:
            # there are repeated variables.
            varsum_terms = linear_term_dict_type()
            for v in var_list:
                varsum_terms.update_from_item_value(v, 1)
        return varsum_terms

    def _sum_with_seq(self, sum_args):
        for z in sum_args:
            if not isinstance(z, Var):
                x_seq_all_variables = False
                break
        else:
            x_seq_all_variables = True

        if x_seq_all_variables:
            return self._sum_vars(sum_args)
        else:
            return self._sum_with_iter(args=sum_args)

    def _sumsq(self, args):
        accumulated_ct = 0
        number_validation_fn = self._checker.get_number_validation_fn()
        qcc = self._quad_factory.term_dict_type()
        lcc = self._linear_factory.term_dict_type()

        for item in args:
            if isinstance(item, Var):
                qcc.update_from_item_value(VarPair(item, item), 1)
            elif isinstance(item, MonomialExpr):
                mcoef = item._coef
                # noinspection PyPep8
                mvar = item._dvar
                qcc.update_from_item_value(VarPair(mvar, mvar), mcoef ** 2)

            elif isinstance(item, LinearExpr):
                cst = item.get_constant()
                accumulated_ct += cst ** 2
                for lv1, lk1 in item.iter_terms():
                    for lv2, lk2 in item.iter_terms():
                        if lv1 is lv2:
                            qcc.update_from_item_value(VarPair(lv1, lv1), lk1 * lk1)
                        elif lv1._index < lv2._index:
                            qcc.update_from_item_value(VarPair(lv1, lv2), 2 * lk1 * lk2)
                        else:
                            pass

                    if cst:
                        lcc.update_from_item_value(lv1, 2 * cst * lk1)
            elif isinstance(item, _IAdvancedExpr):
                fvar = item.functional_var
                qcc.update_from_item_value(VarPair(fvar), 1)

            elif isinstance(item, ZeroExpr):
                pass

            elif is_number(item):
                safe_item = number_validation_fn(item) if number_validation_fn else item
                accumulated_ct += safe_item ** 2

            else:
                self._model.fatal("Model.sumsq() expects numbers/variables/linear expressions, got: {0!s}", item)

        return self._to_expr(qcc, lcc, constant=accumulated_ct)

    def sumsq(self, sum_args):
        if is_iterable(sum_args):
            if is_iterator(sum_args):
                return self._sumsq(sum_args)
            elif isinstance(sum_args, dict):
                return self._sumsq(sum_args.values())
            elif is_numpy_ndarray(sum_args):
                return self._sumsq(sum_args.flat)
            elif is_pandas_series(sum_args):
                return self._sumsq(sum_args.values)

            else:
                return self._sumsq(sum_args)
        elif is_number(sum_args):
            return sum_args ** 2
        else:
            self._model.fatal("Model.sumsq() expects number/iterable/expression, got: {0!s}", sum_args)

    # --- matrix constraint
    @staticmethod
    def generate_df_rows(df):
        for index, row in df.iterrows():
            yield row

    @staticmethod
    def generate_np_matrix_rows(npm):
        for r in npm:
            yield r.tolist()[0]

    def _sparse_matrix_constraints(self, sp_coef_mat, svars, srhs, op):
        range_cts = range(len(srhs))
        lfactory = self._linear_factory
        exprs = [lfactory.linear_expr() for _ in range_cts]
        for e in range(sp_coef_mat.nnz):
            coef = sp_coef_mat.data[e]
            row = sp_coef_mat.row[e]
            col = sp_coef_mat.col[e]
            exprs[row]._add_term(svars[col], coef)
        cts = [lfactory._new_binary_constraint(exprs[r], cmp_op=op, rhs=srhs[r]) for r in range_cts]
        return cts

    def _matrix_constraints(self, coef_mat, svars, srhs, op):

        if is_pandas_dataframe(coef_mat):
            row_gen = self.generate_df_rows(coef_mat)
        elif is_numpy_matrix(coef_mat):
            row_gen = self.generate_np_matrix_rows(coef_mat)
        else:
            row_gen = iter(coef_mat)

        cts = [self._linear_factory._new_binary_constraint(lhs=self._scal_prod(svars, row), cmp_op=op, rhs=rhs)
               for row, rhs in izip(row_gen, srhs)]

        return cts
