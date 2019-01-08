# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore


from docplex.mp.compat23 import izip

from docplex.mp.utils import is_number, is_iterable, is_iterator, is_pandas_series,\
    is_indexable, has_len, is_numpy_ndarray
from docplex.mp.linear import Var, MonomialExpr, AbstractLinearExpr, LinearExpr, ZeroExpr
from docplex.mp.functional import _IAdvancedExpr
from docplex.mp.quad import QuadExpr


class ModelAggregator(object):

    def __init__(self, linear_factory, quad_factory):
        self._linear_factory = linear_factory
        self._quad_factory = quad_factory
        self._model = self._linear_factory._model
        self.zero_expr = self._linear_factory.zero_expr


    def scal_prod(self, terms, coefs=1.0):
        # Testing anumpy array for its logical value will not work:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # we would have to trap the test for ValueError then call any()
        #
        if is_iterable(coefs):
            pass  # ok
        elif is_number(coefs):
            if 0 == coefs:
                return self.zero_expr
            else:
                sum_expr = self.sum(terms)
                return sum_expr * coefs
        else:
            self._model.fatal("scal_prod expects iterable or number, gort: {0!s}", coefs)

        if not is_iterable(terms):
            terms = [terms]

        return self._scal_prod(terms, coefs)

    def _scal_prod(self, terms, coefs, cc_type=LinearExpr.counter_type):
        # INTERNAL
        linear_factory = self._linear_factory
        total_num = 0
        lcc = cc_type()
        qcc = None

        for item, coef in izip(terms, coefs):
            if 0 == coef:
                continue

            safe_coef = linear_factory.to_valid_number(coef, context_msg=lambda : "Model.dot({0!s}..)".format(item))
            if isinstance(item, Var):
                lcc.update_from_item_value(item, safe_coef)

            elif isinstance(item, AbstractLinearExpr):
                total_num += safe_coef * item.constant
                for lv, lk in item.iter_terms():
                    lcc.update_from_item_value(lv, lk * safe_coef)

            elif isinstance(item, QuadExpr):
                if qcc is None:
                    qcc = cc_type()
                for qv, qk in item.iter_quads():
                    qcc.update_from_item_value(qv, qk * safe_coef)
                qlin = item.linear_part
                for v, k in qlin.iter_terms():
                    lcc.update_from_item_value(v, k * safe_coef)

                total_num += safe_coef * qlin.constant

            # --- all is lost ---
            else:
                self._model.fatal("scal_prod accepts variables, expressions, numbers, not: {0!s}", item)

        sorted_terms = LinearExpr._sort_terms_if_needed(self._model, counter=lcc)
        linear_expr = LinearExpr(self._model, e=sorted_terms, constant=total_num, safe=True)
        if qcc:
            return self._quad_factory.new_quad(quad_args=qcc, linexpr=linear_expr, safe=True)
        else:
            return linear_expr


    def sum(self, sum_args):
        if is_iterable(sum_args):
            if is_iterator(sum_args):
                return self._sum_with_iter(sum_args)

            elif isinstance(sum_args, dict):
                # handle dict: sum all values
                return self._sum_with_seq(sum_args.values())

            elif is_indexable(sum_args):
                if has_len(sum_args) and 0 == len(sum_args):
                    return self.zero_expr

                elif is_numpy_ndarray(sum_args):
                    return self._sum_with_iter(sum_args.flat)
                elif is_pandas_series(sum_args):
                    return self.sum(sum_args.values)
                else:
                    return self._sum_with_seq(sum_args)
            else:
                return self._sum_with_seq(sum_args)
        elif is_number(sum_args):
            return sum_args
        else:
            return self._linear_factory._to_linear_expr(sum_args)

    def _sum_with_iter(self, args, cctype=LinearExpr.counter_type):
        accumulated_ct = 0
        lcc = cctype()
        linear_factory = self._linear_factory
        qcc = None
        for item in args:
            if isinstance(item, Var):
                lcc.update_from_item_value(item)
            elif isinstance(item, MonomialExpr):
                lcc.update_from_item_value(item._dvar, item._coef)
            elif isinstance(item, LinearExpr):
                lcc.update(item._get_terms_dict())
                accumulated_ct += item.constant
            elif isinstance(item, _IAdvancedExpr):
                lcc.update_from_item_value(item.functional_var)
            elif isinstance(item, ZeroExpr):
                pass
            elif is_number(item):
                accumulated_ct += linear_factory.to_valid_number(item, context_msg="Model.sum()")
            elif isinstance(item, QuadExpr):
                for v, k in item.linear_part.iter_terms():
                    lcc.update_from_item_value(v, k)
                if qcc is None:
                    qcc = cctype()
                qcc.update(item._quadterms)
                accumulated_ct += item.constant

            else:
                try:
                    le = item.to_linear_expr()
                    lcc.update(le._get_terms_dict())
                    accumulated_ct += le.constant
                except AttributeError:
                    self._model.fatal("Model.sum() expects numbers/variables/expressions, got: {0!s}", item)

        sorted_terms = LinearExpr._sort_terms_if_needed(self._model, lcc)
        linear_sum = LinearExpr(self._model, e=sorted_terms, constant=accumulated_ct, safe=True)
        if qcc:
            return self._quad_factory.new_quad(quad_args=qcc, linexpr=linear_sum, safe=True)
        else:
            return linear_sum

    def _varlist_to_terms(self, var_list,
                          cc_type=LinearExpr.counter_type,
                          term_dict_type=LinearExpr.term_dict_type):
        # INTERNAL: converts a sum of vars to a dict, sorting if needed.
        if self._model._keep_ordering:
            varsum_terms = term_dict_type([(v, 1) for v in var_list])
        else:
            varsum_terms = cc_type(var_list)
        return varsum_terms

    # @profile
    def _sum_with_seq(self, x_list):
        for z in x_list:
            if not isinstance(z, Var):
                x_seq_all_variables = False
                break
        else:
            x_seq_all_variables = True

        if x_seq_all_variables:
            sumvars_terms = self._varlist_to_terms(x_list)
            return LinearExpr(self._model, e=sumvars_terms, safe=True)
        else:
                return self._sum_with_iter(args=x_list)
