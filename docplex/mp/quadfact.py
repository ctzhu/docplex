# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
from docplex.mp.mfactory import _AbstractModelFactory

from docplex.mp.quad import QuadExpr, VarPair
from docplex.mp.linear import Var, MonomialExpr, LinearExpr, ZeroExpr

from docplex.mp.xcounter import FastOrderedCounter


class IQuadFactory(_AbstractModelFactory):
    # INTERNAL

    def new_var_product(self, var, factor):
        raise NotImplementedError  # pragma: no cover

    def new_monomial_product(self, monexpr, factor):
        raise NotImplementedError  # pragma: no cover

    def new_linexpr_product(self, linexpr, factor):
        raise NotImplementedError  # pragma: no cover

    def new_var_square(self, var):
        raise NotImplementedError  # pragma: no cover


class QuadFactory(IQuadFactory):

    def __init__(self, model):
        _AbstractModelFactory.__init__(self, model)
        self._model = model
        self._qterm_type = QuadExpr.qterms_dict_type
        self.zero_expr = model._get_zero_expr()

    def _unexpected_product_error(self, factor1, factor2):
        # INTERNAL
        self._model.fatal("cannot multiply {0!s} by {1!s}", factor1, factor2)

    # TODO: change safe default to False
    def new_quad(self, quad_args, linexpr=None, safe=True):
        return QuadExpr(self._model, quads=quad_args, linexpr=linexpr, safe=safe)

    def new_linear_expr(self, e=0, cst=0):
        return self._model.linear_expr(e, cst)

    def new_var_square(self, var):
        return self.new_quad(quad_args=(var, var, 1))

    def new_var_product(self, var, other):
        # computes and returns the p[roduct var * other
        if isinstance(other, Var):
            return self.new_quad(quad_args=(var, other, 1))
        elif isinstance(other, MonomialExpr):
            mnm_dvar = other._dvar
            mnm_coef = other.coef
            return self.new_quad(quad_args=(var, mnm_dvar, mnm_coef))
        elif isinstance(other, ZeroExpr):
            return other
        elif isinstance(other, LinearExpr):
            linexpr = other
            quad_args = [(VarPair(var, dv), k) for dv, k in linexpr.iter_terms()]
            linexpr_k = linexpr.constant
            if 0 != linexpr_k:
                quad_linexp = linexpr_k * var
            else:
                quad_linexp = None
            return self.new_quad(quad_args, quad_linexp)

        else:
            self._unexpected_product_error(var, other)

    def new_monomial_product(self, mnm, other):
        mnmk = mnm.coef
        if 0 == mnmk:
            return self.zero_expr
        else:
            var_quad = self.new_var_product(mnm.var, other)
            var_quad._scale(mnmk)
            return var_quad

    def new_linexpr_product(self, linexpr, other):
        if isinstance(other, Var):
            return self.new_var_product(other, linexpr)

        elif isinstance(other, MonomialExpr):
            return self.new_monomial_product(other, self)

        elif isinstance(other, LinearExpr):
            cst1 = linexpr.constant
            cst2 = other.constant

            fcc = FastOrderedCounter()
            for lv1, lk1 in linexpr.iter_terms():
                for lv2, lk2 in other.iter_terms():
                    fcc.update_from_item_value(VarPair(lv1, lv2), lk1 * lk2)
            # this is quad
            qlinexpr = self.new_linear_expr()
            # add cst2 * linexp1
            qlinexpr._add_expr_scaled(expr=linexpr, factor=cst2)
            # add cst1 * linexpr2
            qlinexpr._add_expr_scaled(expr=other, factor=cst1)

            # and that's it
            # fix the constant
            qlinexpr.constant = cst1 * cst2
            quad = QuadExpr(self._model, quads=fcc, linexpr=qlinexpr, safe=True)
            return quad

        else:
            self._unexpected_product_error(linexpr, other)


# class NotImplementedQuadFactory(IQuadFactory):
#     def __init__(self, model):
#         _AbstractModelFactory.__init__(self, model)
#
#     # noinspection PyMethodMayBeStatic
#     def _quads_not_implemented_error(self, factor1, factor2):
#         raise DOCplexQuadraticNotImplementedError(factor1, factor2)
#
#     def new_var_product(self, var, factor):
#         assert isinstance(var, Var)
#         self._quads_not_implemented_error(var, factor)
#
#     def new_monomial_product(self, monexpr, factor):
#         assert isinstance(monexpr, MonomialExpr)
#         self._quads_not_implemented_error(monexpr, factor)
#
#     def new_linexpr_product(self, linexpr, factor):
#         assert isinstance(linexpr, LinearExpr)
#         self._quads_not_implemented_error(linexpr, factor)
#
#     def new_var_square(self, var):
#         self._quads_not_implemented_error(var, var)

