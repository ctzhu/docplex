# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
from docplex.mp.mfactory import _AbstractModelFactory

from docplex.mp.constants import ComparisonType
from docplex.mp.utils import is_number, DOCPlexQuadraticArithException
from docplex.mp.quad import QuadExpr, VarPair
from docplex.mp.linear import Var, MonomialExpr, LinearExpr, ZeroExpr
from docplex.mp.constr import QuadraticConstraint


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
        self._qterm_type = QuadExpr._qterms_dict_type
        self.zero_expr = model._get_zero_expr()

    def _unexpected_product_error(self, factor1, factor2):
        # INTERNAL
        self._model.fatal("cannot multiply {0!s} by {1!s}", factor1, factor2)

    def new_quad(self, quad_args=None, linexpr=None, name=None, safe=False):
        return QuadExpr(self._model, quads=quad_args, linexpr=linexpr, name=name, safe=safe)

    def new_linear_expr(self, e=0, cst=0):
        return self._model._linear_expr(e, cst)

    def new_var_square(self, var):
        return self.new_quad(quad_args=(var, var, 1))

    def new_var_product(self, var, other):
        # computes and returns the p[roduct var * other
        if isinstance(other, Var):
            return self.new_quad(quad_args=(var, other, 1), safe=True)
        elif isinstance(other, MonomialExpr):
            mnm_dvar = other._dvar
            mnm_coef = other.coef
            return self.new_quad(quad_args=(var, mnm_dvar, mnm_coef), safe=True)
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
            return self.new_quad(quad_args, quad_linexp, safe=True)

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
            self._unexpected_product_error(linexpr, other)\


    def new_le_constraint(self, e, rhs, ctname=None):
        return self._new_qconstraint(e, ComparisonType.LE, rhs, name=ctname)

    def new_eq_constraint(self, e, rhs, ctname=None):
        return self._new_qconstraint(e, ComparisonType.EQ, rhs, name=ctname)

    def new_ge_constraint(self, e, rhs, ctname=None):
        return self._new_qconstraint(e, ComparisonType.GE, rhs, name=ctname)

    def _new_qconstraint(self, lhs, ctype, rhs, name=None):
        # noinspection PyPep8
        left_expr  = self._to_expr(lhs, context="QuadraticConstraint.left_expr")
        right_expr = self._to_expr(rhs, context="QuadraticConstraint.right_expr")
        self._model._check_both_in_selfmodel(left_expr, right_expr, "new_binary_constraint")
        ct = QuadraticConstraint(self._model, left_expr, ctype, right_expr, name)
        left_expr.notify_used(ct)
        right_expr.notify_used(ct)
        return ct

    def _to_expr(self, e, context=None):
        # INTERNAL
        if hasattr(e, "iter_terms"):
            return e
        elif is_number(e):
            return self._model._lfactory.constant_expr(cst=e, context=context)
        else:
            try:
                return e.to_linear_expr()
            except DOCPlexQuadraticArithException:
                return e
            except AttributeError:
                self._model.fatal("cannot convert to expression: {0!r}", e)
