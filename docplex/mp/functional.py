# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

from docplex.mp.basic import Expr

# do NOT import Model -> circular


# noinspection PyAbstractClass
class FunctionalExpr(Expr):

    def __init__(self, model, argument_expr, name=None):
        Expr.__init__(self, model, name)
        self._argument_expr = model._to_linear_expr(argument_expr)
        self._f_var = None
        self._x_var = None
        self._aux_vars = []  # put here all auxiliary variables
        self._aux_cts = []   # put here auxiliary constraints
        self._allocate_x_var_if_necessary()

    def _new_generated_free_continuous_var(self, name=None):
        self_model = self.model
        inf = self_model.infinity
        return self._new_generated_continuous_var(lb=-inf, ub=+inf, name=name)

    def _new_generated_continuous_var(self, lb=None, ub=None, name=None):
        var = self.model.continuous_var(lb=lb, ub=ub, name=name)
        var.notify_origin(self)
        return var

    def _new_generated_binary_var(self, name=None):
        bvar = self.model.binary_var(name=name)
        bvar.notify_origin(self)
        return bvar

    def _new_generated_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        ind = self.model.add_indicator(binary_var,linear_ct, active_value, name)
        ind.notify_origin(self)
        return ind

    def _post_generated_ct(self, ct):
        # posts a onstraint and marks it as generated.
        ct = self.model.add_constraint(ct)
        ct.notify_origin(self)
        return ct

    def _allocate_x_var_if_necessary(self):
        if self._argument_expr.is_variable():
            self._x_var = next(self._argument_expr.iter_variables())
        else:
            self._x_var = self._new_generated_free_continuous_var()
            self._post_generated_ct(self._x_var == self._argument_expr)

    def is_resolved(self):
        return self._f_var is not None

    def to_linear_expr(self):
        # make sure it has been resolved, then return the f_var
        self.ensure_resolved()
        return self.model._monomial_expr(self._f_var)

    def ensure_resolved(self):
        if self._f_var is None:
            # 1. create the var (once!)
            self._f_var = self._create_functional_var()
            # 2. post the link between the fvar and the argument expr
            self._resolve_function()

    def _create_functional_var(self):
        return self._new_generated_free_continuous_var()

    def _resolve_function(self):
        raise NotImplementedError

    def eval(self, numarg):
        raise NotImplementedError

    def _get_function_symbol(self):
        # redefine this to get the function symbol
        return "F"

    @property
    def functional_var(self):
        self.ensure_resolved()
        return self._f_var

    @property
    def argument_expr(self):
        return self._argument_expr

    @property
    def function_symbol(self):
        return self._get_function_symbol()

    def to_string(self):
        return "{0:s}({1!s})".format(self.function_symbol, self._argument_expr)

    def __str__(self):
        return self.to_string()

    def contains_var(self, dvar):
        return dvar == self.functional_var or self.argument_expr.contains_var(dvar)

    def iter_variables(self):
        return self.generate_variables()

    def generate_variables(self):
        self_functional_var = self.functional_var
        if self_functional_var is not None:
            yield self_functional_var
        for v in self._argument_expr.iter_variables():
            yield v

    # -- arithmetic operators
    def times(self, e):
        return self.functional_var.times(e)

    def __mul__(self, e):
        return self.times(e)

    def __rmul__(self, e):
        return self.times(e)

    def plus(self, e):
        return self.functional_var.plus(e)

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.plus(e)

    def minus(self, e):
        return self.functional_var.minus(e)

    def __sub__(self, e):
        return self.functional_var.minus(e)

    def __rsub__(self, other):
        return

    def solution_value(self):
        arg_value = self._argument_expr.solution_value
        return self.eval(arg_value)



class AbsExpr(FunctionalExpr):

    _abs_instance_counter = 0

    def __init__(self, model, argument_expr):
        FunctionalExpr.__init__(self, model, argument_expr)

    def _get_function_symbol(self):
        return "abs"

    def clone(self):
        return AbsExpr(self.model, self.argument_expr.clone())

    def _create_functional_var(self):
        return self._new_generated_continuous_var(name="_abs_%d" % AbsExpr._abs_instance_counter)

    def _resolve_function(self):
        self_f_var = self._f_var
        AbsExpr._abs_instance_counter += 1
        abs_count = AbsExpr._abs_instance_counter
        # 1 create two vars
        self.positive_var = self._new_generated_continuous_var(lb=0, name="_abs_pp_%d" % abs_count)
        self.negative_var = self._new_generated_continuous_var(lb=0, name="_abs_np_%d" % abs_count)
        # F(x) = p + n
        self._post_generated_ct(self_f_var == self.positive_var + self.negative_var)
        # x = p-n
        self._post_generated_ct(self._argument_expr == self.positive_var - self.negative_var)
        # link sos with sign
        self._plus_var = self._new_generated_binary_var(name="_abs_is_p_%d" % abs_count)
        self._minus_var = self._new_generated_binary_var(name="_abs_is_n_%d" % abs_count)
        self._post_generated_ct(self._plus_var + self._minus_var == 1)
        self._new_generated_indicator(self._plus_var, self.negative_var <= 0, active_value=1)
        self._new_generated_indicator(self._minus_var, self.positive_var <= 0, active_value=1)

    def eval(self, numarg):
        return abs(numarg)
