# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

from docplex.mp.basic import Expr
from docplex.mp.utils import is_iterable, is_iterator

# do NOT import Model -> circular


# noinspection PyAbstractClass
class _IAdvancedExpr(Expr):
    def __init__(self, model, name=None):
        Expr.__init__(self, model, name)
        self._f_var = None

    def _new_generated_free_continuous_var(self, name=None):
        # INTERNAL
        self_model = self.model
        inf = self_model.infinity
        return self._new_generated_continuous_var(lb=-inf, ub=+inf, name=name)

    def _new_generated_continuous_var(self, lb=None, ub=None, name=None):
        # INTERNAL
        var = self.model.continuous_var(lb=lb, ub=ub, name=name)
        var.notify_origin(self)
        return var

    def _new_generated_binary_var(self, name=None):
        bvar = self.model.binary_var(name=name)
        bvar.notify_origin(self)
        return bvar

    def _new_generated_binary_varlist(self, keys, name=None):
        bvars = self.model.binary_var_list(keys, name)
        for bv in bvars:
            bv.notify_origin(self)
        return bvars

    def _new_generated_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        ind = self.model.add_indicator(binary_var, linear_ct, active_value, name)
        ind.notify_origin(self)
        return ind

    def _post_generated_ct(self, ct):
        # posts a onstraint and marks it as generated.
        ct = self.model.add_constraint(ct)
        ct.notify_origin(self)
        return ct

    def to_linear_expr(self):
        # make sure it has been resolved, then return 1 * fvar
        return self.model.linear_expr(self._get_resolved_f_var())

    def _get_resolved_f_var(self):
        if self._f_var is None:
            # 1. create the var (once!)
            self._f_var = self._create_functional_var()
            # 2. post the link between the fvar and the argument expr
            self._resolve()
        return self._f_var

    def _is_resolved(self):
        return self._f_var is not None

    def _create_functional_var(self):
        # add a unique counter suffix cf. RTC
        unique_counter = self.model._new_unique_counter()
        if unique_counter == 0:
            # the first time, we use the the raw name, else we suffix #<unique>
            fname = self.to_string()
        else:
            fname = "%s#%d" % (self.to_string(), unique_counter)
        return self._new_generated_free_continuous_var(name=fname)

    @property
    def functional_var(self):
        return self._get_resolved_f_var()

    def _resolve(self):
        raise NotImplementedError  # pragma : no cover

    def _get_function_symbol(self):
        # redefine this to get the function symbol
        raise NotImplementedError  # pragma : no cover

    @property
    def function_symbol(self):
        return self._get_function_symbol()

    def __str__(self):
        return self.to_string()

    def to_string(self, **kwargs):
        raise NotImplementedError  # pragma : no cover

    # -- arithmetic operators
    def __mul__(self, e):
        return self.functional_var.__mul__(e)

    def __rmul__(self, e):
        return self.functional_var.__mul__(e)

    def __add__(self, e):
        return self.functional_var.__add__(e)

    def __radd__(self, e):
        return self.functional_var.__add__(e)

    def __sub__(self, e):
        return self.functional_var.__sub__(e)

    def __rsub__(self, e):
        return self.functional_var.__rsub__(e)

    def __neg__(self):
        # the "-e" unary minus returns a linear expression
        return self.functional_var.__neg__()

    #  ---

    def __ge__(self, e):
        return self.functional_var.__ge__(e)

    def __le__(self, e):
        return self.functional_var.__le__(e)

    def __eq__(self, e):
        return self.functional_var.__eq__(e)

    def _allocate_arg_var_if_necessary(self, arg_expr):
        # INTERNAL
        # allocates a new variables if only the argument expr is not a variable
        # and returns it
        arg_var = None
        try:
            if arg_expr.is_variable():
                arg_var = next(arg_expr.iter_variables())
        except AttributeError:
            arg_var = None

        if arg_var is None:
            arg_var = self._new_generated_free_continuous_var()
            self._post_generated_ct(arg_var == arg_expr)
        return arg_var


# noinspection PyAbstractClass
class FunctionalExpr(_IAdvancedExpr):

    def __init__(self, model, argument_expr, name=None):
        _IAdvancedExpr.__init__(self, model, name)
        self._argument_expr = model._to_linear_expr(argument_expr)
        self._x_var = self._allocate_arg_var_if_necessary(argument_expr)

    @property
    def argument_expr(self):
        return self._argument_expr

    def eval(self, numarg):
        raise NotImplementedError  # pragma : no cover

    def is_discrete(self):
        return self._argument_expr.is_discrete()

    def to_string(self):
        return "{0:s}({1!s})".format(self.function_symbol, self._argument_expr)

    def contains_var(self, dvar):
        return dvar == self.functional_var or self._argument_expr.contains_var(dvar)

    def iter_variables(self):
        return self.generate_variables()

    def generate_variables(self):
        self_functional_var = self.functional_var
        if self_functional_var is not None:
            yield self_functional_var
        for v in self._argument_expr.iter_variables():
            yield v


class AbsExpr(FunctionalExpr):
    def copy(self, target_model, var_mapping):
        pass

    def __init__(self, model, argument_expr):
        FunctionalExpr.__init__(self, model, argument_expr)

    def _get_function_symbol(self):
        return "abs"

    def clone(self):
        return AbsExpr(self.model, self._argument_expr)

    def _resolve(self):
        self_f_var = self._f_var
        abs_index = self.model._new_unique_counter()
        # 1 create two vars
        self.positive_var = self._new_generated_continuous_var(lb=0, name="_abs_pp_%d" % abs_index)
        self.negative_var = self._new_generated_continuous_var(lb=0, name="_abs_np_%d" % abs_index)
        # F(x) = p + n
        self._post_generated_ct(self_f_var == self.positive_var + self.negative_var)
        # x = p-n
        self._post_generated_ct(self._argument_expr == self.positive_var - self.negative_var)
        # link vars with sign
        self._plus_var = self._new_generated_binary_var(name="_abs_is_p_%d" % abs_index)
        self._minus_var = self._new_generated_binary_var(name="_abs_is_n_%d" % abs_index)
        self._post_generated_ct(self._plus_var + self._minus_var == 1)
        self._new_generated_indicator(self._plus_var, self.negative_var <= 0, active_value=1)
        self._new_generated_indicator(self._minus_var, self.positive_var <= 0, active_value=1)

    def eval(self, numarg):
        return abs(numarg)

    def _get_solution_value(self):
        if self._is_resolved():
            raw = self._f_var.solution_value
        else:
            raw = abs(self._argument_expr._get_solution_value())
        return self._round_if_discrete(raw)

    def __repr__(self):
        return "docplex.mp.AbsExpr({0:s})".format(self._argument_expr.truncated_str())


# noinspection PyAbstractClass
class _SequenceExpr(_IAdvancedExpr):
    # INTERNAL: base class for functional exprs with a sequence argument (e.g. min/max)

    def __init__(self, model, exprs, name=None):
        _IAdvancedExpr.__init__(self, model, name)
        if is_iterable(exprs) or is_iterator(exprs):
            self._exprs = [model._to_linear_expr(e) for e in exprs]
        else:
            self._exprs = [model._to_linear_expr(exprs)]
        # allocate xvars iff necessary
        self._xvars = [self._allocate_arg_var_if_necessary(e) for e in self._exprs]

    def is_discrete(self):
        return all(map(lambda ex: ex.is_discrete(), self._exprs))

    def _get_args_string(self, sep=","):
        return sep.join(e.truncated_str() for e in self._exprs)

    def to_string(self):
        # generic: format expression arguments with holophraste
        str_args = self._get_args_string()
        return "{0}({1!s})".format(self.function_symbol, str_args)

    def iter_exprs(self):
        return iter(self._exprs)

    def _generate_variables(self):
        # INTERNAL: variable generator scanning all expressions
        # may return the same variable twice (or more)
        # use varset() if you need the set.
        for e in self._exprs:
            for v in e.iter_variables():
                yield v

    def iter_variables(self):
        return self._generate_variables()

    def _get_solution_value(self):
        if self._is_resolved():
            raw = self._f_var.solution_value
        else:
            raw = self.compute_solution_value()
        return self._round_if_discrete(raw_value=raw)

    def compute_solution_value(self):
        raise NotImplementedError


class MinimumExpr(_SequenceExpr):
    """ An expression that represents the minimum of a sequence of expressions.

    This expression can be used in all arithmetic operations.
    After a solve, the value of this expression is equal to the minimum of the values
    of its argument expressions.
    """

    def copy(self, target_model, var_mapping):
        pass

    def __init__(self, model, exprs, name=None):
        _SequenceExpr.__init__(self, model, exprs, name)

    def clone(self):
        return MinimumExpr(self.model, self._exprs, self.name)

    def _get_function_symbol(self):
        return "min"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.MinExpr({0!s})".format(str_args)

    def _resolve(self):
        self_min_var = self.functional_var
        self_x_vars = self._xvars
        nb_args = len(self_x_vars)
        if 0 == nb_args:
            self._f_var.set_bounds(0, 0)
        elif 1 == nb_args:
            self._post_generated_ct(self_min_var == self._xvars[0])
        else:
            for xv in self_x_vars:
                self._post_generated_ct(self_min_var <= xv)
            # allocate N _generated_ binaries
            z_vars = self._new_generated_binary_varlist(keys=nb_args)
            # sos?
            self._post_generated_ct(self.model.sum(z_vars) == 1)
            # indicators
            for i in range(nb_args):
                z = z_vars[i]
                x = self_x_vars[i]
                self._new_generated_indicator(binary_var=z, linear_ct=(self_min_var >= x))

    def compute_solution_value(self):
        return min(expr._get_solution_value() for expr in self._exprs)


class MaximumExpr(_SequenceExpr):
    """ An expression that represents the maximum of a sequence of expressions.

    This expression can be used in all arithmetic operations.
    After a solve, the value of this expression is equal to the minimum of the values
    of its argument expressions.
    """

    def __init__(self, model, exprs, name=None):
        _SequenceExpr.__init__(self, model, exprs, name)

    def clone(self):
        return MaximumExpr(self.model, self._exprs, self.name)

    def _get_function_symbol(self):
        return "max"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.MaxExpr({0!s})".format(str_args)

    def _resolve(self):
        self_max_var = self._f_var
        self_x_vars = self._xvars
        nb_args = len(self_x_vars)
        if 0 == nb_args:
            self._f_var.set_bounds(0, 0)  # what else ??
        elif 1 == nb_args:
            self._post_generated_ct(self_max_var == self._xvars[0])
        else:
            for xv in self_x_vars:
                self._post_generated_ct(self_max_var >= xv)
            # allocate N binaries
            z_vars = self._new_generated_binary_varlist(keys=nb_args)
            # sos?
            self._post_generated_ct(self.model.sum(z_vars) == 1)
            # indicators
            for i in range(nb_args):
                z = z_vars[i]
                x = self_x_vars[i]
                self._new_generated_indicator(binary_var=z, linear_ct=(self_max_var <= x))

    def compute_solution_value(self):
        return max(expr._get_solution_value() for expr in self._exprs)
