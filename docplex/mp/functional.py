# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

from docplex.mp.basic import Expr
from docplex.mp.constants import SOSType
from docplex.mp.operand import LinearOperand

from docplex.mp.utils import is_iterable, is_iterator

# do NOT import Model -> circular

# change this flag to generate named objects
# by default all generated objects will have no name
use_debug_names = False


def get_name_if_debug(name):
    return name if use_debug_names else None


# noinspection PyAbstractClass
class _IAdvancedExpr(Expr, LinearOperand):
    # INTERNAL class
    # parent class for all nonlinear expressions.
    __slots__ = ('_f_var', '_resolved')

    def __init__(self, model, name=None):
        Expr.__init__(self, model, name)
        self._f_var = None
        self._resolved = False

    def to_linear_expr(self):
        return self._get_resolved_f_var()

    def iter_terms(self):
        yield self._get_resolved_f_var(), 1

    iter_sorted_terms = iter_terms

    def iter_variables(self):
        # do we need to create it here?
        yield self._get_resolved_f_var()

    def unchecked_get_coef(self, dvar):
        return 1 if dvar is self._f_var else 0

    def _new_generated_free_continuous_var(self, name=None):
        # INTERNAL
        inf = self._model.infinity
        return self._new_generated_continuous_var(lb=-inf, ub=inf, name=name)

    def _new_generated_continuous_var(self, lb=None, ub=None, name=None):
        # INTERNAL
        m = self._model
        var = m._lfactory.new_var(m.continuous_vartype, lb=lb, ub=ub, varname=name)
        var.notify_origin(self)
        return var

    def _new_generated_binary_var(self, name=None):
        m = self._model
        bvar = m._lfactory.new_var(m.binary_vartype, varname=name)
        bvar.notify_origin(self)
        return bvar

    def _new_generated_binary_varlist(self, keys, name=None):
        bvars = self.model.binary_var_list(keys, name)
        for bv in bvars:
            bv.notify_origin(self)
        return bvars

    def new_generated_sos1(self, dvars):
        sos1 = self.model._add_sos(dvars, SOSType.SOS1)
        sos1.notify_origin(self)
        return sos1

    def _new_generated_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        ind = self._model._lfactory.new_indicator_constraint(binary_var, linear_ct, active_value, name)
        ind.notify_origin(self)
        self._model.add(ind)
        return ind

    def _new_generated_binary_ct(self, lhs, rhs, sense='EQ'):
        # posts a constraint and marks it as generated.
        m = self._model
        ct = m._lfactory.new_binary_constraint(lhs=lhs, sense=sense, rhs=rhs)
        m._post_constraint(ct)
        ct.notify_origin(self)
        return ct

    def _post_generated_cts(self, cts):
        # takes a sequence of constraints
        # posts a constraint and marks it as generated.
        self._model._lfactory._post_constraint_block(cts)
        for c in cts:
            c.notify_origin(self)
        return cts

    def _get_resolved_f_var(self):
        self._ensure_resolved()
        return self._f_var

    def _get_allocated_f_var(self):
        if self._f_var is None:
            self._f_var = self._create_functional_var()
        return self._f_var

    def resolve(self):
        self._ensure_resolved()

    def _ensure_resolved(self):
        if self._f_var is None:
            # 1. create the var (once!)
            self._f_var = self._create_functional_var()
            # 2. post the link between the fvar and the argument expr
        if not self._resolved:
            self._resolve()
            self._resolved = True

    def _is_resolved(self):
        return self._resolved and self._f_var is not None

    def _create_functional_var(self, named=True):
        if named:
            # add a unique counter suffix cf. RTC
            unique_counter = self._model._new_unique_counter()
            if unique_counter == 0:
                # the first time, we use the the raw name, else we suffix #<unique>
                fname = self.to_string()
            else:
                fname = "%s#%d" % (self.to_string(), unique_counter)
        else:
            fname = None
        return self._new_generated_free_continuous_var(name=fname)

    @property
    def functional_var(self):
        return self._get_resolved_f_var()

    as_var = functional_var

    def square(self):
        return self.functional_var.square()

    def _resolve(self):
        raise NotImplementedError  # pragma: no cover

    def _get_function_symbol(self):
        # redefine this to get the function symbol
        raise NotImplementedError  # pragma: no cover

    @property
    def function_symbol(self):
        return self._get_function_symbol()

    def __str__(self):
        return self.to_string()

    def to_string(self, **kwargs):
        raise NotImplementedError  # pragma: no cover

    # -- arithmetic operators
    def __mul__(self, e):
        return self.functional_var.__mul__(e)

    def __rmul__(self, e):
        return self.functional_var.__mul__(e)

    def __div__(self, e):
        return self.divide(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.divide(e)  # pragma: no cover

    def divide(self, e):
        return self.functional_var.divide(e)

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

    def _allocate_arg_var_if_necessary(self, arg_expr):
        # INTERNAL
        # allocates a new variables if only the argument expr is not a variable
        # and returns it
        try:

            arg_var = arg_expr.as_variable()
        except AttributeError:
            arg_var = None

        if arg_var is None:
            arg_var = self._new_generated_free_continuous_var()
            self._new_generated_binary_ct(arg_var, arg_expr)
        return arg_var


# noinspection PyAbstractClass
class FunctionalExpr(_IAdvancedExpr):
    def __init__(self, model, argument_expr, name=None):
        _IAdvancedExpr.__init__(self, model, name)
        self._argument_expr = model._lfactory._to_linear_operand(argument_expr)
        self._x_var = self._allocate_arg_var_if_necessary(argument_expr)

    @property
    def argument_expr(self):
        return self._argument_expr

    def eval(self, numarg):
        raise NotImplementedError  # pragma: no cover

    def is_discrete(self):
        return self._argument_expr.is_discrete()

    def to_string(self):
        return "{0:s}({1!s})".format(self.function_symbol, self._argument_expr)


class AbsExpr(FunctionalExpr):
    def copy(self, target_model, var_mapping):
        copied_arg_expr = self._argument_expr.copy(target_model, var_mapping)
        return AbsExpr(model=target_model, argument_expr=copied_arg_expr)

    def __init__(self, model, argument_expr):
        FunctionalExpr.__init__(self, model, argument_expr)
        #self._ensure_resolved_f_var()

    def _get_function_symbol(self):
        return "abs"

    def clone(self):
        return AbsExpr(self.model, self._argument_expr)

    # noinspection PyArgumentEqualDefault,PyArgumentEqualDefault
    def _resolve(self):
        self_f_var = self._f_var
        assert self_f_var
        m = self.model
        abs_index = m._new_unique_counter()
        abs_names = ["_abs_pp_%d" % abs_index, "_abs_np_%d" % abs_index] if use_debug_names else [None, None]
        # 1. allocate two variables in one pass.
        positive_var = self._new_generated_continuous_var(lb=0, name=abs_names[0])
        negative_var = self._new_generated_continuous_var(lb=0, name=abs_names[1])

        # F(x) = p + n
        ct1 = (self_f_var == positive_var + negative_var)
        # sos
        self.sos = self.new_generated_sos1(dvars=[positive_var, negative_var])
        # # x = p-n
        ct2 = (self._argument_expr == positive_var - negative_var)

        self._post_generated_cts([ct1, ct2])
        # store
        self.positive_var = positive_var
        self.negative_var = negative_var

    def eval(self, numarg):
        return abs(numarg)

    def _get_solution_value(self, s=None):
        raw = abs(self._argument_expr._get_solution_value(s))
        return self._round_if_discrete(raw)

    def __repr__(self):
        return "docplex.mp.AbsExpr({0:s})".format(self._argument_expr.truncated_str())


# noinspection PyAbstractClass
class _SequenceExpr(_IAdvancedExpr):
    # INTERNAL: base class for functional exprs with a sequence argument (e.g. min/max)

    def __init__(self, model, exprs, name=None):
        _IAdvancedExpr.__init__(self, model, name)
        if is_iterable(exprs) or is_iterator(exprs):
            self._exprs = exprs
        else:
            self._exprs = [model._lfactory._to_linear_operand(exprs)]
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
        yield self._get_resolved_f_var()

    def iter_variables(self):
        return self._generate_variables()

    def contains_var(self, dvar):
        return dvar is self._f_var

    def _get_solution_value(self, s=None):
        fvar = self._f_var
        if self._is_resolved() and (not s or fvar in s):
            raw = fvar._get_solution_value(s)
        else:
            raw = self.compute_solution_value(s)
        return self._round_if_discrete(raw_value=raw)

    def compute_solution_value(self, s):
        raise NotImplementedError  # pragma: no cover

    def copy(self, target_model, var_mapping):
        copied_exprs = [expr.copy(target_model, var_mapping) for expr in self._exprs]
        return self.__class__(target_model, copied_exprs, self.name)

    def clone(self):
        # generic clone
        return self.__class__(self.model, self._exprs, self.name)


class MinimumExpr(_SequenceExpr):
    """ An expression that represents the minimum of a sequence of expressions.

    This expression can be used in all arithmetic operations.
    After a solve, the value of this expression is equal to the minimum of the values
    of its argument expressions.
    """

    def __init__(self, model, exprs, name=None):
        _SequenceExpr.__init__(self, model, exprs, name)
        #self._ensure_resolved_f_var()

    def _get_function_symbol(self):
        return "min"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.MinExpr({0!s})".format(str_args)

    def _resolve(self):
        self_min_var = self._f_var
        assert self_min_var
        self_x_vars = self._xvars
        nb_args = len(self_x_vars)
        if 0 == nb_args:
            self._f_var.set_bounds(0, 0)
        elif 1 == nb_args:
            self._new_generated_binary_ct(self_min_var, self._xvars[0])
        else:
            cts = []
            for xv in self_x_vars:
                cts.append(self_min_var <= xv)
            # allocate N _generated_ binaries
            z_vars = self._new_generated_binary_varlist(keys=nb_args)
            # sos?
            cts.append(self.model.sum(z_vars) == 1)
            self._post_generated_cts(cts)
            # indicators
            for i in range(nb_args):
                z = z_vars[i]
                x = self_x_vars[i]
                # need a block generation of indicators
                self._new_generated_indicator(binary_var=z, linear_ct=(self_min_var >= x))

    def compute_solution_value(self, s):
        return min(expr._get_solution_value(s) for expr in self._exprs)


class MaximumExpr(_SequenceExpr):
    """ An expression that represents the maximum of a sequence of expressions.

    This expression can be used in all arithmetic operations.
    After a solve, the value of this expression is equal to the minimum of the values
    of its argument expressions.
    """

    def __init__(self, model, exprs, name=None):
        _SequenceExpr.__init__(self, model, exprs, name)
        #self._ensure_resolved_f_var()

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
            self._new_generated_binary_ct(self_max_var, self._xvars[0])
        else:
            for xv in self_x_vars:
                self._new_generated_binary_ct(self_max_var, xv, 'GE')
            # allocate N binaries
            z_vars = self._new_generated_binary_varlist(keys=nb_args)
            # sos?
            self._new_generated_binary_ct(self.model.sum(z_vars), 1)
            # indicators
            for i in range(nb_args):
                z = z_vars[i]
                x = self_x_vars[i]
                self._new_generated_indicator(binary_var=z, linear_ct=(self_max_var <= x))

    def compute_solution_value(self, s):
        return max(expr._get_solution_value(s) for expr in self._exprs)


class _LogicalSequenceExpr(_SequenceExpr):

    def _create_functional_var(self):
        # the resulting variable is a binary variable...
        unique_counter = self._model._new_unique_counter()
        if unique_counter == 0:
            # the first time, we use the the raw name, else we suffix #<unique>
            fname = self.to_string()
        else:
            fname = "%s#%d" % (self.to_string(), unique_counter)
        return self._new_generated_binary_var(name=fname)

    def __init__(self, model, exprs, name=None):
        _IAdvancedExpr.__init__(self, model, name)
        assert is_iterable(exprs) or is_iterator(exprs)
        self._exprs = exprs
        # never allocate vars: arguments --are-- binary variables.
        self._xvars = exprs
        #self._ensure_resolved_f_var()

    def is_discrete(self):
        return True

    precision = 1e-5


class LogicalAndExpr(_LogicalSequenceExpr):

    def _get_function_symbol(self):
        return "and"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.LogicalAndExpr({0!s})".format(str_args)

    def compute_solution_value(self, s):
        # return 1/0 not True/False
        threshold = 1 - self.precision
        return 1 if all(ex._get_solution_value(s) >= threshold for ex in self._exprs) else 0

    def _resolve(self):
        self_and_var = self._f_var
        self_x_vars = self._xvars

        if self_x_vars:
            cts = [(self_and_var <= xv) for xv in self_x_vars]
            m = self._model
            cts.append(len(self_x_vars) * self_and_var >= m._aggregator._sum_with_seq(self._xvars))
            self._post_generated_cts(cts)


class LogicalOrExpr(_LogicalSequenceExpr):

    def _get_function_symbol(self):
        return "or"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.LogicalOrExpr({0!s})".format(str_args)

    def compute_solution_value(self, s):
        # return 1/0 not True/False
        threshold = 1 - self.precision
        return 1 if any(ex._get_solution_value(s) >= threshold for ex in self._exprs) else 0

    def _resolve(self):
        self_or_var = self._f_var
        self_x_vars = self._xvars

        if self_x_vars:
            cts = [(xv <= self_or_var) for xv in self_x_vars]
            m = self._model
            cts.append(self_or_var <= m._aggregator._sum_with_seq(self._xvars))
            self._post_generated_cts(cts)
        self._resolved = True


class PwlExpr(FunctionalExpr):

    def __init__(self, model, pwl_func, argument_expr, usage_counter, y_var=None,  add_counter_suffix=True, resolve=True):
        FunctionalExpr.__init__(self, model, argument_expr)
        self._pwl_func = pwl_func
        self._usage_counter = usage_counter
        self._f_var = y_var
        if pwl_func.name:
            # ?
            if add_counter_suffix:
                self.name = '{0}_{1!s}'.format(self._pwl_func.name, self._usage_counter)
            else:
                self.name = self._pwl_func.name
        if resolve:
            self._ensure_resolved()

    def _get_function_symbol(self):
        pwl_name = self._pwl_func.get_name()
        return pwl_name or '_pwl{0}'.format(self._usage_counter)

    def _get_solution_value(self, s=None):
        raw = self._f_var._get_solution_value(s)
        return self._round_if_discrete(raw)

    def iter_variables(self):
        for v in self._argument_expr.iter_variables():
            yield v
        yield self._get_resolved_f_var()

    def _resolve(self):
        mdl = self._model
        pwl_constraint = mdl._lfactory.new_pwl_constraint(self, self.get_name())
        mdl._add_pwl_constraint_internal(pwl_constraint)


    @property
    def pwl_func(self):
        return self._pwl_func

    @property
    def usage_counter(self):
        return self._usage_counter

    def __repr__(self):
        return "docplex.mp.PwlExpr({0:s}, {1:s})".format(self._get_function_symbol(),
                                                         self._argument_expr.truncated_str())

    def copy(self, target_model, var_map):
        copied_pwl_func = var_map[self.pwl_func]
        copied_x_var = var_map[self._x_var]
        copied_pwl_expr = PwlExpr(target_model, copied_pwl_func, copied_x_var, self.usage_counter)
        copied_pwl_expr_f_var = var_map.get(self._f_var)
        if copied_pwl_expr_f_var:
            copied_pwl_expr._f_var = copied_pwl_expr_f_var
            # Need to set the _origin attribute of the copied var
            copied_pwl_expr_f_var._origin = copied_pwl_expr
        return copied_pwl_expr
