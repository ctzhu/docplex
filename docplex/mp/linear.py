# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint: disable=too-many-lines
from __future__ import print_function

import operator
from collections import OrderedDict

from enum import Enum

from docplex.mp.basic import ModelingObjectBase
from docplex.mp.basic import ModelingObject, Expr
from docplex.mp.utils import *


class VarType(object):
    """VarType()

    This class defines the various types of decision variables.

        This class should never be instantiated. Three instances of
        variable types are available for Binary, Integer and Continuous types.

    """

    def __init__(self, short_name, lb, ub):
        self._short_name = short_name
        self._lb = lb
        self._ub = ub

    @property
    def short_name(self):
        """
        This property returns a short name string for the type.
        """
        return self._short_name

    @property
    def default_lb(self):
        """  This property returns the default lower bound for the type.
        """
        return self._lb

    @property
    def default_ub(self):
        """  This property returns the default upper bound for the type.
        """
        return self._ub

    def compute_lb(self, candidate_lb):
        # INTERNAL
        raise NotImplementedError  # pragma: no cover

    def compute_ub(self, candidate_ub):
        # INTERNAL
        raise NotImplementedError  # pragma: no cover

    def is_discrete(self):
        """
        Returns:
            True if the type is a discrete type.
        """
        raise NotImplementedError  # pragma: no cover

    def accept_value(self, numeric_value):
        """ Returns True if the `numeric_value` is valid for the type.

        Accepted values depend on the type:

        - Binary type accepts only 0 or 1.

        - Integer type accepts only integers.

        - Continuous type accepts any floating-point number within -inf and +inf, where inf
          is the model's infinity value.

        This method never raises any exception.

        :param numeric_value: The candidate value.

        :return: True if the candidate value is valid, else False.

        """
        raise NotImplementedError  # pragma: no cover

    # compat
    def is_continuous(self):
        # INTERNAL
        return not self.is_discrete()

    def to_string(self):
        """
        Returns:
            A string representation of the type.
        """
        return "VarType_%s" % self.short_name

    def __str__(self):
        return self.to_string()


class BinaryVarType(VarType):
    """ Binary variable type.
    """

    def __init__(self):
        """ Constructor
        """
        VarType.__init__(self, short_name="binary", lb=0, ub=1)

    def compute_lb(self, candidate_lb):
        return 0.0

    def compute_ub(self, candidate_ub):
        return 1.0

    def is_discrete(self):
        return True

    def accept_value(self, numeric_value):
        return numeric_value == 0 or numeric_value == 1


class ContinuousVarType(VarType):
    """ Continuous variable type.
    """

    def __init__(self, plus_infinity=1e+20):
        """ Constructor
        """
        VarType.__init__(self, short_name="float", lb=0, ub=plus_infinity)
        self._plus_infinity = plus_infinity
        self._minus_infinity = - plus_infinity

    def compute_ub(self, candidate_ub):
        return self._plus_infinity if candidate_ub > self._plus_infinity else float(candidate_ub)

    def compute_lb(self, candidate_lb):
        return self._minus_infinity if candidate_lb < self._minus_infinity else float(candidate_lb)

    def is_discrete(self):
        return False

    def accept_value(self, numeric_value):
        return self._minus_infinity <= numeric_value <= self._plus_infinity


class IntegerVarType(VarType):
    """ Integer variable type.
    """

    def __init__(self, plus_infinity=1e+20):
        """ Constructor
        """
        VarType.__init__(self, short_name="int", lb=0, ub=plus_infinity)
        self._plus_infinity = plus_infinity

    def compute_ub(self, candidate_ub):
        iub = min(candidate_ub, self._plus_infinity)
        if int(iub) != iub:
            return float(math.floor(iub))
        else:
            return float(iub)

    def compute_lb(self, candidate_lb):
        ilb = max(candidate_lb, -self._plus_infinity)
        if int(ilb) != ilb:
            return float(math.ceil(ilb))
        else:
            return float(ilb)

    def is_discrete(self):
        return True

    def accept_value(self, numeric_value):
        return is_int(numeric_value) or numeric_value == int(numeric_value)


class Var(ModelingObject):
    """Var()

    This class models decision variables.
    Decision variables are instantiated by :class:`docplex.mp.model.Model` methods such as :func:`docplex.mp.model.Model.var`.

    """
    def __init__(self, model, vartype, name,
                 lb=None, ub=None,
                 _safe_domain=False,
                 is_automatic_name=False,
                 container=None):
        ModelingObject.__init__(self, model, name, is_automatic_name=is_automatic_name,
                                container=container)
        self._vartype = vartype
        self.__id = None  # cache the id() for perf

        if _safe_domain:
            # this is called by the var_list
            self._lb = lb
            self._ub = ub
        elif ub is None and lb is None:
            self._lb = vartype.default_lb
            self._ub = vartype.default_ub
        else:
            used_lb = vartype.default_lb if lb is None else vartype.compute_lb(lb)
            used_ub = vartype.default_ub if ub is None else vartype.compute_ub(ub)
            if used_lb <= used_ub:
                self._lb = used_lb
                self._ub = used_ub
            else:
                model.fatal("Variable: {0} has empty domain: [{1}..{2}]", name, lb, ub)

    # noinspection PyUnusedLocal
    def copy(self, new_model, var_mapping):
        return var_mapping[self]

    def equals(self, other):
        if type(other) != Var:
            return False
        # --
        if type(self.vartype) != other.vartype:
            return False
        if self.name != other.name:
            return False
        if self.lb != other.lb:
            return False
        if self.ub != other.ub:
            return False
        return True

    def typecheck_initial_value(self, numeric_value):
        if not self.vartype.accept_value(numeric_value):
            return False
        return self.lb <= numeric_value <= self.ub

    def check_name(self, new_name):
        ModelingObject.check_name(self, new_name)
        if not is_string(new_name) or not new_name:
            self.fatal("Variable name accepts only non-empty strings, got: {0!s}", new_name)
        elif new_name.find(' ') >= 0:
            self.warning("Variable name contains blank space, var: {0!s}, name: {1!s}", self, new_name)

    def __hash__(self):
        # INTERNAL Necessary for python 3
        self_hash = self.__id
        if self_hash is None:
            # beware: using anything else than id() here may generate calls to __eq__
            # which will generate a constraint, not a boolean!
            self_hash = id(self)
            self.__id = self_hash
        return self_hash

    def to_linear_expr(self):
        # INTERNAL
        return self._get_model().linear_expr(self)

    def set_name(self, new_name):
        # INTERNAL
        self.model.set_var_name(self, new_name)

    def get_lb(self):
        """ This property is used to get or set the lower bound of the variable.

        Possible values for the lower bound depend on the variable type. Binary variables
        accept only 0 or 1 as bounds. An integer variable will convert the lower bound value to the
        ceiling integer value of the argument.
        """
        return self._lb

    def _set_lb(self, lb):
        self.model.set_var_lb(self, lb)

    def set_lb(self, lb):
        self._set_lb(lb)
        return self._lb

    lb = property(get_lb, _set_lb)

    def _internal_set_lb(self, lb):
        # Internal, used only by the model
        self._lb = lb

    def _internal_set_ub(self, ub):
        # INTERNAL
        self._ub = ub

    def get_ub(self):
        """ This property is used to get or set the upper bound of the variable.

        Possible values for the upper bound depend on the variable type. Binary variables
        accept only 0 or 1 as bounds. An integer variable will convert the upper bound value to the
        floor integer value.

        To reset the upper bound to its default infinity value, use model.infinity.
        """
        return self._ub

    def _set_ub(self, ub):
        self.model.set_var_ub(self, ub)

    def set_ub(self, ub):
        self._set_ub(ub)
        return self._ub

    ub = property(get_ub, _set_ub)

    def has_free_lb(self):
        return self.model.is_free_lb(self._lb)

    def has_free_ub(self):
        return self.model.is_free_ub(self._ub)

    def is_free(self):
        return self.has_free_lb() and self.has_free_ub()

    @property
    def vartype(self):
        """ This property returns the variable type, an instance of :class:`VarType`.

        """
        return self._vartype

    def has_type(self, vartype):
        # internal
        return type(self._vartype) == vartype

    def is_binary(self):
        """
        returns:
            True if the variable has Binary type.
        """
        return self.has_type(BinaryVarType)

    def is_integer(self):
        """
        Returns:
            True if the variable has Integer type.
        """
        return self.has_type(IntegerVarType)

    def is_continuous(self):
        """
        Returns:
            True if the variable has Continuous type.
        """
        return self.has_type(ContinuousVarType)

    def is_discrete(self):
        """
        Returns:
            True if the variable has type Binary or Integer.
        """
        return self.is_binary() or self.is_integer()

    @property
    def float_precision(self):
        return 0 if self.is_discrete() else self._model.float_precision

    def get_value(self):
        # for compatibility only: use solution_value instead
        print("* get_value() is deprecated, use property solution_value instead")  # pragma: no cover
        return self.solution_value  # pragma: no cover

    @property
    def solution_value(self):
        """ This property returns the solution value of the variable.

        Raises:
            DOCplexException
                if the model has not been solved succesfully.

        """
        self._check_model_has_solution()
        return self._get_solution_value()

    @property
    def unchecked_solution_value(self):
        # INTERNAL
        return self._get_solution_value()

    def _get_solution_value(self):
        return self._get_model()._get_solution().get_value(self)

    def __le__(self, e):
        expr = self.to_linear_expr()
        return expr.le_constraint(e)

    def __eq__(self, e):
        expr = self.to_linear_expr()
        return expr.eq_constraint(e)

    def __ge__(self, e):
        return self.to_linear_expr().ge_constraint(e)

    def __ne__(self, other):
        # INTERNAL: For now, not supported
        self.model.unsupported_neq_error(self, other)


    def __gt__(self, e):
        self.model.unsupported_operator_error(self, ">", e)

    def __lt__(self, e):
        self.model.unsupported_operator_error(self, "<", e)

    def __mul__(self, e):
        return self.times(e)

    def times(self, e):
        if is_number(e):
            if 0 == e:
                return self._model._get_zero_expr()
            else:
                return MonomialExpr(self._model, self, e)
        elif isinstance(e, _ZeroExpr):
            return self._model._get_zero_expr()
        elif AbstractLinearExpr.is_var_operand(e):
            raise DOCplexQuadraticNotImplementedError(self, e)
        else:
            return self.to_linear_expr()._multiply(e)

    def __rmul__(self, e):
        return self.times(e)

    def __add__(self, e):
        return self.plus(e)

    def plus(self, e):
        return self.to_linear_expr()._add(e)

    def __radd__(self, e):
        return self.plus(e)

    def __sub__(self, e):
        return self.minus(e)

    def minus(self, e):
        return self.to_linear_expr()._subtract(e)

    def __rsub__(self, e):
        # e - self
        expr = self.model._to_linear_expr(e)  # makes a clone.
        return expr._subtract(self)

    def divide(self, e):
        expr = self.to_linear_expr()
        return expr._divide(e)

    def __div__(self, e):
        return self.divide(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.divide(e)  # pragma: no cover

    def __rtruediv__(self, e):
        # for py3
        self.fatal("Variable {0!s} cannot be used as denominator of {1!s}", self, e)  # pragma: no cover

    def __rdiv__(self, e):
        self.fatal("Variable {0!s} cannot be used as denominator of {1!s}", self, e)

    def __pos__(self):
        # the "+e" unary plus is syntactic sugar
        return self

    def __neg__(self):
        # the "-e" unary minus returns a linear expression
        return self.model._monomial_expr(self, -1)

    def __int__(self):
        """ Converts a decision variable to a integer number.

        This is only possible for discrete variables,
        and when the model has been solved successfully.
        If the model has been solved, returns the variable's solution value.

        Raises:
            DOCplexException
                if the model has not been solved successfully.
            DOCplexException
                if the variable is not discrete.
        """

        if self.is_continuous():
            self.fatal("Cannot convert continuous variable value to int: {0!s}", self)
        return int(self.solution_value)

    def __float__(self):
        """ Converts a decision variable to a floating-point number.

        This is only possible when the model has been solved successfully.
        Otherwise an exception is raised.
        If the model has been solved, returns the variable's solution value.

        Raises:
            DOCplexException
                if the model has not been solved successfully.
        """
        return float(self.solution_value)

    def to_bool(self):
        """to_bool()

        Converts a variable value to True or False.

        This is only possible for discrete variables and assumes there is a solution.

        Raises:
            DOCplexException 
                if the model has not been solved successfully.
            DOCplexException 
                if the variable is not discrete.

        Returns:
            True if the variable value is nonzero, else False.
        """
        if not self.is_discrete():
            self.fatal("boolean conversion only for discrete variables, type is {0!s}", self.vartype)
        value = self.solution_value  # this property checks for a solution.
        return value != 0

    # def __nonzero__(self):
    # return self.to_boolean(precision=1e-5)

    def __str__(self):
        """
        returns:
            A string representation of the variable.

        """
        return self.to_string()

    def to_string(self):
        print_name = self.name or 'x_%s[%d]' % (self.vartype.short_name, id(self))
        return print_name

    def to_stringio(self, oss):
        oss.write(self.to_string())

    def __repr__(self):
        if self.has_user_name():
            return 'docplex.mp.linear.Var<%s>(%s)' % (self.vartype.short_name, self.name)
        else:
            return 'docplex.mp.linear.Var<%s>' % self.vartype.short_name


# noinspection PyAbstractClass
class AbstractLinearExpr(Expr):
    __slots__ = ()

    def __init__(self, model):
        Expr.__init__(self, model=model)  # pragma: no cover

    def __getitem__(self, dvar):
        return self.unchecked_get_coef(dvar)

    @staticmethod
    def is_var_operand(e):
        return isinstance(e, Var) or isinstance(e, MonomialExpr) or isinstance(e, LinearExpr)

    def unchecked_get_coef(self, dvar):
        raise NotImplementedError  # pragma: no cover

    def get_coef(self, dvar):
        """ Returns the coefficient of a variable in the expression.

        Note:
            If the variable is not present in the expression, the function returns 0.

        :param dvar: The variable for which the coefficient is being queried.

        :return: A floating-point number.
        """
        self.model.typecheck_var(dvar)
        return self.unchecked_get_coef(dvar)

    def __iter__(self):
        # INTERNAL: this is necessary to prevent expr from being an iterable.
        # as it follows getitem protocol, it can mistakenly be interpreted as an iterable
        # but this would make sum loop forever.
        raise TypeError

    def is_constant(self):
        # redefine this for subclasses.
        return False  # pragma: no cover

    def is_discrete(self):
        raise NotImplementedError  # pragma: no cover

    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        raise NotImplementedError  # pragma: no cover

    def is_variable(self):
        # return True if the expression is infact one variable.
        # if True, assume you can replace the expression by the first variable
        # returned by iter_variables()
        return False


class _ZeroExpr(AbstractLinearExpr):

    def _get_solution_value(self):
        return 0

    # INTERNAL
    __slots__ = ()

    def __init__(self, model):
        ModelingObjectBase.__init__(self, model)

    def clone(self):
        return self  # this is not cloned.

    def copy(self, target_model, var_map):
        return _ZeroExpr(target_model)

    def to_linear_expr(self):
        return self  # this is a linear expr.

    def number_of_variables(self):
        return 0

    def iter_variables(self):
        return iter([])

    def is_constant(self):
        return True

    def is_discrete(self):
        return True

    def unchecked_get_coef(self, dvar):
        return 0

    def contains_var(self, dvar):
        return False

    @property
    def constant(self):
        # for compatibility
        return 0

    # noinspection PyMethodMayBeStatic
    def _get_coefs(self):
        return []

    def negate(self):
        pass

    # noinspection PyMethodMayBeStatic
    def plus(self, e):
        return e

    def times(self, _):
        return self

    # noinspection PyMethodMayBeStatic
    def minus(self, e):
        expr = e.to_linear_expr().clone()
        expr.negate()
        return expr

    def to_string(self, nb_digits=None, prod_symbol='', use_space=False):
        return "0"

    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        oss.write(self.to_string())

    # arithmetic
    def __sub__(self, other):
        return self._subtract(other)

    def _subtract(self, other):
        # return -other
        other_expr = self.model.linear_expr(other)
        other_expr.negate()
        return other_expr

    def __rsub__(self, e):
        # e - 0 = e !
        return e

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __div__(self, other):
        return self._divide(other)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.__div__(e)  # pragma: no cover

    def _divide(self, other):
        self.model.typecheck_as_denominator(numerator=self, denominator=other)
        return self

    def __repr__(self):
        return "docplex.mp.linear.ZeroExpr()"

    def equals_expr(self, other):
        return isinstance(other, _ZeroExpr)

    def __ge__(self, other):
        return self._get_model().ge_constraint(self, other)

    def __le__(self, other):
        return self._get_model().le_constraint(self, other)

    def __eq__(self, other):
        return self._get_model().eq_constraint(self, other)


class MonomialExpr(AbstractLinearExpr):
    def _get_solution_value(self):
        raw = self.coef * self._dvar.solution_value
        return self._round_if_discrete(raw)

    # INTERNAL class
    __slots__ = ("_dvar", "_coef")

    # noinspection PyMissingConstructor
    def __init__(self, model, dvar, coeff):
        self._model = model  # faster than to call recursively init methods...
        self._name = None
        self._dvar = dvar
        self._coef = coeff

    def number_of_variables(self):
        return 1

    @property
    def var(self):
        return self._dvar

    @property
    def coef(self):
        return self._coef

    @property
    def constant(self):
        # for compatibility
        return 0

    def is_variable(self):
        return 1 == self._coef

    def clone(self):
        return MonomialExpr(self.model, self._dvar, self._coef)

    def copy(self, target_model, var_mapping):
        copy_var = var_mapping[self._dvar]
        return MonomialExpr(target_model, dvar=copy_var, coeff=self._coef)

    def iter_variables(self):
        return iter([self._dvar])

    def _get_coefs(self):
        return [float(self._coef)]

    def var_set(self):
        return {self._dvar}

    def unchecked_get_coef(self, dvar):
        return self._coef if dvar is self._dvar else 0

    def contains_var(self, dvar):
        return self._dvar is dvar

    def is_normal_form(self):
        # INTERNAL
        return self._coef != 0

    def is_discrete(self):
        return self._dvar.is_discrete() and is_int(self._coef)

    # arithmetics
    def negate(self):
        self._coef = - self._coef
        return self

    def plus(self, e):
        if e is 0:
            return self
        else:
            return self.to_linear_expr()._add(e)

    def minus(self, e):
        expr = self.to_linear_expr()
        expr._subtract(e)
        return expr

    def times(self, e):
        if e is 1:
            return self
        elif is_number(e):
            # e might be a fancy numpy type
            if 1 == float(e):
                return self
            else:
                # return a fresh instance
                return self.model._monomial_expr(self._dvar, self._coef * e)
        elif isinstance(e, LinearExpr):
            return e.times(self)
        elif isinstance(e, Var):
            raise DOCplexQuadraticNotImplementedError(self, e)
        else:
            expr = self.to_linear_expr()
            return expr._multiply(e)

    def quotient(self, e):
        self.model.typecheck_as_denominator(e, self)
        inverse = 1.0 / float(e)
        return self.model._monomial_expr(self._dvar, self._coef * inverse)

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.__add__(e)

    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        return self.model._to_linear_expr(e).minus(self)

    def __neg__(self):
        opposite = self.clone()
        return opposite.negate()

    def __mul__(self, e):
        return self.times(e)

    def __rmul__(self, e):
        return self.times(e)

    def __div__(self, e):
        return self.quotient(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.__div__(e)  # pragma: no cover

    def __rtruediv__(self, e):
        # for py3
        self.model.cannot_be_used_as_denominator_error(self, e)  # pragma: no cover

    def __rdiv__(self, e):
        self.model.cannot_be_used_as_denominator_error(self, e)

    def __le__(self, other):
        return self.to_linear_expr().le_constraint(other)

    def __eq__(self, other):
        return self.to_linear_expr().eq_constraint(other)

    def __ge__(self, other):
        return self.to_linear_expr().ge_constraint(other)

    def __lt__(self, other):
        # unsupported operator
        self.model.unsupported_operator_error(self, "<", other)

    def __gt__(self, other):
        # unsupported operator
        self.model.unsupported_operator_error(self, ">", other)

    def equals_expr(self, other):
        if isinstance(other, MonomialExpr):
            return self.var is other.var and self.coef == other.coef
        elif isinstance(other, LinearExpr):
            expr = other
            if expr.constant == 0 and expr.number_of_variables() == 1:
                other_first_var = next(other.iter_variables())
                return self.var is other_first_var and self.coef == other[other_first_var]
        else:
            return False

    # conversion
    def to_linear_expr(self):
        self_as_dict = {self._dvar: self._coef}
        return self.model.linear_expr(e=self_as_dict)

    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        self_coef = self._coef
        if self_coef != 1:
            if self_coef < 0:
                oss.write('-')
                self_coef = - self_coef
            if self_coef != 1:
                self._num_to_stringio(oss, num=self_coef, ndigits=nb_digits)
            if use_space:
                oss.write(' ')
            if prod_symbol:
                oss.write(prod_symbol)
                if use_space:
                    oss.write(' ')
        oss.write(var_namer(self._dvar))

    def __repr__(self):
        return "docplex.mp.MonomialExpr(%s)" % self.to_string()

    # @property
    # def solution_value(self):
    #     raw = self.coef * self._dvar.solution_value
    #     return self._round_if_discrete(raw)


class LinearExpr(AbstractLinearExpr):
    """LinearExpr()

    This class models linear expressions.
    This class is not intended to be instantiated. Expressions are built
    either using operators or using Model.linear_expr().

    """
    _private_instance_counter = 0
    _private_clone_counter = 0

    # what type to use for merging dicts
    counter_type = ExprCounter

    # what type to use for storing terms
    term_dict_type = OrderedDict

    @staticmethod
    def number_of_instances():
        return LinearExpr._private_instance_counter  # pragma: no cover

    def to_linear_expr(self):
        return self

    def _get_terms_dict(self):
        # INTERNAL
        return self.__terms

    def _get_coefs(self):
        # INTERNAL: returns list of coefs in key order
        # return self.__terms.values()
        # PY3 dict views do not support indexing but Wrapper REQUIRES indexing
        # note that we must return floats for cplex.
        return [float(v) for v in itervalues(self.__terms)]

    def __typecheck_terms_dict(self, terms):
        """
        INTERNAL: check a given dictionary of terms for (var, float)
        :param terms:
        :return:
        """
        if not isinstance(terms, dict):
            self.fatal("expecting expression terms as python dict, got: {0!s}", terms)
        self_model = self.model
        for (v, k) in iteritems(terms):
            self_model.typecheck_var(v)
            self_model.typecheck_num(k, 'LinearExpr:importTerms')


    def _assign_terms(self, terms, is_safe=False, assume_normalized=False):
        if not is_safe:
            self.__typecheck_terms_dict(terms)
        if assume_normalized:
            self.__terms = terms
        else:
            # must put back to normal form
            self.__terms = self.term_dict_type([(k, v) for k, v in iteritems(terms) if v != 0])
        return self

    def _new_terms_dict(self, dict_type=term_dict_type, *args):
        # INTERNAL: builds a new terms dict.
        return dict_type(*args)

    __slots__ = ("_constant", "__terms", "private_instance_counter")

    def __init__(self, model, e=None, constant=0, name=None, safe=False):
        Expr.__init__(self, model, name)
        # a global counter for performance measurement
        LinearExpr._private_instance_counter += 1
        # "calling LinearExpr ctor, k=%d" % LinearExpr.InstanceCounter)
        if not safe and 0 != constant:
            model.typecheck_num(constant, 'LinearExpr()')
        self._constant = constant

        if isinstance(e, dict):
            if safe:
                self.__terms = e
            else:
                self_terms = self._new_terms_dict()
                for (v, k) in iteritems(e):
                    model.typecheck_var(v)
                    model.typecheck_num(k, 'LinearExpr')
                    if k != 0:
                        self_terms[v] = k
                self.__terms = self_terms
            return
        else:
            self.__terms = self._new_terms_dict()

        if isinstance(e, Var):
            self.__terms[e] = 1
        elif is_number(e):
            self._constant += e
        elif isinstance(e, MonomialExpr):
            self._add_term(e.var, e.coef)
        elif isinstance(e, tuple):
            if len(e) != 2:
                self.fatal("Only tuples of len 2 are accepted to build an expression, got: {0!s}", e)
            elif isinstance(e[0], Var):
                model.typecheck_num(e[1], 'LinearExpr')
                self._add_term(e[0], e[1])
            elif isinstance(e[1], Var):
                model.typecheck_num(e[0], 'LinearExpr')
                self._add_term(e[1], e[0])
            else:
                self.fatal("Not a (variable, value) tuple: {0!s}", e)
        elif isinstance(e, LinearExpr):
            self._constant = e.constant
            self.__terms = e._get_terms_dict()

        elif is_iterable(e) and not is_string(e):
            # assume only variables in iteration?:
            for o in e:
                if isinstance(o, Var):
                    self.__terms[o] = 1
                else:
                    self.fatal('only variable sequences are accepted to build an expression, got: {0!s}', e)
        else:
            self.fatal("Cannot convert {1!s} to docplex.mp.LinearExpr, instance: {0!s}", repr(e), type(e).__name__)

    def clone(self):
        """
        Returns:
            A copy of the expression on the same model.
        """
        LinearExpr._private_clone_counter += 1
        cloned_terms = self.term_dict_type(self.__terms)  # faster than copy() on OrderedDict()
        cloned = LinearExpr(model=self.model, e=cloned_terms, constant=self._constant, safe=True)
        # cloned = self.model.linear_expr(self._constant)
        # cloned.__terms = cloned_terms
        return cloned

    def copy(self, target_model, var_mapping):
        # INTERNAL

        copied_terms = self._new_terms_dict()
        for v, k in self.iter_terms():
            copied_terms[var_mapping[v]] = k
        copied_expr = LinearExpr(model=target_model, e=copied_terms, constant=self.constant, safe=True)
        return copied_expr

    def negate(self):
        """ Takes the negation of an expression.

        Changes the expression by replacing each variable coefficient and the constant term
        by its opposite.
        """
        self._constant = - self._constant
        self_terms = self.__terms
        for (v, k) in iteritems(self.__terms):
            self_terms[v] = -k

    def _clear(self):
        """ Clears the expression.

        All variables and coefficients are removed and the constant term is set to zero.
        """
        self._constant = 0
        self.__terms = self._new_terms_dict()

    def equals_constant(self, scalar):
        """
        Args:
            scalar (float): A floating-point number.
        Returns:
            True if the expression equals this constant term.
        """
        return self.is_constant() and (scalar == self._constant)

    def is_zero(self):
        return self.equals_constant(0)

    def is_one(self):
        return self.equals_constant(1)

    def is_constant(self):
        """
        Checks if the expression is a constant.

        Returns:
            True if the expression consists of only a constant term.
        """
        return not self.__terms

    def is_variable(self):
        # INTERNAL: returns True if expression is in fact a variable (1*x)
        return 0 == self.constant \
               and 1 == self.number_of_variables() \
               and 1 == next(itervalues(self.__terms))

    def is_normal_form(self):
        # INTERNAL
        for k in self.__terms.values():
            if 0 == k:
                return False
        return True

    def number_of_variables(self):
        return len(self.__terms)

    def unchecked_get_coef(self, dvar):
        # INTERNAL
        return self.__terms.get(dvar, 0)

    def _set_coef(self, dvar, acoef):
        # INTERNAL
        if 0 != acoef:
            self.__terms[dvar] = acoef
        else:
            self._remove_term(dvar)

    def add_term(self, dvar, coeff):
        """
        Adds a term (variable and coefficient) to the expression.

        Args:
            dvar (:class:`Var`): A decision variable.
            coeff (float): A floating-point number.

        Returns:
            The modified expression itself.
        """
        self.model.typecheck_var(dvar)
        self._add_term(dvar, coeff)
        return self

    def _add_term(self, dvar, coef=1):
        # INTERNAL
        if coef != 0:
            self_terms = self.__terms
            if dvar not in self_terms:
                self_terms[dvar] = coef
            else:
                new_coef = coef + self_terms[dvar]
                if new_coef:
                    self_terms[dvar] = new_coef
                else:
                    del self_terms[dvar]

    def _add_var(self, dvar):
        old_coeff = self.__terms.get(dvar, 0)
        new_coef = 1 + old_coeff
        if new_coef:  # nonzero is True
            self.__terms[dvar] = new_coef
        else:
            del self.__terms[dvar]

    def remove_term(self, dvar):
        """ Removes a term associated with a variable from the expression.

        Args:
            dvar (:class:`Var`): A decision variable.

        Returns:
            The modified expression.

        """
        self.model.typecheck_var(dvar)
        self._remove_term(dvar)
        return self

    def _remove_term(self, dvar):
        # INTERNAL
        del self.__terms[dvar]

    def _get_constant(self):
        """
        This property is used to get or set the constant term of the expression.
        """
        return self._constant

    def _set_constant(self, numval):
        self._constant = numval

    constant = property(_get_constant, _set_constant)

    def iter_variables(self):
        """  Iterates over all variables mentioned in the linear expression.

        Returns:
            An iterator object.
        """
        return iter(self.__terms.keys())

    def var_set(self):
        # INTERNAL
        return set(self.__terms.keys())

    def contains_var(self, dvar):
        """ Checks whether a decision variable is part of an expression.

        Args:
            dvar (:class:`Var`): A decision variable.

        Returns:
            True if `dvar` is mentioned in the expression with a nonzero coefficient.
        """
        return (dvar in self.__terms) and (self.__terms[dvar] != 0)

    def equals_expr(self, other):
        if is_number(other):
            return self.equals_number(other)
        elif isinstance(other, LinearExpr):
            if self.constant != other.constant:
                return False
            if self.number_of_variables() != other.number_of_variables():
                return False
            for dvar in self.iter_variables():
                if self[dvar] != other[dvar]:
                    return False
            else:
                return True
        else:
            return False

    def equals_number(self, nb):
        return self.is_constant() and (nb == self._constant)

    def equals_var(self, dvar):
        return 0 == self.constant and 1 == len(self.__terms) and 1 == self.unchecked_get_coef(dvar)

    # noinspection PyPep8
    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        # INTERNAL
        c = 0
        # noinspection PyPep8Naming
        SP = ' '

        for v, coeff in iteritems(self.__terms):
            if 0 == coeff:
                continue

            # 1 separator
            if use_space and c > 0:
                oss.write(SP)

            # ---
            # sign is printed if  non-first OR negative
            # at the end of this block coeff is positive
            wrote_sign = False
            if coeff < 0 or c > 0:
                oss.write('-' if coeff < 0 else '+')
                wrote_sign = True
                if coeff < 0:
                    coeff = -coeff
            # ---
            varname = var_namer(v)
            if 1 != coeff:
                if wrote_sign and use_space:
                    oss.write(SP)
                self._num_to_stringio(oss, coeff, nb_digits)
                if prod_symbol:
                    oss.write(prod_symbol)
            elif wrote_sign and c > 0 and use_space:
                oss.write(SP)
            elif not use_space and varname[0].isdigit():
                oss.write(SP)
            oss.write(varname)
            c += 1

        k = self.constant
        if c == 0:
            self._num_to_stringio(oss, k, nb_digits)
        elif k != 0:
            if k < 0:
                sign = '-'
                k = -k
            else:
                sign = '+'
            if use_space: oss.write(SP)
            oss.write(sign)
            if use_space: oss.write(SP)
            self._num_to_stringio(oss, k, nb_digits)

    def _add_expr(self, other_expr):
        """
        Internal, assume other_expr is an expression.
        :param other_expr:
        :return:
        """
        self.constant += other_expr.constant
        # merge term dictionaries
        for v, k in other_expr.iter_terms():
            # use unchecked version
            self._add_term(v, k)

    # --- algebra methods always modify self.
    def _add(self, e):
        # INTERNAL: modifies self.
        if isinstance(e, Var):
            self._add_var(e)
        elif is_number(e):
            self._constant += e
        elif isinstance(e, LinearExpr):
            self._add_expr(e)
        elif isinstance(e, MonomialExpr):
            self._add_term(e._dvar, e._coef)
        elif isinstance(e, _ZeroExpr):
            pass
        elif isinstance(e, list):
            for elt in e:
                self._add(elt)
        else:
            try:
                self._add(e.to_linear_expr())
            except AttributeError:
                self.fatal("Unexpected argument for add: {0!s}", e)
        return self

    def iter_terms(self):
        """ Iterates over the terms in the expression.

        Returns:
            An iterator over the (variable, coefficient) pairs in the expression.
        """
        return iteritems(self.__terms)

    def _subtract(self, e):
        """
        INTERNAL: Subtracts to self, self will be modified.
        :param e:
        :return:
        """
        if isinstance(e, Var):
            self._add_term(e, -1)
        elif is_number(e):
            if e != 0:
                self._constant -= e
        elif isinstance(e, LinearExpr):
            if e.is_zero():
                return self
            else:
                # 1. decr constant
                self.constant -= e.constant
                # merge term dictionaries 
                for v, k in e.iter_terms():
                    self._add_term(v, -k)
        elif isinstance(e, MonomialExpr):
            self._add_term(e.var, -e.coef)
        elif isinstance(e, list):
            # do not use is_iterable as string will loop
            for elt in e:
                self._subtract(elt)
        elif isinstance(e, _ZeroExpr):
            pass
        else:
            try:
                self._subtract(e.to_linear_expr())
            except AttributeError:
                self.fatal("Unexpected argument for subtract: {0!s}", e)
        return self

    def _scale(self, factor, dictype=term_dict_type):
        # INTERNAL: used my multiply
        # this method modifies self.
        if 0 == factor:
            self._clear()
        elif factor != 1:
            self._constant *= factor
            self_terms = self.__terms
            for v, k in iteritems(self_terms):
                self_terms[v] = k * factor

    def _multiply(self, e):
        if is_number(e):
            self._scale(e)
        elif isinstance(e, LinearExpr):
            if e.is_constant():
                self._scale(e.constant)
            else:
                raise DOCplexQuadraticNotImplementedError(self, e)
        elif isinstance(e, Var):
            if self.is_constant():
                return e.times(self._constant)
            else:
                raise DOCplexQuadraticNotImplementedError(self, e)
        elif isinstance(e, MonomialExpr):
            if self.is_constant():
                return self.model._monomial_expr(e._dvar, e._coef * self._constant)
            else:
                raise DOCplexQuadraticNotImplementedError(self, e)
        elif isinstance(e, _ZeroExpr):
            return self.model._get_zero_expr()
        else:
            self.fatal("Multiply expects variable, expr or number, got: {0!s}", e)
        return self

    def _divide(self, e):
        self.model.typecheck_as_denominator(e, numerator=self)
        inverse = 1.0 / float(e)
        return self._multiply(inverse)

    # operator-based API
    def opposite(self):
        cloned = self.clone()
        cloned.negate()
        return cloned

    def plus(self, e):
        cloned = self.clone()
        cloned._add(e)
        return cloned

    def minus(self, e):
        cloned = self.clone()
        cloned._subtract(e)
        return cloned

    def times(self, e):
        cloned = self.clone()
        return cloned._multiply(e)

    def quotient(self, e):
        cloned = self.clone()
        cloned._divide(e)
        return cloned

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.plus(e)

    def __iadd__(self, e):
        return self._add(e)

    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        cloned = self.clone()
        cloned._subtract(e)
        return cloned.opposite()

    def __isub__(self, e):
        return self._subtract(e)

    def __neg__(self):
        return self.opposite()

    def __mul__(self, e):
        return self.times(e)

    def __imul__(self, e):
        return self._multiply(e)

    def __div__(self, e):
        return self.quotient(e)

    def __truediv__(self, e):
        return self.__div__(e)  # pragma: no cover

    def __rtruediv__(self, e):
        self.fatal("Expression {0!s} cannot be used as divider of {1!s}", self, e)  # pragma: no cover

    def __rmul__(self, e):
        return self.times(e)

    def __str__(self):
        return self.to_string()

    def __le__(self, other):
        return self.le_constraint(other)

    def __eq__(self, other):
        return self.eq_constraint(other)

    def __lt__(self, other):
        self.model.unsupported_operator_error(self, "<", other)

    def __gt__(self, other):
        self.model.unsupported_operator_error(self, ">", other)

    def __ge__(self, other):
        return self.ge_constraint(other)

    def le_constraint(self, other):
        return self._get_model().le_constraint(self, other)

    def eq_constraint(self, other):
        return self._get_model().eq_constraint(self, other)

    def ge_constraint(self, other):
        return self._get_model().ge_constraint(self, other)

    @property
    def solution_value(self):
        """ This property returns the solution value of the variable.

        Raises:
            DOCplexException
                if the model has not been solved.
        """
        self._check_model_has_solution()
        return self._get_solution_value()

    def get_value(self):
        # DEPRECATED
        return self.solution_value  # pragma: no cover

    def _get_solution_value(self):
        # INTERNAL: no checks
        val = self._constant
        for var, koef in self.iter_terms():
            val += koef * var.unchecked_solution_value
        return self._round_if_discrete(val)

    def is_discrete(self):
        """ Checks if the expression contains only discrete variables and coefficients.

        Example:
            If X is an integer variable, X, X+1, 2X+3 are discrete
            but X+0.3, 1.5X, 2X + 0.7 are not.

        Returns:
            True if the expression contains only discrete variables and coefficients.
        """
        self_cst = self._constant
        if self_cst != int(self_cst):
            return False

        for v, k in self.iter_terms():
            if not v.is_discrete() or not is_int(k):
                return False
        else:
            return True

    def __repr__(self):
        return "docplex.mp.LinearExpr({0})".format(self.truncated_str())


class LinearConstraintType(Enum):
    """This enumerated class defines the various types of linear constraints:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.
    """
    LE, EQ, GE = range(1, 4)

    # NOTE: Never add a static field in an enum class: it would be interpreted as an other enum

    @property
    def short_name(self):
        return _LinearConstraintTypeUtils.get_name(self)

    def get_operator_symbol(self):
        """ Returns a string operator for the constraint.

        Example:
            Returns string "<=" for a e1 <= e2 constraint.

        Returns:
            A string describing the logical operator used in the constraint.
        """
        return _LinearConstraintTypeUtils.get_operator(self)


class _LinearConstraintTypeUtils(object):
    # INTERNAL.
    # raison d'etre: cannot add static fields in an enumerated type, so we deport them elsewhere

    @staticmethod
    def _get_map_value(cttype, attribute_map, default_value=None):
        if cttype not in attribute_map:
            raise DOcplexException("unexpected constraint type: {0!s}".format(cttype))  # pragma: no cover

        return attribute_map.get(cttype, default_value)

    _operator_symbol_map = {LinearConstraintType.LE: "<=",
                            LinearConstraintType.EQ: "==",
                            LinearConstraintType.GE: ">="}
    _name_map = {LinearConstraintType.LE: "LE",
                 LinearConstraintType.EQ: "EQ",
                 LinearConstraintType.GE: "GE"}

    _python_op_map = {LinearConstraintType.LE: operator.le,
                      LinearConstraintType.EQ: operator.eq,
                      LinearConstraintType.GE: operator.ge}

    @staticmethod
    def get_operator(cttype):
        return _LinearConstraintTypeUtils._get_map_value(cttype, _LinearConstraintTypeUtils._operator_symbol_map)

    @staticmethod
    def get_name(cttype):
        return _LinearConstraintTypeUtils._get_map_value(cttype, _LinearConstraintTypeUtils._name_map)

    @staticmethod
    def get_python_operator_fn(cttype):
        """
        Returns the Python operator function associated with the ct.
        For example, returns operator.ge(a, b) for a GE constraint.
        :param cttype: A binary constraint type (LE, EQ, GE.
        :return: A Python operator function.
        """
        return _LinearConstraintTypeUtils._get_map_value(cttype, _LinearConstraintTypeUtils._python_op_map)


class AbstractConstraint(ModelingObject):
    __slots__ = ()

    def __init__(self, model, name=None):
        ModelingObject.__init__(self, model, name)

    def _unsupported_relational_op(self, op_string, other):
        self.fatal("Cannot use relational operator {1} on linear constraint: {0!s}", self, op_string)

    def __le__(self, e):
        self._unsupported_relational_op("<=", e)

    def __ge__(self, e):
        self._unsupported_relational_op(">=", e)

    def __lt__(self, e):
        self._unsupported_relational_op("<", e)

    def __gt__(self, e):
        self._unsupported_relational_op(">", e)

    # def __eq__(self, e):
    # self._unsupported_relational_op("==", e)

    def _no_linear_ct_in_logical_test_error(self):
        raise TypeError("cannot convert a constraint to boolean: {0!s}".format(self))

    def __nonzero__(self):
        self._no_linear_ct_in_logical_test_error()

    def __bool__(self):
        # python 3 version of nonzero
        self._no_linear_ct_in_logical_test_error()  # pragma: no cover

    def iter_variables(self):
        raise NotImplementedError  # pragma: no cover

    def copy(self, target_model, var_map):
        raise NotImplementedError  # pragma: no cover

    # noinspection PyMethodMayBeStatic
    def notify_deleted(self):
        # INTERNAL
        pass  # pragma: no cover

    def short_typename(self):
        return "constraint"

    def is_trivial(self):
        return False


# noinspection PyAbstractClass
class AbstractLinearConstraint(AbstractConstraint):
    # INTERNAL:  base class for linear constraints

    def __init__(self, model, name=None):
        AbstractConstraint.__init__(self, model, name)

    def copy(self, tagret_model, var_map):
        raise NotImplementedError  # pragma: no cover

    def get_var_coef(self, dvar):
        """ Returns the coefficient of the variable in the constraint.
             Calls the unsafe fastGetCoef method once type has been checked.
        """
        self.model.typecheck_var(dvar)
        return self.fast_get_coef(dvar)

    def fast_get_coef(self, var):
        """ Redefine this method for concrete sub-classes."""
        raise NotImplementedError  # pragma: no cover

    def rhs(self):
        """ Redefine this for any concrete constraint class."""
        raise NotImplementedError  # pragma: no cover


# noinspection PyAbstractClass
class AbstractLogicalConstraint(AbstractConstraint):
    # INTERNAL Root class for all logical stuff constraints

    def __init__(self, model, name=None):
        AbstractConstraint.__init__(self, model, name)


class LinearConstraint(AbstractLinearConstraint):
    """ The class that models all constraints of the form `<expr1> <OP> <expr2>`.
    """
    __slots__ = ("_ctype", "_left_expr", "_right_expr")

    def __init__(self, model, left_expr, ctype, right_expr, name=None):
        AbstractLinearConstraint.__init__(self, model, name)
        self._ctype = ctype
        self._left_expr = model._to_linear_expr(left_expr)
        self._right_expr = model._to_linear_expr(right_expr)

        model._check_both_in_selfmodel(self._left_expr, self._right_expr, "linear constraint")

    def equals(self, other):
        if type(other) != LinearConstraint:
            return False
        if self._ctype != other.type:
            return False
        if not self._left_expr.equals_expr(other.left_expr):
            return False
        if not self._right_expr.equals_expr(other.right_expr):
            return False

        return True

    def is_trivial(self):
        # Checks whether the constraint is equivalent to a comparison between numbers.
        # For example, x <= x+1 is trivial, but 1.5 X <= X + 1 is not.
        self_left_expr = self._left_expr
        self_right_expr = self._right_expr
        if self_right_expr.is_constant():
            return self_left_expr.is_constant()
        elif self_left_expr.is_constant():
            return self_right_expr.is_constant()
        else:
            for lv in self_left_expr.iter_variables():
                if self_left_expr[lv] != self_right_expr[lv]:
                    return False
            for rv in self_right_expr.iter_variables():
                if self_left_expr[rv] != self_right_expr[rv]:
                    return False
            return True

    def is_trivial_feasible(self):
        return self.is_trivial() and self._is_trivially_feasible()

    def is_trivial_infeasible(self):
        return self.is_trivial() and self._is_trivially_infeasible()

    def _is_trivially_feasible(self):
        # INTERNAL : assume self is trivial()
        op_func = _LinearConstraintTypeUtils.get_python_operator_fn(self.type)
        return op_func(self.left_expr.constant, self.right_expr.constant) if op_func else False

    def _is_trivially_infeasible(self):
        # INTERNAL: assume self is trivial .
        op_func = _LinearConstraintTypeUtils.get_python_operator_fn(self.type)
        return not op_func(self.left_expr.constant, self.right_expr.constant) if op_func else False

    def copy(self, target_model, var_map):
        copied_left = self.left_expr.copy(target_model, var_map)
        copied_right = self.right_expr.copy(target_model, var_map)
        return LinearConstraint(target_model, copied_left, self.type, copied_right, self.name)

    @property
    def type(self):
        """ This property returns the type of the constraint; type is an enumerated value
        of type :class:`LinearConstraintType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.

        """
        return self._ctype

    @property
    def left_expr(self):
        """ This property returns the left expression in the constraint.

        Example:
            (X+Y <= Z+1) has left expression (X+Y).
        """
        return self._left_expr

    @property
    def right_expr(self):
        """ This property returns the right expression in the constraint.

        Example:
            (X+Y <= Z+1) has right expression (Z+1).
        """
        return self._right_expr

    def to_string(self):
        """ Returns a string representation of the constraint.

        The operators in this representation are the usual operators <=, ==, and >=.

        Example:
            The constraint (X+Y <= Z+1) is represented as "X+Y <= Z+1".

        Returns:
            A string.

        """
        return "%s %s %s" % (self.left_expr.to_string(),
                             self._ctype.get_operator_symbol(),
                             str(self.right_expr))

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        user_name = self.name if self.has_user_name() else ""
        typename = self.type.short_name
        sleft = self._left_expr.truncated_str()
        sright = self._right_expr.truncated_str()
        return "docplex.mp.linear.LinearConstraint[{0}]({1!s},{2},{3!s})". \
            format(user_name, sleft, typename, sright)

    def __le__(self, e):
        # INTERNAL: define ranges with operators.
        # Beware one must use parentheses as in r = (1 <= x) <= 2
        # Chained comparisons like: 1 <= x <= 2 will fail as Python
        # generates an "and" of two constraints (1<=x) and (x<=2) but
        # constraints _cannot_ be converted to booleans.
        if not is_number(e):
            self.fatal("operator <= on constraint requires numeric argument, got: {0!s}", e)
        if self.type is LinearConstraintType.GE:
            rhs = self.right_expr
            if rhs.is_constant():
                range_min = rhs.constant
                range_max = float(e)
                return self.model.range_constraint(range_min, self.left_expr, range_max)
            else:
                self.fatal("operator <= requires a constraint with numeric RHS, rhs is: {0!s}", rhs)
        else:
            self.fatal("operator <= is only allowed for LE constraints, type is: {0!s}", self.type)

    def __ge__(self, e):
        # INTERNAL: define ranges with operators.
        # Beware one must use parentheses as in r = (1 <= x) <= 2
        # Chained comparisons like: 1 <= x <= 2 will fail as Python
        # generates an "and" of two constraints (1<=x) and (x<=2) but
        # constraints _cannot_ be converted to booleans.
        if not is_number(e):
            self.fatal("operator >= on constraints requires number argument, got: {0!s}", e)
        if self.type is LinearConstraintType.LE:
            rhs = self.right_expr
            if rhs.is_constant():
                range_max = rhs.constant
                range_min = float(e)
                return self.model.range_constraint(range_min, self.left_expr, range_max)
            else:
                self.fatal("operator >= requires a constraint with numeric RHS, got: {0!s}", rhs)
        else:
            self.fatal("operator >= is only allowed for GE constraints, not {0!s}", self.type)

    def iter_variables(self):
        """  Iterates over all variables mentioned in the constraint.

        *Note:* Includes variables that are mentioned with a zero coefficient. For example,
        the iterator on the following constraint:

         X <= X+Y + 1

        will return X and Y, although X is mentioned with a zero coefficient.

        Returns:
            An iterator object.
        """
        if self._right_expr.is_constant():
            return self._left_expr.iter_variables()
        elif self._left_expr.is_constant():
            return self._right_expr.iter_variables()
        else:
            # noinspection PyPep8
            if self._model._keep_ordering:
                return self._ordered_var_iter()
            else:
                left_vars = self.left_expr.var_set()
                right_vars = self.right_expr.var_set()
                union_vars = left_vars.union(right_vars)
                return iter(union_vars)

    def _ordered_var_iter(self):
        # INTERNAL
        e1 = self._left_expr._get_terms_dict()
        e2 = self._right_expr._get_terms_dict()
        od1 = OrderedDict(e1)
        od1.update(e2)
        return iter(od1)

    def contains_var(self, dvar):
        return self._left_expr.contains_var(dvar) or self._right_expr.contains_var(dvar)

    def fast_get_coef(self, dvar):
        return self._left_expr[dvar] - self._right_expr[dvar]

    def rhs(self):
        right_cst = self._right_expr.constant
        left_cst = self._left_expr.constant
        return right_cst - left_cst

    def _iter_net_coeffs(self, ordering=False):
        # INTERNAL
        if self._right_expr.is_constant():
            return self._left_expr.iter_terms()
        elif self._left_expr.is_constant():
            self_right_expr = self._right_expr
            opposite_right_coeffs = {v: -k for v, k in self_right_expr.iter_terms()}
            return iteritems(opposite_right_coeffs)
        elif ordering:
            # build an ordered dict of left terms updated by right terms
            e1 = self._left_expr._get_terms_dict()
            # we must create an extra od here
            od = OrderedDict(e1)
            #  update od no need to create yet another od
            for v, rk in self._right_expr.iter_terms():
                if v in od:
                    od[v] -= rk
                else:
                    od[v] = -rk
            # od2 = OrderedDict({v: -k for v, k in self._right_expr.iter_terms()})
            # od.update(od2)
            return iteritems(od)
        else:
            diff_coeffs = {v: self._left_expr[v] - self._right_expr[v] for v in self.iter_variables()}
            return iteritems(diff_coeffs)

    def _generate_net_coefs(self):
        for v in self.iter_variables():
            net_k = self._left_expr[v] - self._right_expr[v]
            yield v, net_k


class _DummyFeasibleConstraint(LinearConstraint):
    # INTERNAL
    def __init__(self, model, zero_expr):
        LinearConstraint.__init__(self, model, zero_expr, LinearConstraintType.EQ, 0, "_dummy_feasible_ct_")

    def __repr__(self):
        return "docplex.mp.linear.LinearConstraint._TrivialFeasible"


class _DummyInfeasibleConstraint(LinearConstraint):
    # INTERNAL
    def __init__(self, model, zero):
        LinearConstraint.__init__(self, model, zero, LinearConstraintType.EQ, 1, "_dummy_infeasible_ct_")

    def __repr__(self):
        return "docplex.mp.linear.LinearConstraint.TrivialInfeasible"


class RangeConstraint(AbstractLinearConstraint):
    """ This class models range constraints.

    A range constraint states that an expression must stay between two
    values, `lb` and `ub`.

    This class is not meant to be instantiated by the user.
    To create a range constraint, use the factory method :func:`docplex.mp.model.Model.add_range`
    defined on :class:`docplex.mp.model.Model`.

    """

    def __init__(self, model, expr, lb, ub, name=None):
        AbstractLinearConstraint.__init__(self, model, name)
        model.typecheck_num(lb, 'RangeConstraint.lb')
        model.typecheck_num(ub, 'RangeConstraint.ub')
        self.__ub = ub
        self.__lb = lb
        self.__expr = model._to_linear_expr(expr)

    def equals(self, other):
        if type(other) != RangeConstraint:
            return False
        if self.__lb != other.lb:
            return False
        if self.__ub != other.ub:
            return False
        if not self.__expr.equals_expr(other.expr):
            return False

        return True

    def short_typename(self):
        return "range"

    def is_trivial(self):
        return self.__expr.is_constant()

    def _is_trivially_feasible(self):
        # INTERNAL : assume self is trivial()
        expr_num = self.__expr.constant
        return self.__lb <= expr_num <= self.__ub

    def _is_trivially_infeasible(self):
        # INTERNAL : assume self is trivial()
        expr_num = self.__expr.constant
        return expr_num < self.__lb or expr_num > self.__ub

    @property
    def expr(self):
        """ This property returns the linear expression of the range constraint.
        """
        return self.__expr

    @property
    def lb(self):
        """ This property returns the lower bound of the range constraint.

        """
        return self.__lb

    @property
    def ub(self):
        """ This property returns the upper bound of the range constraint.

        """
        return self.__ub

    def is_valid(self):
        return self.__ub >= self.__lb

    def iter_variables(self):
        """Iterates over all the variables of the range constraint.

        Returns:
           An iterator object.
        """
        return self.__expr.iter_variables()

    def contains_var(self, dvar):
        return self.__expr.contains_var(dvar)

    def rhs(self):
        # INTERNAL
        return self.__lb - self.__expr.constant

    def fast_get_coef(self, dvar):
        # INTERNAL
        return self.__expr.unchecked_get_coef(dvar)

    def copy(self, target_model, var_map):
        copied_expr = self.expr.copy(target_model, var_map)
        copied_range = RangeConstraint(target_model, copied_expr, self.lb, self.ub, self.name)
        return copied_range

    def to_string(self):
        return "{0} <= {1!s} <= {2}".format(self.__lb, self.__expr, self.__ub)

    def __str__(self):
        """ Returns a string representation of the range constraint.

        Example:
            1 <= x+y+z <= 3 represents the range constraint where the expression (x+y+z) is 
            constrained to stay between 1 and 3.

        Returns:
            A string.
        """
        return self.to_string()

    def __repr__(self):
        printable_name = self.name if self.has_user_name() else ""
        return "docplex.mp.linear.RangeConstraint[{0}]({1},{2!s},{3})".format(printable_name, self.lb, self.__expr,
                                                                              self.ub)


class IndicatorConstraint(AbstractLogicalConstraint):
    """ This class models indicator constraints.

    An indicator constraint links (one-way) the value of a binary variable to the satisfaction of a linear constraint.
    If the binary variable equals the active value, then the constraint is satisfied, but otherwise the constraint
    may or may not be satisfied.

    This class is not meant to be instantiated by the user.

    To create an indicator constraint, use the factory method :func:`docplex.mp.model.Model.add_indicator`
    defined on :class:`docplex.mp.model.Model`.

    """

    def __init__(self, model, binary_var, linear_ct, active_value=1, name=None):
        AbstractLogicalConstraint.__init__(self, model, name)
        self._binary_var = binary_var
        self._linear_ct = linear_ct
        self._active_value = active_value
        if linear_ct.is_trivial():
            self.warning("Using trivial linear constraint in indicator, binary: {0!s}, ct: {1!s}", binary_var,
                         linear_ct)

    def equals(self, other):
        if type(other) != IndicatorConstraint:
            return False

        if self._active_value != other.active_value:
            return False
        if not self._binary_var.equals(other.binary_var):
            return False
        if not self._linear_ct.equals(other.linear_ct):
            return False

        return True

    def short_typename(self):
        return "indicator"

    @property
    def active_value(self):
        return self._active_value

    @property
    def indicator_var(self):
        return self._binary_var

    @property
    def linear_constraint(self):
        return self._linear_ct

    @property
    def logical_rhs(self):
        """
        This property returns the target right-hand side used to trigger the linear constraint. Returns 1 if not complemented, else 0.
        """
        return self.active_value

    def copy(self, target_model, var_map):
        copied_binary = var_map[self.indicator_var]
        copied_linear_ct = self.linear_constraint.copy(target_model, var_map)
        copied_indicator = IndicatorConstraint(target_model, copied_binary, copied_linear_ct, self.active_value,
                                               self.name)
        return copied_indicator

    def invalidate(self):
        """
        Sets the binary variable to the opposite of its active value.
        Typically used by indicator constraints with a trivial infeasible linear part.
        For example, z=1 -> 4 <= 3 sets z to 0 and
        z=0 -> 4 <= 3 sets z to 1.
        This is equivalent to if z=a => False, then z *cannot* be equal to a.
        """
        if self.active_value is 0:
            # set to 1 : lb = 1
            self.indicator_var.lb = 1
        elif self.active_value is 1:
            # set to 0 ub = 0
            self.indicator_var.ub = 0
        else:
            self.fatal("Unexpected active value for indicator constraint {0!s}, expecting 0 or 1", self)

    def iter_variables(self):
        yield self._binary_var
        for v in self._linear_ct.iter_variables():
            yield v

    def contains_var(self, dvar):
        return dvar is self._binary_var or self._linear_ct.contains_var(dvar)

    def to_string(self):
        """
        Displays the indicator constraint in the LP style:
        z = 1 -> x+y+z == 2
        """
        return "{0!s} = {1} -> {2!s}".format(self._binary_var, self.logical_rhs, self.linear_constraint)

    def __str__(self):
        return self.to_string()


class ObjectiveSense(Enum):
    """
    This enumerated class defines the two types of objectives, Minimize and Maximize.
    """
    Minimize, Maximize = 1, 2

    # static method: which type is the default.
    @staticmethod
    def default_sense():
        return ObjectiveSense.Minimize

    def is_minimize(self):
        return self is ObjectiveSense.Minimize

    def is_maximize(self):
        return self is ObjectiveSense.Maximize

    def verb(self):
        # INTERNAL
        return "minimize" if self.is_minimize() else "maximize" if self.is_maximize() else "WHAT???"

    def action(self):
        return "%sing" % self.verb()[:-1]

    @staticmethod
    def parse(text, errorhandler, do_raise=False):
        if not text or not is_string(text):
            if do_raise:
                errorhandler.fatal("cannot read objective sense from: <{}>", text)
            else:
                errorhandler.error("cannot read objective sense from: <{}>", text)
                return None
        else:
            lower_text = text.lower()
            if lower_text in {"minimize", "min"}:
                return ObjectiveSense.Minimize
            elif lower_text in {"maximize", "max"}:
                return ObjectiveSense.Maximize
            else:
                errorhandler.fatal("Text is not recognized as objective sense: {0}, expecting min"" or max", (text,))


# --- version with global functions


def _docplex_extract_model(e, do_raise=True):
    try:
        model = e.model
        return model
    except AttributeError:
        if do_raise:
            raise DOcplexException("object has no model attribute: {0!s}", e)
        else:
            return None


def docplex_sum(x_seq):
    if is_iterable(x_seq):
        if not x_seq:
            return 0
        elif isinstance(x_seq, dict):
            return _docplex_sum_with_seq(x_seq.values())
        elif is_indexable(x_seq):
            return _docplex_sum_with_seq(x_seq)
        else:
            return _docplex_sum_with_seq(list(x_seq))
    elif x_seq:
        mdl = _docplex_extract_model(x_seq, do_raise=False)
        return mdl.to_linear_expr(x_seq) if mdl else x_seq
    else:
        return 0


def _docplex_sum_with_seq(x_list):
    shared_model = None
    for x in x_list:
        try:
            model = x.model
            if not shared_model:
                shared_model = model
            else:
                if model != shared_model:
                    raise DOcplexException("Cannot mix objects belonging to different models")
        except AttributeError:
            pass
    if shared_model:
        return shared_model.sum(x_list)
    else:
        # try a python sum ?
        return sum(x_list)
