# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint: disable=too-many-lines
from __future__ import print_function

from collections import OrderedDict

from six import iteritems

from docplex.mp.constants import ComparisonType
from docplex.mp.compat23 import unitext
from docplex.mp.basic import ModelingObject, Expr, ModelingObjectBase
from docplex.mp.vartype import BinaryVarType, IntegerVarType, ContinuousVarType
from docplex.mp.utils import *
from docplex.mp.xcounter import ExprCounter


class Var(ModelingObject):
    """Var()

    This class models decision variables.
    Decision variables are instantiated by :class:`docplex.mp.model.Model` methods such as :func:`docplex.mp.model.Model.var`.

    """

    __slots__ = ("_vartype", "_lb", "_ub", "__id")

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
            used_lb = vartype.get_default_lb() if lb is None else vartype.compute_lb(lb)
            used_ub = vartype.get_default_ub() if ub is None else vartype.compute_ub(ub)
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

    def accept_initial_value(self, numeric_value):
        if not self.vartype.accept_value(numeric_value):
            return False
        return self.lb <= numeric_value <= self.ub

    def check_name(self, new_name):
        ModelingObject.check_name(self, new_name)
        if not is_string(new_name) or not new_name:
            self.fatal("Variable name accepts only non-empty strings, got: {0!s}", new_name)
        elif new_name.find(' ') >= 0:
            self.warning("Variable name contains blank space, var: {0!s}, name: \'{1!s}\'", self, new_name)

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
        expr = self._get_model()._linear_expr(self)
        expr._transient = True
        return expr
        # return MonomialExpr(self._get_model(), self, coeff=1)

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
        floor integer value of the argument.

        To reset the upper bound to its default infinity value, use :func:`docplex.mp.model.Model.infinity`.
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

    def get_vartype(self):
        return self._vartype

    def set_vartype(self, new_vartype):
        return self._model.set_var_type(self, new_vartype)


    def has_type(self, vartype):
        # internal
        return type(self._vartype) == vartype

    def is_binary(self):
        """ Checks if the variable is binary.

        Returns:
            Boolean: True if the variable is of type Binary.
        """
        return self.has_type(BinaryVarType)

    def is_integer(self):
        """ Checks if the variable is integer.

        Returns:
            Boolean: True if the variable is of type Integer.
        """
        return self.has_type(IntegerVarType)

    def is_continuous(self):
        """ Checks if the variable is continuous.

        Returns:
            Boolean: True if the variable is of type Continuous.
        """
        return self.has_type(ContinuousVarType)

    def is_discrete(self):
        """  Checks if the variable is discrete.

        Returns:
            Boolean: True if the variable is of  type Binary or Integer.
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

    def get_constraint_factory(self, arg):
        # INTERNAL
        try:
            if arg.is_quad_expr():
                return self._model._qfactory
        except AttributeError:
            pass
        return self._model._lfactory

    def le_constraint(self, rhs):
        return self.get_constraint_factory(rhs).new_le_constraint(self, rhs)


    def eq_constraint(self, rhs):
        return self.get_constraint_factory(rhs).new_eq_constraint(self, rhs)

    def ge_constraint(self, rhs):
        return self.get_constraint_factory(rhs).new_ge_constraint(self, rhs)

    def __le__(self, e):
        return self.le_constraint(e)

    def __eq__(self, e):
        return self.eq_constraint(e)

    def __ge__(self, e):
        return self.ge_constraint(e)

    def __ne__(self, other):
        # INTERNAL: For now, not supported
        self.model.unsupported_neq_error(self, other)


    def __gt__(self, e):
        self.model.unsupported_relational_operator_error(self, ">", e)

    def __lt__(self, e):
        self.model.unsupported_relational_operator_error(self, "<", e)

    def __mul__(self, e):
        return self.times(e)


    def times(self, e):
        if is_number(e):
            if 0 == e:
                return self.zero_expr()
            elif 1 == e:
                return self
            else:
                return MonomialExpr(self._model, self, e, checked_num=True)
        elif isinstance(e, ZeroExpr):
            return e
        elif isinstance(e, (Var, Expr)):
            return self._model._qfactory.new_var_product(self, e)
        else:
            return self.to_linear_expr().multiply(e)

    def __rmul__(self, e):
        return self.times(e)

    def __add__(self, e):
        return self.plus(e)

    def plus(self, e):
        if is_number(e):
            if e == 0:
                return self
        try:
            return self.to_linear_expr().add(e)
        except DOCPlexQuadraticArithException:
            return e.plus(self)

    def __radd__(self, e):
        return self.plus(e)

    def __sub__(self, e):
        return self.minus(e)

    def minus(self, e):
        if is_number(e):
            if e == 0:
                return self
        try:
            return self.to_linear_expr().subtract(e)
        except DOCPlexQuadraticArithException:
            return e.rminus(self)

    def __rsub__(self, e):
        # e - self
        expr = self._get_model()._to_linear_expr(e, force_clone=True)  # makes a clone.
        return expr.subtract(self)

    def divide(self, e):
        if is_number(e):
            if e == 1:
                return self
        expr = self.to_linear_expr()
        return expr.divide(e)

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
        return self.model._lfactory.new_monomial_expr(self, -1)

    def __pow__(self, power):
        # INTERNAL
        # power must be checke in {0, 1, 2}
        self.model.typecheck_as_power(self, power)
        if 0 == power:
            return 1
        elif 1 == power:
            return self
        else:
            return self.square()

    def square(self):
        return self._model._qfactory.new_var_square(self)

    def __int__(self):
        """ Converts a decision variable to a integer number.

        This is only possible for discrete variables,
        and when the model has been solved successfully.
        If the model has been solved, returns the variable's solution value.

        Returns:
            int: The variable's solution value.

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

        This is only possible when the model has been solved successfully,
        otherwise an exception is raised.
        If the model has been solved, it returns the variable's solution value.

        Returns:
            float: The variable's solution value.
        Raises:
            DOCplexException
                if the model has not been solved successfully.
        """
        return float(self.solution_value)

    def to_bool(self):
        """ Converts a variable value to True or False.

        This is only possible for discrete variables and assumes there is a solution.

        Raises:
            DOCplexException 
                if the model has not been solved successfully.
            DOCplexException 
                if the variable is not discrete.

        Returns:
            Boolean: True if the variable value is nonzero, else False.
        """
        if not self.is_discrete():
            self.fatal("boolean conversion only for discrete variables, type is {0!s}", self.vartype)
        value = self.solution_value  # this property checks for a solution.
        return value != 0

    # def __nonzero__(self):
    # return self.to_boolean(precision=1e-5)

    def __str__(self):
        """
        Returns:
            string: A string representation of the variable.

        """
        return self.to_string()

    def _set_vartype_internal(self, new_vartype):
        # INTERNAL
        self._vartype = new_vartype

    def _name_marker(self):
        if self.is_generated():
            return "?"  # ? for generated variables e.g. _x1?
        elif self.has_automatic_name():
            return "*"  # anonymous but not generated _x1*
        else:
            return ""   # no suffix for usernames

    def to_string(self):
        str_name = self.name or '%s_var_%d' % (self.vartype.short_name, self.unchecked_index)
        return str_name

    def to_full_string(self):
        str_name = self.name or '%s_var_%d' % (self.vartype.short_name, self.unchecked_index)
        name_mark = self._name_marker()
        self_vartype, self_lb, self_ub = self._vartype, self._lb, self._ub
        if self_vartype.is_default_domain(self_lb, self_ub):
            str_bounds = ""
        else:
            str_bounds = "[%g..%g]" % (self_lb, self_ub)
        return "{1}{2}{3}".format(self_vartype.one_letter_symbol(), str_name, name_mark, str_bounds)

    def to_stringio(self, oss):
        oss.write(self.to_string())

    def __repr__(self):
        repr_name = self.name or "NO_NAME"
        name_mark = self._name_marker()
        self_vartype, self_lb, self_ub = self._vartype, self._lb, self._ub
        if self_vartype.is_default_domain(self_lb, self_ub):
            repr_bounds = ""
        else:
            repr_bounds = ",lb=%g,ub=%g" % (self_lb, self_ub)
        return 'docplex.mp.linear.Var(type={0},name={1}{2}{3})'.\
            format(self_vartype.one_letter_symbol(), repr_name, name_mark, repr_bounds)


# noinspection PyAbstractClass
class AbstractLinearExpr(Expr):
    __slots__ = ()

    def __init__(self, model):
        Expr.__init__(self, model=model)  # pragma: no cover

    def __getitem__(self, dvar):
        return self.unchecked_get_coef(dvar)

    def unchecked_get_coef(self, dvar):
        raise NotImplementedError  # pragma: no cover

    def iter_variables(self):
        """
        Iterates over all variables in the expression.

        Returns:
            iterator: An iterator over all variables present in the expression.
        """
        raise NotImplementedError  # pragma: no cover

    def iter_terms(self):
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


    def is_variable(self):
        # return True if the expression is infact one variable.
        # if True, assume you can replace the expression by the first variable
        # returned by iter_variables()
        return False

    def __lt__(self, e):
        self.model.unsupported_relational_operator_error(self, "<", e)

    def __gt__(self, e):
        self.model.unsupported_relational_operator_error(self, ">", e)

    def __le__(self, other):
        return self.le_constraint(other)

    def __eq__(self, other):
        return self.eq_constraint(other)

    def __ge__(self, other):
        return self.ge_constraint(other)

    def le_constraint(self, other):
        return self.get_constraint_factory(other).new_le_constraint(self, other)

    def eq_constraint(self, other):
        return self.get_constraint_factory(other).new_eq_constraint(self, other)

    def ge_constraint(self, other):
        return self.get_constraint_factory(other).new_ge_constraint(self, other)

    def iter_quads(self):
        return iter_emptyset()


class MonomialExpr(AbstractLinearExpr):
    def _get_solution_value(self):
        raw = self.coef * self._dvar.solution_value
        return self._round_if_discrete(raw)

    # INTERNAL class
    __slots__ = ("_dvar", "_coef")

    # noinspection PyMissingConstructor
    def __init__(self, model, dvar, coeff, checked_num=False, safe=False):
        self._model = model  # faster than to call recursively init methods...
        self._name = None
        self._dvar = dvar
        # check perf on that
        if safe:
            self._coef = coeff
        else:
            self._coef = model._lfactory.to_valid_number(coeff,
                                                         checked_num=checked_num,
                                                         context_msg=lambda: dvar.name + ".times()")

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
        # INTERNAL
        return 1 == self._coef

    def clone(self):
        return MonomialExpr(self.model, self._dvar, self._coef, safe=True)

    def copy(self, target_model, var_mapping):
        copy_var = var_mapping[self._dvar]
        return MonomialExpr(target_model, dvar=copy_var, coeff=self._coef, safe=True)

    def iter_variables(self):
        yield self._dvar

    def iter_terms(self):
        yield self._dvar, self._coef

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
            try:
                return self.to_linear_expr().add(e)
            except DOCPlexQuadraticArithException:
                return e.plus(self)

    def minus(self, e):
        try:
            expr = self.to_linear_expr()
            expr.subtract(e)
            return expr
        except DOCPlexQuadraticArithException:
            return e.rminus(self)

    def times(self, e):
        if is_number(e):
            # e might be a fancy numpy type
            if 1 == e:
                return self
            elif 0 == e:
                return self.zero_expr()
            else:
                # return a fresh instance
                return self.model._lfactory.new_monomial_expr(self._dvar, self._coef * e)
        elif isinstance(e, LinearExpr):
            return e.times(self)
        elif isinstance(e, Var):
            return self.model._qfactory.new_var_product(e, self)
        elif isinstance(e, MonomialExpr):
            return self.model._qfactory.new_monomial_product(self, e)
        else:
            expr = self.to_linear_expr()
            return expr.multiply(e)



    def square(self):
        return self.model._qfactory.new_monomial_product(self, self)

    def quotient(self, e):
        self.model.typecheck_as_denominator(e, self)
        inverse = 1.0 / float(e)
        return self.model._lfactory.new_monomial_expr(self._dvar, self._coef * inverse)

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


    add = plus
    subtract = minus
    divide = quotient
    multiply = times

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
        terms = LinearExpr.term_dict_type()
        terms[self._dvar] = self._coef
        e = LinearExpr(self._model, e=terms, safe=True)
        e._transient = True
        return e

    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        self_coef = self._coef
        if self_coef != 1:
            if self_coef < 0:
                oss.write(u'-')
                self_coef = - self_coef
            if self_coef != 1:
                self._num_to_stringio(oss, num=self_coef, ndigits=nb_digits)
            if use_space:
                oss.write(u' ')
            if prod_symbol:
                oss.write(unitext(prod_symbol))
                if use_space:
                    oss.write(u' ')
        oss.write(unitext(var_namer(self._dvar)))

    def __repr__(self):
        return "docplex.mp.MonomialExpr(%s)" % self.to_string()


class LinearExpr(AbstractLinearExpr):
    """LinearExpr()

    This class models linear expressions.
    This class is not intended to be instantiated. Expressions are built
    either using operators or using `Model.linear_expr()`.

    """

    # what type to use for merging dicts
    counter_type = ExprCounter

    # what type to use for storing terms
    term_dict_type = OrderedDict


    @staticmethod
    def _sort_terms_if_needed(mdl, counter, term_dict_type=term_dict_type):
        # INTERNAL
        if not mdl._keep_ordering:
            return counter
        elif isinstance(counter, term_dict_type):
            return counter
        else:
            # normalize by sorting variables by increasing indices
            sorted_items = sorted(counter.items(), key=lambda vk: vk[0].get_index())
            return term_dict_type(sorted_items)

    def to_linear_expr(self):
        return self

    def _get_terms_dict(self):
        # INTERNAL
        return self.__terms

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

    __slots__ = ('_constant', '__terms', '_transient', '_refcnt')

    def __init__(self, model, e=None, constant=0, name=None, safe=False):
        Expr.__init__(self, model, name)
        # a global counter for performance measurement
        model._linexpr_instance_counter += 1
        # "calling LinearExpr ctor, k=%d" % LinearExpr.InstanceCounter)
        if not safe and 0 != constant:
            model.typecheck_num(constant, 'LinearExpr()')
        self._constant = constant
        self._transient = False
        self._refcnt = 0

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

        if e is None:
            pass
        elif isinstance(e, Var):
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
            # note that transient is not kept.
            self._constant = e.constant
            self.__terms = self.term_dict_type(e._get_terms_dict())  # copy

        elif is_iterable(e) and not is_string(e):
            for o in e:
                model.typecheck_var(o)
                self.__terms[o] = 1

        else:
            self.fatal("Cannot convert {1!s} to docplex.mp.LinearExpr, instance: {0!s}", repr(e), type(e).__name__)

    def _check_mutable(self):
        # INTERNAL
        if self._refcnt > 0:
            self.fatal("An expression used in constraints is not mutable: {0!s}", self)

    def notify_used(self, user):
        # INTERNAL
        self._refcnt += 1

    def _is_mutable(self):
        return 0 == self._refcnt

    def clone_if_necessary(self):
        #  INTERNAL
        if self._transient and not self._model._clone_transient_exprs and self._is_mutable():
            return self
        else:
            return self.clone()

    def set_name(self, name):
        Expr.set_name(self, name)
        # an expression with a name is not transient any more
        if name:
            self._transient = False

    def _get_name(self):
        return self._name

    name = property(_get_name, set_name)

    def is_transient(self):
        # INTERNAL
        return self._transient

    def clone(self):
        """
        Returns:
            A copy of the expression on the same model.
        """
        self._model._linexpr_clone_counter += 1
        cloned_terms = self.term_dict_type(self.__terms)  # faster than copy() on OrderedDict()
        cloned = LinearExpr(model=self._model, e=cloned_terms, constant=self._constant, safe=True)
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

        Note:
            This method does not create any new expression but modifies the `self` instance.

        Returns:
            The modified self.

        """
        self._check_mutable()
        self._constant = - self._constant
        self_terms = self.__terms
        for v, k in iteritems(self_terms):
            self_terms[v] = -k
        return self

    def _clear(self):
        """ Clears the expression.

        All variables and coefficients are removed and the constant term is set to zero.
        """
        self._check_mutable()
        self._constant = 0
        self.__terms = self._new_terms_dict()

    def equals_constant(self, scalar):
        """ Checks if the expression equals a constant term.

        Args:
            scalar (float): A floating-point number.
        Returns:
            Boolean: True if the expression equals this constant term.
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
            Boolean: True if the expression consists of only a constant term.
        """
        return not self.__terms

    def is_variable(self):
        # INTERNAL: returns True if expression is in fact a variable (1*x)
        terms = self.__terms
        return 0 == self.constant and 1 == len(terms) and 1 == next(itervalues(terms))

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
        self._check_mutable()
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
        self._check_mutable()
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
        self._check_mutable()
        # INTERNAL
        del self.__terms[dvar]

    def get_constant(self):
        """
        This property is used to get or set the constant term of the expression.
        """
        return self._constant

    def set_constant(self, numval):
        self._check_mutable()
        self._constant = numval

    constant = property(get_constant, set_constant)

    def iter_variables(self):
        """  Iterates over all variables mentioned in the linear expression.

        Returns:
            An iterator object.
        """
        return iter(self.__terms.keys())

    def contains_var(self, dvar):
        """ Checks whether a decision variable is part of an expression.

        Args:
            dvar (:class:`Var`): A decision variable.

        Returns:
            Boolean: True if `dvar` is mentioned in the expression with a nonzero coefficient.
        """
        return dvar in self.__terms

    def equals_expr(self, other):
        if is_number(other):
            return self.is_constant() and other == self.constant
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

    # noinspection PyPep8
    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        # INTERNAL
        # Writes unicode repsentation of self
        c = 0
        # noinspection PyPep8Naming
        SP = u' '

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
                oss.write(u'-' if coeff < 0 else u'+')
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
                    oss.write(unitext(prod_symbol))
            elif wrote_sign and c > 0 and use_space:
                oss.write(SP)
            elif not use_space and varname[0].isdigit():
                oss.write(SP)
            oss.write(unitext(varname))
            c += 1

        k = self.constant
        if c == 0:
            self._num_to_stringio(oss, k, nb_digits)
        elif k != 0:
            if k < 0:
                sign = u'-'
                k = -k
            else:
                sign = u'+'
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
        self._check_mutable()
        self.constant += other_expr.constant
        # merge term dictionaries
        for v, k in other_expr.iter_terms():
            # use unchecked version
            self._add_term(v, k)

    def _add_expr_scaled(self, expr, factor):
        self._check_mutable()
        # INTERNAL
        if 0 != factor:
            self.constant += expr.constant * factor
            for v, k in expr.iter_terms():
                # use unchecked version
                self._add_term(v, k * factor)

    # --- algebra methods always modify self.
    def add(self, e):
        """ Adds an expression to self.

        Note:
            This method does not create an new expression but modifies the `self` instance.

        Args:
            e: The expression to be added. Can be a variable, an expression, or a number.

        Returns:
            The modified self.

        See Also:
            The method :func:`plus` to compute a sum without modifying the self instance.
        """
        self._check_mutable()
        if isinstance(e, Var):
            self._add_var(e)
        elif isinstance(e, LinearExpr):
            self._add_expr(e)
        elif isinstance(e, MonomialExpr):
            self._add_term(e._dvar, e._coef)
        elif isinstance(e, ZeroExpr):
            pass
        elif is_number(e):
            self._constant += e
        elif isinstance(e, Expr) and e.is_quad_expr():
            raise DOCPlexQuadraticArithException
        else:
            try:
                self.add(e.to_linear_expr())
            except AttributeError:
                self._unsupported_binary_operation(self, "+", e)

        return self

    def iter_terms(self):
        """ Iterates over the terms in the expression.

        Returns:
            An iterator over the (variable, coefficient) pairs in the expression.
        """
        return iteritems(self.__terms)

    def subtract(self, e):
        """ Subtracts an expression from this expression.
        Note:
            This method does not create a new expression but modifies the `self` instance.

        Args:
            e: The expression to be subtracted. Can be either a variable, an expression, or a number.

        Returns:
            The modified self.

        See Also:
            The method :func:`minus` to compute a difference without modifying the `self` instance.
        """
        self._check_mutable()
        if isinstance(e, Var):
            self._add_term(e, -1)
        elif is_number(e):
            if e != 0:
                self._constant -= e
        elif isinstance(e, LinearExpr):
            if e.is_constant() and 0 == e.get_constant():
                return self
            else:
                # 1. decr constant
                self.constant -= e.constant
                # merge term dictionaries 
                for v, k in e.iter_terms():
                    self._add_term(v, -k)
        elif isinstance(e, MonomialExpr):
            self._add_term(e.var, -e.coef)
        elif isinstance(e, ZeroExpr):
            pass
        elif isinstance(e, Expr) and e.is_quad_expr():
            raise DOCPlexQuadraticArithException
        else:
            try:
                self.subtract(e.to_linear_expr())
            except AttributeError:
                self._unsupported_binary_operation(self, "-", e)
        return self

    def _scale(self, factor):
        # INTERNAL: used my multiply
        # this method modifies self.
        self._check_mutable()
        if 0 == factor:
            self._clear()
        elif factor != 1:
            self._constant *= factor
            self_terms = self.__terms
            for v, k in iteritems(self_terms):
                self_terms[v] = k * factor

    def multiply(self, e):
        """ Multiplies this expression by an expression.

        Note:
            This method does not create a new expression but modifies the `self` instance.

        Args:
            e: The expression that is used to multiply `self`.

        Returns:
            The modified `self`.

        See Also:
            The method :func:`times` to compute a multiplication without modifying the `self` instance.
        """
        self._check_mutable()
        if is_number(e):
            self._scale(e)
        elif isinstance(e, LinearExpr):
            if e.is_constant():
                self._scale(e.constant)
            else:
                return self.model._qfactory.new_linexpr_product(self, e)

        elif isinstance(e, Var):
            if self.is_constant():
                return e.times(self._constant)
            else:
                return self.model._qfactory.new_var_product(e, self)
        elif isinstance(e, MonomialExpr):
            if self.is_constant():
                return self.model._lfactory.new_monomial_expr(e._dvar, e._coef * self._constant)
            else:
                return self.model._qfactory.new_linexpr_product(self, e)
        elif isinstance(e, ZeroExpr):
            return self.zero_expr()
        else:
            self.fatal("Multiply expects variable, expr or number, got: {0!s}", e)
        return self

    def square(self):
        return self.model._qfactory.new_linexpr_product(self, self)

    def divide(self, e):
        """ Divides this expression by an operand.

        Args:
            e: The operand by which the self expression is divided. Only nonzero numbers are permitted.

        Note:
            This method does not create a new expression but modifies the `self` instance.

        Returns:
            The modified `self`.
        """
        self.model.typecheck_as_denominator(e, numerator=self)
        inverse = 1.0 / float(e)
        return self.multiply(inverse)

    # operator-based API
    def opposite(self):
        cloned = self.clone_if_necessary()
        cloned.negate()
        return cloned

    def plus(self, e):
        """ Computes the sum of the expression and some operand.

        Args:
            e: the expression to add to self. Can be either a variable, an expression or a number.

        Returns:
            a new expression equal to the sum of the self expression and `e`

        Note:
            This method doe snot modify self.
        """
        cloned = self.clone_if_necessary()
        try:
            return cloned.add(e)
        except DOCPlexQuadraticArithException:
            return e.plus(self)

    def minus(self, e):
        cloned = self.clone_if_necessary()
        try:
            return cloned.subtract(e)
        except DOCPlexQuadraticArithException:
            return e.rminus(self)

    def times(self, e):
        """ Computes the multiplication of this expression with an operand.

        Note:
            This method does not modify the `self` instance but returns a new expression instance.

        Args:
            e: The expression that is used to multiply `self`.

        Returns:
            A new instance of expression.
        """
        cloned = self.clone_if_necessary()
        return cloned.multiply(e)

    def quotient(self, e):
        """ Computes the division of this expression with an operand.

        Note:
            This method does not modify the `self` instance but returns a new expression instance.

        Args:
            e: The expression that is used to modify `self`. Only nonzero numbers are permitted.

        Returns:
            A new instance of expression.
        """
        cloned = self.clone_if_necessary()
        cloned.divide(e)
        return cloned

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.plus(e)

    def __iadd__(self, e):
        return self.add(e)

    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        cloned = self.clone_if_necessary()
        cloned.subtract(e)
        cloned.negate()
        return cloned

    def __isub__(self, e):
        return self.subtract(e)

    def __neg__(self):
        return self.opposite()

    def __mul__(self, e):
        return self.times(e)

    def __imul__(self, e):
        return self.multiply(e)

    def __div__(self, e):
        return self.quotient(e)

    def __truediv__(self, e):
        return self.__div__(e)  # pragma: no cover

    def __rtruediv__(self, e):
        self.fatal("Expression {0!s} cannot be used as divider of {1!s}", self, e)  # pragma: no cover

    def __rmul__(self, e):
        return self.times(e)

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
            Boolean: True if the expression contains only discrete variables and coefficients.
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

LinearConstraintType = ComparisonType

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


class ZeroExpr(AbstractLinearExpr):

    def _get_solution_value(self):
        return 0

    def is_zero(self):
        return True

    # INTERNAL
    __slots__ = ()

    def __init__(self, model):
        ModelingObjectBase.__init__(self, model)

    def clone(self):
        return self  # this is not cloned.

    def copy(self, target_model, var_map):
        return ZeroExpr(target_model)

    def to_linear_expr(self):
        return self  # this is a linear expr.

    def number_of_variables(self):
        return 0

    def iter_variables(self):
        return iter_emptyset()

    def iter_terms(self):
        return iter_emptyset()

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


    def negate(self):
        pass

    # noinspection PyMethodMayBeStatic
    def plus(self, e):
        return e


    def times(self, _):
        return self

    # noinspection PyMethodMayBeStatic
    def minus(self, e):
        return -e
        # expr = e.to_linear_expr().clone()
        # expr.negate()
        # return expr

    def to_string(self, nb_digits=None, prod_symbol='', use_space=False):
        return "0"

    def to_stringio(self, oss, nb_digits, prod_symbol, use_space, var_namer=lambda v: v.name):
        oss.write(self.to_string())

    # arithmetic
    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        # e - 0 = e !
        return e

    def __neg__(self):
        return self

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
        return isinstance(other, ZeroExpr)

    def square(self):
        return self

    # arithmetci to self
    add = plus
    subtract = minus
    tmultiply = times

    def _scale(self, factor):
        return self
