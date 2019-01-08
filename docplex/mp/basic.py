# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import sys
from io import StringIO

from enum import Enum

from docplex.mp.operand import Operand
from docplex.mp.utils import is_number, is_string, str_holo


class ModelingObjectBase(object):
    """ModelingObjectBase()

    Parent class for all modeling objects (variables and constraints).

    This class is not intended to be instantiated directly.
    """
    
    __array_priority__ = 100

    __slots__ = ("_name", "_model")

    def __init__(self, model, name=None):
        self._name = name
        self._model = model

    @property
    def name(self):
        """ This property is used to get or set the name of the modeling object.

        """
        return self._name

    @name.setter
    def name(self, new_name):
        self.set_name(new_name)

    def get_name(self):
        return self._name

    def set_name(self, name):
        self.check_name(name)
        self._set_name(name)

    def _set_name(self, name):
        self._name = name

    def _get_safe_name(self):
        # INTERNAL: always return a string
        return self._name or ''

    def check_name(self, new_name):
        # INTERNAL: basic method for checking names.
        pass  # pragma: no cover

    def has_name(self):
        """ Checks whether the object has a name.

        Returns:
            True if the object has a name.

        """
        return self._name is not None

    def has_user_name(self):
        """ Checks whether the object has a valid name given by the user.

        Returns:
            True if the object has a valid name given by the user.

        """
        return self.has_name()

    @property
    def model(self):
        """
        This property returns the :class:`docplex.mp.model.Model` to which the object belongs.
        """
        return self._model

    def _get_model(self):
        return self._model

    def get_linear_factory(self):
        return self._model._lfactory

    def get_quadratic_factory(self):
        return self._model._qfactory

    def is_in_model(self, model):
        return model and self._model is model

    def is_model_ordered(self):
        return self._model._keep_ordering

    def _check_model_has_solution(self):
        self.model.check_has_solution()

    def fatal(self, msg, *args):
        self.error_handler.fatal(msg, args)

    def error(self, msg, *args):
        self.error_handler.error(msg, args)

    def warning(self, msg, *args):
        self.error_handler.warning(msg, args)

    def trace(self, msg, *args):
        self.error_handler.trace(msg, args)

    @property
    def error_handler(self):
        return self._model.error_handler

    def truncated_str(self):
        return str_holo(self, maxlen=self._model._max_repr_len)

    def zero_expr(self):
        # INTERNAL
        return self._model._lfactory.new_zero_expr()

    def _unsupported_binary_operation(self, lhs, op, rhs):
        self.fatal("Unsupported operation: {0!s} {1:s} {2!s}", lhs, op, rhs)

    def is_quad_expr(self):
        return False

    def __unicode__(self):
        return self.to_string()

    def __str__(self):
        if sys.version_info[0] == 2:
            return self.__unicode__().encode('utf-8')
        else:  # pragma: no cover
            return self.__unicode__()


class ModelingObject(ModelingObjectBase):
    __slots__ = ("_index", "_origin", "_container")

    def is_valid_index(self, idx):
        # INTERNAL: This is where the valid index check is performed
        return idx >= 0

    _invalid_index = -2

    # @profile
    def __init__(self, model, name=None, index=_invalid_index):
        #  ModelingObjectBase.__init__(self, model, name)
        self._model = model
        self._name = name
        self._index = index

    def is_generated(self):
        """ Checks whether this object has been generated by another modeling object.

        If so, the origin object is stored in the ``_origin`` attribute.

        Returns:
            True if the objects has been generated.
        """
        return hasattr(self, '_origin') and self._origin is not None

    def notify_origin(self, origin):
        if origin is not None:
            self._origin = origin

    def origin(self):
        return getattr(self, '_origin', None)

    def __hash__(self):
        return id(self)

    @property
    def unchecked_index(self):
        return self._index

    def get_index(self):
        return self._index

    def set_index(self, idx):
        self._index = idx

    def has_valid_index(self):
        return self._index >= 0

    def _set_invalid_index(self):
        self._index = self._invalid_index

    @property
    def safe_index(self):
        if not self.has_valid_index():
            self.fatal("Modeling object {0!s} has invalid index: {1:d}", self, self._index)  # pragma: no cover
        return self._index

    index = property(get_index, set_index)

    def get_container(self):
        # INTERNAL
        return getattr(self, '_container', None)


class Expr(ModelingObjectBase, Operand):
    """Expr()

    Parent class for all expression classes.
    """
    __slots__ = ()

    def __init__(self, model, name=None):
        ModelingObjectBase.__init__(self, model, name)

    def clone(self):
        raise NotImplementedError  # pragma: no cover

    def copy(self, target_model, var_mapping):
        # internal
        raise NotImplementedError  # pragma: no cover

    def number_of_variables(self):
        """
        Returns:
            integer: The number of variables in the expression.
        """
        return sum(1 for _ in self.iter_variables())  # pragma: no cover

    def contains_var(self, dvar):
        """ Checks whether a variable is present in the expression.

        :param: dvar (:class:`docplex.mp.linear.Var`): A decision variable.

        Returns:
            Boolean: True if the variable is present in the expression, else False.
        """
        for v in self.iter_variables():
            if dvar is v:
                return True
        else:
            return False

    def to_string(self, nb_digits=None, use_space=False):
        oss = StringIO()
        if nb_digits is None:
            nb_digits = self.model.float_precision
        self.to_stringio(oss, nb_digits=nb_digits, use_space=use_space)
        return oss.getvalue()

    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.name):
        raise NotImplementedError  # pragma: no cover

    def _num_to_stringio(self, oss, num, ndigits=None, print_sign=False, force_plus=False, use_space=False):
        k = num
        if print_sign:
            if k < 0:
                sign = u'-'
                k = -k
            elif k > 0 and force_plus:
                # force a plus
                sign = u'+'
            else:
                sign = None
            if use_space:
                oss.write(u' ')
            if sign:
                oss.write(sign)
            if use_space:
                oss.write(u' ')
        # INTERNAL
        ndigits = ndigits or self.model.float_precision
        if k == int(k):
            oss.write(u'%d' % k)
        else:
            # use second arg as nb digits:
            oss.write(u"{0:.{1}f}".format(k, ndigits))

    # def __pos__(self):
    #     # + e is identical to e
    #     return self

    def is_discrete(self):
        raise NotImplementedError  # pragma: no cover

    def is_zero(self):
        return False

    constant = property(Operand.get_constant)

    @property
    def float_precision(self):
        return 0 if self.is_discrete() else self.model.float_precision

    def _round_if_discrete(self, raw_value):
        if self.is_discrete():
            return self.model.round_nearest(raw_value)
        else:
            return raw_value

    def _get_solution_value(self, s=None):
        # INTERNAL: compute solution value.
        raise NotImplementedError  # pragma: no cover

    @property
    def solution_value(self):
        self._check_model_has_solution()
        return self._get_solution_value()

    def __ne__(self, other):
        self.model.unsupported_neq_error(self, other)

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
        # redefine for each class of expression
        return None  # pragma: no cover

    def __gt__(self, e):
        """ The strict > operator is not supported
        """
        self.model.unsupported_relational_operator_error(self, ">", e)

    def __lt__(self, e):
        """ The strict < operator is not supported
        """
        self.model.unsupported_relational_operator_error(self, "<", e)


# --- Priority class used for relaxation
class Priority(Enum):
    """
    This enumerated class defines the priorities: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH, MANDATORY.
    """

    def __new__(cls, value, print_name):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = value
        obj._print_name = print_name
        return obj

    VERY_LOW = 100, 'Very Low'
    LOW      = 200, 'Low'
    MEDIUM   = 300, 'Medium'
    HIGH     = 400, 'High'
    VERY_HIGH = 500, 'Very High'
    MANDATORY = 999999999, 'Mandatory'

    @staticmethod
    def default_priority():
        return Priority.MEDIUM

    def __repr__(self):
        return 'Priority<{}>'.format(self.name)

    def print_name(self):
        return self._print_name

    def cplex_preference(self):
        return self._get_geometric_preference_factor(base=10.0)

    def _get_geometric_preference_factor(self, base):
        # INTERNAL: returns a CPLEX preference factor as a power of "base"
        # MEDIUM priority is the balance point with a preference of 1.
        assert is_number(base)
        medium_index = Priority.MEDIUM.value / 100
        if self.is_mandatory():
            return 1e+20
        else:
            # pylint complains about no value member but is wrong!
            diff = self.value / 100 - medium_index
            factor = 1.0
            pdiff = diff if diff >= 0 else -diff
            for _ in range(0, int(pdiff)):
                factor *= base
            return factor if diff >= 0 else 1.0 / factor

    def less_than(self, other):
        assert isinstance(other, Priority)
        return self.value < other.value

    def __lt__(self, other):
        return self.less_than(other)

    def __gt__(self, other):
        return other.less_than(self)

    def is_mandatory(self):
        return self == Priority.MANDATORY

    @classmethod
    def parse(cls, arg, logger, accept_none=True, do_raise=True):
        ''' Converts its argument to a ``Priority`` object.

        Returns `default_priority` if the text is not a string, empty, or does not match.

        Args;
            arg: The argument to convert.

            logger: An error logger

            accept_none: True if None is a possible value. Typically,
                Constraint.set_priority accepts None as a way to
                remove the constraint's own priority.

            do_raise: A Boolean flag indicating if an exception is to be raised if the value
                is not recognized.

        Returns:
            A Priority enumerated object.
        '''
        if isinstance(arg, cls):
            return arg
        elif is_string(arg):
            key = arg.lower()
            # noinspection PyTypeChecker
            for p in cls:
                if key == p.name.lower() or key == str(p.value):
                    return p
            else:
                if do_raise:
                    logger.fatal('String does not match priority type: {}', arg)
                else:
                    logger.error('String does not match priority type: {}', arg)
                    return None
                return None
        elif accept_none and arg is None:
            return None
        else:
            logger.fatal('Cannot convert to a priority: {0!s}'.format(arg))
# ---


# noinspection PyUnusedLocal,PyPropertyAccess

class _SubscriptionMixin(object):

    __slots__ = ()
    # INTERNAL:
    # This class is absolutely not meant to be directly instantiated
    # but used as a mixin

    @classmethod
    def _new_empty_subscribers(cls):
        return []

    def notify_used(self, user):
        # INTERNAL
        self._subscribers.append(user)

    def notify_unsubscribed(self, subscriber):
        # 1 find index
        sx = None
        for s, sc in enumerate(self._subscribers):
            if sc is subscriber:
                del self._subscribers[s]
                break

    def clear_subscribers(self):
        self._subscribers = []

    def is_in_use(self):
        return bool(self._subscribers)

    def is_used_by(self, obj):
        # lists are not optimal here, but we favor insertion: append is faster than set.add
        for sc in self._subscribers:
            if obj is sc:
                return True
        else:
            return False

    def notify_modified(self, event):
        for s in self._subscribers:
            s.notify_expr_modified(self, event)

    def iter_subscribers(self):
        return iter(self._subscribers)

    def notify_replaced(self, new_expr):
        for s in self._subscribers:
            s.notify_expr_replaced(self, new_expr)

    def grab_subscribers(self, other):
        # grab subscribers from another expression
        # typically when an expression is replaced by another.
        for s in other.iter_subscribers():
            self._subscribers.append(s)
        # delete all subscriptions on old
        other.clear_subscribers()


class _BendersAnnotatedMixin(object):
    __slots__ = ()

    def set_benders_annotation(self, group):
        self._model.set_benders_annotation(self, group)

    def get_benders_annotation(self):
        return self._model.get_benders_annotation(self)
