# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
import sys

from docplex.mp.basic import ModelingObject, Priority
from docplex.mp.constants import ComparisonType
from docplex.mp.utils import is_number


class AbstractConstraint(ModelingObject):
    # noinspection PyRedundantParentheses
    __slots__ = ('_priority')

    def __init__(self, model, name=None):
        ModelingObject.__init__(self, model, name)
        self._priority = None

    def get_priority(self):
        return self._priority

    def set_priority(self, newprio):
        self._priority = Priority._parse(newprio, logger=self.error_handler, accept_none=True)

    priority = property(get_priority, set_priority)

    def set_mandatory(self):
        ''' Sets the constraint as mandatory.

        This prevents relaxation from relaxing this constraint.
        To revert this, set the priority to any non-mandatory priprity, or None.
        '''
        self._priority = Priority.MANDATORY

    def is_mandatory(self):
        return Priority.MANDATORY == self._priority


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

    def contains_var(self, dvar):
        return any(dvar is v for v in self.iter_variables())

    def copy(self, target_model, var_map):
        raise NotImplementedError  # pragma: no cover

    def compute_infeasibility(self, slack):
        raise NotImplementedError  # pragma: no cover

    # noinspection PyMethodMayBeStatic
    def notify_deleted(self):
        # INTERNAL
        self._index = self._invalid_index

    def short_typename(self):
        return "constraint"

    def is_trivial(self):
        return False

    def is_linear(self):
        return False

    def is_quadratic(self):
        return False

    @property
    def slack_value(self):
        """ Returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.

        Returns:
            The slack value of the constraint in the latest solve (a float value).
        """
        return self._model.slack_values(cts=self)


# noinspection PyAbstractClass
class BinaryConstraint(AbstractConstraint):
    __slots__ = ("_ctype", "_left_expr", "_right_expr")

    def __init__(self, model, left_expr, ctype, right_expr, name=None):
        AbstractConstraint.__init__(self, model, name)
        self._ctype = ctype
        # noinspection PyPep8
        self._left_expr  = left_expr
        self._right_expr = right_expr

    def get_constraint_type(self):
        return self._ctype

    @property
    def type(self):
        """ This property returns the type of the constraint; type is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

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
        left_string = self.left_expr.to_string()
        right_string = self.right_expr.to_string()
        return u"%s %s %s" % (left_string,
                              self._ctype.operator_symbol,
                              right_string)

    def rhs(self):
        right_cst = self._right_expr.get_constant()
        left_cst = self._left_expr.get_constant()
        return right_cst - left_cst

    def __repr__(self):
        classname = self.__class__.__name__
        user_name = self._get_safe_name()
        typename = self.type.short_name
        sleft = self._left_expr.truncated_str()
        sright = self._right_expr.truncated_str()
        return "docplex.mp.linear.{0}[{1}]({2!s},{3},{4!s})". \
            format(classname, user_name, sleft, typename, sright)

    def _is_trivially_feasible(self):
        # INTERNAL : assume self is trivial()
        op_func = self.type.python_operator
        return op_func(self.left_expr.constant, self.right_expr.constant) if op_func else False

    def _is_trivially_infeasible(self):
        # INTERNAL: assume self is trivial .
        op_func = self.type.python_operator
        return not op_func(self.left_expr.constant, self.right_expr.constant) if op_func else False

    def is_trivial_feasible(self):
        return self.is_trivial() and self._is_trivially_feasible()

    def is_trivial_infeasible(self):
        return self.is_trivial() and self._is_trivially_infeasible()

    def _generate_opposite_linear_coefs(self, expr):
        for v, k in expr.iter_terms():
            yield v, -k


    def _iter_net_linear_coefs2(self, left_expr, right_expr):
        # INTERNAL
        if right_expr.is_constant():
            return left_expr.iter_terms()
        elif left_expr.is_constant():
            return self._generate_opposite_linear_coefs(right_expr)
        else:
            return self._generate_net_linear_coefs2(left_expr, right_expr)

    def iter_variables(self):
        """  Iterates over all variables mentioned in the constraint.

        *Note:* This includes variables that are mentioned with a zero coefficient. For example,
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
            return self.generate_ordered_vars()

    def generate_ordered_vars(self):
        left_expr = self.left_expr
        for lv in left_expr.iter_variables():
            yield lv
        for rv in self.right_expr.iter_variables():
            if rv not in left_expr:
                yield rv

    def _generate_net_linear_coefs2(self, left_expr, right_expr):
        # INTERNAL
        for lv, lk in left_expr.iter_terms():
            net_k = lk - right_expr[lv]
            if 0 != net_k:
                yield lv, net_k
        for rv, rk in right_expr.iter_terms():
            if rv not in left_expr and 0 != rk:
                yield rv, -rk

    def _generate_net_linear_coefs(self):
        return self._generate_net_linear_coefs2(self._left_expr, self._right_expr)




class LinearConstraint(BinaryConstraint):
    """ The class that models all constraints of the form `<expr1> <OP> <expr2>`,
            where <expr1> and <expr2> are linear expressions.
    """
    __slots__ = ()

    def __init__(self, model, left_expr, ctype, right_expr, name=None):
        BinaryConstraint.__init__(self, model, left_expr, ctype, right_expr, name)

    def is_linear(self):
        return True

    def copy(self, target_model, var_map):
        copied_left = self.left_expr.copy(target_model, var_map)
        copied_right = self.right_expr.copy(target_model, var_map)
        copy_name = self.name if self.has_user_name() else None
        return LinearConstraint(target_model, copied_left, self.type, copied_right, copy_name)


    @property
    def type(self):
        """ This property returns the type of the constraint; type is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

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
        return BinaryConstraint.to_string(self)


    def compute_infeasibility(self, slack):
        ctype = self._ctype
        if ctype == ComparisonType.EQ:
            infeas = slack
        elif ComparisonType.LE == ctype:
            infeas = slack if slack <= 0 else 0
        elif ComparisonType.GE == ctype:
            infeas = slack if slack >= 0 else 0
        else:
            infeas = 0
        return infeas


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
                if self_left_expr.unchecked_get_coef(lv) != self_right_expr.unchecked_get_coef(lv):
                    return False
            for rv in self_right_expr.iter_variables():
                if self_left_expr.unchecked_get_coef(rv) != self_right_expr.unchecked_get_coef(rv):
                    return False
            return True


    def __le__(self, e):
        # INTERNAL: define ranges with operators.
        # Beware one must use parentheses as in r = (1 <= x) <= 2
        # Chained comparisons like: 1 <= x <= 2 will fail as Python
        # generates an "and" of two constraints (1<=x) and (x<=2) but
        # constraints _cannot_ be converted to booleans.
        if not is_number(e):
            self.fatal("operator <= on constraint requires numeric argument, got: {0!s}", e)
        if self.type is ComparisonType.GE:
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
        if self.type is ComparisonType.LE:
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

        *Note:* This includes variables that are mentioned with a zero coefficient. For example,
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
            return self.generate_ordered_vars()

    def generate_ordered_vars(self):
        left_expr = self.left_expr
        for lv in left_expr.iter_variables():
            yield lv
        for rv in self.right_expr.iter_variables():
            if rv not in left_expr:
                yield rv

    def fast_get_coef(self, dvar):
        return self._left_expr[dvar] - self._right_expr[dvar]


    def iter_net_linear_coefs(self):
        # INTERNAL
        left_expr = self._left_expr
        right_expr = self._right_expr
        if right_expr.is_constant():
            return left_expr.iter_terms()
        elif left_expr.is_constant():
            return self._generate_opposite_linear_coefs(right_expr)
        else:
            return self._generate_net_linear_coefs2(left_expr, right_expr)




class _DummyFeasibleConstraint(LinearConstraint):
    # INTERNAL
    def __init__(self, model, zero_expr, name=None):
        ctname = name or "dummy_feasible_ct"
        LinearConstraint.__init__(self, model,
                                  left_expr=zero_expr,
                                  ctype=ComparisonType.EQ,
                                  right_expr=zero_expr,
                                  name=ctname)

    def __repr__(self):
        return "docplex.mp.linear.LinearConstraint._TrivialFeasible"


class _DummyInfeasibleConstraint(LinearConstraint):
    # INTERNAL
    def __init__(self, model, zero, one):
        LinearConstraint.__init__(self, model,
                                  left_expr=zero,
                                  ctype=ComparisonType.EQ,
                                  right_expr=one,
                                  name="_dummy_infeasible_ct_")

    def __repr__(self):
        return "docplex.mp.linear.LinearConstraint.TrivialInfeasible"


class RangeConstraint(AbstractConstraint):
    """ This class models range constraints.

    A range constraint states that an expression must stay between two
    values, `lb` and `ub`.

    This class is not meant to be instantiated by the user.
    To create a range constraint, use the factory method :func:`docplex.mp.model.Model.add_range`
    defined on :class:`docplex.mp.model.Model`.

    """

    def __init__(self, model, expr, lb, ub, name=None):
        AbstractConstraint.__init__(self, model, name)
        model.typecheck_num(lb, 'RangeConstraint.lb')
        model.typecheck_num(ub, 'RangeConstraint.ub')
        self.__ub = ub
        self.__lb = lb
        self.__expr = expr

    def is_linear(self):
        return True

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

    def compute_infeasibility(self, slack):
        # compatible with cplex...
        return -slack


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

    def rhs(self):
        # INTERNAL
        return self.__lb - self.__expr.constant

    def fast_get_coef(self, dvar):
        # INTERNAL
        return self.__expr.unchecked_get_coef(dvar)

    def copy(self, target_model, var_map):
        copied_expr = self.expr.copy(target_model, var_map)
        copy_name = self.name if self.has_user_name() else None
        copied_range = RangeConstraint(target_model, copied_expr, self.lb, self.ub, copy_name)
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
        printable_name = self._get_safe_name()
        return "docplex.mp.linear.RangeConstraint[{0}]({1},{2!s},{3})".format(printable_name, self.lb, self.__expr,
                                                                              self.ub)


class IndicatorConstraint(AbstractConstraint):
    """ This class models indicator constraints.

    An indicator constraint links (one-way) the value of a binary variable to the satisfaction of a linear constraint.
    If the binary variable equals the active value, then the constraint is satisfied, but otherwise the constraint
    may or may not be satisfied.

    This class is not meant to be instantiated by the user.

    To create an indicator constraint, use the factory method :func:`docplex.mp.model.Model.add_indicator`
    defined on :class:`docplex.mp.model.Model`.

    """
    __slots__ = ('_binary_var', '_linear_ct', '_active_value')

    def __init__(self, model, binary_var, linear_ct, active_value=1, name=None):
        AbstractConstraint.__init__(self, model, name)
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
        copy_name = self.name if self.has_user_name() else None
        copied_indicator = IndicatorConstraint(target_model,
                                               copied_binary,
                                               copied_linear_ct,
                                               self.active_value,
                                               copy_name)
        return copied_indicator

    def invalidate(self):
        """
        Sets the binary variable to the opposite of its active value.
        Typically used by indicator constraints with a trivial infeasible linear part.
        For example, z=1 -> 4 <= 3 sets z to 0 and
        z=0 -> 4 <= 3 sets z to 1.
        This is equivalent to if z=a => False, then z *cannot* be equal to a.
        """
        if 0 == self.active_value:
            # set to 1 : lb = 1
            self.indicator_var.lb = 1
        elif 1 == self.active_value:
            # set to 0 ub = 0
            self.indicator_var.ub = 0
        else:
            self.fatal("Unexpected active value for indicator constraint: {0!s}, value is: {1!s}, expecting 0 or 1",  # pragma: no cover
                       self, self.active_value)  # pragma: no cover

    def iter_variables(self):
        yield self._binary_var
        for v in self._linear_ct.iter_variables():
            yield v

    def to_string(self):
        """
        Displays the indicator constraint in the LP style:
        z = 1 -> x+y+z == 2

        Returns:
            A string.
        """
        return "{0!s} = {1} -> {2!s}".format(self._binary_var, self.logical_rhs, self.linear_constraint)

    def __str__(self):
        return self.to_string()

    def compute_infeasibility(self, slack):
        pass


class QuadraticConstraint(BinaryConstraint):
    """ The class models quadratic constraints.

        Quadratic constraints are of the form `<qexpr1> <OP> <qexpr2>`,
        where at least one of <qexpr1> or <qexpr2> is a quadratic expression.

    """
    def copy(self, target_model, var_map):
        # noinspection PyPep8
        copied_left_expr  = self.left_expr.copy(target_model, var_map)
        copied_right_expr = self.right_expr.copy(target_model, var_map)
        copy_name = self.name if self.has_user_name() else None
        return QuadraticConstraint(target_model, copied_left_expr, self.type, copied_right_expr, copy_name)

    def compute_infeasibility(self, slack):
        pass

    def is_quadratic(self):
        return True

    __slots__ = ()

    def __init__(self, model, left_expr, ctype, right_expr, name=None):
        BinaryConstraint.__init__(self, model=model, left_expr=left_expr,
                                  ctype=ctype,
                                  right_expr=right_expr,
                                  name=name)

    def is_trivial(self):
        for qv, nqk in self.iter_net_quads():
            if 0 != nqk:
                return False
        # now check linear parts

        for lv, lk in self.iter_net_linear_coefs():
            if 0 != lk:
                return False
        return True

    def iter_net_linear_coefs(self):
        linear_left = self._left_expr.get_linear_part()
        linear_right = self._right_expr.get_linear_part()
        return self._iter_net_linear_coefs2(linear_left, linear_right)

    def iter_net_quads(self):
        # INTERNAL
        left_expr = self._left_expr
        right_expr = self._right_expr
        if not right_expr.is_quad_expr():
            return left_expr.iter_quads()
        elif not left_expr.is_quad_expr():
            return right_expr.iter_opposite_quads()
        else:
            return self.generate_ordered_net_quads(left_expr, right_expr)

    def generate_ordered_net_quads(self, qleft, qright):
        # left first, then right
        for lqv, lqk in qleft.iter_quads():
            net_k = lqk - qright._get_quadratic_coefficient_from_var_pair(lqv)
            if 0 != net_k:
                yield lqv, net_k
        for rqv, rqk in qright.iter_quads():
            if not qleft.contains_quad(rqv) and 0 != rqk:
                yield rqv, -rqk

