# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
from docplex.mp.basic import ModelingObject, Priority
from docplex.mp.constants import ComparisonType, UpdateEvent
from docplex.mp.utils import is_number, iter_emptyset


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

    def iter_exprs(self):
        # INTERNAL
        return iter_emptyset()

    def contains_var(self, dvar):
        return any(dvar is v for v in self.iter_variables())

    def copy(self, target_model, var_map):
        raise NotImplementedError  # pragma: no cover

    def compute_infeasibility(self, slack):  # pragma: no cover
        # INTERNAL: only used when json has no infeasibility info.
        return slack

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

    def _get_slack_value(self):
        return self._model.slack_values(cts=self)

    def _get_dual_value(self):
        # INTERNAL
        # Note that dual values are only avilable for LP problems,
        # so can be calle donly on linear or range constraints.
        return self._model.dual_values(cts=self)

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        pass  # pragma: no cover

    def notify_expr_replaced(self, expr, new_expr):
        # INTERNAL
        pass  # pragma: no cover

    def get_indicator(self):
        return getattr(self, '_ind', None)


# noinspection PyAbstractClass
class BinaryConstraint(AbstractConstraint):
    __slots__ = ("_ctsense", "_left_expr", "_right_expr")

    def _internal_set_sense(self, new_sense):
        self._ctsense = new_sense

    def __init__(self, model, left_expr, ctsense, right_expr, name=None):
        AbstractConstraint.__init__(self, model, name)
        self._ctsense = ctsense
        # noinspection PyPep8
        self._left_expr = left_expr
        self._right_expr = right_expr

    @property
    def type(self):
        """ This property returns the type of the constraint; type is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.

        """
        return self._ctsense

    def cplex_code(self):
        return self._ctsense._cplex_code

    def get_left_expr(self):
        """ This property returns the left expression in the constraint.

        Example:
            (X+Y <= Z+1) has left expression (X+Y).
        """
        return self._left_expr

    def get_right_expr(self):
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
        left_string = self._left_expr.to_string()
        right_string = self._right_expr.to_string()
        self_name = self.name
        if self_name:
            return u"%s: %s %s %s" % (self_name,
                                      left_string,
                                      self._ctsense.operator_symbol,
                                  right_string)
        else:
            return u"%s %s %s" % (left_string,
                                  self._ctsense.operator_symbol,
                                  right_string)


    def cplex_num_rhs(self):
        # INTERNAL
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
        return op_func(self._left_expr.get_constant(), self._right_expr.get_constant()) if op_func else False

    def _is_trivially_infeasible(self):
        # INTERNAL: assume self is trivial .
        op_func = self.type.python_operator
        return not op_func(self._left_expr.get_constant(), self._right_expr.get_constant()) if op_func else False

    def is_trivial_feasible(self):
        return self.is_trivial() and self._is_trivially_feasible()

    def is_trivial_infeasible(self):
        return self.is_trivial() and self._is_trivially_infeasible()

    def _generate_opposite_linear_coefs(self, expr):
        for v, k in expr.iter_sorted_terms():
            yield v, -k

    def _iter_net_linear_coefs2(self, left_expr, right_expr):
        # INTERNAL
        if right_expr.is_constant():
            return left_expr.iter_sorted_terms()
        elif left_expr.is_constant():
            return self._generate_opposite_linear_coefs(right_expr)
        else:
            return self._generate_net_linear_coefs2_sorted(left_expr, right_expr)

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
        left_expr = self._left_expr
        for lv in left_expr.iter_variables():
            yield lv
        for rv in self._right_expr.iter_variables():
            if not left_expr.contains_var(rv):
                yield rv

    @staticmethod
    def _generate_net_linear_coefs2_sorted(left_expr, right_expr):
        # INTERNAL
        for lv, lk in left_expr.iter_sorted_terms():
            net_k = lk - right_expr.unchecked_get_coef(lv)
            if net_k:
                yield lv, net_k
        for rv, rk in right_expr.iter_sorted_terms():
            if not left_expr.contains_var(rv) and rk:
                yield rv, -rk

    @staticmethod
    def _generate_net_linear_coefs2_unsorted(left_expr, right_expr):
        # INTERNAL
        for lv, lk in left_expr.iter_terms():
            net_k = lk - right_expr.unchecked_get_coef(lv)
            yield lv, net_k
        for rv, rk in right_expr.iter_terms():
            if not left_expr.contains_var(rv):
                yield rv, -rk

    def _generate_net_linear_coefs_sorted(self):
        return self._generate_net_linear_coefs2_sorted(self._left_expr, self._right_expr)

    def notify_deleted(self):
        # INTERNAL
        super(BinaryConstraint, self).notify_deleted()
        self._left_expr.notify_unsubscribed(self)
        self._right_expr.notify_unsubscribed(self)

    def iter_exprs(self):
        return iter([self._left_expr, self._right_expr])

    def get_expr_from_pos(self, pos):
        if pos is 0:
            return self._left_expr
        elif pos is 1:
            return self._right_expr
        else:  # pragma: no cover
            self.fatal('Unexpected expression position: {0!r}, expecting 0 or 1', pos)

    def set_expr_from_pos(self, pos, new_expr):
        if pos is 0:
            self._left_expr = new_expr
        elif pos is 1:
            self._right_expr = new_expr
        else:  # pragma: no cover
            self.fatal('Unexpected expression position: {0!r}, expecting 0 or 1', pos)


class LinearConstraint(BinaryConstraint):
    """ The class that models all constraints of the form `<expr1> <OP> <expr2>`,
            where <expr1> and <expr2> are linear expressions.
    """
    __slots__ = ('_ind',)  # for enclosing indicator if any

    def __init__(self, model, left_expr, ctsense, right_expr, name=None):
        BinaryConstraint.__init__(self, model, left_expr, ctsense, right_expr, name)
        left_expr.notify_used(self)
        right_expr.notify_used(self)

    def is_linear(self):
        return True

    def copy(self, target_model, var_map):
        copied_left = self.left_expr.copy(target_model, var_map)
        copied_right = self.right_expr.copy(target_model, var_map)
        copy_name = self.name if self.has_user_name() else None
        return LinearConstraint(target_model, copied_left, self.type, copied_right, copy_name)

    def get_sense(self):
        """ This property is used to get or set the sense of the constraint; sense is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.

        """
        return self._ctsense

    def set_sense(self, arg_newtype):
        self.get_linear_factory().set_linear_constraint_type(self, arg_newtype)

    sense = property(get_sense, set_sense)

    # compatibility
    type = property(get_sense, set_sense)

    def get_left_expr(self):
        """ This property returns the left expression in the constraint.

        Example:
            (X+Y <= Z+1) has left expression (X+Y).
        """
        return self._left_expr

    def get_right_expr(self):
        """ This property returns the right expression in the constraint.

        Example:
            (X+Y <= Z+1) has right expression (Z+1).
        """
        return self._right_expr

    def set_right_expr(self, new_rexpr):
        self.get_linear_factory().set_linear_constraint_right_expr(ct=self, new_rexpr=new_rexpr)

    right_expr = property(get_right_expr, set_right_expr)

    def set_left_expr(self, new_lexpr):
        self.get_linear_factory().set_linear_constraint_left_expr(ct=self, new_lexpr=new_lexpr)

    left_expr = property(get_left_expr, set_left_expr)

    # aliases
    lhs = left_expr
    rhs = right_expr

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        if event:
            if event is UpdateEvent.LinExprPromotedToQuad:
                self.fatal('Cannot change constraint from linear to quadratic: {0}', expr)
            else:
                self.get_linear_factory().update_linear_constraint_exprs(ct=self)

    def notify_expr_replaced(self, old_expr, new_expr):
        # INTERNAL
        # TODO: quads are not allowed here...
        if old_expr is self._left_expr:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(lct=self, pos=0, new_expr=new_expr, update_subscribers=False)
        elif old_expr is self._right_expr:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(lct=self, pos=1, new_expr=new_expr, update_subscribers=False)
        else:
            # should not happen
            pass
        # new expr takes al subscribers from old expr
        new_expr.grab_subscribers(old_expr)

    def to_string(self):
        """ Returns a string representation of the constraint.

        The operators in this representation are the usual operators <=, ==, and >=.

        Example:
            The constraint (X+Y <= Z+1) is represented as "X+Y <= Z+1".

        Returns:
            A string.

        """
        return BinaryConstraint.to_string(self)

    def compute_infeasibility(self, slack):  # pragma: no cover
        ctsense = self._ctsense
        if ctsense == ComparisonType.EQ:
            infeas = slack
        elif ComparisonType.LE == ctsense:
            infeas = slack if slack <= 0 else 0
        elif ComparisonType.GE == ctsense:
            infeas = slack if slack >= 0 else 0
        else:
            infeas = 0
        return infeas

    def _get_index_scope(self):
        return self._model._linct_scope

    def is_trivial(self):
        # Checks whether the constraint is equivalent to a comparison between numbers.
        # For example, x <= x+1 is trivial, but 1.5 X <= X + 1 is not.
        self_left_expr = self._left_expr
        self_right_expr = self._right_expr
        if self_right_expr.is_constant():
            for rv, rk in self_left_expr.iter_terms():
                if rk:
                    return False
            else:
                return True

        elif self_left_expr.is_constant():
            for lv, lk in self_left_expr.iter_terms():
                if lk:
                    return False
            else:
                return True
        else:
            for _, nk in BinaryConstraint._generate_net_linear_coefs2_unsorted(self_left_expr, self_right_expr):
                if nk:
                    return False
            else:
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

    @property
    def dual_value(self):
        """ This property returns the dual value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._get_dual_value()

    @property
    def slack_value(self):
        """ This property returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._model.slack_values(cts=self)

    def generate_ordered_vars(self):
        # INTERNAL
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
            return left_expr.iter_sorted_terms()
        elif left_expr.is_constant():
            return self._generate_opposite_linear_coefs(right_expr)
        else:
            return self._generate_net_linear_coefs2_sorted(left_expr, right_expr)


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
        self._ub = ub
        self._lb = lb
        self._expr = expr

    def is_linear(self):
        return True

    def cplex_code(self):
        return 'R'

    def _get_index_scope(self):
        return self._model._linct_scope

    def short_typename(self):
        return "range"

    def is_trivial(self):
        return self._expr.is_constant()

    def _is_trivially_feasible(self):
        # INTERNAL : assume self is trivial()
        expr_num = self._expr.constant
        return self._lb <= expr_num <= self._ub

    def _is_trivially_infeasible(self):
        # INTERNAL : assume self is trivial()
        expr_num = self._expr.constant
        return expr_num < self._lb or expr_num > self._ub

    def compute_infeasibility(self, slack):  # pragma: no cover
        # compatible with cplex...
        return -slack

    def get_expr(self):
        """ This property returns the linear expression of the range constraint.
        """
        return self._expr

    def set_expr(self, new_expr):
        self.get_linear_factory().set_range_constraint_expr(self, new_expr)

    expr = property(get_expr, set_expr)

    def get_lb(self):
        """ This property returns the lower bound of the range constraint.

        """
        return self._lb

    def set_lb(self, new_lb):
        self._model.typecheck_num(new_lb)
        self.get_linear_factory().set_range_constraint_lb(self, new_lb)

    lb = property(get_lb, set_lb)

    def get_ub(self):
        """ This property returns the upper bound of the range constraint.

        """
        return self._ub

    def set_ub(self, new_ub):
        self._model.typecheck_num(new_ub)
        self.get_linear_factory().set_range_constraint_ub(self, new_ub)

    ub = property(get_ub, set_ub)

    def _internal_set_lb(self, new_lb):
        self._lb = new_lb

    def _internal_set_ub(self, new_ub):
        self._ub = new_ub

    def is_feasible(self):
        return self._ub >= self._lb

    @property
    def dual_value(self):
        """ This property returns the dual value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._get_dual_value()

    @property
    def slack_value(self):
        """ This property returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._model.slack_values(cts=self)

    def iter_variables(self):
        """Iterates over all the variables of the range constraint.

        Returns:
           An iterator object.
        """
        return self._expr.iter_variables()

    def iter_exprs(self):
        yield self._expr

    def cplex_num_rhs(self):
        # INTERNAL
        return self._lb - self._expr.get_constant()

    def copy(self, target_model, var_map):
        copied_expr = self.expr.copy(target_model, var_map)
        copy_name = self.name if self.has_user_name() else None
        copied_range = RangeConstraint(target_model, copied_expr, self.lb, self.ub, copy_name)
        return copied_range

    def to_string(self):
        return "{0} <= {1!s} <= {2}".format(self._lb, self._expr, self._ub)

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
        return "docplex.mp.linear.RangeConstraint[{0}]({1},{2!s},{3})".format(printable_name, self.lb, self._expr,
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
        # connect exprs
        for expr in linear_ct.iter_exprs():
            expr.notify_used(self)
        linear_ct._ind = self

    def _get_index_scope(self):
        return self._model._indicator_scope

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
            self.fatal("Unexpected active value for indicator constraint: {0!s}, value is: {1!s}, expecting 0 or 1",
                       # pragma: no cover
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

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        self.get_linear_factory().update_indicator_constraint_expr(self, event, expr)


class QuadraticConstraint(BinaryConstraint):
    """ The class models quadratic constraints.

        Quadratic constraints are of the form `<qexpr1> <OP> <qexpr2>`,
        where at least one of <qexpr1> or <qexpr2> is a quadratic expression.

    """

    def copy(self, target_model, var_map):
        # noinspection PyPep8
        copied_left_expr = self.left_expr.copy(target_model, var_map)
        copied_right_expr = self.right_expr.copy(target_model, var_map)
        copy_name = self.name if self.has_user_name() else None
        return QuadraticConstraint(target_model, copied_left_expr, self.type, copied_right_expr, copy_name)

    def is_quadratic(self):
        return True

    def _get_index_scope(self):
        return self._model._quadct_scope

    __slots__ = ()

    def __init__(self, model, left_expr, ctsense, right_expr, name=None):
        BinaryConstraint.__init__(self, model=model, left_expr=left_expr,
                                  ctsense=ctsense,
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
            return left_expr.iter_sorted_quads()
        elif not left_expr.is_quad_expr():
            return right_expr.iter_opposite_ordered_quads()
        else:
            return self.generate_ordered_net_quads(left_expr, right_expr)

    def generate_ordered_net_quads(self, qleft, qright):
        # left first, then right
        for lqv, lqk in qleft.iter_sorted_quads():
            net_k = lqk - qright._get_quadratic_coefficient_from_var_pair(lqv)
            if 0 != net_k:
                yield lqv, net_k
        for rqv, rqk in qright.iter_sorted_quads():
            if not qleft.contains_quad(rqv) and 0 != rqk:
                yield rqv, -rqk

    def _set_left_expr(self, new_left_expr):
        self.get_quadratic_factory().set_quadratic_constraint_expr_from_pos(self, pos=0, new_expr=new_left_expr)

    left_expr = property(BinaryConstraint.get_left_expr, _set_left_expr)

    def _set_right_expr(self, new_right_expr):
        self.get_quadratic_factory().set_quadratic_constraint_expr_from_pos(self, pos=1, new_expr=new_right_expr)

    right_expr = property(BinaryConstraint.get_right_expr, _set_right_expr)

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        self.get_quadratic_factory().update_quadratic_constraint(self, expr, event)

    @property
    def slack_value(self):
        """ This property returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._model.slack_values(cts=self)

    def get_sense(self):
        """ This property is used to get or set the sense of the constraint; sense is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote quadratic expressions.

        """
        return self._ctsense

    def set_sense(self, arg_newsense):
        self.get_quadratic_factory().set_quadratic_constraint_type(self, arg_newsense)

    sense = property(get_sense, set_sense)

    # compat
    type = property(get_sense, set_sense)

    def has_net_quadratic_term(self):
        # INTERNAL
        return any(nk for _, nk in self.iter_net_quads())


class PwlConstraint(AbstractConstraint):
    """ This class models piecewise linear constraints.

    This class is not meant to be instantiated by the user.
    To create a piecewise constraint, use the factory method :func:`docplex.mp.model.Model.piecewise`
    defined on :class:`docplex.mp.model.Model`.

    """

    def __init__(self, model, pwl_expr, name=None):
        AbstractConstraint.__init__(self, model, name)
        self.__pwl_expr = pwl_expr
        self.__pwl_func = pwl_expr.pwl_func
        self.__expr = pwl_expr._x_var
        self.__y = None
        self.__usage_counter = pwl_expr.usage_counter

    @property
    def expr(self):
        """ This property returns the linear expression of the piecewise linear constraint.
        """
        return self.__expr

    @property
    def pwl_func(self):
        """ This property returns the piecewise linear function of the piecewise linear constraint.
        """
        return self.__pwl_func

    @property
    def y(self):
        """ This property returns the output variable associated with the piecewise linear constraint.
        """
        if self.__y is None:
            self.__y = self.__pwl_expr.functional_var
        return self.__y

    @property
    def usage_counter(self):
        """ This property returns the usage counter of the piecewise linear function associated with the
        piecewise linear constraint.
        """
        return self.__usage_counter

    def _get_index_scope(self):
        return self._model._pwl_scope

    def iter_variables(self):
        """Iterates over all the variables of the piecewise linear constraint.

        Returns:
           An iterator object.
        """
        y = self.y
        yield y
        for v in self.expr.iter_variables():
            if v is not y:
                yield v

    def copy(self, target_model, var_map):
        # Internal: copy must not be invoked on PwlConstraint.
        raise NotImplementedError  # pragma: no cover

    def notify_deleted(self):
        # INTERNAL
        super(PwlConstraint, self).notify_deleted()
        self.model._allpwl.remove(self)

    def to_string(self):
        return "{0} == {1!s}({2!s})".format(self.y, self.pwl_func.get_name(), self.expr)

    def __str__(self):
        """ Returns a string representation of the piecewise linear constraint.

        Example:
            `y == pwl_name(x + z)` represents the piecewise linear constraint where the variable `y` is
            constrained to be equal to the value of the piecewise linear function whose name is 'pwl_name'
            applied to the expression (x + z).

        Returns:
            A string.
        """
        return self.to_string()

    def __repr__(self):
        printable_name = self._get_safe_name()
        return "docplex.mp.linear.PwlConstraint[{0}]({1},{2!s},{3})".format(printable_name, self.y,
                                                                            self.pwl_func, self.expr)
