# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.utils import iter_emptyset, is_number
from docplex.mp.constants import ComparisonType


class Operand(object):
    __slots__ = ()

    def get_constant(self):
        return 0

    def is_constant(self):
        return False

    # --- basic subscription api
    def notify_used(self, user):
        pass

    def notify_unsubscribed(self, subscriber):
        pass

    def is_in_use(self):
        return False

    def notify_modified(self, event):
        pass
    # ---

    def get_linear_part(self):
        return self

    def __le__(self, rhs):
        return self._model._qfactory.new_xconstraint(lhs=self, rhs=rhs, comparaison_type=ComparisonType.LE)

    def __eq__(self, rhs):
        return self._model._qfactory.new_xconstraint(lhs=self, rhs=rhs, comparaison_type=ComparisonType.EQ)

    def __ge__(self, rhs):
        return self._model._qfactory.new_xconstraint(lhs=self, rhs=rhs, comparaison_type=ComparisonType.GE)

    le = __le__
    eq = __eq__
    ge = __ge__


class LinearOperand(Operand):
    # no ctor as used in multiple inheritance
    __slots__ = ()


    def unchecked_get_coef(self, dvar):
        raise NotImplementedError  # pragma: no cover

    def iter_variables(self):
        """
        Iterates over all variables in the expression.

        Returns:
            iterator: An iterator over all variables present in the operand.
        """
        raise NotImplementedError  # pragma: no cover

    def iter_terms(self):
        # iterates over alllinear terms, if any
        return iter_emptyset()

    iter_sorted_terms = iter_terms

    def number_of_terms(self):
        return sum(1 for _ in self.iter_terms())

    def iter_quads(self):
        return iter_emptyset()

    def is_constant(self):
        # redefine this for subclasses.
        return False  # pragma: no cover

    def is_variable(self):
        # return True if the expression is infact one variable.
        # if True, assume you can replace the expression by the first variable
        # returned by iter_variables()
        return False

    def is_zero(self):
        return False

    # no strict comparisons
    def __lt__(self, e):
        self.model.unsupported_relational_operator_error(self, "<", e)

    def __gt__(self, e):
        self.model.unsupported_relational_operator_error(self, ">", e)

    def __contains__(self, dvar):
        """Overloads operator `in` for an expression and a variable.

        :param: dvar (:class:`docplex.mp.linear.Var`): A decision variable.

        Returns:
            Boolean: True if the variable is present in the expression, else False.
        """
        return self.contains_var(dvar)

    def contains_var(self, dvar):
        raise NotImplementedError  # pragma: no cover
