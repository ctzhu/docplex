# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import math

from docplex.mp.utils import is_int


class VarType(object):
    """VarType()

    This abstract class is the parent class for all types of decision variables.

    This class must never be instantiated.
    Specialized sub-classes are defined for each type of decision variable.

    """

    def __init__(self, short_name, lb, ub):
        self._short_name = short_name
        self._lb = lb
        self._ub = ub

    @property
    def short_name(self):
        """ This property returns a short name string for the type.
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
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True if the type is a discrete type.
        """
        raise NotImplementedError  # pragma: no cover

    def accept_value(self, numeric_value):
        """ Checks if the `numeric_value` is valid for the type.

        Accepted values depend on the type:

        - Binary type accepts only 0 or 1.

        - Integer type accepts only integers.

        - Continuous type accepts any floating-point number within -inf and +inf, where inf
          is the model's infinity value.

        This method never raises an exception.

        Args:
            numeric_value: The candidate value.

        Returns:
            Boolean: True if the candidate value is valid, else False.

        """
        raise NotImplementedError  # pragma: no cover

    def to_string(self):
        """
        Returns:
            string: A string representation of the type.
        """
        return "VarType_%s" % self.short_name

    def __str__(self):
        return self.to_string()

    def is_default_domain(self, domain_lb, domain_ub):
        # INTERNAL
        # returns True if [domain_lb..domain_ub] is identical to the vartype's default domain.
        return self._lb == domain_lb and self._ub == domain_ub

    def one_letter_symbol(self):
        # INTERNAL: returns B,I,C
        raise NotImplementedError

    def __eq__(self, other):
        return type(other) == type(self)

    def __ne__(self, other):
        return type(other) != type(self)

    def hash_vartype(self):
        return hash(self.one_letter_symbol())


class BinaryVarType(VarType):
    """BinaryVarType()

        This class models the binary variable type and
        is not meant to be instantiated. Each model contains one instance of
        this type.
    """

    def __init__(self):
        VarType.__init__(self, short_name="binary", lb=0, ub=1)

    def compute_lb(self, candidate_lb):
        # INTERNAL
        return 0

    def compute_ub(self, candidate_ub):
        # INTERNAL
        return 1

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True as this is a discrete type.
        """
        return True

    def accept_value(self, numeric_value):
        """ Checks if `numeric_value` equals 0 or 1.

        Args:
            numeric_value: The candidate value.

        Returns:
            Boolean: True if `numeric_value` equals 0 or 1.
        """
        return 0 == numeric_value or 1 == numeric_value

    def one_letter_symbol(self):
        return "B"

    def __hash__(self):
        return VarType.hash_vartype(self)


class ContinuousVarType(VarType):
    """ContinuousVarType()

        This class models the continuous variable type and
        is not meant to be instantiated. Each model contains one instance of this type.
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="float", lb=0, ub=plus_infinity)
        self._plus_infinity = plus_infinity
        self._minus_infinity = - plus_infinity

    def compute_ub(self, candidate_ub):
        return self._plus_infinity if candidate_ub > self._plus_infinity else float(candidate_ub)

    def compute_lb(self, candidate_lb):
        return self._minus_infinity if candidate_lb < self._minus_infinity else float(candidate_lb)

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: False as this type is not a discrete type.
        """
        return False

    def accept_value(self, numeric_value):
        """ Checks if the value is within the minus infinity to positive infinity range.

        Args:
            numeric_value: The candidate value.

        Returns:
            Boolean: True if the candidate value is a valid floating-point number
            with respect to the model's infinity.
        """
        return self._minus_infinity <= numeric_value <= self._plus_infinity

    def one_letter_symbol(self):
        return "C"

    def __hash__(self):
        return VarType.hash_vartype(self)


class IntegerVarType(VarType):
    """IntegerVarType()
    This class models the integer variable type and
    is not meant to be instantiated. Each models contains one instance
    of this type.

    """

    def __init__(self, plus_infinity=1e+20):
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
        """  Checks if this is a discrete type.

        Returns:
            Boolean: True as this is a discrete type.
        """
        return True

    def accept_value(self, numeric_value):
        """ Redefines the generic `accept_value` method.

        A value is valid if is an integer and belongs to the variable's domain.

        Args:
            numeric_value: The numeric value being tested.

        Returns:
            True if the value is valid for the type, else False.
        """
        return is_int(numeric_value) or numeric_value == int(numeric_value)

    def one_letter_symbol(self):
        return "I"

    def __hash__(self):
        return VarType.hash_vartype(self)
