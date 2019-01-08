# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from docplex.mp.utils import is_int
from docplex.mp.error_handler import docplex_fatal

class VarType(object):
    """VarType()

    This abstract class is the parent class for all types of decision variables.

    This class must never be instantiated.
    Specialized sub-classes are defined for each type of decision variable.

    """

    def __init__(self, short_name, lb, ub, cplex_typecode):
        self._short_name = short_name
        self._lb = lb
        self._ub = ub
        self._cpx_typecode = cplex_typecode

    def get_cplex_typecode(self):
        # INTERNAL
        return self._cpx_typecode

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

    def get_default_lb(self):
        return self._lb

    @property
    def default_ub(self):
        """  This property returns the default upper bound for the type.
        """
        return self._ub

    def get_default_ub(self):
        return self._ub

    def resolve_lb(self, candidate_lb):
        if candidate_lb is None:
            return self._lb
        else:
            return self.compute_lb(candidate_lb)

    def resolve_ub(self, candidate_ub):
        if candidate_ub is None:
            return self._ub
        else:
            return self.compute_ub(candidate_ub)

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
        # """ Checks if the `numeric_value` is valid for the type.
        #
        # Accepted values depend on the type:
        #
        # - Binary type accepts only 0 or 1.
        #
        # - Integer type accepts only integers.
        #
        # - Continuous type accepts any floating-point number within -inf and +inf, where inf
        #   is the model's infinity value.
        #
        # This method never raises an exception.
        #
        # Args:
        #     numeric_value: The candidate value.
        #
        # Returns:
        #     Boolean: True if the candidate value is valid, else False.
        #
        # """
        raise NotImplementedError  # pragma: no cover

    def accept_domain_value(self, candidate_value, lb, ub):
        # INTERNAL: check that a value is OK w.r.t the ttype and a domain [lb,ub]
        return self.accept_value(candidate_value) and (lb <= candidate_value <= ub)

    def to_string(self):
        """
        Returns:
            string: A string representation of the type.
        """
        return "VarType_%s" % self.short_name

    def __str__(self):
        return self.to_string()

    def is_default_lb(self, domain_lb):
        return domain_lb == self._lb

    def is_default_ub(self, domain_ub):
        # INTERNAL
        # use equality here: output user data whenever different from default
        return domain_ub == self._ub

    def one_letter_symbol(self):
        # INTERNAL: returns B,I,C
        return self.get_cplex_typecode()

    def __eq__(self, other):
        return type(other) == type(self)

    def __ne__(self, other):
        return type(other) != type(self)

    def hash_vartype(self):   # pragma: no cover
        return hash(self.get_cplex_typecode())


class BinaryVarType(VarType):
    """BinaryVarType()

        This class models the binary variable type and
        is not meant to be instantiated. Each model contains one instance of
        this type.
    """

    def __init__(self):
        VarType.__init__(self, short_name="binary", lb=0, ub=1, cplex_typecode='B')

    def compute_lb(self, candidate_lb):
        # INTERNAL
        return 0 if candidate_lb <= 0 else 1

    def resolve_ub(self, candidate_ub):
        if candidate_ub is None:
            return 1
        elif candidate_ub >= 1:
            return 1
        else:
            return 0

    def resolve_lb(self, candidate_ub):
        if candidate_ub is None:
            return 0
        elif candidate_ub <= 0:
            return 0
        else:
            return 1

    def compute_ub(self, candidate_ub):
        # INTERNAL
        return 1 if candidate_ub >= 1 else 0

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True as this is a discrete type.
        """
        return True

    def accept_value(self, numeric_value):
        # """ Checks if `numeric_value` equals 0 or 1.
        #
        # Args:
        #     numeric_value: The candidate value.
        #
        # Returns:
        #     Boolean: True if `numeric_value` equals 0 or 1.
        # """
        return 0 == numeric_value or 1 == numeric_value

    def __hash__(self):  # pragma: no cover
        return VarType.hash_vartype(self)


class ContinuousVarType(VarType):
    """ContinuousVarType()

        This class models the continuous variable type and
        is not meant to be instantiated. Each model contains one instance of this type.
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="float", lb=0, ub=plus_infinity, cplex_typecode='C')
        self._plus_infinity = plus_infinity
        self._minus_infinity = - plus_infinity

    def compute_ub(self, candidate_ub):
        if candidate_ub >= self._plus_infinity:
            return self._plus_infinity
        else:
            return candidate_ub

    def compute_lb(self, candidate_lb):
        if 0 == candidate_lb:
            return 0
        elif candidate_lb <= self._minus_infinity:
            return self._minus_infinity
        else:
            return candidate_lb

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: False because this type is not a discrete type.
        """
        return False

    def accept_value(self, numeric_value):
        # """ Checks if the value is within the minus infinity to positive infinity range.
        #
        # Args:
        #     numeric_value: The candidate value.
        #
        # Returns:
        #     Boolean: True if the candidate value is a valid floating-point number
        #     with respect to the model's infinity.
        # """
        return self._minus_infinity <= numeric_value <= self._plus_infinity

    def __hash__(self):  # pragma: no cover
        return VarType.hash_vartype(self)


class IntegerVarType(VarType):
    """IntegerVarType()
    This class models the integer variable type and
    is not meant to be instantiated. Each models contains one instance
    of this type.

    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="int", lb=0, ub=plus_infinity, cplex_typecode='I')
        self._plus_infinity = plus_infinity
        self._minus_infinity = -plus_infinity

    def compute_ub(self, candidate_ub):
        if candidate_ub >= self._plus_infinity:
            return self._plus_infinity
        else:
            return candidate_ub

    def compute_lb(self, candidate_lb):
        if candidate_lb <= self._minus_infinity:
            return self._minus_infinity
        else:
            return candidate_lb

    def is_discrete(self):
        """  Checks if this is a discrete type.

        Returns:
            Boolean: True as this is a discrete type.
        """
        return True

    def accept_value(self, numeric_value):
        # """ Redefines the generic `accept_value` method.
        #
        # A value is valid if is an integer and belongs to the variable's domain.
        #
        # Args:
        #     numeric_value: The numeric value being tested.
        #
        # Returns:
        #     True if the value is valid for the type, else False.
        # """
        return is_int(numeric_value) or numeric_value == int(numeric_value)

    def __hash__(self):  # pragma: no cover
        return VarType.hash_vartype(self)


class SemiContinuousVarType(VarType):
    """SemiContinuousVarType()

            This class models the :index:`semi-continuous` variable type and
            is not meant to be instantiated. 
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="semi", lb=1e-6, ub=plus_infinity, cplex_typecode='S')
        self._plus_infinity = plus_infinity

    def compute_ub(self, candidate_ub):
        return self._plus_infinity if candidate_ub >= self._plus_infinity else float(candidate_ub)

    def compute_lb(self, candidate_lb):
        if candidate_lb <= 0:
            docplex_fatal(
                'semi-continuous variable expects strict positive lower bound, not: {0}'.format(candidate_lb))
        return candidate_lb

    def is_default_lb(self, domain_lb):
        # any lb is nondefault
        return False

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: False because this type is not a discrete type.
        """
        return False

    def accept_value(self, numeric_value):
        # """ Checks if the value is within the minus infinity to positive infinity range.
        #
        # Args:
        #     numeric_value: The candidate value.
        #
        # Returns:
        #     Boolean: True if the candidate value is a valid floating-point number
        #     with respect to the model's infinity.
        # """
        return 0 <= numeric_value <= self._plus_infinity

    def accept_domain_value(self, candidate_value, lb, ub):
        # INTERNAL: check that a value is OK w.r.t the ttype and a domain [lb,ub]
        return 0 == candidate_value or (lb <= candidate_value <= ub)

    def __hash__(self):  # pragma: no cover
        return VarType.hash_vartype(self)


class SemiIntegerVarType(VarType):
    """SemiIntegerVarType()

            This class models the :index:`semi-integer` variable type and
            is not meant to be instantiated.
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="semi", lb=1e-6, ub=plus_infinity, cplex_typecode='N')
        self._plus_infinity = plus_infinity

    def compute_ub(self, candidate_ub):
        return self._plus_infinity if candidate_ub >= self._plus_infinity else float(candidate_ub)

    def compute_lb(self, candidate_lb):
        if candidate_lb <= 0:
            docplex_fatal('semi-integer variable expects strict positive lower bound, not: {0}'.format(candidate_lb))
        return candidate_lb

    def is_default_lb(self, domain_lb):
        # any lb is nondefault
        return False

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True because this type is an integer type.
        """
        return True

    def accept_value(self, numeric_value):
        if 0 == numeric_value:
            return True
        return numeric_value >= 0 and (is_int(numeric_value) or numeric_value == int(numeric_value))

    def accept_domain_value(self, candidate_value, lb, ub):
        # INTERNAL: check that a value is OK w.r.t the ttype and a domain [lb,ub]
        return 0 == candidate_value or (lb <= candidate_value <= ub)

    def __hash__(self):  # pragma: no cover
        return VarType.hash_vartype(self)
