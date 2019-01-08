# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from enum import Enum

from docplex.mp.error_handler import docplex_fatal
from docplex.mp.utils import is_string

import operator


class ComparisonType(Enum):
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
        return _ComparisonUtils.get_name(self)

    def get_operator_symbol(self):
        """ Returns a string operator for the constraint.

        Example:
            Returns string "<=" for a e1 <= e2 constraint.

        Returns:
            string: A string describing the logical operator used in the constraint.
        """
        return _ComparisonUtils.get_operator(self)


class _ComparisonUtils(object):
    # INTERNAL.
    # raison d'etre: cannot add static fields in an enumerated type, so we deport them elsewhere

    @staticmethod
    def _get_map_value(cttype, attribute_map, default_value=None):
        if cttype not in attribute_map:
            docplex_fatal("unexpected constraint type: {0!s}".format(cttype))  # pragma: no cover

        return attribute_map.get(cttype, default_value)

    _operator_symbol_map = {ComparisonType.LE: "<=",
                            ComparisonType.EQ: "==",
                            ComparisonType.GE: ">="}
    _name_map = {ComparisonType.LE: "LE",
                 ComparisonType.EQ: "EQ",
                 ComparisonType.GE: "GE"}

    _python_op_map = {ComparisonType.LE: operator.le,
                      ComparisonType.EQ: operator.eq,
                      ComparisonType.GE: operator.ge}

    @staticmethod
    def get_operator(cttype):
        return _ComparisonUtils._get_map_value(cttype, _ComparisonUtils._operator_symbol_map)

    @staticmethod
    def get_name(cttype):
        return _ComparisonUtils._get_map_value(cttype, _ComparisonUtils._name_map)

    @staticmethod
    def get_python_operator_fn(cttype):
        """
        Returns the Python operator function associated with the ct.
        For example, returns operator.ge(a, b) for a GE constraint.
        :param cttype: A binary constraint type (LE, EQ, GE.
        :return: A Python operator function.
        """
        return _ComparisonUtils._get_map_value(cttype, _ComparisonUtils._python_op_map)



class RelaxerMode(Enum):
    MinSum, OptSum, MinInf, OptInf = range(4)


    @staticmethod
    def compute_mode(optimize, use_sum_infeas=True):
        # which forced mode for feasopt? switch this flag for testing
        # with 12.6.2 and nurse, INF is very slow...
        if use_sum_infeas:
            if optimize:
                new_mode = RelaxerMode.OptSum
            else:
                new_mode = RelaxerMode.MinSum
        else:
            if optimize:  # pragma: no cover
                new_mode = RelaxerMode.MinSum  # pragma: no cover
            else:  # pragma: no cover
                new_mode = RelaxerMode.MinInf  # pragma: no cover
        return new_mode


# class ConstraintType(Enum):
#     Variable_lower_bound, Variable_upper_bound, Linear_constraint, Quadratic_constraint, \
#         Special_ordered_set_constraint, Indicator_constraint = 1, 2, 3, 4, 5, 6


class ConflictStatus(Enum):
    """
    This enumerated class defines the conflict status types
    """
    Excluded, Possible_member, Possible_member_lower_bound, Possible_member_upper_bound, \
        Member, Member_lower_bound, Member_upper_bound = -1, 0, 1, 2, 3, 4, 5


class CplexCtSenseToPython(object):
    # INTERNAL
    sense_map = {'G': operator.ge, 'L': operator.le, 'E': operator.eq}

    @staticmethod
    def cplex_ctsense_to_python_op(cpx_sense, _sense_map=sense_map):
        return _sense_map[cpx_sense]


class SOSType(Enum):
    """This enumerated class defines the SOS types:

        - SOS1 for SOS type 1

        - SOS1 for SOS type 2.
    """
    SOS1, SOS2 = 1,2

    def lower(self):
        return self.name.lower()

    @staticmethod
    def parse(arg,
              sos1_tokens=frozenset(['1', 'sos1']),
              sos2_tokens=frozenset(['2', 'sos2'])):
        if isinstance(arg, SOSType):
            return arg
        elif 1 == arg:
            return SOSType.SOS1
        elif 2 == arg:
            return SOSType.SOS2
        elif is_string(arg):
            arg_lower = arg.lower()
            if arg_lower in sos1_tokens:
                return SOSType.SOS1
            elif arg_lower in sos2_tokens:
                return SOSType.SOS2

        docplex_fatal("Cannot convert to SOS type: {0!s} - expecting 1|2|'sos1'|'sos2'", arg)

    def _cpx_sos_type(self):
        # INTERNAL
        return str(self.value)

    def min_size(self):
        # INTERNAL
        return self.value + 1

    def __repr__(self):
        return 'docplex.mp.SOSType.{0}'.format(self.name)
