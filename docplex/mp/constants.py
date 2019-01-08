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
    LE = 1, '<=', 'L', operator.le
    EQ = 2, '==', 'E', operator.eq
    GE = 3, '>=', 'G', operator.ge

    def __new__(cls, code, operator_symbol, cplex_kw, python_op):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj._cplex_code = cplex_kw
        obj._op_symbol = operator_symbol
        obj._pyop = python_op
        return obj

    # NOTE: Never add a static field in an enum class: it would be interpreted as an other enum

    @property
    def short_name(self):
        return self.name

    @property
    def operator_symbol(self):
        """ Returns a string operator for the constraint.

        Example:
            Returns string "<=" for a e1 <= e2 constraint.

        Returns:
            string: A string describing the logical operator used in the constraint.
        """
        return self._op_symbol

    @property
    def python_operator(self):
        return self._pyop


class RelaxationMode(Enum):
    """ This enumerated type describes the different strategies for model relaxation: MinSum, OptSum, MinInf, OptInf, MinQuad, OptQuad.

    Relaxation algorithms work in two phases: in the first phase, they attempt to find a
    feasible solution while making minimal changes to the model (according to a metric). In a second phase, they
    attempt to find an optimal solution while keeping the relaxation at the minimal value found in phase 1.

    Values of this type define two aspects of the algorithm:
     - whether or not they should continue to a second phase: all OptXxx values continue to a second phase, and
       MinXxx values stop at phase 1.
    - which metric to use for evaluating the relaxation in the first phase. There are three metrics:
        - the sum of relaxations for OptSum, MinSum, or
        - the total number of constraints being relaxed for OptInf, MinInf, or
        - the sum of squares of relaxations for OptQuad, MinQuad.

    """

    MinSum, OptSum, MinInf, OptInf, MinQuad, OptQuad = range(6)


    @staticmethod
    def parse(arg):
        # INTERNAL
        # noinspection PyTypeChecker
        for m in RelaxationMode:
            if arg == m or arg == m.value:
                return m
            elif is_string(arg):
                if arg == str(m.value) or arg.lower() == m.name.lower():
                    return m
        else:
            docplex_fatal('cannot parse this as a relaxation mode: {0!r}'.format(arg))


    @staticmethod
    def get_no_optimization_mode(mode):
        assert isinstance(mode, RelaxationMode)
        # even values are MinXXX modes
        relax_code = mode.value
        if 0 == relax_code % 2:
            return mode
        else:
            # OptXXX is 2k+1 when MinXXX is 2k
            return RelaxationMode(relax_code - 1)

    def __repr__(self):
        return 'docplex.mp.RelaxationMode.{0}'.format(self.name)





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
    SOS1, SOS2 = 1, 2

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
