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
    def cplex_code(self):
        return self._cplex_code

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

    @classmethod
    def parse(cls, arg, do_raise=True):
        # INTERNAL
        # noinspection PyTypeChecker
        for cmp in cls:
            if arg == cmp or arg == cmp.value:
                return cmp
            elif is_string(arg):
                if arg == str(cmp.value) \
                        or arg.lower() == cmp.name.lower() \
                    or arg == cmp._cplex_code:
                    return cmp
        else:
            if do_raise:
                docplex_fatal('cannot convert this to a comparison type: {0!r}'.format(arg))
            else:
                return None

    @classmethod
    def cplex_ctsense_to_python_op(cls, cpx_sense):
        return cls.parse(cpx_sense).python_operator

    @classmethod
    def almost_compare(cls, lval, op, rval, eps):
        if op is cls.LE:
            # lval <= rval with eps tolerance means lval-rval <= e
            return lval -rval <= eps
        elif op is cls.GE:
            # lval >= rval with eps tolerance means lval-rval >= -eps
            return lval - rval >= -eps
        elif op is cls.EQ:
            return abs(lval - rval) <= eps
        else:
            raise TypeError



class RelaxationMode(Enum):
    """ This enumerated type describes the different strategies for model relaxation: MinSum, OptSum, MinInf, OptInf, MinQuad, OptQuad.

    Relaxation algorithms work in two phases: In the first phase, they attempt to find a
    feasible solution while making minimal changes to the model (according to a metric). In the second phase, they
    attempt to find an optimal solution while keeping the relaxation at the minimal value found in phase 1.

    Values of this type define two aspects of the algorithm:
        - whether or not they should continue to a second phase: all OptXxx values continue to a
          second phase, and MinXxx values stop at phase 1.
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


class ConflictStatus(Enum):
    """
    This enumerated class defines the conflict status types.
    """
    Excluded, Possible_member, Possible_member_lower_bound, Possible_member_upper_bound, \
        Member, Member_lower_bound, Member_upper_bound = -1, 0, 1, 2, 3, 4, 5


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


class SolveAttribute(Enum):
    duals = 1, False, True
    slacks = 2, False, True
    reduced_costs = 3, True, True

    def __new__(cls, code, is_for_vars, requires_solve):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.requires_vars = is_for_vars
        obj.requires_solve = requires_solve
        return obj

    @classmethod
    def parse(cls, arg, do_raise=True):
        # INTERNAL
        # noinspection PyTypeChecker
        for m in cls:
            if arg == m or arg == m.value:
                return m
            elif is_string(arg):
                if arg == str(m.value) or arg.lower() == m.name.lower():
                    return m
        else:
            if do_raise:
                docplex_fatal('cannot convert this to a solve attribute: {0!r}'.format(arg))
            else:
                return None


class UpdateEvent(Enum):
    # INTERNAL
    NoOp = 0
    #
    # Linear ct
    LinearConstraintCoef = 1
    LinearConstraintRhs = 2
    LinearConstraintGlobal = 3  # logical and of Coef + Rhs
    LinearConstraintType = 4

    # Range
    RangeConstraintBounds = 5
    RangeConstraintExpr = 6
    # Expr
    ExprConstant = 8
    LinExprCoef = 16
    LinExprGlobal = 24
    LinExprPromotedToQuad = 25  # objective is ok, constraint wont support this.

    # Quad
    QuadExprQuadCoef = 32
    QuadExprGlobal = 64

    # Ind
    IndicatorLinearConstraint = 128

    # Quadct
    QuadraticConstraintGlobal = 256

    def __bool__(self):
        return True if self.value else False


class ObjectiveSense(Enum):
    """
    This enumerated class defines the two types of objectives, `Minimize` and `Maximize`.
    """
    Minimize, Maximize = 1, 2

    def is_minimize(self):
        return self is ObjectiveSense.Minimize

    def is_maximize(self):
        return self is ObjectiveSense.Maximize

    @property
    def cplex_coef(self):
        return 1 if self.is_minimize() else -1

    def verb(self):
        # INTERNAL
        return "minimize" if self.is_minimize() else "maximize" if self.is_maximize() else "WHAT???"

    def action(self):
        # INTERNAL
        # minimize -> minimizing, maximize -> maximizing...
        return "%sing" % self.verb()[:-1]

    @staticmethod
    def parse(arg, logger=None, default_sense=None):
        if isinstance(arg, ObjectiveSense):
            return arg

        elif is_string(arg):
            lower_text = arg.lower()
            if lower_text in {"minimize", "min"}:
                return ObjectiveSense.Minimize
            elif lower_text in {"maximize", "max"}:
                return ObjectiveSense.Maximize
            elif default_sense:
                logger.error(
                    "Text is not recognized as objective sense: {0}, expecting \"min\" or \"max\" - using default {1:s}",
                    (arg, default_sense))
                return default_sense
            elif logger:
                logger.fatal("Text is not recognized as objective sense: {0}, expecting ""min"" or ""max", (arg,))
            else:
                docplex_fatal("Text is not recognized as objective sense: {0}, expecting ""min"" or ""max".format(arg))

        elif arg == 1:
            return ObjectiveSense.Minimize
        elif -1 == arg:
            return ObjectiveSense.Maximize

        elif default_sense:
            return default_sense
        elif logger:
            if not default_sense:
                logger.fatal("cannot convert: <{}> to objective sense", (arg,))
            else:
                logger.warning("cannot convert: <{0!r}> to objective sense - using default: {1!s}", (arg, default_sense))
        else:
            docplex_fatal("cannot convert: <{}> to objective sense".format(arg))


# noinspection PyPep8
class CplexScope(Enum):

    def __new__(cls, code, prefix):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.prefix = prefix
        return obj
    # INTERNAL
    VAR_SCOPE       = 0, 'x'
    LINEAR_CT_SCOPE = 1, 'c'
    IND_CT_SCOPE    = 2, 'ic'
    QUAD_CT_SCOPE   = 3, 'qc'
    PWL_CT_SCOPE    = 4, 'pwl'
    SOS_SCOPE       = 5, 'sos'
    UNKNOWN_SCOPE   = 9999, '?'