# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
from docplex.mp.utils import is_number

import math

class StaticTypeChecker(object):

    @staticmethod
    def typecheck_as_power(mdl, e, power):
        # INTERNAL: checks <power> is 0,1,2
        if power < 0 or power > 2:
            mdl.fatal("Cannot raise {0!s} to the power {1}. A variable's exponent must be 0, 1 or 2.", e, power)

    @staticmethod
    def cannot_be_used_as_denominator_error(mdl, denominator, numerator):
        mdl.fatal("{1!s} / {0!s} : operation not supported, only numbers can be denominators", denominator, numerator)

    @classmethod
    def typecheck_as_denominator(cls, mdl, denominator, numerator):
        if not is_number(denominator):
            cls.cannot_be_used_as_denominator_error(mdl, denominator, numerator)
        else:
            float_e = float(denominator)
            if 0 == float_e:
                mdl.fatal("Zero divide on {0!s}", numerator)

    @classmethod
    def typecheck_discrete_expression(cls, logger, expr, msg):
        if not expr.is_discrete():
            logger.fatal('{0}, expression: ({1!s}) is not discrete', msg, expr)

    @classmethod
    def typecheck_discrete_constraint(cls, logger, ct, msg):
        if not ct.is_discrete():
            logger.fatal('{0}, {1!s} is not discrete', msg, ct)

    @classmethod
    def typecheck_added_constraint(cls, mdl, ct):
        if not ct.has_valid_index():
            mdl.fatal("Constraint: {0!s} has not been added to any model".format(ct))
        elif mdl is not ct.model:
            mdl.fatal("Constraint: {0!s} belongs to a different model".format(ct))

    @classmethod
    def mul_quad_lin_error(cls, logger, f1, f2):
        logger.fatal(
            "Cannot multiply {0!s} by {1!s}, some terms would have degree >= 3. Maximum polynomial degree is 2.",
            f1, f2)

    @classmethod
    def typecheck_callable(cls, logger, arg, msg):
        if not callable(arg):
            logger.fatal(msg)

    @classmethod
    def typecheck_num_nan_inf(cls, logger, arg, caller=None):
        # check for a "real" number, not a NaN, not infinity
        caller_string = "{0}: ".format(caller) if caller is not None else ""
        if not is_number(arg):
            logger.fatal("{0}Expecting number, got: {1!r}", caller_string, arg)
        elif math.isnan(arg):
            logger.fatal("{0}NaN value detected", caller_string)
        elif math.isinf(arg):
            logger.fatal("{0}Infinite value detected", caller_string)