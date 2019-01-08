# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore


from docplex.mp.utils import is_int, is_number, is_iterable, is_string

from docplex.mp.basic import Priority
from docplex.mp.vartype import VarType
from docplex.mp.linear import Var, Expr
from docplex.mp.constr import AbstractConstraint
from docplex.mp.progress import ProgressListener

import math


class IDocplexTypeChecker(object):

    def typecheck_iterable(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_valid_index(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_vartype(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_var(self, obj):
        raise NotImplementedError  # pragma: no cover

    def typecheck_operand(self, obj):
        raise NotImplementedError  # pragma: no cover

    def typecheck_constraint(self, obj):
        raise NotImplementedError  # pragma: no cover

    def typecheck_linear_constraint(self, obj):
        raise NotImplementedError  # pragma: no cover

    def typecheck_zero_or_one(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_int(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_num(self, arg, caller=None):
        raise NotImplementedError  # pragma: no cover

    def typecheck_string(self, arg, accept_empty=False, accept_none=False):
        raise NotImplementedError  # pragma: no cover

    def typecheck_priority(self, prio):
        raise NotImplementedError  # pragma: no cover

    def typecheck_as_power(self, e, power):
        raise NotImplementedError  # pragma: no cover

    def typecheck_in_model(self, model, mobj, header=''):
        raise NotImplementedError  # pragma: no cover

    def cannot_be_used_as_denominator_error(self, denominator, numerator):
        raise NotImplementedError  # pragma: no cover

    def typecheck_as_denominator(self, denominator, numerator):
        raise NotImplementedError  # pragma: no cover

    def typecheck_key_seq(self, keys, accept_empty_seq=False):
        raise NotImplementedError  # pragma: no cover

    def to_valid_number(self, e, checked_num=False, context_msg=None, infinity=1e+20):
        raise NotImplementedError  # pragma: no cover

    def typecheck_progress_listener(self, arg):
        raise NotImplementedError  # pragma: no cover


# noinspection PyAbstractClass
class DOcplexLoggerTypeChecker(IDocplexTypeChecker):
    def __init__(self, logger):
        self._logger = logger


    def fatal(self, msg, *args):
        self._logger.fatal(msg, args)

    def error(self, msg, *args):
        self._logger.error(msg, args)

    def warning(self, msg, *args):
        self._logger.warning(msg, args)

    def info(self, msg, *args):
        self._logger.info(msg, args)


class DOcplexTypeChecker(DOcplexLoggerTypeChecker):


    def __init__(self, logger):
        DOcplexLoggerTypeChecker.__init__(self, logger)

    def typecheck_iterable(self, arg):
        # INTERNAL: checks for an iterable
        if not is_iterable(arg):
            self.fatal("Expecting iterable, got: {0!s}", arg)

    # safe checks.
    def typecheck_valid_index(self, arg):
        if arg < 0:
            self.fatal("Invalid index: {0!s}", arg)

    def typecheck_vartype(self, arg):
        # INTERNAL: check for a valid vartype
        if not isinstance(arg, VarType):
            self.fatal("Not a variable type: {0!s}, type: {1!s}", arg, type(arg))
        return True

    def typecheck_var(self, obj):
        # INTERNAL: check for Var instance
        if not isinstance(obj, Var):
            self.fatal("Expecting decision variable, got: {0!s} type: {1!s}", obj, type(obj))


    def typecheck_constraint(self, obj):
        if not isinstance(obj, AbstractConstraint):
            self.fatal("Expecting constraint, got: {0!s} with type: {1!s}", obj, type(obj))

    def typecheck_linear_constraint(self, obj):
        if not isinstance(obj, AbstractConstraint):
            self.fatal("Expecting constraint, got: {0!s} with type: {1!s}", obj, type(obj))
        if not obj.is_linear():
            self.fatal("Expectinglinear constraint, got: {0!s} with type: {1!s}", obj, type(obj))

    def typecheck_zero_or_one(self, arg):
        if arg != 0 and arg != 1:
            self.fatal("expecting 0 or 1, got: {0!s}", arg)


    def typecheck_int(self, arg):
        if not is_int(arg):
            self.fatal('Expecting integer, got: {0!s}', arg)


    def typecheck_num(self, arg, caller=None):
        caller_string = "{0}: ".format(caller) if caller is not None else ""
        if not is_number(arg):
            self.fatal("{0}Expecting number, got: {1!r}", caller_string, arg)
        elif math.isnan(arg):
            self.fatal("{0}NaN value detected", caller_string)


    def typecheck_string(self, arg, accept_empty=False, accept_none=False):
        if is_string(arg):
            if not accept_empty and 0 == len(arg):
                self.fatal("A nonempty string is not allowed here")
        elif arg is None and not accept_none:
            self.fatal("expecting string, got: None")

    def typecheck_priority(self, prio):
        if not isinstance(prio, Priority):
            self.fatal('expecting priority, got: {0!r}'.format(prio))

    def typecheck_as_power(self, e, power):
        # INTERNAL: checks <power> is 0,1,2
        if power < 0 or power > 2:
            self.fatal("Cannot raise {0!s} to the power {1}. A variable's exponent must be 0, 1 or 2.", e, power)

    def typecheck_in_model(self, model, mobj, header=''):
        # produces message of the type: "constraint ... does not belong to model
        if mobj.model != model:
            self.fatal("{0}{2!s} does not belong to model {1}: {2!s}".format(header, model.name, mobj))

    def cannot_be_used_as_denominator_error(self, denominator, numerator):
        self.fatal("{1!s} / {0!s} : operation not supported, only numbers can be denominators", denominator, numerator)

    def typecheck_as_denominator(self, denominator, numerator):
        if not is_number(denominator):
            self.cannot_be_used_as_denominator_error(denominator, numerator)
        else:
            # float_e = float(denominator)
            if 0 == denominator:
                self.fatal("Zero divide on {0!s}", numerator)


    def typecheck_key_seq(self, keys, accept_empty_seq=False):
        if not keys:
            if accept_empty_seq:
                return []
            else:
                self.fatal("No keys to index the variables.")
        else:
            if any(k is None for k in keys):
                self.fatal("A variable key cannot be None, see: {0!s}", keys)


    def to_valid_number(self, e, checked_num=False, context_msg=None, infinity=1e+20):
        if not checked_num and not is_number(e):
            self.fatal("Not a number: {}".format(e))
        elif math.isnan(e):
            msg = "NaN value found in expression"
            try:
                msg = "{0}: {1}".format(context_msg(), msg)
            except TypeError:
                msg = "{0}: {1}".format(context_msg, msg)
            self.fatal(msg)
        elif -infinity <= e <= infinity:
            return e
        elif e >= infinity:
            return infinity
        else:
            return -infinity


    @staticmethod
    def _is_operand(arg, accept_numbers=True):
        return isinstance(arg, (Expr, Var)) or (accept_numbers and is_number(arg))


    def typecheck_operand(self, arg, caller=None, accept_numbers=True):
        if not self._is_operand(arg, accept_numbers=accept_numbers):
            caller_str = "{0}: ".format(caller) if caller else ""
            accept_str = "Expecting expr/var"
            if accept_numbers:
                accept_str += "/number"
            self.fatal("{0}{1}, got: {2!r}", caller_str, accept_str, arg)

    def typecheck_progress_listener(self, arg):
        if not isinstance(arg, ProgressListener):
            self.fatal('not  aprogress listener: {0!r}', arg)


class DummyTypeChecker(IDocplexTypeChecker):
    # noinspection PyUnusedLocal
    def __init__(self, logger):
        pass

    def typecheck_iterable(self, arg):
        pass  # pragma: no cover

    def typecheck_valid_index(self, arg):
        pass  # pragma: no cover

    def typecheck_vartype(self, arg):
        pass  # pragma: no cover

    def typecheck_var(self, obj):
        pass  # pragma: no cover

    def typecheck_operand(self, obj):
        pass  # pragma: no cover

    def typecheck_constraint(self, obj):
        pass  # pragma: no cover

    def typecheck_linear_constraint(self, obj):
        pass  # pragma: no cover

    def typecheck_zero_or_one(self, arg):
        pass  # pragma: no cover

    def typecheck_int(self, arg):
        pass  # pragma: no cover

    def typecheck_num(self, arg, caller=None):
        pass  # pragma: no cover

    def typecheck_string(self, arg, accept_empty=False, accept_none=False):
        pass  # pragma: no cover

    def typecheck_priority(self, prio):
        pass  # pragma: no cover

    def typecheck_as_power(self, e, power):
        pass  # pragma: no cover

    def typecheck_in_model(self, model, mobj, header=''):
        pass  # pragma: no cover

    def cannot_be_used_as_denominator_error(self, denominator, numerator):
        pass  # pragma: no cover

    def typecheck_as_denominator(self, denominator, numerator):
        pass  # pragma: no cover

    def typecheck_key_seq(self, keys, accept_empty_seq=False):
        pass  # pragma: no cover

    def to_valid_number(self, e, checked_num=False, context_msg=None, infinity=1e+20):
        # returns raw argument
        return e

    def typecheck_progress_listener(self, arg):
        pass  # pragma: no cover


_tck_map = {'default': DOcplexTypeChecker,
            'none': DummyTypeChecker}


def get_typechecker(arg, logger):
    key = arg.lower() if arg else 'default'
    checker_type = _tck_map.get(key, DOcplexTypeChecker)
    return checker_type(logger)
