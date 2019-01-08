# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Generated automatically

"""
This module contains the functions that allows to construct all operations
and constraints that can be used in a CPO model.

There is one Python function per CPO operation or constraint.
As many operations have multiple combinations of parameters,
the functions of this module are declared with a variable number of arguments.
The valid combinations are detailed in the description of the function.
"""

from docplex.cp.catalog import *
from docplex.cp.expression import CpoExpr
from docplex.cp.expression import _create_operation as create_op
import collections

try:
    import __builtin__ as builtin  # Python 2.7
except ImportError:
    import builtins as builtin     # Python 3


###############################################################################
##  Private methods
###############################################################################

def _expand(arg):
    """ Expand an argument if it is an iterator
    Args:
        arg: Argument to check
    Returns:
        Argument, expanded as list if it is an iterator (recursively)
    """
    if isinstance(arg, collections.Iterator):
       return [_expand(x) for x in arg]
    return arg

def _no_cpo_args(largs):
    """ Check if a list of arguments does not contain any CPO expression
    Returns:
        True if list of arguments does not contain any CPO expression
    """
    for x in largs:
        if isinstance(x, CpoExpr) or isinstance(x, (list, tuple)) and all(isinstance(v, CpoExpr) for v in x):
            return False
    return True


###############################################################################
##  Expression construction methods
###############################################################################


def abs(*args):
    """ Computes the absolute value of an expression.

    Function *abs* computes the absolute value of an integer or floating-point expression *x*.  The type of the function is
    the same as the type of its argument.  *abs(x)* is a more efficient way of writing *max(x, -x)*.

    Args:
        x: Integer or floating-point expression for which the absolute value is to be computed.

    Possible argument and return type combinations are:

     * (integer expression) => integer expression
     * (float expression) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    args = [_expand(x) for x in args]
    if _no_cpo_args(args):
        return builtin.abs(*args)
    return create_op(Oper_abs, args)


def abstraction(*args):
    """ Returns a constraint that abstracts the values of one array as values in another array.

    For constraint programming: returns a constraint that abstracts the
    values of expressions contained in one array to expressions contained in
    another array.
    
    This function returns a constraint that abstracts the values of the
    elements of one array of expressions (called *x*) in a model into the
    abstract value of another array of expressions (called *y*). In other
    words, for each element *x[i]*, there is an expression *y[i]*
    corresponding to the abstraction of *x[i]* with respect to an array of
    numeric *values*. That is:
    
     * *x[i] = v* with *v* in *values* if and only if *y[i] = v*
     * *x[i] = v* with *v* not in *values* if and only if *y[i] = abstractValue*
    
    This constraint maintains a many-to-one mapping that makes it possible to
    define constraints that impinge only on a particular set of values from the
    domains of expressions. The abstract value (specified by
    *abstractValue*) must not be in the domain of *x[i]*.

    Args:
        y: An array of abstracted integer expressions.
        x: An array of reference integer expressions.
        values: An array of integer values to be abstracted.
        abstractValue: An escape value.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of integer expressions, array of integers, integer constant) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_abstraction, args)


def all_diff(*args):
    """ Ensures a number of expressions contain no duplicate values.

    The *all_diff* function returns a constraint which ensures that no
    two expressions in the array *x* can have equivalent values.

    Args:
        x: An array of integer expressions.

    Possible argument and return type combinations are:

     * (array of integer expressions) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_all_diff, args)


def all_min_distance(*args):
    """ Constraint on the minimum absolute distance between a pair of integer expressions in an array.

    This constraint makes sure that the absolute distance between any pair
    of integer expressions in *exprs* will be greater than or equal to the
    given integer *distance*. In short, for any *i*, *j* distinct indices of *exprs* , it
    enforces *abs(exprs[i] - exprs[j]) >= distance*.

    Args:
        exprs: Array of integer expressions.
        distance: Value used to constrain the distance between two elements of exprs.

    Possible argument and return type combinations are:

     * (array of integer expressions, integer constant) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_all_min_distance, args)


def allowed_assignments(*args):
    """ Explicitly defines possible assignments on one or more integer expressions.

    This Boolean expression (which is interpreted as a constraint outside of
    an expression) determines whether the assignment to a single expression *expr*
    or to an array of expressions *exprs* is contained within the value set
    *values* or the tuple set *tuples* respectively.  The Boolean expression
    will be true if and only if (depending on the signature):
     * the value of the expression *expr* is present in the array *values*.
     * the values of the expressions *exprs* are present in the tuple set *tuples*.
    
    The order of the constrained variables in the array *exprs* is
    important because the same order is respected in the tuple set *tuples*.

    Args:
        expr: An integer expression to be constrained.
        values: An integer array giving the possible values of expr.
        exprs: An array of integer expressions to be constrained.
        tuples: The set of tuples defining possible value assignments to exprs.

    Possible argument and return type combinations are:

     * (integer expression, array of integers) => boolean expression
     * (array of integer expressions, set of tuples) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_allowed_assignments, args)


def alternative(*args):
    """ Creates an alternative constraint between interval variables.

    This function creates an alternative constraint between interval variable
    *interval* and the set of interval variables in *array*. If no *cardinality*
    expression is specified, if *interval* is present, then one and only
    one of the intervals in *array* will be selected by the alternative constraint
    to be present and the start and end values of *interval* will be the same as the
    ones of the selected interval. If a *cardinality* expression is
    specified, *cardinality* intervals in *array* will be selected by the
    alternative constraint to be present and the selected intervals will have the
    same start and end value as interval variable *interval*. Interval variable *interval* is
    absent if and only if all interval variables in *array* are absent.

    Args:
        interval: Interval variable.
        array: Array of interval variables.
        cardinality: Cardinality of the alternative constraint. By default, when this optional argument is not specified, a unit cardinality is assumed (cardinality=1).

    Possible argument and return type combinations are:

     * (interval variable, array of interval variables, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_alternative, args)


def always_constant(*args):
    """ This constraint ensures a constant state for a state function on an interval.

    This function returns a constraint that ensures that *function* is defined everywhere on the interval (either interval
    variable *interval* when it is present or fixed interval [*start*,*end*)) and remains constant over this interval.
    
    Generally speaking, the optional Boolean values *isStartAligned* and *isEndAligned* allow synchronization of start and
    end with the intervals of the state function:
     * When *isStartAligned* is true, it means that start must be the start of an interval of the state function.
     * When *isEndAligned* is true, it means that end must be the end of an interval of the state function.

    Args:
        function: Constrained state function.
        interval: Interval variable during which function is constrained.
        start: Start of the fixed interval [start,end) during which function is constrained.
        end: End of the fixed interval [start,end) during which function is constrained.
        isStartAligned: Boolean flag that states whether the interval is start aligned (default: no alignment).
        isEndAligned: Boolean flag that states whether the interval is end aligned (default: no alignment).

    Possible argument and return type combinations are:

     * (state function, interval variable, boolean integer (0, 1) [=0], boolean integer (0, 1) [=0]) => constraint
     * (state function, integer time, integer time, boolean integer (0, 1) [=0], boolean integer (0, 1) [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_always_constant, args)


def always_equal(*args):
    """ This constraint fixes a given state for a state function during a variable or fixed interval.

    This function returns a constraint that ensures that *function* is defined everywhere on the interval (either interval
    variable *interval* when it is present or fixed interval [*start*,*end*)) and remains equal to value *val* over this
    interval.
    
    Generally speaking, the optional Boolean values *isStartAligned* and *isEndAligned* allow synchronization of start and
    end with the intervals of the state function:
     * When *isStartAligned* is true, it means that start must be the start of an interval of the state function.
     * When *isEndAligned* is true, it means that end must be the end of an interval of the state function.

    Args:
        function: Constrained state function.
        interval: Interval variable during which function is constrained.
        start: Start of the fixed interval [start,end) during which function is constrained.
        end: End of the fixed interval [start,end) during which function is constrained.
        val: Value of function during the interval.
        isStartAligned: Boolean flag that states whether the interval is start aligned (default: no alignment).
        isEndAligned: Boolean flag that states whether the interval is end aligned (default: no alignment).

    Possible argument and return type combinations are:

     * (state function, interval variable, positive integer, boolean integer (0, 1) [=0], boolean integer (0, 1) [=0]) => constraint
     * (state function, integer time, integer time, positive integer, boolean integer (0, 1) [=0], boolean integer (0, 1) [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_always_equal, args)


def always_in(*args):
    """ These constraints restrict the possible values of a *cumulExpr* or
    *stateFunction* to a particular range during a variable or fixed interval.

    These functions return a constraints that restricts the possible values of *function* to a particular range
    [*min*,*max*] during an interval variable *interval* or a fixed interval [*start*,*end*). In the case of an interval
    variable *interval*, this constraint is active only when the interval variable is present, if the interval is absent the
    constraint is always satisfied, regardless of the value of *function*. When the constraint is posted on a state
    function, the range constraint holds only on the segments where the state function is defined.

    Args:
        function: Constrained cumul expression or state function.
        interval: Interval variable during which function is constrained.
        start: Start of the fixed interval [start,end) during which function is constrained.
        end: End of the fixed interval [start,end) during which function is constrained.
        min: Minimum of the allowed range for values of function during the interval.
        max: Maximum of the allowed range for values of function during the interval.

    Possible argument and return type combinations are:

     * (cumul expression, interval variable, positive integer, positive integer) => constraint
     * (cumul expression, integer time, integer time, positive integer, positive integer) => constraint
     * (state function, interval variable, positive integer, positive integer) => constraint
     * (state function, integer time, integer time, positive integer, positive integer) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_always_in, args)


def always_no_state(*args):
    """ This constraint ensures that a state function is undefined on an interval.

    This function returns a constraint that ensures that *function* is undefined everywhere on the interval (either interval
    variable *interval* when it is present or fixed interval [*start*,*end*)). This constraint will ensure, in particular,
    that no interval variable that requires the function to be defined (see *always_equal*, *always_constant*) can overlap
    with interval variable *interval* or fixed interval [*start*,*end*)).

    Args:
        function: Constrained state function.
        interval: Interval variable during which function is constrained.
        start: Start of the fixed interval [start,end) during which function is constrained.
        end: End of the fixed interval [start,end) during which function is constrained.

    Possible argument and return type combinations are:

     * (state function, interval variable) => constraint
     * (state function, integer time, integer time) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_always_no_state, args)


def before(*args):
    """ Constrains an interval variable to be before another interval variable in a sequence.

    This function returns a constraint that states that whenever both interval variables *interval1* and *interval2* are
    present,
    *interval1* must be ordered before *interval2* in the sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval1: First interval variables.
        interval2: Second interval variables.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, interval variable) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_before, args)


def bool_abstraction(*args):
    """ Creates a constraint that abstracts the values of one array as Boolean values in another array.

    This function creates and returns a constraint that abstracts an array
    of integer expressions in a model. It differs from *abstraction* in
    that elements each *y[i]* is Boolean.
    
    Like *abstraction*, for each element *x[i]* there is an expression *y[i]*
    corresponding to the abstraction of *x[i]* with respect to the
    *values* array. That is,
     * *x[i] = v* with *v* in *values* if and only if *y[i] = true()*
     * *x[i] = v* with *v* not in *values* if and only if *y[i] = false()*
    
    This constraint maintains a many-to-one mapping that makes it possible
    to define constraints that impinge only on a particular set of values
    from the domains of constrained variables.

    Args:
        y: An array of abstracted integer expressions.
        x: An array of reference integer expressions.
        values: An array of integer values to abstract.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of integer expressions, array of integers) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_bool_abstraction, args)


def constant(*args):
    """ Creates an expression for operation *constant*.

    Possible argument and return type combinations are:

     * (integer constant) => integer constant
     * (float constant) => float constant

    Returns:
        An expression of type float constant or integer constant
    """
    return create_op(Oper_constant, args)


def coordinate_piecewise_linear(*args):
    """ Creates an expression for operation *coordinatePiecewiseLinear*.

    Possible argument and return type combinations are:

     * (float expression, float constant, array of floats, array of floats, float constant) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_coordinate_piecewise_linear, args)


def count(*args):
    """ Returns the number of occurrences of a given value found in a given integer expression array.

    This expression counts how many of the expressions in *exprs* take the value *v*.

    Args:
        exprs: An array of integer expressions.
        v: The value for which occurrences must be counted.

    Possible argument and return type combinations are:

     * (array of integer expressions, integer constant) => integer expression
     * (integer constant, array of integer expressions) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_count, args)


def count_different(*args):
    """ Creates an expression for operation *countDifferent*.

    Possible argument and return type combinations are:

     * (array of integer expressions) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_count_different, args)


def cumul_range(*args):
    """ Limits the range of a cumul function expression.

    This function returns a constraint that restricts the possible values of cumul *function* to belong to a range
    [*min*,*max*].

    Args:
        function: Cumul function expression.
        min: Minimum of the range of allowed values for the cumul function.
        max: Maximum of the range of allowed values for the cumul function.

    Possible argument and return type combinations are:

     * (cumul expression, integer expression, integer expression) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_cumul_range, args)


def diff(*args):
    """ Creates an expression for operation *diff*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => boolean expression
     * (float expression, float expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_diff, args)


def distribute(*args):
    """ Calculates and/or constrains the distribution of values taken by an array
    of integer expressions.

    The *distribute* constraint is used to count the number of occurrences of
    several values in an array of constrained expressions. You can also
    use *distribute* to force a set of constrained expressions to assume
    values in such a way that only a limited number of the constrained
    expressions can assume each value.
    
    More precisely, for any index *i* of *counts*, *counts[i]* is equal to
    the number of expressions in *exprs* who have value of *values[i]*.
    When using the signature which has *values* missing, then the values
    counted are assumed to be a set spanning from 0 up to the size of the
    *counts* array, less one.

    Args:
        counts: An array of integer expressions representing, for each element of values, its cardinality in exprs.
        values: An integer array containing values to count.
        exprs: An array of integer expressions for which value occurrences must be counted.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of integers, array of integer expressions) => constraint
     * (array of integer expressions, array of integer expressions) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_distribute, args)


def domain_max(*args):
    """ Creates an expression for operation *domainMax*.

    Possible argument and return type combinations are:

     * () => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_domain_max, args)


def domain_min(*args):
    """ Creates an expression for operation *domainMin*.

    Possible argument and return type combinations are:

     * () => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_domain_min, args)


def domain_size(*args):
    """ Creates an expression for operation *domainSize*.

    Possible argument and return type combinations are:

     * () => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_domain_size, args)


def element(*args):
    """ This function returns an element of a given array indexed by an
    integer expression for use in a constraint or another expression.

    This function returns an expression for use in a constraint or other
    expression. The semantics of this expression are: when *subscript*
    takes the value *i*, then the value of the expression is equal to
    *array[i]*.

    Args:
        subscript: An integer expression used to subscript the array.
        array: An array in which an element will be selected using subscript.

    Possible argument and return type combinations are:

     * (array of integers, integer expression) => integer expression
     * (array of integer expressions, integer expression) => integer expression
     * (array of floats, integer expression) => float expression
     * (integer expression, array of integers) => integer expression
     * (integer expression, array of integer expressions) => integer expression
     * (integer expression, array of floats) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    return create_op(Oper_element, args)


def end_at_end(*args):
    """ Constrains the delay between the ends of two interval variables.

    The function *end_at_end* constrains interval variables *a* and *b* in the
    following way. If both intervals *a* and *b* are present then interval *b* must
    end exactly at *end_of(a)+delay*. If *a* or *b* is absent then the
    constraint is automatically satisfied.
    
    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: Interval variables.
        b: Interval variables.
        delay: Exact delay between ends of a and b. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_end_at_end, args)


def end_at_start(*args):
    """ Constrains the delay between the end of one interval variable and start of another one.

    The function *end_at_start* constrains interval variables *predecessor* and
    *successor* in the following way. If both intervals *predecessor* and
    *successor* are present then interval *successor* must start exactly at
    *end_of(predecessor)+delay*. If *predecessor* or *successor* is absent then the
    constraint is automatically satisfied.
    
    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        predecessor: Interval variables.
        successor: Interval variables.
        delay: Exact delay between end of predecessor and start of successor. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_end_at_start, args)


def end_before_end(*args):
    """ Constrains the minimum delay between the ends of two interval variables.

    The function *end_before_end* constrains interval variables *predecessor* and
    *successor* in the following way. If both interval variables *predecessor* and
    *successor* are present then *successor* cannot end before
    *end_of(predecessor)+minDelay*. If *predecessor* or *successor* is absent then
    the constraint is automatically satisfied.
    
    The default value for *minDelay* is zero. It is possible to specify a
    negative *minDelay*; in this case *successor* can actually end before the end
    of *predecessor* but still not sooner than *end_of(predecessor)+minDelay*.

    Args:
        predecessor: Interval variable which ends before.
        successor: Interval variable which ends after.
        minDelay: The minimal delay between end of predecessor and end of successor. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_end_before_end, args)


def end_before_start(*args):
    """ Constrains minimum delay between the end of one interval variable and start of another one.

    the function *end_before_start* constrains interval variables *predecessor* and
    *successor* in the following way. If both interval variables *predecessor* and
    *successor* are present then *successor* cannot start before
    *end_of(predecessor)+minDelay*. If *predecessor* or *successor* is absent then
    the constraint is automatically satisfied.
    
    The default value for *minDelay* is zero. It is possible to specify even
    negative *minDelay*, in this case *successor* can actually start before the end
    of *predecessor* but still not sooner than *end_of(predecessor)+minDelay*.

    Args:
        predecessor: Interval variable which ends before.
        successor: Interval variable which starts after.
        minDelay: The minimal delay between end of predecessor and start of successor. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_end_before_start, args)


def end_eval(*args):
    """ Evaluates *segmentedFunction* at the end of an interval variable.

    Evaluate *function* at the end of interval variable *interval*. However if *interval*
    is absent then it does not have any defined end and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue: Value to return if interval variable interval is absent.

    Possible argument and return type combinations are:

     * (interval variable, segmented function, float constant [=0]) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_end_eval, args)


def end_of(*args):
    """ Returns the end of specified interval variable.

    This function returns an integer expression that is equal to end of the interval
    variable *interval* if it is present. If it is absent then the value of the
    expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue: Value to return if the interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (interval variable, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_end_of, args)


def end_of_next(*args):
    """ Returns an integer expression that represents the end of the interval variable that is next.

    This function returns an integer expression that represents the end of the interval variable
    that is next to *interval* in *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_end_of_next, args)


def end_of_prev(*args):
    """ Returns an integer expression that represents the end of the interval variable that is previous.

    This function returns an integer expression that represents the end of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_end_of_prev, args)


def equal(*args):
    """ Creates an expression for operation *equal*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => boolean expression
     * (float expression, float expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_equal, args)


def equal_or_escape(*args):
    """ Creates an expression for operation *equalOrEscape*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression, integer constant) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_equal_or_escape, args)


def exotic_object(*args):
    """ Creates an expression for operation *exoticObject*.

    Possible argument and return type combinations are:

     * () => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_exotic_object, args)


def explicit_value_eval(*args):
    """ Creates an expression for operation *explicitValueEval*.

    Possible argument and return type combinations are:

     * (array of integers, array of floats, float constant [=0]) => evaluator of integer value

    Returns:
        An expression of type evaluator of integer value
    """
    return create_op(Oper_explicit_value_eval, args)


def explicit_var_eval(*args):
    """ Creates an expression for operation *explicitVarEval*.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of floats, float constant [=0]) => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_explicit_var_eval, args)


def exponent(*args):
    """ Returns the exponent of its argument

    The *exponent* function returns the exponentiation of its argument.

    Args:
        floatExpr: A floating point expression.

    Possible argument and return type combinations are:

     * (float expression) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_exponent, args)


def false(*args):
    """ Returns a false Boolean expression.

    This function returns a false Boolean expression (*boolExpr*). CP Optimizer
    usually eliminates *false()* from expressions using partial evaluation.
    
    The function *false()* does not have any particular purpose except for being a
    filler.

    Possible argument and return type combinations are:

     * () => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_false, args)


def first(*args):
    """ Constrains an interval variable to be the first in a sequence.

    This function returns a constraint that states that whenever interval variable *interval* is present,
    it must be ordered first in the sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval: Interval variable.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_first, args)


def float_div(*args):
    """ Creates an expression for operation *floatDiv*.

    Possible argument and return type combinations are:

     * (float expression, float expression) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_float_div, args)


def forbid_end(*args):
    """ Forbids an interval variable to end during specified regions.

    In the declaration of interval_variable it is only possible to specify
    a range of possible end times. This function allows the user to specify more
    precisely when the interval variable can end. In particular, the interval
    variable can end only at point *t* such that the function has non-zero value at
    *t-1*. When the interval variable is absent then this constraint is
    automatically satisfied (since such interval variable does not't have any start at
    all).
    
    Note the difference between *t* (end time of the interval variable) and *t-1*
    (the point when the function value is checked). It simplifies the sharing of the
    same function in constraints *forbid_start* and *forbid_end*. It also allows one to
    use the same function as *intensity* parameter of
    interval_variable.

    Args:
        interval: Interval variable being restricted.
        function: If the function has value 0 at point t-1 then the interval variable interval cannot end at t.

    Possible argument and return type combinations are:

     * (interval variable, step function) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_forbid_end, args)


def forbid_extent(*args):
    """ Forbids an interval variable to overlap with specified regions.

    This function allows specification of forbidden regions that the interval variable
    *interval* cannot overlap with. In particular, if interval variable *interval* is present and
    if *function* has value 0 during interval *[a,b)* (i.e. *[a,b)* is a
    forbidden region) then either *end <= a* (*interval* ends before the
    forbidden region) or *b <= start* (*interval* starts after the forbidden
    region).
    
    If the interval variable *interval* is absent then the constraint is automatically
    satisfied (the interval does not exist therefore it cannot overlap with any
    region).

    Args:
        interval: Interval variable being restricted.
        function: Forbidden regions corresponds to step of the function that have value 0.

    Possible argument and return type combinations are:

     * (interval variable, step function) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_forbid_extent, args)


def forbid_start(*args):
    """ Forbids an interval variable to start during specified regions.

    This constraint restricts possible start times of interval variable using a
    step function. The interval variable can start only at points where the
    function value is not zero. When the interval variable is absent then this
    constraint is automatically satisfied (since such interval variable does not
    have any start at all).
    
    In declaration of interval_variable it is only possible to specify
    a range of possible start times. This function allows more
    precise specification of when the interval variable can start.

    Args:
        interval: Interval variable being restricted.
        function: If the function has value 0 at point t then the interval variable interval cannot start at t.

    Possible argument and return type combinations are:

     * (interval variable, step function) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_forbid_start, args)


def forbidden_assignments(*args):
    """ Explicitly defines forbidden assignments for one or more integer expressions.

    This function can be used to define simple constraints based
    on explicitly giving the forbidden assignments for a single expression or
    a small group of expressions.  The function returns a Boolean expression
    that represents the truth value of the statement: the values of the
    variables *exprs* is a combination not present in *tuples*.
    
    
    This Boolean expression (which is interpreted as a constraint outside of
    an expression) determines whether the assignment to a single expression *expr*
    or to an array of expressions *exprs* is not contained within the value set
    *values* or the tuple set *tuples* respectively.  The Boolean expression
    will be true if and only if (depending on the signature):
     * the value of the expression *expr* is not present in the array *values*.
     * the values of the expressions *exprs* are not present in the tuple set *tuples*.
    
    The order of the constrained variables in the array *exprs* is
    important because the same order is respected in the tuple set *tuples*.

    Args:
        expr: An integer expression.
        values: An integer array defining forbidden values of expr.
        exprs: An array of integer expressions.
        tuples: Specifies the combinations of forbidden values of the expressions exprs.

    Possible argument and return type combinations are:

     * (integer expression, array of integers) => boolean expression
     * (array of integer expressions, set of tuples) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_forbidden_assignments, args)


def greater(*args):
    """ Creates an expression for operation *greater*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => boolean expression
     * (float expression, float expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_greater, args)


def greater_or_equal(*args):
    """ Creates an expression for operation *greaterOrEqual*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => boolean expression
     * (float expression, float expression) => boolean expression
     * (positive integer, cumul expression) => constraint
     * (cumul expression, positive integer) => constraint
     * (cumul expression, integer expression) => constraint
     * (integer expression, cumul expression) => constraint

    Returns:
        An expression of type boolean expression or constraint
    """
    return create_op(Oper_greater_or_equal, args)


def height_at_end(*args):
    """ Returns the contribution of an interval variable to a cumul function at its end point.

    Whenever interval variable *interval* is present, this function returns an integer expression that represents the total
    contribution of the end of interval variable *interval* to the cumul *function*. When interval variable *interval* is
    absent, this function returns a constant integer expression equal to *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        function: Cumul function expression.
        absentValue: Value to return if the interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (interval variable, cumul expression, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_height_at_end, args)


def height_at_start(*args):
    """ Returns the contribution of an interval variable to a cumul function at its start point.

    Whenever interval variable *interval* is present, this function returns an integer expression that represents the total
    contribution of the start of interval variable *interval* to the cumul *function*. When interval variable *interval* is
    absent, this function returns a constant integer expression equal to *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        function: Cumul function expression.
        absentValue: Value to return if the interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (interval variable, cumul expression, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_height_at_start, args)


def if_then(*args):
    """ Creates an expression for operation *ifThen*.

    Possible argument and return type combinations are:

     * (boolean expression, boolean expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_if_then, args)


def impact_of_last_branch(*args):
    """ Creates an expression for operation *impactOfLastBranch*.

    Possible argument and return type combinations are:

     * () => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_impact_of_last_branch, args)


def int_div(*args):
    """ Creates an expression for operation *intDiv*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_int_div, args)


def inverse(*args):
    """ Constrains elements of one array to be inverses of another.

    This function creates an inverse constraint such that if the length of
    the arrays *f* and *invf* is *n*, then this function returns a
    constraint that ensures that:
     * for all *i* in the interval *[0, n-1]*, *invf[f[i* == i*
     * for all *j* in the interval *[0, n-1]*, *f[invf[j* == j*

    Args:
        f: An integer expression array.
        invf: An integer expression array.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of integer expressions) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_inverse, args)


def isomorphism(*args):
    """ Returns an isomorphism constraint between two sets of interval variables.

    This function creates an isomorphism constraint between the set of interval variables in the array *array1* and the set
    of interval variables in the array *array2*. If an integer expression array *map* is used, it is used to reflect the
    mapping of the intervals of *array1* on the intervals of *array2*, that is, interval variable *array2[i]*, if present,
    is mapped on interval variable *array1[map[i**. If *array2[i]* is absent, index *map[i]* takes value *absentValue*.

    Args:
        array1: The first isomorphic sets of interval variables.
        array1: The second isomorphic sets of interval variables.
        map: Array of integer expressions mapping intervals of array2 on array1.
        absentValue: Value of map[i] when array2[i] is absent.

    Possible argument and return type combinations are:

     * (array of interval variables, array of interval variables, array of integer expressions [=0], integer constant [=0]) => constraint
     * (array of interval variables, array of interval variables, integer constant [=0], array of integer expressions [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_isomorphism, args)


def last(*args):
    """ Constrains an interval variable to be the last in a sequence.

    This function returns a constraint that states that whenever interval variable *interval* is present,
    it must be ordered last in the sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval: Interval variable.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_last, args)


def length_eval(*args):
    """ Evaluates *segmentedFunction* using the length of an interval variable.

    Evaluate *function* for the x value equal to the length of interval variable *interval*. If *interval* is absent then it
    does not have any defined length and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue: Value to return if interval variable interval is absent.

    Possible argument and return type combinations are:

     * (interval variable, segmented function, float constant [=0]) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_length_eval, args)


def length_of(*args):
    """ Returns the length of specified interval variable.

    This function returns an integer expression that is equal to the length (*end -
    start*) of the interval variable *interval* if it is present. If it is absent, then
    the value of the expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue: Value to return if the interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (interval variable, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_length_of, args)


def length_of_next(*args):
    """ Returns an integer expression that represents the length of the interval variable that is next.

    This function returns an integer expression that represents the length of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_length_of_next, args)


def length_of_prev(*args):
    """ Returns an integer expression that represents the length of the interval variable that is previous.

    This function returns an integer expression that represents the length of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_length_of_prev, args)


def less(*args):
    """ Creates an expression for operation *less*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => boolean expression
     * (float expression, float expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_less, args)


def less_or_equal(*args):
    """ Creates an expression for operation *lessOrEqual*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => boolean expression
     * (float expression, float expression) => boolean expression
     * (cumul expression, positive integer) => constraint
     * (positive integer, cumul expression) => constraint
     * (cumul expression, integer expression) => constraint
     * (integer expression, cumul expression) => constraint

    Returns:
        An expression of type boolean expression or constraint
    """
    return create_op(Oper_less_or_equal, args)


def lexicographic(*args):
    """ Returns a constraint which maintains two arrays to be lexicographically ordered.

    The *lexicographic* function returns a constraint which
    maintains two arrays to be lexicographically ordered.
    
    More specifically, *lexicographic(x, y)* maintains that *x* is less
    than or equal to *y* in the lexicographical sense of the term. This
    means that either both arrays are equal or that there exists *i <
    size(x)* such that for all *j < i*, *x[j] = y[j]* and *x[i] < y[i]*.

    Args:
        x: An array of integer expressions.
        y: An array of integer expressions.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of integer expressions) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_lexicographic, args)


def log(*args):
    """ Returns the logarithm of the input.

    The function *log* computes the logarithm of *x*.

    Args:
        x: A floating-point expression.

    Possible argument and return type combinations are:

     * (float expression) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_log, args)


def logical_and(*args):
    """ Creates an expression for operation *and*.

    Possible argument and return type combinations are:

     * (boolean expression, boolean expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_logical_and, args)


def logical_not(*args):
    """ Creates an expression for operation *not*.

    Possible argument and return type combinations are:

     * (boolean expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_logical_not, args)


def logical_or(*args):
    """ Creates an expression for operation *or*.

    Possible argument and return type combinations are:

     * (boolean expression, boolean expression) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_logical_or, args)


def max(*args):
    """ Computes the maximum of a pair or array of integer or floating-point expressions.

    The *max* function returns an expression which has the same value as the
    maximum of the supplied arguments.  The return type corresponds to the
    type of arguments supplied.

    Args:
        a: An array of integer or floating-point expressions from which the maximum is to be computed.
        x: First integer or floating-point expressions from which the maximum is to be computed.
        y: Second integer or floating-point expressions from which the maximum is to be computed.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression
     * (array of integer expressions) => integer expression
     * (float expression, float expression) => float expression
     * (array of float expressions) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    args = [_expand(x) for x in args]
    if _no_cpo_args(args):
        return builtin.max(*args)
    return create_op(Oper_max, args)


def maximize(*args):
    """ A function to specify an optimization problem.  It asks CP Optimizer to
    seek to maximize the value of an expression.

    The function *maximize* specifies to CP Optimizer a floating-point expression
    whose value is sought to be maximized.  When this function is used and
    the problem is feasible, CP Optimizer will generate
    one or more feasible solutions to the problem, with subsequent solutions having
    a larger value of *expr* than preceding ones.  The search terminates when
    either the optimality of the last solution is proved, a search limit is
    exhausted, or the search is aborted.

    Args:
        expr: The expression whose value is to be maximized.

    Possible argument and return type combinations are:

     * (float expression) => objective
     * (array of float expressions) => objective

    Returns:
        An expression of type objective
    """
    return create_op(Oper_maximize, args)


def maximize_static_lex(*args):
    """ A function to specify an optimization problem.  It asks CP Optimizer to
    seek to lexicographically maximize the values of a number of expressions.

    The function *maximize_static_lex* specifies to CP Optimizer a number of
    floating-point expressions whose values are sought to be maximized in a
    lexicographic fashion.  When this function is used and
    the problem is feasible, CP Optimizer will generate one or more
    feasible solutions to the problem, with subsequent solutions having
    a lexicographically larger value of *exprs* than preceding ones.
    This means that a new solution replaces the preceding one as incumbent if
    the value of criterion *exprs[i]* is greater than in the preceding solution,
    so long as the values of criteria *exprs[0..i-1]* are not less than in the
    preceding solution.  In particular, this means that the newer solution is
    preferable even if there are arbitrary reductions in the values of criteria
    after position *i* in *exprs*, as compared with the preceding solution.
    The search terminates when either the optimality of the last solution
    is proved, a search limit is exhausted, or the search is aborted.

    Args:
        exprs: An array of floating-point expressions whose values are to be lexicographically maximized.

    Possible argument and return type combinations are:

     * (array of float expressions) => objective

    Returns:
        An expression of type objective
    """
    return create_op(Oper_maximize_static_lex, args)


def member(*args):
    """ Creates an expression for operation *member*.

    Possible argument and return type combinations are:

     * (integer expression, array of integers) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_member, args)


def min(*args):
    """ Computes the minimum of a pair or array of integer or floating-point expressions.

    The *min* function returns an expression which has the same value as the
    minimum of the supplied arguments.  The return type corresponds to the
    type of arguments supplied.

    Args:
        a: An array of integer or floating-point expressions from which the minimum is to be computed.
        x: First integer or floating-point expressions from which the minimum is to be computed.
        y: Second integer or floating-point expressions from which the minimum is to be computed.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression
     * (array of integer expressions) => integer expression
     * (float expression, float expression) => float expression
     * (array of float expressions) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    args = [_expand(x) for x in args]
    if _no_cpo_args(args):
        return builtin.min(*args)
    return create_op(Oper_min, args)


def minimize(*args):
    """ A function to specify an optimization problem.  It asks CP Optimizer to
    seek to minimize the value of an expression.

    The function *minimize* specifies to CP Optimizer a floating-point expression
    whose value is sought to be minimized.  When this function is used and
    the problem is feasible, CP Optimizer will generate
    one or more feasible solutions to the problem, with subsequent solutions having
    a smaller value of *expr* than preceding ones.  The search terminates when
    either the optimality of the last solution is proved, a search limit is
    exhausted, or the search is aborted.

    Args:
        expr: The expression whose value is to be minimized.

    Possible argument and return type combinations are:

     * (float expression) => objective
     * (array of float expressions) => objective

    Returns:
        An expression of type objective
    """
    return create_op(Oper_minimize, args)


def minimize_static_lex(*args):
    """ A function to specify an optimization problem.  It asks CP Optimizer to
    seek to lexicographically minimize the values of a number of expressions.

    The function *minimize_static_lex* specifies to CP Optimizer a number of
    floating-point expressions whose values are sought to be minimized in a
    lexicographic fashion.  When this function is used and
    the problem is feasible, CP Optimizer will generate
    one or more feasible solutions to the problem, with subsequent solutions having
    a lexicographically smaller value of *exprs* than preceding ones.
    This means that a new solution replaces the preceding one as incumbent if
    the value of criterion *exprs[i]* is less than in the preceding solution,
    so long as the values of criteria *exprs[0..i-1]* are not greater than in the
    preceding solution.  In particular, this means that the newer solution is
    preferable even if there are arbitrary increases in the values of criteria
    after position *i* in *exprs*, as compared with the preceding solution.
    The search terminates when either the optimality of the last solution
    is proved, a search limit is exhausted, or the search is aborted.

    Args:
        exprs: An array of floating-point expressions whose values are to be lexicographically minimized.

    Possible argument and return type combinations are:

     * (array of float expressions) => objective

    Returns:
        An expression of type objective
    """
    return create_op(Oper_minimize_static_lex, args)


def minus(*args):
    """ Creates an expression for operation *minus*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression
     * (integer expression) => integer expression
     * (float expression, float expression) => float expression
     * (float expression) => float expression
     * (cumul expression, cumul expression) => cumul expression
     * (cumul expression) => cumul expression

    Returns:
        An expression of type cumul expression, float expression or integer expression
    """
    return create_op(Oper_minus, args)


def mod(*args):
    """ Creates an expression for operation *mod*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_mod, args)


def no_overlap(*args):
    """ Constrains a set of interval variables not to overlap each others.

    This function returns a constraint over a set of interval variables {*a1*, ..., *an*} that states that all the present
    intervals in the set are pairwise non-overlapping. It means that whenever both interval variables *ai* and *aj*, i!=j
    are present, *ai* is constrained to end before the start of *aj* or *aj* is constrained to end before the start of *ai*.
    
    If the no-overlap constraint has been built on an interval sequence variable *sequence*, it means that the no-overlap
    constraint works on the set of interval variables {*a1*, ..., *an*} of the sequence and that the order of interval
    variables of the sequence will describe the order of the non-overlapping intervals. That is, if *ai* and *aj*, i!=j are
    both present and if *ai* appears before *aj* in the sequence value, then *ai* is constrained to end before the start of
    *aj*. If a transition matrix *distanceMatrix* is specified and if *tpi* and *tpj* respectively denote the types of
    interval variables *ai* and *aj* in the *sequence*, it means that a minimal distance *distanceMatrix[tpi,tpj]* is to be
    maintained between the end of *ai* and the start of *aj*. If Boolean flag *isDirect* is true, the transition distance
    holds between an interval and its immediate successor in the sequence otherwise, if *isDirect* is false (default), the
    transition distance holds between an interval and all its successors in the sequence.

    Args:
        intervals: An array of interval variables.
        sequence: A sequence variable.
        distanceMatrix: An optional transition matrix defining the transition distance between consecutive interval variables.
        isDirect: A Boolean flag stating whether the distance specified in the transition matrix distanceMatrix holds between direct successors (isDirect=1) or also between indirect successors (isDirect=0, default).

    Possible argument and return type combinations are:

     * (sequence variable, transition matrix [=0], boolean integer (0, 1) [=0]) => constraint
     * (array of interval variables) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_no_overlap, args)


def not_member(*args):
    """ Creates an expression for operation *notMember*.

    Possible argument and return type combinations are:

     * (integer expression, array of integers) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_not_member, args)


def overlap_length(*args):
    """ Returns the length of the overlap of two interval variables.

    The first function returns an integer expression that represents the length of the overlap of interval variable
    *interval1* and the interval variable *interval2* whenever the interval variables *interval1* and *interval2* are
    present. When one of the interval variables *interval1* or *interval2* is absent, the function returns the constant
    integer value *absentValue* (zero by default).
    
    The second function returns an integer expression that represents the length of the overlap of interval variable
    *interval* and the constant interval [*start*, *end*) whenever the interval variable *interval* is present. When the
    interval variable *interval* is absent, the function returns the constant integer value *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        interval1: Interval variable.
        interval2: Interval variable.
        start: Start value of a fixed interval.
        end: End value of a fixed interval.
        absentValue: Value to return if some interval variable is absent.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer constant [=0]) => integer expression
     * (interval variable, integer time, integer time, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_overlap_length, args)


def pack(*args):
    """ Maintains the load on a set of containers given objects sizes and assignments.

    The *pack* constraint is used to represent sub-problems where the requirement
    is to assign objects to containers such that the capacities or minimum fill levels of the containers are respected. 
    Let's assume we have *n* objects
    and *m* containers.  The sizes of the array arguments of *pack* must correspond
    to these constants.  That is, *load* must be of size *m*, whereas *where*
    and *size* must be of size *n*.
    Given assignments to the *where* expressions,
    the *pack* constraint will calculate the values of the *load* and *used*
    expressions.
    
    All counting is done from 0, and so
    the interpretation of 5 being assigned to *where[3]* is that object 3
    (the 4th object) is placed into container 5 (the 6th container). This will be
    reflected by the inclusion of the size of object 3 (*size[3]*) being
    included in the calculation of the value of *load[5]*.
    
    Naturally, all the arguments (with the exception of *size*) can
    be constrained by additional constraints.  The most common form is to limit
    the capacity of a container.  For example, to limit container 2
    to a capacity of 15, one would write *load[2] <= 15*.  Minimum fill level
    requirements can also be specified this way: for example *load[2] >= 12*.
    Other more esoteric constraints are possible, for example to say that only
    an even number of containers can be used: *(used % 2) == 0*.  If *used*
    is omitted from the signature, then it will not be possible to specifically
    constrain the number of containers used.

    Args:
        load: An array of integer expressions, each element representing the load (total size of the objects inside) the corresponding container.
        where: An array of integer expressions, each element representing in which container the corresponding object is placed.
        size: An array of integers, each element representing the size of the corresponding object.
        used: (optional) An integer expression indicating the number of used containers.  That is, the number of containers with at least one object inside.

    Possible argument and return type combinations are:

     * (array of integer expressions, array of integer expressions, array of integers, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_pack, args)


def phase(*args):
    """ Creates an expression for operation *phase*.

    Possible argument and return type combinations are:

     * (array of integer expressions) => search phase
     * (array of interval variables) => search phase

    Returns:
        An expression of type search phase
    """
    return create_op(Oper_phase, args)


def plus(*args):
    """ Creates an expression for operation *plus*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression
     * (float expression, float expression) => float expression
     * (cumul expression, cumul expression) => cumul expression

    Returns:
        An expression of type cumul expression, float expression or integer expression
    """
    return create_op(Oper_plus, args)


def power(*args):
    """ Creates an expression for operation *power*.

    Possible argument and return type combinations are:

     * (float expression, float expression) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_power, args)


def presence_of(*args):
    """ Returns the presence status of specified interval variable.

    This function returns the Boolean expression that represents the presence status of
    the interval variable *interval*. If *interval* is present then the value of the expression
    is 1; if *interval* is absent then the value is 0.
    
    Use *presence_of* to express logical relationships between interval variables.
    Note that the most effective are binary relations such as
    *presence_of(x)=>presence_of(y)* because CP Optimizer is able to take them into
    account during propagation of other constraints such as *end_before_start* or *no_overlap*.
    
    The function *presence_of* can be also used to compute cost associated with
    execution/non-execution of an interval.

    Args:
        interval: Interval variable.

    Possible argument and return type combinations are:

     * (interval variable) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_presence_of, args)


def previous(*args):
    """ Constrains an interval variable to be previous to another interval variable in a sequence.

    This function returns a constraint that states that whenever both interval variables *interval1* and *interval2* are
    present,
    *interval1* must be the interval variable that is previous to *interval2* in the sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval1: Interval variable.
        interval2: Interval variable.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, interval variable) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_previous, args)


def pulse(*args):
    """ Returns an elementary cumul function of constant value between the start and the end of an interval.

    This function returns an elementary cumul function expression that is equal to a value *h* everywhere between the start
    and the end of an interval variable *interval* or a fixed interval [*start*,*end*). The function is equal to 0 outside
    of the interval. When interval variable *interval* is absent, the function is the constant zero function. When a range
    [*heightMin*, *heightMax*) is specified it means that the height value *h* of the pulse is part of the decisions of the
    problem and will be fixed by the engine within this specified range.

    Args:
        interval: Interval variable contributing to the cumul function.
        h: Non-negative integer representing the height of the contribution.
        heightMin: Non-negative integer representing the minimum of the range of possible values for the height of the contribution.
        heightMax: Non-negative integer representing the maximum of the range of possible values for the height of the contribution.
        start: Start of the fixed interval [start,end) contributing to the cumul function.
        end: End of the fixed interval [start,end) contributing to the cumul function.

    Possible argument and return type combinations are:

     * (integer time, integer time, positive integer) => cumul atom
     * (interval variable, positive integer) => cumul atom
     * (interval variable, positive integer, positive integer) => cumul atom

    Returns:
        An expression of type cumul atom
    """
    return create_op(Oper_pulse, args)


def range(*args):
    """ Restricts the bounds of an integer or floating-point expression.

    This Boolean expression (which is interpreted as a constraint outside of
    an expression) determines whether the value of expression *x* is
    inside the range *[lb, ub]*.  The returned expression will be true if and
    only if *x* is no less than *lb* and no greater than *ub*.
    *range(y, a, b)* is also a more efficient form of
    writing *a <= y && y <= b*.

    Args:
        x: The integer or floating-point expression.
        lb: The lower bound.
        ub: The upper bound.

    Possible argument and return type combinations are:

     * (integer expression, float constant, float constant) => boolean expression
     * (float expression, float constant, float constant) => boolean expression

    Returns:
        An expression of type boolean expression
    """
    args = [_expand(x) for x in args]
    if _no_cpo_args(args):
        return builtin.range(*args)
    return create_op(Oper_range, args)


def same_common_sub_sequence(*args):
    """ Creates an expression for operation *sameCommonSubSequence*.

    Possible argument and return type combinations are:

     * (sequence variable, sequence variable, array of interval variables, array of interval variables, boolean integer (0, 1)) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_same_common_sub_sequence, args)


def same_common_subsequence(*args):
    """ This function creates a same-common-subsequence constraint between two sequence variables.

    This function creates a same-common-subsequence constraint between sequence variables *seq1* and *seq2*.
    
    If no interval variable array is specified as argument, the sequence variables *seq1* and *seq2* should be of the same
    size and the mapping between interval variables of the two sequences is given by the order of the interval variables in
    the arrays *array1* and *array2* used in the definition of the sequences.
    
    If interval variable arrays *array1* and *array2* are used, these arrays define the mapping between interval variables
    of the two sequences.
    
    The constraint states that the sub-sequences defined by *seq1* and *seq2* by only considering the pairs of present
    intervals (*array1[i]*,*array2[i]*) are identical modulo the mapping between intervals *array1[i]* and *array2[i]*.

    Args:
        seq1: First constrained sequence variables.
        seq2: Second constrained sequence variables.
        array1: First array of interval variables defining the mapping between the two sequence variables.
        array2: Second array of interval variables defining the mapping between the two sequence variables.

    Possible argument and return type combinations are:

     * (sequence variable, sequence variable) => constraint
     * (sequence variable, sequence variable, array of interval variables, array of interval variables) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_same_common_subsequence, args)


def same_sequence(*args):
    """ This function creates a same-sequence constraint between two sequence variables.

    This function creates a same-sequence constraint between sequence variables *seq1 and seq2*. Sequence variables *seq1*
    and *seq2* should be of the same size *n*. If no array of interval variables is specified, the mapping between interval
    variables of the two sequences is given by the order of the interval variables in the arrays *array1* and *array2* used
    in the definition of the sequences. If some arrays are specified, they are used to define the mapping. The constraint
    states that the two sequences *seq1* and *seq2* are identical modulo a mapping between intervals *array1[i]* and
    *array2[i]*.

    Args:
        seq1: First constrained sequence variables.
        seq2: Second constrained sequence variables.
        array1: First array of interval variables defining the mapping between the two sequence variables.
        array2: Second array of interval variables defining the mapping between the two sequence variables.

    Possible argument and return type combinations are:

     * (sequence variable, sequence variable) => constraint
     * (sequence variable, sequence variable, array of interval variables, array of interval variables) => constraint
     * (sequence variable, sequence variable, boolean integer (0, 1)) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_same_sequence, args)


def scal_prod(*args):
    """ Returns the scalar product of two vectors.

    The function *scal_prod* returns an integer or floating-point expression that
    represents the scalar product of two vectors *x* and *y*. Depending on the
    type of *x* and *y* the result is either integer or floating-point
    expression.
    
    The versions with constant arrays (*intArray* or *floatArray*) are preferred
    because they can be slightly faster.

    Args:
        x: First input array (vector) to be multiplied.
        y: Second input array (vector) to be multiplied.

    Possible argument and return type combinations are:

     * (array of integers, array of integer expressions) => integer expression
     * (array of integer expressions, array of integers) => integer expression
     * (array of integer expressions, array of integer expressions) => integer expression
     * (array of floats, array of float expressions) => float expression
     * (array of float expressions, array of floats) => float expression
     * (array of float expressions, array of float expressions) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    return create_op(Oper_scal_prod, args)


def search_phase(*args):
    """ Creates an expression for operation *searchPhase*.

    Possible argument and return type combinations are:

     * (array of integer expressions) => search phase
     * (chooser of integer variable, chooser of integer value) => search phase
     * (array of integer expressions, chooser of integer variable, chooser of integer value) => search phase
     * (array of interval variables) => search phase
     * (array of sequence variables) => search phase

    Returns:
        An expression of type search phase
    """
    return create_op(Oper_search_phase, args)


def search_phase_int(*args):
    """ Creates an expression for operation *searchPhaseInt*.

    Possible argument and return type combinations are:

     * (array of integer expressions) => search phase
     * (chooser of integer variable, chooser of integer value) => search phase
     * (array of integer expressions, chooser of integer variable, chooser of integer value) => search phase

    Returns:
        An expression of type search phase
    """
    return create_op(Oper_search_phase_int, args)


def search_phase_rank(*args):
    """ Creates an expression for operation *searchPhaseRank*.

    Possible argument and return type combinations are:

     * (array of sequence variables [=0]) => search phase

    Returns:
        An expression of type search phase
    """
    return create_op(Oper_search_phase_rank, args)


def search_phase_set_times(*args):
    """ Creates an expression for operation *searchPhaseSetTimes*.

    Possible argument and return type combinations are:

     * (array of interval variables [=0]) => search phase

    Returns:
        An expression of type search phase
    """
    return create_op(Oper_search_phase_set_times, args)


def select_largest(*args):
    """ Creates an expression for operation *selectLargest*.

    Possible argument and return type combinations are:

     * (float constant, evaluator of integer variable) => selector of integer variable
     * (evaluator of integer variable, float constant [=0]) => selector of integer variable
     * (float constant, evaluator of integer value) => selector of integer value
     * (evaluator of integer value, float constant [=0]) => selector of integer value
     * (evaluator of integer variable, integer constant, float constant) => selector of integer variable
     * (evaluator of integer value, integer constant, float constant) => selector of integer value

    Returns:
        An expression of type selector of integer value or selector of integer variable
    """
    return create_op(Oper_select_largest, args)


def select_random_value(*args):
    """ Creates an expression for operation *selectRandomValue*.

    Possible argument and return type combinations are:

     * () => selector of integer value

    Returns:
        An expression of type selector of integer value
    """
    return create_op(Oper_select_random_value, args)


def select_random_var(*args):
    """ Creates an expression for operation *selectRandomVar*.

    Possible argument and return type combinations are:

     * () => selector of integer variable

    Returns:
        An expression of type selector of integer variable
    """
    return create_op(Oper_select_random_var, args)


def select_smallest(*args):
    """ Creates an expression for operation *selectSmallest*.

    Possible argument and return type combinations are:

     * (float constant, evaluator of integer variable) => selector of integer variable
     * (evaluator of integer variable, float constant [=0]) => selector of integer variable
     * (float constant, evaluator of integer value) => selector of integer value
     * (evaluator of integer value, float constant [=0]) => selector of integer value
     * (evaluator of integer variable, integer constant, float constant) => selector of integer variable
     * (evaluator of integer value, integer constant, float constant) => selector of integer value

    Returns:
        An expression of type selector of integer value or selector of integer variable
    """
    return create_op(Oper_select_smallest, args)


def sequence(*args):
    """ Constrains the number of occurrences of the values taken by the different subsets of consecutive *k* variables.

    This constraint ensures:
     * that *cards[i]* will be equal to the number of occurrences of the value *values[i]* in the array *vars*,
     * and that each sequence of *width* consecutive variables (like *vars[j+1]*, *vars[j+2]*, ..., *vars[j+width]*) takes at least *min* and at most *max* values of the array *values*.

    Args:
        min: The minimum number of allowable values.
        max: The maximum number of allowable values.
        width: The size of the sequences of consecutive variables.
        vars: The array of variables.
        values: The array of values.
        cards: The array of cardinality variables.

    Possible argument and return type combinations are:

     * (integer constant, integer constant, integer constant, array of integer expressions, array of integers, array of integer expressions) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_sequence, args)


def size_eval(*args):
    """ Evaluates *segmentedFunction* using the size of an interval variable.

    Evaluate *function* for the x value equal to the size of interval variable *interval*. If *interval*
    is absent then it does not have any defined size and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue: Value to return if interval variable interval is absent.

    Possible argument and return type combinations are:

     * (interval variable, segmented function, float constant [=0]) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_size_eval, args)


def size_of(*args):
    """ Returns the size of a specified interval variable.

    This function returns an integer expression that is equal to size of the interval
    variable *interval* if it is present. If it is absent then the value of the
    expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue: Value to return if the interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (interval variable, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_size_of, args)


def size_of_next(*args):
    """ Returns an integer expression that represents the size of the interval variable that is next.

    This function returns an integer expression that represents the size of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_size_of_next, args)


def size_of_prev(*args):
    """ Returns an integer expression that represents the size of the interval variable that is previous.

    This function returns an integer expression that represents the size of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_size_of_prev, args)


def slope_piecewise_linear(*args):
    """ Creates an expression for operation *slopePiecewiseLinear*.

    Possible argument and return type combinations are:

     * (float expression, array of floats, array of floats, float constant, float constant) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_slope_piecewise_linear, args)


def span(*args):
    """ Creates a span constraint between interval variables.

    This function creates a span constraint between an interval variable *interval*
    and a set of interval variables in *array*. This constraint states that
    *interval* when it is present spans over all present intervals from the
    *array*. That is: *interval* starts together with the first present
    interval from *array* and ends together with the last one. Interval *interval*
    is absent if and only if all intervals in *array* are absent.

    Args:
        interval: Spanning interval variable.
        array: Array of spanned interval variables.

    Possible argument and return type combinations are:

     * (interval variable, array of interval variables) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_span, args)


def spread(*args):
    """ Creates an expression for operation *spread*.

    Possible argument and return type combinations are:

     * (array of integer expressions, float expression, float expression) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_spread, args)


def square(*args):
    """ Returns the square of the input.

    Function *square* computes the square of *x*. Depending on the type of *x* the
    result is an integer or a floating-point expression.

    Args:
        x: Integer or floating-point expression.

    Possible argument and return type combinations are:

     * (integer expression) => integer expression
     * (float expression) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    return create_op(Oper_square, args)


def standard_deviation(*args):
    """ Creates a constrained numeric expression equal
    to the standard deviation of the values of the variables in an array.

    This function creates a new constrained numeric expression which is
    equal to the standard deviation of the values of the variables in the
    array *x*.  The mean of the values of the variables in the array x is
    constrained to be in the interval [meanLB, meanUB].

    Args:
        x: An array of integer expressions.
        meanLB: A lower bound on the mean of the array.
        meanUB: An upper bound on the mean of the array.

    Possible argument and return type combinations are:

     * (array of integer expressions, float constant, float constant) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_standard_deviation, args)


def start_at_end(*args):
    """ Constrains the delay between the start of one interval variable and end of another one.

    The function *start_at_end* constrains interval variables *a* and *b* in the
    following way. If both intervals *a* and *b* are present then interval *b* must
    end exactly at *start_of(a)+delay*. If *a* or *b* is absent then the constraint
    is automatically satisfied.
    
    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: First interval variables.
        b: Second interval variables.
        delay: Exact delay between start of a and end of b. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_start_at_end, args)


def start_at_start(*args):
    """ Constrains the delay between the starts of two interval variables.

    The function *start_at_start* constrains interval variables *a* and *b* in the
    following way. If both intervals *a* and *b* are present then interval *b* must
    start exactly at *start_of(a)+delay*. If *a* or *b* is absent then the
    constraint is automatically satisfied.
    
    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: First interval variables.
        b: Second interval variables.
        delay: Exact delay between starts of a and b. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_start_at_start, args)


def start_before_end(*args):
    """ Constrains minimum delay between the start of one interval variable and end of another one.

    The function *start_before_end* constrains interval variables *predecessor* and
    *successor* in the following way. If both interval variables *predecessor* and
    *successor* are present then *successor* cannot end before
    *start_of(predecessor)+minDelay*. If *predecessor* or *successor* is absent then
    the constraint is automatically satisfied.
    
    The default value for *minDelay* is zero. It is possible to specify a
    negative *minDelay*; in this case *successor* can actually end before the start
    of *predecessor* but still not sooner than *start_of(predecessor)+minDelay*.

    Args:
        predecessor: Interval variable which starts before.
        successor: Interval variable which ends after.
        minDelay: The minimal delay between start of predecessor and end of successor. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_start_before_end, args)


def start_before_start(*args):
    """ Constrains the minimum delay between starts of two interval variables.

    The function *start_before_start* constrains interval variables *predecessor* and
    *successor* in the following way. If both interval variables *predecessor* and
    *successor* are present then *successor* cannot start before
    *start_of(predecessor)+minDelay*. If *predecessor* or *successor* is absent then
    the constraint is automatically satisfied.
    
    The default value for *minDelay* is zero. It is possible to specify even
    negative *minDelay*, in this case *successor* can actually start before the start
    of *predecessor* but still not sooner than *start_of(predecessor)+minDelay*.

    Args:
        predecessor: Interval variable which starts before.
        successor: Interval variable which starts after.
        minDelay: The minimal delay between start of predecessor and start of successor. If not specified then zero is used.

    Possible argument and return type combinations are:

     * (interval variable, interval variable, integer expression [=0]) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_start_before_start, args)


def start_eval(*args):
    """ Evaluates *segmentedFunction* at the start of an interval variable.

    Evaluates *function* at the start of interval variable *interval*. However if *interval*
    is absent then it does not have any defined start and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue: Value to return if interval variable interval is absent.

    Possible argument and return type combinations are:

     * (interval variable, segmented function, float constant [=0]) => float expression

    Returns:
        An expression of type float expression
    """
    return create_op(Oper_start_eval, args)


def start_of(*args):
    """ Returns the start of a specified interval variable.

    This function returns an integer expression that is equal to start of the interval
    variable *interval* if it is present. If it is absent, then the value of the
    expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue: Value to return if the interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (interval variable, integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_start_of, args)


def start_of_next(*args):
    """ Returns an integer expression that represents the start of the interval variable that is next.

    This function returns an integer expression that represents the start of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_start_of_next, args)


def start_of_prev(*args):
    """ Returns an integer expression that represents the start of the interval variable that is previous.

    This function returns an integer expression that represents the start of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_start_of_prev, args)


def step_at(*args):
    """ Returns an elementary cumul function of constant value after a given point.

    This function returns an elementary cumul function expression that is equal to 0 before point *t* and equal to *h* after
    point *t*.

    Args:
        t: Integer.
        h: Non-negative integer representing the height of the contribution after point t.

    Possible argument and return type combinations are:

     * (integer time, positive integer) => cumul atom

    Returns:
        An expression of type cumul atom
    """
    return create_op(Oper_step_at, args)


def step_at_end(*args):
    """ Returns an elementary cumul function of constant value after the end of an interval.

    This function returns an elementary cumul function expression that, whenever interval variable *interval* is present, is
    equal to 0 before the end of *interval* and equal to *h* after the end of *interval*. When a range [*heightMin*,
    *heightMax*) is specified it means that the height value *h* of the function is part of the decisions of the problem and
    will be fixed by the engine within this specified range. When interval variable *interval* is absent, the function is
    the constant zero function.

    Args:
        interval: Interval variable contributing to the cumul function.
        h: Non-negative integer representing the height of the contribution.
        heightMin: Non-negative integer representing the the minimum of the range of possible values for the height of the contribution.
        heightMax: Non-negative integer representing the maximum of the range of possible values for the height of the contribution.

    Possible argument and return type combinations are:

     * (interval variable, positive integer) => cumul atom
     * (interval variable, positive integer, positive integer) => cumul atom

    Returns:
        An expression of type cumul atom
    """
    return create_op(Oper_step_at_end, args)


def step_at_start(*args):
    """ Returns an elementary cumul function of constant value after the start of an interval.

    This function returns an elementary cumul function expression that, whenever interval variable *interval* is present, is
    equal to 0 before the start of *interval* and equal to *h* after the start of *interval*. When a range [*heightMin*,
    *heightMax*) is specified it means that the height value *h* of the function is part of the decisions of the problem and
    will be fixed by the engine within this specified range. When interval variable *interval* is absent, the function is
    the constant zero function.

    Args:
        interval: Interval variable contributing to the cumul function.
        h: Non-negative integer representing the height of the contribution.
        heightMin: Non-negative integer representing the the minimum of the range of possible values for the height of the contribution.
        heightMax: Non-negative integer representing the maximum of the range of possible values for the height of the contribution.

    Possible argument and return type combinations are:

     * (interval variable, positive integer) => cumul atom
     * (interval variable, positive integer, positive integer) => cumul atom

    Returns:
        An expression of type cumul atom
    """
    return create_op(Oper_step_at_start, args)


def strong(*args):
    """ A model annotation to encourage CP Optimizer to produce stronger (higher inference) constraints.

    The *strong* constraint strengthens the model on the expressions *x*.
    This is done by creating an *allowed_assignments* constraint in place
    of the *strong* constraint during presolve. Only the assignments to
    the expressions which do not result in an immediate inconsistency are
    added to the tuple set of the *allowed_assignments* constraint.
    
    Constraints that can be identified as redundant (when taken together
    with this new constraint) are removed from the model during presolve.
    This is the case for constraints that are only over the variables of
    the array given as argument.

    Args:
        x: An array of integer expressions over which propagation is to be strengthened.

    Possible argument and return type combinations are:

     * (array of integer expressions) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_strong, args)


def sum(*args):
    """ Returns the sum of the input.

    The function *sum* computes the sum of *x*. Depending on the type of *x* the
    result is an integer or a floating-point expression.

    Args:
        x: An array of integer or floating-point expressions.

    Possible argument and return type combinations are:

     * (array of integer expressions) => integer expression
     * (array of float expressions) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    args = [_expand(x) for x in args]
    if _no_cpo_args(args):
        return builtin.sum(*args)
    return create_op(Oper_sum, args)


def synchronize(*args):
    """ Creates a synchronization constraint between interval variables.

    This function creates a synchronization constraint between an interval variable *interval* and a set of interval
    variables in *array*. This constraint makes all present intervals in *array* start and end together with *interval*, if
    it is present.

    Args:
        interval: Interval variable.
        array: Array of interval variables synchronized with interval.

    Possible argument and return type combinations are:

     * (interval variable, array of interval variables) => constraint

    Returns:
        An expression of type constraint
    """
    return create_op(Oper_synchronize, args)


def times(*args):
    """ Creates an expression for operation *times*.

    Possible argument and return type combinations are:

     * (integer expression, integer expression) => integer expression
     * (float expression, float expression) => float expression

    Returns:
        An expression of type float expression or integer expression
    """
    return create_op(Oper_times, args)


def true(*args):
    """ Returns a true Boolean expression.

    This function returns a true Boolean expression (*boolExpr*). CP Optimizer
    usually eliminates *true()* from expressions using partial evaluation.
    
    The function *true()* does not have any particular purpose except for being a
    filler.

    Possible argument and return type combinations are:

     * () => boolean expression

    Returns:
        An expression of type boolean expression
    """
    return create_op(Oper_true, args)


def type_of_next(*args):
    """ Returns an integer expression that represents the type of the interval variable that is next.

    This function returns an integer expression that represents the type of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_type_of_next, args)


def type_of_prev(*args):
    """ Returns an integer expression that represents the type of the interval variable that is previous.

    This function returns an integer expression that represents the type of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.

    Possible argument and return type combinations are:

     * (sequence variable, interval variable, integer constant [=0], integer constant [=0]) => integer expression

    Returns:
        An expression of type integer expression
    """
    return create_op(Oper_type_of_prev, args)


def value(*args):
    """ Creates an expression for operation *value*.

    Possible argument and return type combinations are:

     * () => evaluator of integer value

    Returns:
        An expression of type evaluator of integer value
    """
    return create_op(Oper_value, args)


def value_impact(*args):
    """ Creates an expression for operation *valueImpact*.

    Possible argument and return type combinations are:

     * () => evaluator of integer value

    Returns:
        An expression of type evaluator of integer value
    """
    return create_op(Oper_value_impact, args)


def value_index(*args):
    """ Creates an expression for operation *valueIndex*.

    Possible argument and return type combinations are:

     * (array of integers, float constant [=-1]) => evaluator of integer value

    Returns:
        An expression of type evaluator of integer value
    """
    return create_op(Oper_value_index, args)


def value_index_eval(*args):
    """ Creates an expression for operation *valueIndexEval*.

    Possible argument and return type combinations are:

     * (array of integers, float constant [=-1]) => evaluator of integer value

    Returns:
        An expression of type evaluator of integer value
    """
    return create_op(Oper_value_index_eval, args)


def value_success_rate(*args):
    """ Creates an expression for operation *valueSuccessRate*.

    Possible argument and return type combinations are:

     * () => evaluator of integer value

    Returns:
        An expression of type evaluator of integer value
    """
    return create_op(Oper_value_success_rate, args)


def var_impact(*args):
    """ Creates an expression for operation *varImpact*.

    Possible argument and return type combinations are:

     * () => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_var_impact, args)


def var_index(*args):
    """ Creates an expression for operation *varIndex*.

    Possible argument and return type combinations are:

     * (array of integer expressions, float constant [=-1]) => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_var_index, args)


def var_index_eval(*args):
    """ Creates an expression for operation *varIndexEval*.

    Possible argument and return type combinations are:

     * (array of integer expressions, float constant [=-1]) => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_var_index_eval, args)


def var_local_impact(*args):
    """ Creates an expression for operation *varLocalImpact*.

    Possible argument and return type combinations are:

     * (integer constant [=-1]) => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_var_local_impact, args)


def var_success_rate(*args):
    """ Creates an expression for operation *varSuccessRate*.

    Possible argument and return type combinations are:

     * () => evaluator of integer variable

    Returns:
        An expression of type evaluator of integer variable
    """
    return create_op(Oper_var_success_rate, args)



###############################################################################
##  Operators overloading
###############################################################################

CpoExpr.__ne__ = diff
CpoExpr.__eq__ = equal
CpoExpr.__div__ = float_div
CpoExpr.__truediv__ = float_div
CpoExpr.__rdiv__ = lambda x, y: float_div(y, x)
CpoExpr.__rtruediv__ = lambda x, y: float_div(y, x)
CpoExpr.__gt__ = greater
CpoExpr.__ge__ = greater_or_equal
CpoExpr.__floordiv__ = int_div
CpoExpr.__rfloordiv__ = lambda x, y: int_div(y, x)
CpoExpr.__lt__ = less
CpoExpr.__le__ = less_or_equal
CpoExpr.__and__ = logical_and
CpoExpr.__rand__ = lambda x, y: logical_and(y, x)
CpoExpr.__invert__ = logical_not
CpoExpr.__or__ = logical_or
CpoExpr.__ror__ = lambda x, y: logical_or(y, x)
CpoExpr.__sub__ = minus
CpoExpr.__rsub__ = lambda x, y: minus(y, x)
CpoExpr.__mod__ = mod
CpoExpr.__rmod__ = lambda x, y: mod(y, x)
CpoExpr.__add__ = plus
CpoExpr.__radd__ = lambda x, y: plus(y, x)
CpoExpr.__pow__ = power
CpoExpr.__rpow__ = lambda x, y: power(y, x)
CpoExpr.__mul__ = times
CpoExpr.__rmul__ = lambda x, y: times(y, x)
CpoExpr.__neg__ = minus
CpoExpr.__pos__ = lambda x: x

