# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the basic classes representing the expressions required to
describe a constraint programming model.

In particular, it defines the following classes:

 * :class:`CpoExpr`: the root class of each model expression node,
 * :class:`CpoIntVar`: representation of an integer variable,
 * :class:`CpoIntervalVar`: representation of an interval variable,
 * :class:`CpoSequenceVar`: representation of an interval variable,
 * :class:`CpoSequenceVar`: representation of an interval variable,
 * :class:`CpoTransitionMatrix`: representation of a transition matrix,
 * :class:`CpoTupleSet`: representation of a tuple set,
 * :class:`CpoStateFunction`: representation of a state function.

None of these classes should be created explicitly.
There are various factory functions to do so, such as:

 * :meth:`integer_var`, :meth:`integer_var_list`, :meth:`integer_var_dict` to create integer variable(s),
 * :meth:`interval_var`, :meth:`interval_var_list` to create an interval variable(s),
 * :meth:`sequence_var` to create a sequence variable,
 * etc.

Moreover, some automatic conversions are also provided.
For example, a list of tuples of integers is automatically converted into a tuple set.
"""

from docplex.cp.utils import *
from docplex.cp.catalog import *
from docplex.cp.config import context

import math
import collections
import threading


###############################################################################
## Constants
###############################################################################

INT_MAX = (2**53 - 1)  # (2^53 - 1) for 64 bits, (2^31 - 1) for 32 bits
""" Maximum integer value. """

INT_MIN = -INT_MAX
""" Minimum integer value. """

# Name generator for integer variables
_INTEGER_VARIABLE_ID_ALLOCATOR = SafeIdAllocator('_INT_')

# Name generator for interval variables
_INTERVAL_VARIABLE_ID_ALLOCATOR = SafeIdAllocator('_ITV_')

# Name generator for sequence variables
_SEQUENCE_VARIABLE_ID_ALLOCATOR = SafeIdAllocator('_SEQ_')

# Name generator for state functions
_STATE_FUNCTION_ID_ALLOCATOR = SafeIdAllocator('_FUN_')

# Name generator for all other variables
_VARIABLE_ID_ALLOCATOR = SafeIdAllocator('_VAR_')

# Name generator for shared expressions
_EXPRESSION_ID_ALLOCATOR = SafeIdAllocator('_EXP_')

# Name generator for constraints
_CONSTRAINT_ID_ALLOCATOR = SafeIdAllocator('_CTR_')

# Floating point precision to verify equality of floats
_FLOATING_POINT_PRECISION = 1e-9

# Domain for binary variables
_BINARY_DOMAIN = ((0, 1),)


###############################################################################
## Public expression classes
###############################################################################

class CpoExpr(object):
    """ This class is an abstract class that represents any CPO expression node.

    It does not contain links to children expressions that are implemented in extending classes.
    However, method allowing to access to children is provided with default return value.
    """
    __slots__ = ('type',       # Expression result type
                 'name',       # Name of the expression (None if none)
                 'nbrefs',     # Number of references on this expression
                )

    # To force possible numpy operators overloading to get CPO expressions as main operand
    __array_priority__ = 100

    # Expression name generator
    __name_generator__ = _EXPRESSION_ID_ALLOCATOR

    def __init__(self, type, name):
        """ Constructor:

        Args:
            type:   Expression type.
            name:   Expression name.
        """
        # super(CpoExpr, self).__init__()
        self.type = type
        self.name = name
        self.nbrefs = 0
        
    def __hash__(self):
        """ Redefinition of hash function (mandatory for Python 3)
        """
        return int(id(self) / 16)

    '''
    # These functions are required with standard 'pickle'
    # But DO NOT activate them with cloudpickle
    def __getstate__(self):
        """ Build a picklable object from this object
        """
        return dict((k, getattr(self, k, None)) for k in self.__slots__)

    def __setstate__(self, data):
        """ Fill object from its pickle form
        """
        for (k, v) in data.iteritems():
            setattr(self, k, v)
    '''

    def set_name(self, name):
        """ Set the name of the expression.

        Args:
            name: Expression name, possibly None.
        """
        assert (name is None) or is_string(name), "Argument 'name' should be a string or None, not '{}' of type {}".format(name, type(name))
        if name is not None:
            name = make_unicode(name)
        self.name = name
            
    def get_name(self):
        """ Return the name of the expression.

        Returns:
            Name of the expression, possibly None.
        """
        return self.name
            
    def has_name(self):
        """ Check if the expression has a name.

        Returns:
            True if the expression has a name, False otherwise.
        """
        return (self.name is not None)

    @classmethod
    def _generate_name(cls):
        """ Generate a name for this type of expression
        Return
            Unused name for this type of object
        """
        return cls.__name_generator__.allocate()

    def _set_generated_name(self):
        """ Generate a name for this type of expression (with corresponding generator)
        """
        self.name = make_unicode(self._generate_name())

    def get_type(self):
        """ Return the type of this expression.

        Returns:
            Expression type descriptor.
        """
        return self.type
            
    def is_type(self, xtyp):
        """ Check if the type of this expression is a given one

        Args:
            xtyp:  Expected type
        Returns:
            True if expression type is the expected one
        """
        return self.type == xtyp

    def is_kind_of(self, tp):
        """ Checks if the type of this expression type is compatible with another type.

        Args:
            tp: Other type to check.
        Returns:
           True if this expression type is a kind of tp.
        """
        # Check if required type is the same
        return self.type.is_kind_of(tp)

    def is_variable(self):
        """ Checks if this expression is a variable.

        Returns:
            True if this expression is a variable, False otherwise.
        """
        return self.type.is_variable()
            
    def is_constant(self):
        """ Checks if this expression is a constant, possibly array.

        Returns:
            True if this expression is a constant, False otherwise.
        """
        return self.type.is_constant()

    def is_atom_constant(self):
        """ Checks if this expression is an atomic constant (boolean, int, float).

        Returns:
            True if this expression is an atomic constant.
        """
        return False

    def is_constraint_or_bool_expr(self):
        """ Checks if this expression is a constraint or a boolean expression

        Returns:
            True if this expression is a constraint or a boolean expression
        """
        return self.type in (Type_Constraint, Type_BoolExpr)

    def get_priority(self):
        """ Return the operation priority of this expression.

        Returns:
            Operation priority, -1 for none.
        """
        return -1

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, () if none
        """
        return(())

    def is_leaf(self):
        """ Checks if this expression is a leaf (no children)

        Returns:
            True if this expression is a leaf
        """
        return not self._get_children()

    def get_max_depth(self):
        """ Gets the maximum expression depth.

        Returns:
            Max expression depth.
        """
        depth = 1
        stack = [[self, -1]]
        while stack:
            selem = stack[-1]
            expr = selem[0]
            if not isinstance(expr, CpoExpr):
                stack.pop()
            else:
                chlds = getattr(expr, "operands", None)
                if chlds is None:
                    chlds = getattr(expr, "value", None)
                cnx = selem[1] + 1
                if (not isinstance(chlds, (list, tuple))) or (cnx >= len(chlds)):
                    depth = max(depth, len(stack))
                    stack.pop()
                else:
                    selem[1] = cnx
                    stack.append([chlds[cnx], -1])
        return depth

    def equals(self, other):
        """ Checks the equality of this expression with another object.

        Implementation is required with a different name than __eq__ name because this function is already
        overloaded to construct model expression with operator '=='.

        Args:
            other: Other object to compare with.
        Return:
            True if 'other' is equal to this object, False otherwise.
        """
        return _is_equal_expressions(self, other)

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Return:
            True if 'other' is equal to this object, False otherwise.
        """
        return (type(self) == type(other)) and (self.type == other.type) and (self.name == other.name)

    def _incr_ref_count(self):
        """ Increase reference count on this expression and create a name if more than one
        """
        # Increment reference count
        self.nbrefs += 1
        if (self.nbrefs > 1) and not(self.is_atom_constant()):
            # Add expression id if none
            if self.name is None:
                self.name = self._generate_name()


    def __str__(self):
        """ Convert this expression into a string """
        name = self.get_name()
        if is_string(name):
            return to_printable_symbol(name) + " = " + _to_string(self)
        else:
            return _to_string(self)


class CpoValue(CpoExpr):
    """ CPO model expression node representing a constant value. """
    __slots__ = ('value',  # Python value of the constant
                )

    def __init__(self, value, type):
        """ Constructor

        Args:
            value:  Constant value.
            type :  Value type.
        """
        assert isinstance(type, CpoType), "Argument 'type' should be a CpoType"
        super(CpoValue, self).__init__(type, None)
        if type.is_array_of_expr():
            value = _update_references(value)
        self.value = value

    def get_value(self):
        """ Return the value of the constant.

        Returns:
            Value of the constant.
        """
        return self.value

    def is_atom_constant(self):
        """ Checks if this expression is an atomic constant (boolean, int or float).

        Returns:
            True if this expression is an atomic constant.
        """
        return self.type.is_constant_atom()

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        # Call super
        if not super(CpoValue, self)._equals(other):
            return False
        # For array of expr, managed as a recursive call by _equals_expressions().
        if self.type.is_array_of_expr():
            return True
        # Check value
        return _is_equal_values(self.value, other.value)

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, () if none
        """
        if self.type.is_array_of_expr():
            return self.value
        return(())


class CpoExprList(list):
    """ List of CPO expressions.

    This extension of a standard Python list overwrites __getitem__ to call :func:`docplex.cp.modeler.element`
    constraint if the index is a CPO integer expression.

    This object is returned by the constructor method :meth:`integer_var_list` in this module.
    """

    def __init__(self):
        super(CpoExprList, self).__init__()

    def __getitem__(self, nx):
        """ Overloading of operator '[]' to create a CPO element() expression if index is a CPO integer expression.

        Args:
            nx: Element index
        Returns:
            If the index is a CPO integer expression, returns a CPO element() expression.
            Otherwise, returns the element corresponding to the index.
        """
        if isinstance(nx, CpoExpr) and nx.is_kind_of(Type_IntExpr):
            return(_create_operation(Oper_element, (nx, self)))
        return super(CpoExprList, self).__getitem__(nx)


class CpoFunctionCall(CpoExpr):
    """ This class represent all model expression nodes that call a predefined modeler function.

    All modeling functions are available in module :mod:`docplex.cp.modeler`.
    """
    __slots__ = ('signature',  # Signature of the operation (None if none)
                 'operands',   # List of operand expressions, or Python value for constants
                )

    def __init__(self, sign, oprnds):
        """ Constructor

        Args:
            sign:   Operation signature (children is operands).
            oprnds: List of operand expressions.
        """
        assert isinstance(sign, CpoSignature), "Argument 'sign' should be a CpoSignature"
        super(CpoFunctionCall, self).__init__(sign.get_returned_type(), None)
        self.signature = sign

        # Check no toplevel constraints
        if oprnds:
            if any(e.is_type(Type_Constraint) for e in oprnds):
                raise CpoException("A constraint can not be operand of an expression.")
            self.operands = _update_references(oprnds)
        else:
            self.operands = ()

    def get_signature(self):
        """ Return the signature of the expression.

        Return:
            Expression signature.
        """
        return self.signature

    def get_operation(self):
        """ Return the operation of the expression signature.

        Returns:
            Expression operation, None if none.
        """
        return self.signature.operation

    def is_operation(self, op):
        """ Checks if the operation of the expression signature is an expected one.

        ARgs:
            op:  Expected operation
        Returns:
            True if expression operation is the expected one, False otherwise.
        """
        return self.signature.operation == op

    def get_priority(self):
        """ Return the priority of the operation of this expression.

        Returns:
            Operation priority, -1 for none.
        """
        return self.signature.get_priority()

    def get_operands(self):
        """ Return the operands of this expression.

        Returns:
            List of operand expressions, () if no operands.
        """
        return self.operands

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        return super(CpoFunctionCall, self)._equals(other) and (self.signature == other.signature)

    def _get_children(self):
        """ Return the list of children expressions if any

        Returns:
            List of children expressions, () if none
        """
        return self.operands


class CpoVariable(CpoExpr):
    """ This class is an abstract class extended by all expression nodes that represent a CPO variable.
    """

    # Expression name generator
    __name_generator__ = _VARIABLE_ID_ALLOCATOR

    def __init__(self, type, name):
        """ Constructor:

        Args:
            type:   Expression type.
            name:   Variable name.
        """
        # Check name length
        if name is None:
            name = self._generate_name()
        super(CpoVariable, self).__init__(type, name)
        self.alias = None

    def is_variable(self):
        """ Checks if this expression is a variable.

        Returns:
            Always True.
        """
        return True


class CpoIntVar(CpoVariable):
    """ This class represents an *integer variable* that can be used in a CPO model.
    """
    __slots__ = ('domain',  # Variable domain
                 )
    
    # Expression name generator
    __name_generator__ = _INTEGER_VARIABLE_ID_ALLOCATOR

    def __init__(self, dom, name):
        # Private constructor
        super(CpoIntVar, self).__init__(Type_IntVar, name)
        self.domain = dom
        
    def set_domain(self, dom):
        """ Sets the domain of the variable.

        The domain of the variable is a list or tuple of:

           * discrete integer values,
           * list or tuple of 2 integers representing an interval.

        For example, here are valid domain definitions:

           set_domain([1, 3, 4, 5, 9])
           set_domain([1, (3, 5), 9])

        Args:
            dom: List of integers or interval tuples representing the variable domain.
        """
        self.domain = _check_arg_domain(dom, 'dom')
    
    def get_domain(self):
        """ Gets the domain of the variable.

        Returns:
            List of integers or interval tuples representing the variable domain.
        """
        return self.domain
    

    def equals(self, other):
        """ Checks if this expression is equivalent to another

        Args:
            other: Other object to compare with.
        Return:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        return super(CpoIntVar, self).equals(other) and (self.domain == other.domain)


###############################################################################
## Scheduling expressions
###############################################################################

INTERVAL_MAX = (INT_MAX // 2) - 1
""" Maximum interval variable range value """

INTERVAL_MIN = -INTERVAL_MAX
""" Minimum interval variable range value """

INFINITY = float('inf')
""" Infinity """

DEFAULT_INTERVAL = (0, INTERVAL_MAX)
""" Default interval. """

# Different interval variable presence states
_PRES_PRESENT   = "present"   # Always present
_PRES_ABSENT    = "absent"    # Always absent
_PRES_OPTIONAL  = "optional"  # Present or absent, choice made by the solver


class CpoIntervalVar(CpoVariable):
    """ This class represents an *interval variable* that can be used in a CPO model.
    """
    __slots__ = ('start',        # Start domain
                 'end',          # End domain
                 'length',       # Length domain
                 'size',         # Size domain
                 'intensity',    # Specifies relation between size and length of the interval.
                 'granularity',  # Scale of the intensity function (int)
                 'presence',     # Presence requirement (in _PRES_*)
                 )

    # Expression name generator
    __name_generator__ = _INTERVAL_VARIABLE_ID_ALLOCATOR

    def __init__(self, start, end, length, size, intensity, granularity, presence, name):
        # Private constructor
        super(CpoIntervalVar, self).__init__(Type_IntervalVar, name)
        self.start   = start
        self.end     = end
        self.length  = length
        self.size    = size
        self.presence = presence
        self.granularity = granularity
        self.intensity = intensity
        if intensity is not None:
            intensity._incr_ref_count()

    def set_start(self, intv):
        """ Sets the start interval.

        Args:
            intv: Start of the interval (single integer or interval expressed as a tuple of 2 integers).
        """
        self.start = _check_arg_interval(intv, "intv")

    def get_start(self):
        """ Gets the start of the interval.

        Returns:
            Start of the interval (interval expressed as a tuple of 2 integers).
        """
        return self.start

    def set_start_min(self, mn):
        """ Sets the minimum value of the start interval.

        Args:
            mn: Min value of the start of the interval.
        """
        self.start = (mn, self.start[1])

    def set_start_max(self, mx):
        """ Sets the maximum value of the start interval.

        Args:
            mx: Max value of the start of the interval.
        """
        self.start = (self.start[0], mx)

    def set_end(self, intv):
        """ Sets the end of the interval.

        Args:
            intv: End of the interval (single integer or interval expressed as a tuple of 2 integers).
        """
        self.end = _check_arg_interval(intv, "intv")

    def get_end(self):
        """ Gets the end of the interval.

        Returns:
            End of the interval (interval expressed as a tuple of 2 integers)
        """
        return self.end

    def set_end_min(self, mn):
        """ Sets the minimum value of the end interval.

        Args:
            mn: Min value of the end of the interval.
        """
        self.end = (mn, self.end[1])

    def set_end_max(self, mx):
        """ Sets the maximum value of the end of the interval.

        Args:
            mx: Max value of the end of the interval.
        """
        self.end =(self.end[0], mx)

    def set_length(self, intv):
        """ Sets the length interval.

        Args:
            intv: Length of the interval (single integer or interval expressed as a tuple of 2 integers).
        """
        self.length = _check_arg_interval(intv, "intv")

    def get_length(self):
        """ Gets the length interval.

        Returns:
            Length of the interval (interval expressed as a tuple of 2 integers).
        """
        return self.length

    def set_length_min(self, mn):
        """ Sets the minimum value of the length interval.

        Args:
            mn: Min value of the length of the interval min value.
        """
        self.length = (mn, self.length[1])

    def set_length_max(self, mx):
        """ Sets the maximum value of the length interval.

        Args:
            mx: Max value of the length of the interval.
        """
        self.length = (self.length[0], mx)

    def set_size(self, intv):
        """ Sets the size of the interval.

        Args:
            intv: Size of the interval (single integer or interval expressed as a tuple of 2 integers).
        """
        self.size = _check_arg_interval(intv, "intv")

    def get_size(self):
        """ Gets the size of the interval.

        Returns:
            Size of the interval (interval expressed as a tuple of 2 integers).
        """
        return self.size

    def set_size_min(self, mn):
        """ Sets the minimum value of the size interval.

        Args:
            mn: Min value of the size of the interval.
        """
        self.size = (mn, self.size[1])

    def set_size_max(self, mx):
        """ Sets the maximum value of the size interval.

        Args:
            mx: Max value of the size of the interval.
        """
        self.size = (self.size[0], mx)

    def set_present(self):
        """ Specifies that this IntervalVar must be present. """
        self.presence = _PRES_PRESENT

    def is_present(self):
        """ Check if this interval variable must be present.

        Returns:
            True if this interval variable must be present, False otherwise.
        """
        return self.presence == _PRES_PRESENT

    def set_absent(self):
        """ Specifies that this interval variable must be absent. """
        self.presence = _PRES_ABSENT

    def is_absent(self):
        """ Check if this interval variable must be absent.

        Returns:
            True if this interval variable must be absent, False otherwise.
        """
        return self.presence == _PRES_ABSENT

    def set_optional(self):
        """ Specifies that this interval variable is optional. """
        self.presence = _PRES_OPTIONAL

    def is_optional(self):
        """ Check if this interval variable is optional.

        Returns:
            True if this interval variable is optional, False otherwise.
        """
        return self.presence == _PRES_OPTIONAL

    def set_intensity(self, intensity):
        """ Sets the intensity function of this interval var.

        Args:
           intensity:  Intensity function (None, or StepFunction).
        """
        _check_arg_intensity(intensity, self.granularity)
        self.intensity = intensity
        if intensity is not None:
            intensity._incr_ref_count()

    def get_intensity(self):
        """ Gets the intensity function of this interval var.

        Returns:
           Intensity function (None, or StepFunction).
        """
        return self.intensity

    def set_granularity(self, granularity):
        """ Sets the scale of the intensity function.

        Args:
            granularity: Scale of the intensity function (integer).
        """
        assert (granularity is None) or (is_int(granularity) and (granularity >= 0)), "Argument 'granularity' should be None or positive integer"
        self.granularity = granularity 

    def get_granularity(self):
        """ Get the scale of the intensity function.

        Returns:
            Scale of the intensity function, None for default (100)
        """
        return self.granularity

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        # Call super
        if not super(CpoIntervalVar, self)._equals(other):
            return False

        # Check same attributes (intensity processed as a children)
        return self.start == other.start and \
               self.end == other.end and \
               self.length == other.length and \
               self.size == other.size and \
               self.granularity == other.granularity and \
               self.presence == other.presence

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, () if none
        """
        return (self.intensity,) if self.intensity else ()


class CpoSequenceVar(CpoVariable):
    """ This class represents an *sequence variable* that can be used in a CPO model.
    """
    __slots__ = ('vars',   # List of variables
                 'types',  # Variable types
                )
    
    # Expression name generator
    __name_generator__ = _SEQUENCE_VARIABLE_ID_ALLOCATOR

    def __init__(self, vars, types=None, name=None):
        """ Creates a new sequence variable.

        This method creates an instance of sequence variable on the set of interval variables defined
        by the array 'vars'.
        A list of non-negative integer types can be optionally  specified.
        List of variables and types must be of the same size and interval variable vars[i] will have type types[i]
        in the sequence variable.

        Args:
            vars:  List of IntervalVars that constitute the sequence.
            types: List of variable types as integers, same size as vars, or None (default).
            name:  Name of the sequence, None for automatic naming.
        """
        # Check  arguments
        if isinstance(vars, CpoValue):
            assert vars.is_type(Type_IntervalVarArray)
            vars = vars.value
        else:
            assert is_array_of_type(vars, CpoIntervalVar), "Argument 'vars' should be an array of CpoIntervalVar"
        if types is not None:
            if isinstance(types, CpoValue):
                assert types.is_type(Type_IntArray)
                types = types.value
            else:
                types = _check_and_expand_interval_tuples('types', types)
            assert len(types) == len(vars), "The array of types should have the same length than the array of variables."
        # Store attributes
        super(CpoSequenceVar, self).__init__(Type_SequenceVar, name)
        self.vars = vars
        self.types = types

    def get_interval_variables(self):
        """ Gets the array of variables.

        Returns:
            Array of interval variables that are in the sequence.
        """
        return self.vars
    
    def get_types(self):
        """ Gets the array of types.

        Returns:
            Array of variable types (array of integers), None if no type defined.
        """
        return self.types
    
    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        # Call super
        if not super(CpoSequenceVar, self)._equals(other):
            return False
        # Check equality of types
        if self.types != other.types:
            return False
        # List of variables is processed as expression children
        return True

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, () if none
        """
        return self.vars

    def __str__(self):
        """ Convert this expression into a string """
        return "SequenceVar({}, types={})".format(self.vars, self.types)


class CpoTransitionMatrix(CpoExpr):
    """ This class represents a *transition matrix* that is used in CPO model to represent transition distances.
    """
    __slots__ = ('size',    # Matrix width/height
                 'matrix',  # Matrix values
                 'is_flat', # Indicates that the list of values is flat
                )
    
    def __init__(self, size=None, values=None, name=None):
        """ Creates a new transition matrix (square matrix of integers).

        A transition matrix is a square matrix of non-negative integers that represents a minimal distance between
        two interval variables.
        An instance of transition matrix can be used in the no_overlap constraint and in state functions.

          * In a no_overlap constraint the transition matrix represents the minimal distance between two
            non-overlapping interval variables.
            The matrix is indexed using the integer types of interval variables in the sequence variable
            of the no_overlap constraint.

          * In a state function, the transition matrix represents the minimal distance between two integer
            states of the function.

        There are two ways to create a transition matrix:

          * Giving only its size. In this case, a transition matrix is created by this constructor with all
            values initialized to zero. Matrix values can then be set using :meth:`set_value` method.

          * Giving the matrix values either as list of rows, each row being a list of integers,
            or as a single list of integers containing the concatenation of rows.
            Given value is not duplicated. Any change on the value is reflected in the transition matrix,
            and any change in the matrix using :meth:`set_value` is reflected in the source object.

        Args:
            size (optional):  Matrix size (width or height).
                    If not given, the `values` argument must be given.
            values (optional):  Matrix value as list of integers or list of rows.
                    If not given, the method `set_value()` should be called to initialize matrix content.
            name (optional):  Name of the matrix. None by default.
        """

        super(CpoTransitionMatrix, self).__init__(Type_TransitionMatrix, name)

        if size is not None:
            assert is_int(size) and size >= 0, "Argument 'size' should be a positive integer"
            self.size = size

        if values is None:
            assert size is not None, "At least 'size' or 'values' should be given"
            # Allocate matrix as flat array
            self.matrix = [0] * (size * size)
            self.is_flat = True
        else:
            assert is_array(values), "Argument 'values' should be an array of integers"
            # Check empty array
            alen = len(values)
            if alen == 0:
                assert size is None or size == 0, "Size should be zero with an empty array of values"
                self.size = 0
                self.is_flat = True
            else:
                if is_int(values[0]):
                    # Assume matrix is a flat array of integers
                    self.is_flat = True
                    assert all(is_int(v) and v >= 0 for v in values), "All matrix values should be positive integers"
                    if size is None:
                        size = int(math.sqrt(alen))
                        assert (alen == (size * size)), "Array of values should have a length that is a perfect square"
                        self.size = size
                    else:
                        assert alen == size * size, "Array of values should have a size equal to size * size"
                else:
                    # Assume matrix is an array of arrays
                    self.is_flat = False
                    if size is None:
                        self.size = alen
                    else:
                        assert (alen == size), "Array of values has a size that is different than the given one"
                    assert all(is_array(r) for r in values), "Array of values should be a list of lists of integers"
                    assert all((len(r) == alen) for r in values), "All values rows should have the same size"
                    assert all(all(is_int(v) and v >= 0 for v in r) for r in values), "All matrix values should be positive integers"
            self.matrix = values

    def get_size(self):
        """ Returns the size of the matrix.

        Returns:
            Matrix size.
        """
        return self.size

    def get_value(self, from_state, to_state):
        """ Returns a value in the transition matrix.

        Args:
            from_state: Index of the from state.
            to_state:   Index of the to state.
        Returns:
            Transition value.
        """
        assert_arg_int_interval(from_state, 0, self.size, "from_state")
        assert_arg_int_interval(to_state, 0, self.size, "to_state")
        if self.is_flat:
            return self.matrix[(from_state * self.size) + to_state]
        else:
            return self.matrix[from_state][to_state]

    def get_all_values(self):
        """ Returns an iterator on all matrix values, in row/column order

        Returns:
            Iterator on all values
        """
        return (self.get_value(f, t) for f in range(self.size) for t in range(self.size))

    def get_matrix(self):
        """ Returns the complete transition matrix.

        Returns:
            Transition matrix as a list of integers that is the concatenation of all matrix rows.
        """
        return self.matrix

    def set_value(self, from_state, to_state, value):
        """ Sets a value in the transition matrix.

        Args:
            from_state: Index of the from state.
            to_state:   Index of the to state.
            value:      Transition value.
        """
        assert_arg_int_interval(from_state, 0, self.size, "from_state")
        assert_arg_int_interval(to_state, 0, self.size, "to_state")
        assert is_int(value) and value >= 0, "Value should be a positive integer"
        if self.is_flat:
            self.matrix[from_state * self.size + to_state] = value
        else:
            self.matrix[from_state][to_state] = value

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        return super(CpoTransitionMatrix, self)._equals(other) and \
               (self.size == other.size) and \
               (self.matrix == other.matrix)

    def __str__(self):
        """ Convert this expression into a string """
        return "TransitionMatrix" + to_string(self.matrix)
    
    
class CpoTupleSet(CpoExpr):
    """ This class is used to represent a set of integer tuples.
    """
    __slots__ = ('size',      # Size of a single tuple
                 'tupleset',  # List of tuples
                 )
    
    def __init__(self, size=-1, name=None):
        """ Constructor

        Args:
            size (optional): Tuple size; default value is -1 for automatic size.
            name (optional): Name of the tuple set. Default is None.
        """
        assert is_int(size), "Argument 'size' should be an int"
        super(CpoTupleSet, self).__init__(Type_TupleSet, name)
        self.size = size
        self.tupleset = []
         
    def get_size(self):
        """ Returns the size of one tuple in this set.

        Returns:
            Tuple size, -1 if undefined.
        """
        return self.size

    def add(self, tpl):
        """ Appends a tuple at the end of this tuple set.

        Args:
            tpl: Tuple to add.
        """
        # Check for intervals
        tpl = _check_and_expand_interval_tuples("tpl", tpl)
        if self.size < 0:
            self.size = len(tpl)
        elif len(tpl) != self.size:
            raise CpoException("You must add only tuples of size " + str(self.size))
        self.tupleset.append(tpl)

    def add_set(self, tpls):
        """ Appends a set of tuples in this tuple set.

        Args:
            tpls: Iterator of tuples to add.
        """
        for t in tpls:
            self.add(t)

    def get_tuple_set(self):
        """ Returns the complete tuple set.

        Returns:
            Tuple set (list of tuples)
        """
        return self.tupleset

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        return super(CpoTupleSet, self)._equals(other) and \
               (self.size == other.size) and \
               (self.tupleset == other.tupleset)

    def __len__(self):
        """ Returns the number of tuples in this tuple set.

        Returns:
            Number of tuples in this tuple set.
        """
        return len(self.tupleset)

    def __str__(self):
        """ Convert this expression into a string """
        return "TupleSet" + to_string(self.tupleset)
    
    
class CpoStateFunction(CpoVariable):
    """ This class represents a *state function* expression node.

    State functions are used by *interval variables* to represent the evolution of a state variable over time.
    """
    __slots__ = ('trmtx',      # Transition matrix
                )

    # Expression name generator
    __name_generator__ = _STATE_FUNCTION_ID_ALLOCATOR

    def __init__(self, trmtx=None, name=None):
        """ Creates a new state function.

        Args:
            trmtx (optional):  Transition matrix.
                               If not given in the constructor, method :meth:`set_transition_matrix` should be called
                               after the constructor.
            name (optional):   Name of the state function.
        """
        # Force name for state functions
        super(CpoStateFunction, self).__init__(Type_StateFunction, name)
        self.set_transition_matrix(trmtx)

    def set_transition_matrix(self, trmtx):
        """ Sets the transition matrix.

        Args:
            trmtx: Transition matrix, None if none.
        """
        if trmtx is not None:
            assert isinstance(trmtx, CpoTransitionMatrix), "Argument 'trmtx' should be a CpoTransitionMatrix"
            trmtx._incr_ref_count()
        self.trmtx = trmtx

    def get_transition_matrix(self):
        """ Returns the transition matrix.

        Returns:
            Transition matrix, None if none.
        """
        return self.trmtx

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        return super(CpoStateFunction, self)._equals(other)
        # Transition matrix is checked as children

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, () if none
        """
        return (self.trmtx,) if self.trmtx else ()

    def __str__(self):
        """ Convert this expression into a string """
        return "StateFunction" + to_string(self.get_transition_matrix())


###############################################################################
## Factory Functions
###############################################################################

def integer_var(min, max=None, name=None):
    """ Creates an integer variable.

    An integer variable is a decision variable with a set of potential values called 'domain of the variable'.
    This domain can be expressed either:

     * as a single interval, with a minimum and a maximum bounds included in the domain,
     * or as an extensive list of values and/or intervals.

    When the domain is given extensively, an interval of the domain is represented by a tuple (min, max).
    Examples of variable domains expressed extensively are:

     * (1, 2, 3, 4)
     * (1, 2, (3, 7), 9)
     * ((1, 2), (7, 9))

    Following integer variable declarations are equivalent:

     * v = integer_var(0, 9, "X")
     * v = integer_var((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "X")
     * v = integer_var((0, (1, 5), (6, 7), 8, 9), "X")
     * v = integer_var(((0, 9)), "X")

    Args:
        min:  Domain min value, or extensive list of values and/or intervals expressed as tuples of integers.
        max (optional):  Domain max value.
                         Default is None and indicates that 'min' should contain an extensive list of values.
        name (optional): Variable name, default is None for automatic name.
    Returns:
        CpoIntVar expression
    """
    return CpoIntVar(_build_int_var_domain(min, max), name)


def integer_var_list(size, min, max=None, name=None):
    """ Creates a list of integer variables.

    This methods creates a list of integer variables whose size is given as first parameter.
    All other parameters are identical to those requested by integer_var() method, that allows to
    create a single integer variable.
    See the documentation of this method for details.

    If a name is given, each variable of the list is created with this
    name concatenated with the index of the variable in the list, starting by zero.

    Args:
        size: Size of the list of variables
        min:  Domain min value, or extensive list of values and/or intervals expressed as tuples of integers.
        max (optional):  Domain max value.
                         Default is None and indicates that 'min' should contain an extensive list of values.
        name (optional): Variable name prefix. If not given, a name prefix is generated automatically.
    Returns:
        List of integer variables.
    """
    if name is None:
        name = CpoIntVar._generate_name() + "_"
    dom = _build_int_var_domain(min, max)
    res = CpoExprList()
    for i in range(size):
        res.append(CpoIntVar(dom, name + str(i)))
    return res


def integer_var_dict(keys, min, max=None, name=None):
    """ Creates a dictionary of integer variables.

    This methods creates a dictionary of integer variables associated to a list of keys given as first parameter.
    All other parameters are identical to those requested by integer_var() method, that allows to
    create a single integer variable.
    See the documentation of this method for details.

    If a name is given, each variable of the list is created with this
    name concatenated with the string representation of the corresponding key.
    The parameter 'name' can also be a function that is called to build the variable name
    with the variable key as parameter.

    Args:
        keys: Iterable of variable keys.
        min:  Domain min value, or extensive list of values and/or intervals expressed as tuples of integers.
        max (optional):  Domain max value.
                         Default is None and indicates that 'min' should contain an extensive list of values.
        name (optional): Variable name prefix, or function to be called on dictionary key (example: str).
                         If not given, a name prefix is generated automatically.
    Returns:
        Dictionary of CpoIntVar objects (OrderedDict).
    """
    if name is None:
        name = CpoIntVar._generate_name() + "_"
    dom = _build_int_var_domain(min, max)
    res = collections.OrderedDict()
    i = 0
    isnamestr = is_string(name)
    for k in keys:
        if isnamestr:
            vname = name + str(i) 
        else:
            vname = name(k)
        res[k] = CpoIntVar(dom, vname)
        i += 1
    return res


def binary_var(name=None):
    """ Creates a binary integer variable.

    An binary variable is an integer variable with domain limited to 0 and 1

    Args:
        name (optional): Variable name, default is None for automatic name.
    Returns:
        CpoIntVar expression
    """
    return CpoIntVar(_BINARY_DOMAIN, name)


def binary_var_list(size, name=None):
    """ Creates a list of binary variables.

    This methods creates a list of binary variables.

    If a name is given, each variable of the list is created with this
    name concatenated with the index of the variable in the list, starting by zero.

    Args:
        size: Size of the list of variables
        name (optional): Variable name prefix. If not given, a name prefix is generated automatically.
    Returns:
        List of binary integer variables.
    """
    return integer_var_list(size, _BINARY_DOMAIN, name=name)


def binary_var_dict(keys, name=None):
    """ Creates a dictionary of binary variables.

    This methods creates a dictionary of binary variables associated to a list of keys given as first parameter.

    If a name is given, each variable of the list is created with this
    name concatenated with the string representation of the corresponding key.
    The parameter 'name' can also be a function that is called to build the variable name
    with the variable key as parameter.

    Args:
        keys: Iterable of variable keys.
        name (optional): Variable name prefix, or function to be called on dictionary key (example: str).
                         If not given, a name prefix is generated automatically.
    Returns:
        Dictionary of CpoIntVar objects (OrderedDict).
    """
    return integer_var_dict(keys, _BINARY_DOMAIN, name=name)


def interval_var(start=DEFAULT_INTERVAL, end=DEFAULT_INTERVAL, length=DEFAULT_INTERVAL, size=DEFAULT_INTERVAL,
                 intensity=None, granularity=None, optional=False, name=None):
    """ Creates an interval variable.

    Represents an interval of integers. Interval variables are used mostly for scheduling to represent a
    task as an interval of time.
    In its most basic form, an interval variable can be seen as a pair of two integer variables start and end
    such that start < end.
    However there is an important difference: the interval variable can be absent to represent the fact that the
    interval does not exist at all (which is different from a zero-length interval).

    Args:
        start (optional):       Allowed range for the start of the interval (single integer or interval expressed as a tuple of 2 integers).
        end (optional):         Allowed range for the end the interval (single integer or interval expressed as a tuple of 2 integers).
        length (optional):      Allowed range for the length the interval (single integer or interval expressed as a tuple of 2 integers).
        size (optional):        Allowed range for the size the interval (single integer or interval expressed as a tuple of 2 integers).
        intensity (optional):   StepFunction that specifies relation between size and length of the interval.
        granularity (optional): Scale of the intensity function.
        optional (optional):    Optional presence indicator.
        name (optional):        Name of the variable. If not given, a name is generated automatically.
    Returns:
        IntervalVar expression.
    """
    start  = _check_arg_interval(start,  "start")
    end    = _check_arg_interval(end,    "end")
    length = _check_arg_interval(length, "length")
    size   = _check_arg_interval(size,   "size")
    _check_arg_intensity(intensity, granularity)
    #presence = _PRES_OPTIONAL if (_check_arg_boolean(optional, "optional") or not present) else _PRES_PRESENT
    presence = _PRES_OPTIONAL if (_check_arg_boolean(optional, "optional")) else _PRES_PRESENT
    return CpoIntervalVar(start, end, length, size, intensity, granularity, presence, name)


def interval_var_list(asize, start=DEFAULT_INTERVAL, end=DEFAULT_INTERVAL, length=DEFAULT_INTERVAL, size=DEFAULT_INTERVAL,
                 intensity=None, granularity=None, optional=False, name=None):
    """ Creates a list of interval variables.

    If a name is given, each variable of the array is created with this
    name concatenated with the index of the variable in the list.

    Args:
        asize:                  Size of the list of variables
        start (optional):       Allowed range for the start of the interval (single integer or interval expressed as a tuple of 2 integers).
        end (optional):         Allowed range for the end the interval (single integer or interval expressed as a tuple of 2 integers).
        length (optional):      Allowed range for the length the interval (single integer or interval expressed as a tuple of 2 integers).
        size (optional):        Allowed range for the size the interval (single integer or interval expressed as a tuple of 2 integers).
        intensity (optional):   StepFunction that specifies relation between size and length of the interval.
        granularity (optional): Scale of the intensity function.
        optional (optional):    Optional presence indicator.
        name (optional):        Name of the variable. If not given, a name is generated automatically.
    Returns:
        List of interval variables.
    """
    start  = _check_arg_interval(start,  "start")
    end    = _check_arg_interval(end,    "end")
    length = _check_arg_interval(length, "length")
    size   = _check_arg_interval(size,   "size")
    _check_arg_intensity(intensity, granularity)
    #presence = _PRES_OPTIONAL if (_check_arg_boolean(optional, "optional") or not present) else _PRES_PRESENT
    presence = _PRES_OPTIONAL if (_check_arg_boolean(optional, "optional")) else _PRES_PRESENT
    if name is None:
        name = CpoIntervalVar._generate_name() + "_"
    res = []
    for i in range(asize):
        res.append(CpoIntervalVar(start, end, length, size, intensity, granularity, presence, name + str(i)))
    return res


def sequence_var(vars, types=None, name=None):
    """ Creates a new sequence variable (list of interval variables).

    This method creates an instance of sequence variable on the set of interval variables defined
    by the array 'vars'.
    A list of non-negative integer types can be optionally  specified.
    List of variables and types must be of the same size and interval variable vars[i] will have type types[i]
    in the sequence variable.

    Args:
        vars:  List of IntervalVars that constitute the sequence.
        types: List of variable types as integers, same size as vars, or None (default).
        name:  Name of the sequence, None for automatic naming.
    Returns:
        IntervalVar expression.
    """
    return CpoSequenceVar(vars, types, name)


def transition_matrix(size=None, values=None, name=None):
    """ Creates a new transition matrix (square matrix of integers).

    A transition matrix is a square matrix of non-negative integers that represents a minimal distance between
    two interval variables.
    An instance of transition matrix can be used in the no_overlap constraint and in state functions.

      * In a no_overlap constraint the transition matrix represents the minimal distance between two
        non-overlapping interval variables.
        The matrix is indexed using the integer types of interval variables in the sequence variable
        of the no_overlap constraint.

      * In a state function, the transition matrix represents the minimal distance between two integer
        states of the function.

    There are two ways to create a transition matrix:

      * Giving only its size. In this case, a transition matrix is created by this constructor with all
        values initialized to zero. Matrix values can then be set using :meth:`set_value` method.

      * Giving the matrix values either as list of rows, each row being a list of integers,
        or as a single list of integers containing the concatenation of rows.
        Given value is not duplicated. Any change on the value is reflected in the transition matrix,
        and any change in the matrix using :meth:`set_value` is reflected in the source object.

    Args:
        size (optional):  Matrix size (width or height).
                If not given, the `values` argument must be given.
        values (optional):  Matrix value as list of integers or list of rows.
                If not given, the method `set_value()` should be called to initialize matrix content.
        name (optional):  Name of the matrix. None by default.
    Returns:
        TransitionMatrix expression.
    """
    return CpoTransitionMatrix(size, values, name)
 
 
def tuple_set(size, name=None):
    """ Create tuple set.

    The tuple set can be created empty, and then filled using methods provided by the TupleSet object.
    Another option is to pass the list of all tuples as first parameter of this function.

    Args:
        size:  Size of one tuple, or initial list of tuples.
        name:  Object name (default is None).
    Returns:
        TupleSet expression.
    """
    # Check if a tupleset is given
    if (is_array(size)):
        res = CpoTupleSet(-1, name)
        res.add_set(size)
        return res
    return CpoTupleSet(size, name)


def state_function(trmtx=None, name=None):
    """ Create a new State Function

    Args:
        trmtx: Transition matrix
        name: Name of the state function
    Returns:
        CpoStateFunction expression
    """
    return CpoStateFunction(trmtx, name)


###############################################################################
##  Public Functions
###############################################################################

# Cache of CPO expressions corresponding to Python values
# This cache is used to retrieve the CPO expression that corresponds to a Python expression
# that is used multiple times in a model.
# This allows to:
#  - speed-up conversion as expression type has not to be recompute again
#  - reduce CPO file length as common expressions are easily identified.
_CACHE_CONTEXT = context.model.cache
_CPO_VALUES_FROM_PYTHON = ObjectCache(_CACHE_CONTEXT.size)

# Lock to protect the map
_CPO_VALUES_FROM_PYTHON_LOCK = threading.Lock()

def build_cpo_expr(val):
    """ Builds an expression from a given Python value.

    This method uses a cache to return the same CpoExpr for the same constant.

    Args:
        val: Value to convert (possibly already an expression).
    Returns:
        Corresponding expression.
    Raises:
        CpoException if conversion is not possible.
    """
    # Check if already a CPO expression
    if isinstance(val, CpoExpr):
        return(val)

    # Check none
    if val is None:
        return None

    #  Check atoms (not cached)
    if is_number(val):
        return create_cpo_expr(val)

    # Check if already in the cache
    cactive = _CACHE_CONTEXT.active
    if cactive:
        _CPO_VALUES_FROM_PYTHON_LOCK.acquire()
        cpval = _CPO_VALUES_FROM_PYTHON.get(val)
        _CPO_VALUES_FROM_PYTHON_LOCK.release()
        if cpval is not None:
            return cpval

    # Expand iterators
    if isinstance(val, collections.Iterator):
        rval = tuple(val)
    else:
        rval = val

    # Build new expression
    cpval = create_cpo_expr(rval)
    if cactive:
        _CPO_VALUES_FROM_PYTHON_LOCK.acquire()
        _CPO_VALUES_FROM_PYTHON.set(val, cpval)
        _CPO_VALUES_FROM_PYTHON_LOCK.release()
    return cpval


def create_cpo_expr(value):
    """ Create a new CP expression from a given Python value

    Args:
        value:  Operation descriptor
    Returns:
        New expression
    Raises:
        CpoException if it is not possible.
    """
    # Determine type
    typ = _get_cpo_type(value)
    if typ is None:
        raise CpoException("Impossible to build a CPO expression with python value '" + to_string(value) + "'")

    # Convert array elements if required
    if typ.is_array_of_expr():
        nval = []
        narr = False
        for v in value:
            if isinstance(v, CpoExpr):
                nval.append(v)
            else:
                nval.append(create_cpo_expr(v))
                narr = True
        return CpoValue(nval if narr else value, typ)

    if typ == Type_TupleSet:
        res = CpoTupleSet()
        res.add_set(value)
        return res

    # Default
    return CpoValue(value, typ)


def _create_operation(oper, params):
    """ Create a new expression that matches an operation descriptor

    Search in the signatures which one matches a set or arguments
    and then create an instance of the returned expression

    Args:
        oper:   Operation descriptor
        params: List of expression parameters
    Returns:
        New expression 
    Raises:
        CpoException if no operation signature matches arguments
    """
    assert isinstance(oper, CpoOperation)

    # Convert arguments in CPO expressions
    args = tuple(map(build_cpo_expr, params))
      
    # Search corresponding signature
    s = _get_matching_signature(oper, args)
    if s is None:
        raise CpoException("The combination of parameters ({}) is not allowed for operation '{}' ({})"
                           .format(", ".join(map(_get_cpo_type_str, args)), oper.get_py_name(), oper.get_cpo_name()))

    # Check arguments values when applicable
    # TODO (range currently not in parameters)
    
    # Create result expression
    return CpoFunctionCall(s, args)


###############################################################################
##  Private Functions
###############################################################################

# Mapping of Python types to CPO types
_PYTHON_TO_CPO_TYPE = {}
for t in BOOL_TYPES:
    _PYTHON_TO_CPO_TYPE[t] = Type_Bool
for t in INTEGER_TYPES:
    _PYTHON_TO_CPO_TYPE[t] = Type_Int
for t in FLOAT_TYPES:
    _PYTHON_TO_CPO_TYPE[t] = Type_Float
_PYTHON_TO_CPO_TYPE[CpoIntVar]           = Type_IntVar
_PYTHON_TO_CPO_TYPE[CpoIntervalVar]      = Type_IntervalVar
_PYTHON_TO_CPO_TYPE[CpoSequenceVar]      = Type_SequenceVar
_PYTHON_TO_CPO_TYPE[CpoTransitionMatrix] = Type_TransitionMatrix
_PYTHON_TO_CPO_TYPE[CpoTupleSet]         = Type_TupleSet
_PYTHON_TO_CPO_TYPE[CpoStateFunction]    = Type_StateFunction


def _check_arg_boolean(arg, name):
    """ Check that an argument is a boolean and raise error if wrong
    Args:
        arg:  Argument value
        name: Argument name
    Returns:
        Boolean to be set
    Raises:
        Exception if argument has the wrong format
    """
    assert is_bool(arg), "Argument '" + name + "' should be a boolean"
    return arg


def _check_arg_interval(arg, name):
    """ Check that an argument is an interval and raise error if wrong
    Args:
        arg:  Argument value
        name: Argument name
    Returns:
        Interval to be set
    Raises:
        Exception if argument has the wrong format
    """
    if is_int(arg):
        return (arg, arg)
    assert isinstance(arg, (list, tuple)) and is_int(arg[0]) and is_int(arg[1]), "Argument '" + name + "' should be an integer or an interval expressed as a tuple"
    return arg


def _check_arg_domain(val, name):
    """ Check that an argument is a correct domain and raise error if wrong
    Args:
        val:  Argument value
        name: Argument name
    Returns:
        Domain to be set
    Raises:
        Exception if argument has the wrong format
    """
    assert is_array(val), "Argument '" + name + "' should be a list of integers and/or intervals (tuples of 2 integers)"
    assert all(is_int(v) or is_interval_tuple(v) for v in val), \
           "Argument '" + name + "' should be a list of integers and/or intervals (tuples of 2 integers)"
    return val


def _build_int_var_domain(min, max):
    """ Create/check integer variable domain from parameters min and max
    Args:
        min:  Domain min value, or extensive list of values and/or intervals expressed as tuples of integers.
        max:  Domain max value, default is None and indicates that 'min' should contain an extensive list of values.
    Returns:
        Valid integer variable domain
    Raises:
        Exception if argument has the wrong format
    """
    if max is None:
        assert is_array(min), "When 'max' is not specified, argument 'min' should be a list of integers and/or intervals (tuples of 2 integers)"
        return(_check_arg_domain(min, 'min'))
    else:
        assert is_int(min) and is_int(max), "Domain 'min' and 'max' values should be int"
        return ((min, max),)


def _check_arg_step_function(arg, name):
    """ Check that an argument is a step function and raise error if wrong
    Args:
        arg:  Argument value
        name: Argument name
    Returns:
        Resulting step function
    Raises:
        Exception if argument has the wrong format
    """
    assert isinstance(arg, CpoExpr) and (arg.get_type() == Type_StepFunction), "Argument '" + name + "' should be a StepFunction"
    return arg


def _check_arg_intensity(intensity, granularity):
    """ Check the intensity parameter of an interval var.
    Args:
       intensity:   Intensity function (None, or StepFunction).
       granularity: Granularity
    """
    if __debug__ and (intensity is not None):
        assert isinstance(intensity, CpoExpr) and (intensity.is_type(Type_StepFunction)), "Interval variable 'intensity' should be None or a StepFunction"
        if granularity is None:
            granularity = 100
        for (s, v) in intensity.get_step_list():
            assert is_int(s), "'intensity' step start should be an integer"
            assert is_int(v) and (v >= 0) and (v <= granularity), "'intensity' step value should be in [0..granularity]"


def _check_and_expand_interval_tuples(name, arr):
    """ Check that a list contains only integers and expand interval tuples if any
    Args:
        name:  Argument name
        arr:   Array of integers and/or intervals
    Returns:
        Array of integers
    Raises:
        Exception if wrong type
    """
    assert isinstance(arr, (list, tuple)), "Argument '{}' (type {}) should be a list of integers or intervals".format(name, type(arr))
    res = None
    for i in range(len(arr)):
        v = arr[i]
        if is_int(v):
            if res:
                res.append(v)
        else:
            assert is_interval_tuple(v), "Argument '{}' (type {}) should be a list of integers or intervals".format(name, type(arr))
            if not res:
                res = arr[:i]
            res.extend(range(v[0], v[1] + 1))
    return res if res else arr


def _get_cpo_type(val):
    """ Determine the CPO type of a given Python value
    Args:
        val: Python value
    Returns:
        Corresponding CPO Type, None if none
    """
    # Check simple types
    ctyp = _PYTHON_TO_CPO_TYPE.get(type(val))
    if ctyp:
        return ctyp

    # Check CPO Expr
    if isinstance(val, CpoExpr):
        return val.get_type()

    # Check numpy Array Scalars (special case when called from overloaded)
    if IS_NUMPY_AVAILABLE and type(val) is numpy.ndarray and not val.shape:
        return _PYTHON_TO_CPO_TYPE.get(val.dtype.type)

    # Check arrays
    if not is_array(val):
        return None
    
    # Check empty Array
    if len(val) == 0:
        return Type_IntArray

    # Get the most common type to all array elements
    gt = None
    for v in val:
        # Determine type of element
        if is_interval_tuple(v):
            # Special case for intervals
            nt = Type_Int
        else:
            nt = _get_cpo_type(v)
            if nt is None:
                return None
        # Combine with global type
        gt = nt if gt is None else gt.get_common_type(nt)
        if gt is None:
            return None
    
    # Determine array type for result element type
    if gt == Type_IntArray:
        return Type_TupleSet
    return gt.get_array_type()
    
    
def _get_cpo_type_str(val):
    """ Get the CPO type name of a value

    Args:
        val: Value
    Returns:
        Value type string in CPO types
    """
    return _get_cpo_type(val).get_name()


def _update_references(oprnds):
    """ Increase reference count on all operand expressions and build a tuple with result

    Args:
        oprnds: Array of operands
    Returns:
        Tuple of operands with nbref incremented
    """
    # Build a tuple if not
    if not isinstance(oprnds, tuple):
        oprnds = tuple(oprnds)

    # Increment reference count
    for c in oprnds:
        c._incr_ref_count()

    return oprnds


def _get_matching_signature(oper, args):
    """ Search the first operation signature matched by a list of arguments

    Args:
        oper: Operation where searching signature
        args: Candidate list of argument expressions
    Returns:
        Matching signature, None if not found
    """
    # Search corresponding signature
    # for s in oper.get_signatures():
    #     if _is_matching_arguments(s, args):
    #         return(s)
    # return None
    return next((s for s in oper.get_signatures() if _is_matching_arguments(s, args)), None)


def _is_matching_arguments(sgn, args):
    """ Check if a list of argument expressions matches this signature

    Args:
        sgn:  Signature descriptor
        args: Candidate list of argument expressions
    Returns:
        True if the arguments are matching signature
    """
    for a, p in zip_longest(args, sgn.get_parameters()):
        if a:
            # Accepted if there is a parameter descriptor that is compatible with argument type
            if not (p and a.get_type().is_kind_of(p.get_type())):
                return False
        else:
            # Argument absent, check that parameter has a default value
            if not p.is_default_value():
                return False
    return True


def _is_equal_expressions(v1, v2):
    """ Check if two expressions can be considered as equivalent.

    This method handles values that can be CPO expressions, None, and manage possible
    differences between number representations.

    It is implemented outside expression objects and use a self-managed stack to avoid too many
    recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

    Args:
        v1:  First CPO expression
        v2:  Second CPO expression
    Returns:
        True if both values are 'equivalent'
    """
    # Initialize expression stack
    estack = [[v1, v2, -1]]  # [expr1, expr2, child index]

    # Loop while expression stack is not empty
    while estack:
        # Get expressions to compare
        edscr = estack[-1]
        v1, v2, cx = edscr

        # Check same object type
        if type(v1) != type(v2):
            return False

        # Check physical equality
        if v1 is v2:
            estack.pop()
            continue

        # Check if expression is a CPO expression
        if isinstance(v1, CpoExpr):
            # Check objects itself
            if (cx < 0) and not v1._equals(v2):
                return False
            # Access children
            ar1 = v1._get_children()
            ar2 = v2._get_children()
            # Check children
            alen = len(ar1)
            if (cx < 0) and (alen != len(ar2)):
                return False
            cx += 1
            if (cx >= alen):
                estack.pop()
                continue
            # Store new child index in descriptor
            edscr[2] = cx
            # Stack children to compare it
            estack.append([ar1[cx], ar2[cx], -1])

        # Else, expressions are Python values
        else:
            if not _is_equal_values(v1, v2):
                return False
            estack.pop()


    # Expressions identical
    return True


def _is_equal_values(v1, v2):
    """ Check if two values can be considered as equivalent.

    This method handles values that can be CPO expressions, None, and manage possible
    differences between number representations.

    Args:
        v1:  First value
        v2:  Second value
    Returns:
        True if both values are 'equivalent'
    """
    # Check obvious cases
    if v1 is v2:
        return True
    # Check specifically CPO expressions (to not call '==' operator on it)
    if isinstance(v1, CpoExpr):
        return v1.equals(v2)
    if isinstance(v2, CpoExpr):
        return False
    # Check floats
    if is_float(v1):
        return is_float(v2) and (abs(v1 - v2) <= _FLOATING_POINT_PRECISION * max(abs(v1), abs(v2)))
    if is_array(v1):
        return is_array(v2) and (len(v1) == len(v2)) and all(_is_equal_values(x1, x2) for x1, x2 in zip(v1, v2))
    if isinstance(v1, dict):
        return isinstance(v2, dict) and (len(v1) == len(v2)) and all(_is_equal_values(v1[k], v2[k]) for k in v1)
    # Check finally basic equality
    return v1 == v2


import docplex.cp.cpo_compiler as compiler
def _to_string(expr):
    """ Build a string representing an expression.

    Args:
        expr:  Expression to convert into string
    Returns:
        String representing this expression
    """
    return compiler.CpoCompiler(None)._compile_expression(expr)
