# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the basic classes representing constraint programming model expressions.

In particular, it defines the following classes:

 * CpoExpr: the root class of each model expression node,
 * CpoIntVar: representation of an integer variable,
 * CpoIntervalVar: representation of an interval variable,
 * CpoSequenceVar: representation of an interval variable,
 * CpoSequenceVar: representation of an interval variable,
 * CpoTransitionMatrix: representation of a transition matrix,
 * CpoTupleSet: representation of a tuple set, and
 * CpoStateFunction: representation of a state function.

None of these classes should be created explicitly.
There are various factory functions to do so, such as:

 * integer_var(), integer_var_list(), integer_var_dict() to create integer variable(s),
 * interval_var(), interval_var_list() to create an interval variable,
 * sequence_var() to create a sequence variable,
 * etc.

Moreover, some automatic conversions are also provided.
For example, a list of tuples of integers is automatically converted into a tuple set.
"""

from docplex.cp.utils import *
from docplex.cp.catalog import *
import math
import collections
import threading


###############################################################################
## Integer expressions
###############################################################################

#INT_MAX = 0x1FFFFFFFFFFFFF  # (2^53 - 1) for 64 bits
INT_MAX = 0x7FFFFFFF         # (2^31 - 1) for 32 bits
""" Maximum integer value. """

INT_MIN = -INT_MAX
""" Minimum integer value. """

class CpoExpr(object):
    """ Root constraint programming model expression.

    This class represents a CPO expression atom. It does not contain links to children expressions
    that are implemented in extending classes. However, access to children is provided with default
    return value.
    """
    # To force possible numpy operators overloading to get CPO expressions as main operand
    __array_priority__ = 100

    __slots__ = ('type',       # Expression result type
                 'name',       # Name of the expression (None if none)
                 'nbrefs',     # Number of references on this expression
                )

    def __init__(self, type, name):
        """ Create a new expression

        Args:
            type:   Expression type.
            name:   Expression name.
        """
        # super(CpoExpr, self).__init__()
        self.type = type
        self.set_name(name)
        self.nbrefs = 0
        
    def __hash__(self):
        """ Redefinition of hash (needed by Python 3)
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
        """ Sets the expression name.

        Args:
            name: Expression name, possibly None.
        """
        assert (name is None) or isinstance(name, str), "Argument 'name' should be a string or None"
        self.name = name
            
    def get_name(self):
        """ Gets the expression name.

        Returns:
            Expression name, possibly None.
        """
        return self.name
            
    def _get_alias_then_name(self):
        """ Get the name of the expression (alias are for variables only)

        Returns:
            Expression name
        """
        return self.name


    def has_name(self):
        """ Checks if the expression has a name.

        Returns:
            True if expression has a name, False otherwise.
        """
        return (self.name is not None)
            
    def get_type(self):
        """ Gets the type of this expression.

        Returns:
            Expression type descriptor.
        """
        return self.type
            
    def is_kind_of(self, tp):
        """ Checks if this expression type is a kind of another type.

        Args:
            tp: Other type to check.
        Returns:
           True if this expression type type is a kind of tp.
        """
        # Check if required type is the same
        return (tp.base_type in self.type.higher_types)

    def is_variable(self):
        """ Checks if this expression is a variable.

        Returns:
            True if this expression is a variable, False otherwise.
        """
        return self.type.is_variable()
            
    def is_constant(self):
        """ Checks if this expression is constant.

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

    def get_priority(self):
        """ Gets the expression operation priority.

        Returns:
            Operation priority, -1 for none.
        """
        return -1

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, None if none
        """
        return None

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
        """ Checks the equality of this expression with another.

        Implementation is required with this name because '==' is already overloaded to construct an expression.

        Args:
            other: Other object to compare with.
        """
        return (type(self) == type(other)) and (self.type == other.type) and (self.name == other.name)

    def _incr_ref_count(self):
        """ Increase reference count on this expression and create a name if more than one

        Args:
            expr: Expression to update
        """
        # Increment reference count
        self.nbrefs += 1
        if (self.nbrefs > 1) and not(self.is_atom_constant()):
            # Add expression id if none
            if self.name is None:
                self.name = _allocate_identifier()

    def _get_string(self):
        """ Get the string representing this expression (without name)
        Returns:
            String representation of this expression
            This default implementation returns the name of the type
        """
        return self.get_type().get_name()

    def _to_string(self, root):
        """ Get the string representing this expression
        Args:
            root: Expression root indicator
        Returns:
            String representation of this expression
        """
        # Check named expression not at root
        name = self.name
        if name:
            if not root:
                return name
            return name + " = " + self._get_string()
        return self._get_string()

    def __str__(self):
        """ Convert this expression into a string """
        return self._to_string(True)


class CpoValue(CpoExpr):
    """ Expression representing a constant. """
    __slots__ = ('value',  # Python value of the constant
                )

    def __init__(self, value, type):
        """ Creates a new constant expression.

        Args:
            value:  Python value.
            vtyp :  Value type.
        """
        assert isinstance(type, CpoType), "Argument 'type' should be a CpoType"
        super(CpoValue, self).__init__(type, None)
        if type.is_array_of_expr():
            value = _update_references(value)
        self.value = value

    def get_value(self):
        """ Gets the value of the constant.

        Returns:
            Value of the constant.
        """
        return self.value

    def is_atom_constant(self):
        """ Checks if this expression is an atomic constant (boolean, int, float).

        Returns:
            True if this expression is an atomic constant.
        """
        return (self.type in (Type_Int, Type_Float, Type_Bool))

    def equals(self, other):
        """ Checks equality of this expression with another.

        Implementation is required with this name because '==' is overloaded to construct expressions.

        Args:
            other: Other object to compare with.
        """
        # Call super
        if not super(CpoValue, self).equals(other):
            return False
        # Check value
        if self.type.is_array_of_expr():
            sval = self.value
            oval = other.value
            if len(oval) != len(sval):
                return False
            for i in range(len(sval)):
                if not sval[i].equals(oval[i]):
                    return False
            return True
        else:
            return self.value == other.value

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, None if none
        """
        if self.type.is_array_of_expr():
            return self.value
        return None

    def _get_string(self):
        """ Get the string representing this expression (without name)
        Returns:
            String representation of this expression
        """
        return str(self.value)


class CpoExprList(list):
    """ List of CPO expressions.

    This extension of a standard Python list overwrites __getitem__ to call element() constraint
    if the index is a CPO integer expression.

    This object is used as returned object by constructor methods integer_var_list() in this module.
    """

    def __init__(self):
        super(CpoExprList, self).__init__()

    def __getitem__(self, nx):
        """ Overloading of [] to create a CPO element() expression if index is a CPO integer expression.

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
    """ Constraint programming model expression representing a function call. """
    __slots__ = ('signature',  # Signature of the operation (None if none)
                 'operands',   # List of operand expressions, or Python value for constants
                )

    def __init__(self, sign, oprnds):
        """ Create a new function call expression

        Args:
            sign:   Operation signature (children is operands).
            oprnds: List of operand expressions.
        """
        assert isinstance(sign, CpoSignature), "Argument 'sign' should be a CpoSignature"
        super(CpoFunctionCall, self).__init__(sign.get_returned_type(), None)
        self.signature = sign

        # Check no toplevel constraints
        if oprnds:
            for e in oprnds:
                if (e.get_type() == Type_Constraint):
                    raise CpoException("The constraint " + str(e) + " can not be member of an expression.")
            self.operands = _update_references(oprnds)
        else:
            self.operands = None

    def get_signature(self):
        """ Gets the expression signature.

        Returns:
            Expression signature.
        """
        return self.signature

    def get_operation(self):
        """ Gets the expression operation.

        Returns:
            Expression operation, None if none.
        """
        return self.signature.operation

    def get_priority(self):
        """ Gets the expression operation priority.

        Returns:
            Operation priority, -1 for none.
        """
        return self.signature.get_priority()

    def get_operands(self):
        """ Gets the expression operands.

        Returns:
            Expression operands (list of child expressions), None if none.
        """
        return self.operands

    def equals(self, other):
        """ Checks the equality of this expression with .

        Implementation is required with this name because '==' is already overloaded to construct an expression.

        Args:
            other: Other object to compare with.
        """
        # Call super
        if not super(CpoFunctionCall, self).equals(other):
            return False

        # Check signature
        if self.signature != other.signature:
            return False

        # Check operands
        if self.signature.is_parameters():
            soprds = self.operands
            ooprds = other.operands
            if len(soprds) != len(ooprds):
                return False
            for i in range(len(soprds)):
                if not soprds[i].equals(ooprds[i]):
                    return False
        return True

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, None if none
        """
        return self.operands

    def _get_string(self):
        """ Get the string representing this expression (without name)
        Returns:
            String representation of this expression
            This default implementation returns the name of the type
        """
        # Check named expression not at root
        opsgn = self.signature
        oprnds = self.operands
        return opsgn.get_operation().get_py_name() + "(" + ", ".join(x._to_string(False) for x in oprnds) + ")"


class CpoVariable(CpoExpr):
    """ Expression representing a variable. """

    def __init__(self, type, name):
        """ Creates a new variable expression.

        Args:
            type:   Expression type.
            name:   Variable name.
        """
        # Check name length
        if name is None:
            name = _allocate_var_name()
        super(CpoVariable, self).__init__(type, name)
        self.alias = None

    def is_variable(self):
        """ Checks if this expression is a variable.

        Returns:
            True if this expression is a variable, False otherwise.
        """
        return True


class CpoIntVar(CpoVariable):
    """ Integer variable. """
    __slots__ = ('domain',  # Variable domain
                 )
    
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
    
    def __str__(self):
        """ Convert this expression into a string """
        return "integer_var(" + self.get_name() + ")"
        

###############################################################################
## Scheduling expressions
###############################################################################

#INTERVAL_MAX = 0xFFFFFFFFFFFFE  # (2^52 - 2) for 64 bits
INTERVAL_MAX = 0x3FFFFFFE        # (2^30 - 2) for 32 bits
""" Maximum interval variable range. """

INTERVAL_MIN = -INTERVAL_MAX
""" Minimum interval variable range. """

INFINITY = float('inf')
""" Infinity. """

DEFAULT_INTERVAL = (0, INTERVAL_MAX)
""" Default interval. """

# Different interval variable presence states
_PRES_PRESENT   = "present"   # Always present
_PRES_ABSENT    = "absent"    # Always absent
_PRES_OPTIONAL  = "optional"  # Present or absent, choice made by the solver


class CpoIntervalVar(CpoVariable):
    """ Interval variable. """
    __slots__ = ('start',        # Start domain
                 'end',          # End domain
                 'length',       # Length domain
                 'size',         # Size domain
                 'intensity',    # Specifies relation between size and length of the interval.
                 'granularity',  # Scale of the intensity function.
                 'presence',     # Presence requirement
                 )

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
            mn: Max value of the start of the interval.
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
            mn: Max value of the end of the interval.
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
            mn: Max value of the length of the interval.
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
            mn: Max value of the size of the interval.
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

    def set_granularity(self, granularity):
        """ Sets the scale of the intensity function.

        Args:
            granularity: Scale of the intensity function (integer).
        """
        assert (granularity is None) or (is_int(granularity) and (granularity >= 0)), "Argument 'granularity' should be None or positive integer"
        self.granularity = granularity 

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, None if none
        """
        return (self.intensity,) if self.intensity else None

    def __str__(self):
        """ Convert this expression into a string """
        return "interval_var(" + self.get_name() + ")"


class CpoSequenceVar(CpoVariable):
    """ Sequence variable. """
    __slots__ = ('vars',   # List of variables
                 'types',  # Variable types
                )
    
    def __init__(self, vars, types=None, name=None):
        """ Creates a new sequence variable.

        Args:
            vars:  Array of IntervalVars that constitute the sequence.
            types: Variable types (same size as vars), default is None.
            name:  Name of the sequence, None for automatic naming.
        """
        # Check  arguments 
        assert is_array_of_type(vars, CpoIntervalVar), "Argument 'vars' should be an array of CpoIntervalVar"
        if types is not None:
            types = _check_and_expand_interval_tuples('types', types)
        # Store attributes
        super(CpoSequenceVar, self).__init__(Type_SequenceVar, name)
        self.vars = vars
        self.types = types

    def get_interval_variables(self):
        """ Gets the array of variables.

        Returns:
            Array of variables.
        """
        return self.vars
    
    def get_types(self):
        """ Gets the array of types.

        Returns:
            Array of types.
        """
        return self.types
    
    def __str__(self):
        """ Convert this expression into a string """
        return "SequenceVar" + to_string(self.vars)


class CpoTransitionMatrix(CpoExpr):
    """ Transition matrix (transition distances). """
    __slots__ = ('size',    # Matrix width/height
                 'matrix',  # Matrix values
                )
    
    def __init__(self, size=None, values=None, name=None):
        """ Creates a new empty transition matrix (square matrix of integers).

        Args:
            size:   Matrix size (width or height).
                    If not given, the `values` argument must be given.
            values: Optional list of matrix values.
                    If not given, the method `set_value()` should be called to initialize matrix content.
            name:   Name of the matrix. None by default.
        """
        super(CpoTransitionMatrix, self).__init__(Type_TransitionMatrix, name)
        if (size is not None):
            assert is_int(size), "Argument 'size' should be an int"
            self.size = size
            if (values is None):
                self.matrix = [0] * (size * size)
            else:
                assert is_array_of_type(values, int) and (len(values) == size * size), "Argument 'values' should be an array of integer of size*size length"
                self.matrix = values
        elif (values is not None):
            assert is_array_of_type(values, int), "Argument 'values' should be an array of integers"
            size = math.sqrt(len(values))
            assert (len(values) == (size * size)), "Argument 'values' should hae a length that is a perfect square"
            self.matrix = values
        else:
            assert False, "At least 'size' or 'values' should be given"
         
    def get_size(self):
        """ Gets the matrix size.

        Returns:
            Matrix size.
        """
        return self.size

    def get_value(self, from_state, to_state):
        """ Gets a value in the transition matrix.

        Args:
            from_state: Index of the from state.
            to_state:   Index of the to state.
        Returns:
            Transition value.
        """
        assert_arg_int_interval(from_state, 0, self.size, "from_state")
        assert_arg_int_interval(to_state, 0, self.size, "to_state")
        return self.matrix[(from_state * self.size) + to_state]

    def get_matrix(self):
        """ Gets the complete transition matrix.

        Returns:
            Transition value (list of lists).
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
        self.matrix[from_state * self.size + to_state] = value

    def __str__(self):
        """ Convert this expression into a string """
        return "TransitionMatrix" + to_string(self.matrix)
    
    
class CpoTupleSet(CpoExpr):
    """ Tuple set. """
    __slots__ = ('size',      # Size of a single tuple
                 'tupleset',  # List of tuples
                 )
    
    def __init__(self, size=-1, name=None):
        """ Creates a new empty tuple set.

        Args:
            size: Tuple size; default value is -1 for automatic size.
        """
        assert is_int(size), "Argument 'size' should be an int"
        super(CpoTupleSet, self).__init__(Type_TupleSet, name)
        self.size = size
        self.tupleset = []
         
    def get_size(self):
        """ Gets the tuple size.

        Returns:
            Tuple size, -1 if undefined.
        """
        return self.size

    def add(self, tpl):
        """ Adds a tuple in this tuple set.

        Args:
            tpl: Tuple to add.
        """
        # Check for intervals
        tpl = _check_and_expand_interval_tuples("tpl", tpl)
        if (self.size < 0):
            self.size = len(tpl)
        elif len(tpl) != self.size:
            raise CpoException("You must add only tuples of size " + str(self.size))
        self.tupleset.append(tpl)

    def add_set(self, tpls):
        """ Adds a set of tuples in this tuple set.

        Args:
            tpls: Iterator of tuples to add.
        """
        for t in tpls:
            self.add(t)

    def get_tuple_set(self):
        """ Gets the complete tuple set.

        Returns:
            Tuple set.
        """
        return self.tupleset

    def __str__(self):
        """ Convert this expression into a string """
        return "TupleSet" + to_string(self.tupleset)
    
    
class CpoStateFunction(CpoVariable):
    """ State function. """
    __slots__ = ('trmtx',      # Transition matrix
                )

    def __init__(self, trmtx=None, name=None):
        """ Creates a new state function.

        Args:
            trmtx:  Transition matrix.
            name:   Name of the state function.
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
        """ Gets the transition matrix.

        Returns:
            Transition matrix, None if none.
        """
        return self.trmtx

    def _get_children(self):
        """ Get the list of children expressions if any

        Returns:
            List of children expressions, None if none
        """
        return (self.trmtx,) if self.trmtx else None

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

    * as a single interval, with a minimum and a maximum values that are included in the domain,
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
        max:  Domain max value, default is None and indicates that 'min' should contain an extensive list of values.
        name: Variable name, default is None for automatic name.
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
        size: List size.
        min:  Domain min value, or extensive list of values and/or intervals expressed as tuples of integers.
        max:  Domain max value, default is None and indicates that 'min' should contain an extensive list of values.
        name: Variable name prefix, default is None for automatic name.
    Returns:
        List of integer variables.
    """
    if name is None:
        name = _allocate_var_name() + "_"
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
        max:  Domain max value, default is None and indicates that 'min' should contain an extensive list of values.
        name: Variable name prefix, default is None for automatic name.
              Or function to be called on dictionary key (example: str).
    Returns:
        Dictionary of IntVars (OrderedDict).
    """
    if name is None:
        name = _allocate_var_name() + "_"
    dom = _build_int_var_domain(min, max)
    res = collections.OrderedDict()
    i = 0
    isnamestr = isinstance(name, str)
    for k in keys:
        if isnamestr:
            vname = name + str(i) 
        else:
            vname = name(k)
        res[k] = CpoIntVar(dom, vname)
        i += 1
    return res


def interval_var(start=DEFAULT_INTERVAL, end=DEFAULT_INTERVAL, length=DEFAULT_INTERVAL, size=DEFAULT_INTERVAL,
                 intensity=None, granularity=None, optional=False, name=None, present=True):
    """ Creates an interval variable.

    Represents an interval of integers. Interval variables are used mostly for scheduling to represent a
    task as an interval of time.
    In its most basic form, an interval variable can be seen as a pair of two integer variables start and end
    such that start ? end.
    However there is an important difference: the interval variable can be absent to represent the fact that the
    interval does not exist at all (which is different from a zero-length interval).

    Args:
        start:       Allowed range for the start of the interval (single integer or interval expressed as a tuple of 2 integers).
        end:         Allowed range for the end the interval (single integer or interval expressed as a tuple of 2 integers).
        length:      Allowed range for the length the interval (single integer or interval expressed as a tuple of 2 integers).
        size:        Allowed range for the size the interval (single integer or interval expressed as a tuple of 2 integers).
        intensity:   StepFunction that specifies relation between size and length of the interval.
        granularity: Scale of the intensity function.
        optional:    Optional presence indicator.
        name:        Name of the variable.
    Deprecated:
        present:     Presence indicator. present=False is equivalent to optional=True)
    Returns:
        IntervalVar expression.
    """
    start  = _check_arg_interval(start,  "start")
    end    = _check_arg_interval(end,    "end")
    length = _check_arg_interval(length, "length")
    size   = _check_arg_interval(size,   "size")
    _check_arg_intensity(intensity, granularity)
    presence = _PRES_OPTIONAL if (_check_arg_boolean(optional, "optional") or not present) else _PRES_PRESENT
    return CpoIntervalVar(start, end, length, size, intensity, granularity, presence, name)


def interval_var_list(asize, start=DEFAULT_INTERVAL, end=DEFAULT_INTERVAL, length=DEFAULT_INTERVAL, size=DEFAULT_INTERVAL,
                 intensity=None, granularity=None, optional=False, name=None, present=True):
    """ Creates a list of interval variables.

    If a name is given, each variable of the array is created with this
    name concatenated with the index of the variable in the list.

    Args:
        asize:       Result list size.
        start:       Allowed range for the start of the interval (single integer or interval expressed as a tuple of 2 integers).
        end:         Allowed range for the end the interval (single integer or interval expressed as a tuple of 2 integers).
        length:      Allowed range for the length the interval (single integer or interval expressed as a tuple of 2 integers).
        size:        Allowed range for the size the interval (single integer or interval expressed as a tuple of 2 integers).
        intensity:   StepFunction that specifies relation between size and length of the interval.
        granularity: Scale of the intensity function.
        optional:    Optional presence indicator.
        name:        Variable name prefix.
    Deprecated:
        present:     Presence indicator. present=False is equivalent to optional=True)
    Returns:
        List of interval variables.
    """
    start  = _check_arg_interval(start,  "start")
    end    = _check_arg_interval(end,    "end")
    length = _check_arg_interval(length, "length")
    size   = _check_arg_interval(size,   "size")
    _check_arg_intensity(intensity, granularity)
    presence = _PRES_OPTIONAL if (_check_arg_boolean(optional, "optional") or not present) else _PRES_PRESENT
    if name is None:
        name = _allocate_var_name() + "_"
    res = []
    for i in range(asize):
        res.append(CpoIntervalVar(start, end, length, size, intensity, granularity, presence, name+str(i)))
    return res


def sequence_var(ivars, types=None, name=None):
    """ Creates a new sequence variable (list of interval variables).

    Args:
        ivars:   Array of interval variables.
        types:   Variable types (same size than vars), default is None.
        name:    Variable name, default is None for automatic name.
    Returns:
        IntervalVar expression.
    """
    if name is None:
        name = _allocate_var_name()
    return CpoSequenceVar(ivars, types, name)


def transition_matrix(size, name=None):
    """ Creates an empty transition matrix.

    The matrix can be filled using the methods provided by the TransitionMatrix object.

    Args:
        size: Size of the square matrix.
        name: Object name (default is None).
    Returns:
        TransitionMatrix expression.
    """
    return CpoTransitionMatrix(size, name)
 
 
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

# Map of CPO expressions corresponding to Python values
_CPO_VALUES_FROM_PYTHON = KeyIdDict()

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

    #  Check iterators
    if isinstance(val, collections.Iterator):
        val = tuple(val)

    # Check if already converted
    _CPO_VALUES_FROM_PYTHON_LOCK.acquire()
    cpval = _CPO_VALUES_FROM_PYTHON.get(val)
    _CPO_VALUES_FROM_PYTHON_LOCK.release()
    if cpval is not None:
        return cpval

    # Build new expression
    cpval = create_cpo_expr(val)
    _CPO_VALUES_FROM_PYTHON_LOCK.acquire()
    _CPO_VALUES_FROM_PYTHON.set(val, cpval)
    _CPO_VALUES_FROM_PYTHON_LOCK.release()
    return cpval


# Constant for True and False
_CONSTANT_TRUE  = CpoValue(True, Type_Bool)
_CONSTANT_FALSE = CpoValue(False, Type_Bool)

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

    # Check special types
    if typ == Type_TupleSet:
        res = CpoTupleSet()
        res.add_set(value)
    elif typ == Type_Bool:
        #res = CpoFunctionCall((Oper_true if value else Oper_false).signatures[0], ())
        res = _CONSTANT_TRUE if value else _CONSTANT_FALSE
    else:
        # Check if array of exprs contains only expressions
        if typ.is_array_of_expr():
            if not all(isinstance(x, CpoExpr) for x in value):
                nval = []
                for v in value:
                    if isinstance(v, CpoExpr):
                        nval.append(v)
                    else:
                        nval.append(create_cpo_expr(v))
                value = nval
        res = CpoValue(value, typ)

    # Return
    return res


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

    # Check if the operation contains a single signature

    # Convert arguments in CPO expressions
    args = tuple(map(build_cpo_expr, params))
      
    # Search corresponding signature
    s = _get_matching_signature(oper, args)
    if s is None:
        raise CpoException("The combination of parameters (" + ", ".join(map(_get_cpo_type_str, args))
                           + ") is not allowed for operation '" + oper.get_py_name() + "'")
    
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
    for v in val:
        if not is_int(v):
            assert is_interval_tuple(v), "Argument '" + name + "' should be a list of integers and/or intervals (tuples of 2 integers)"
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
        assert isinstance(intensity, CpoExpr) and (intensity.get_type() == Type_StepFunction), "Interval variable 'intensity' should be None or a StepFunction"
        for (s, v) in intensity.get_step_list():
            assert is_int(s), "'intensity' step start should be an integer"
            if granularity is None:
                granularity = 100
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
    assert isinstance(arr, (list, tuple)), "Argument '" + name + "' should be a list of integers or intervals"
    res = None
    for i in range(len(arr)):
        v = arr[i]
        if is_int(v):
            if res:
                res.append(v)
        else:
            assert is_interval_tuple(v), "Argument '" + name + "' should be a list of integers or intervals"
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
        return(_PYTHON_TO_CPO_TYPE.get(val.dtype.type))

    # Check arrays
    if not isinstance(val, (list, tuple)):
        return(None)
    
    # Check empty Array
    if (len(val) == 0):
        return(Type_IntArray)

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
                return(None)
        # Combine with global type
        if gt is None:
            gt = nt
        else:
            gt = gt.get_common_type(nt)
        if gt is None:
            return None
    
    # Determine array type for result element type
    if gt == Type_IntArray:
        return(Type_TupleSet)
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
    # Increment reference count
    for c in oprnds:
        c._incr_ref_count()

    # Build a tuple if not
    if not isinstance(oprnds, tuple):
        oprnds = tuple(oprnds)
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
    for s in oper.get_signatures():
        if _is_matching_arguments(s, args):
            return(s)
    return None


def _is_matching_arguments(sgn, args):
    """ Check if a list of argument expressions matches this signature

    Args:
        sgn:  Signature descriptor
        args: Candidate list of argument expressions
    Returns:
        True if the arguments are matching signature
    """
    params = sgn.get_parameters()
    if len(args) > len(params):
        return(False)
    # Check types compatibility
    nbargs = len(args)
    for i in range(0, nbargs):
        if not args[i].get_type().is_kind_of(params[i].get_type()):
            return(False)
    # Check that remaining parameters, if any, have a default value
    for i in range(nbargs, len(params)):
        if (not params[i].is_default_value()):
            return(False)
    return True   


_ANONYMOUS_VAR_ID_ALLOCATOR = SafeIdAllocator('_ANM_')
def _allocate_var_name():
    """ Allocate a new variable name for anonymous variables
    Returns:
        New variable name
    """
    return _ANONYMOUS_VAR_ID_ALLOCATOR.allocate()


_IDENTIFIER_ALLOCATOR = SafeIdAllocator('_ID_')
def _allocate_identifier():
    """ Allocate a new expression identifier
    Returns:
        New identifier
    """
    return _IDENTIFIER_ALLOCATOR.allocate()
