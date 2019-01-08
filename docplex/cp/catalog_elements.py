# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the different object classes used to describe the
CPO catalog of types and functions.
"""

###############################################################################
## Public classes
###############################################################################

class CpoType(object):
    """ CPO type (flavor) descriptor """
    __slots__ = ('name',          # Name of the type
                 'is_var',        # Indicate a type corresponding to a variable
                 'is_cst',        # Indicates that type denotes a constant (possibly array)
                 'is_cst_atom',   # Indicates that type denotes an atomic constant
                 'is_arr',        # Indicates that type describes an array
                 'is_arr_expr',   # Indicates that type describes an array of expressions (not constants)
                 'higher_types',  # List of higher types in the hierarchy
                 'element_type',  # Type of array element (for arrays)
                 'parent_array',  # Type corresponding to an array of this type
                 'base_type',     # Base type to be used for signature matching
                 'id',            # Unique type id (index) used to fasten type links
                 'kind_of_types', # Set of types that are kind of this one
                 'common_types'   # Dictionary of types that are common with this and others.yield
                                  # Key is type name, value is common type with this one.
                )

    def __init__(self, name, isvar=False, iscst=False, htyps=(), eltyp=None, bastyp=None):
        """ Create a new type definition

        Args:
            name:  Name of the type
            isvar: Indicates whether this type denotes a variable
            htyps: List of types higher in the hierarchy
            eltyp: Array element type, None (default) if not array
            iscst: Indicate wether this type denotes a constant
        """
        super(CpoType, self).__init__()
        self.name         = name
        self.is_var       = isvar
        self.is_cst       = iscst
        self.is_cst_atom  = iscst
        self.higher_types = (self,) + htyps
        self.element_type = eltyp
        self.base_type    = bastyp if bastyp else self
        # Process array case
        if (eltyp is not None):
            eltyp.parent_array = self
            self.is_cst  = eltyp.is_constant()
        self.is_arr      = eltyp is not None
        self.is_arr_expr = self.is_array and not(self.is_cst)
        self.parent_array = None

    def get_name(self):
        """ Get the name of the type

        Returns:
            Name of the type
        """
        return self.name

    def is_variable(self):
        """ Check if this type describes a variable

        Returns:
            True if this type describes a variable, False otherwise
        """
        return self.is_var

    def is_constant(self):
        """ Check if this type describes a constant, or an array of constants

        Returns:
            True if this type describes a constant
        """
        return self.is_cst

    def is_constant_atom(self):
        """ Check if this type describes an atomic constant

        Returns:
            True if this type describes an atomic constant
        """
        return self.is_cst_atom

    def is_array(self):
        """ Check if this type describes an array

        Returns:
            True if this type describes an array, False otherwise
        """
        return self.is_arr

    def is_array_of_expr(self):
        """ Check if this type describes an array of expressions (not constants)

        Returns:
            True if this type describes an array of expressions, False otherwise
        """
        return self.is_arr_expr

    def get_element_type(self):
        """ Get the type of an array element, if this type is an array

        Returns:
            Array element type, None if not array
        """
        return self.element_type

    def get_base_type(self):
        """ Get the base type that should be used or signatures

        Returns:
            Base type (in general, returns self)
        """
        return self.base_type

    def is_kind_of(self, tp):
        """ Check if this type is a kind of another type, i.e. other type is in is hierarchy

        Args:
            tp: Other type to check
        Returns:
           True if this type is a kind of tp
        """
        # Check if required type is the same
        # return tp.base_type in self.higher_types
        return self.kind_of_types[tp.id]

    def get_common_type(self, tp):
        """ Get first common type between this and the parameter.
        
        Args:
            tp: Other type 
        Returns:
            The first common type between this and the parameter, None if none
        """
        return self.common_types[tp.id]

    def _compute_common_type(self, tp):
        """ Compute the first common type between this and the parameter.

        Args:
            tp: Other type
        Returns:
            The first common type between this and the parameter, None if none
        """
        # Check if given type is derived
        if tp.base_type is not tp:
            tp = tp.base_type
        # Check if this type is derived
        if self.base_type is not self:
            return self.base_type._compute_common_type(tp)
        # Check direct comparison
        if (self is tp) or tp.is_kind_of(self):
            return(self)
        elif (self.is_kind_of(tp)):
            return(tp)
        # Search common types in ancestors
        for ct in self.higher_types:
            if ct in tp.higher_types:
                return ct
        return None

    def get_array_type(self):
        """ Get the array type with this type as element
        
        Args:
            tp: Other type 
        Returns:
            The type that has this as element type, None if none
        """
        return self.parent_array
        
    def __str__(self):
        """ Convert this object into a string """
        return self.name

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.name == other.name))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)

    def __hash__(self):
        """ Return object hash-code """
        return id(self)


class CpoParam(object):
    """ Descriptor of an operation parameter """
    __slots__ = ('type',   # Parameter CPO type
                 'defval'  # Parameter default value, None if none
                )
    
    def __init__(self, ptyp, dval=None):
        """ Create a new parameter
        
        Args:
            ptyp: Parameter type
            dval: Default value
        """
        super(CpoParam, self).__init__()
        self.type = ptyp
        self.defval = dval
        
    def get_type(self):
        """ Get the parameter type
        
        Returns:
            Parameter type
        """
        return self.type
        
    def is_default_value(self):
        """ Check whether this parameter has a default value
        
        Returns:
            True if parameter has a default value, false otherwise
        """
        return not (self.defval is None)
        
    def get_default_value(self):
        """ Parameter default value
        
        Returns:
            Default value, None if none
        """
        return self.defval
        
    def __str__(self):
        if self.defval is None:
            return self.type.name
        else:
            return self.type.name + "=" + str(self.defval)

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.type == other.type) and (self.defval == other.defval))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)

            
class CpoSignature(object):
    """ Descriptor of a signature of a CPO operation """
    __slots__ = ('rtype',     # Return type
                 'params',    # List of parameter descriptors
                 'operation'  # Parent operation
                )
    
    def __init__(self, rtyp, ptyps):
        """ Create a new signature
        
        Args:
            rtyp:  Returned type
            ptyps: Array of parameter types
        """
        super(CpoSignature, self).__init__()
        self.rtype = rtyp
        
        # Build list of parameters
        lpt = []
        for pt in ptyps:
            if isinstance(pt, CpoParam):
                lpt.append(pt)
            else:
                lpt.append(CpoParam(pt)) 
        self.params = tuple(lpt)
        
    def get_returned_type(self):
        """ Get the type of the expression returned by this signature
        
        Returns:
            Signature return type
        """
        return self.rtype
        
    def is_parameters(self):
        """ Get if there are parameters to this signature
        
        Returns:
            True if this signature has parameters, false otherwise
        """
        return len(self.params) > 0
        
    def get_parameters(self):
        """ Get the list of parameters for this signature

        Returns:
            List of parameter types
        """
        return self.params

    def get_operation(self):
        """ Get the parent operation
        
        Returns:
            Parent operation
        """
        return self.operation
        
    def get_priority(self):
        """ Get the operation priority
        
        Returns:
            Operation priority
        """
        return self.operation.priority
        
    def __str__(self):
        return str(self.rtype) + "[" + ", ".join(map(str, self.params)) + "]"

    def __eq__(self, other):
        """ Check equality of this object with another """
        if self is other:
            return True
        return isinstance(other, self.__class__) and (self.rtype == other.rtype) \
               and (self.operation == other.operation) and (self.params == other.params)

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


class CpoOperation(object):
    """ CPO operation descriptor """
    __slots__ = ('cpname',     # Operation CPO name
                 'pyname',     # Operation python name
                 'keyword',    # Operation keyword (operation symbol)
                 'priority',   # Operator priority, -1 for function call
                 'isoptim',    # Optimization function indicator
                 'signatures'  # List of possible operation signatures
                )

    def __init__(self, cpname, pyname, kwrd, prio, signs):
        """ Create a new operation
        
        Args:
            cpname:  Operation CPO name
            pyname:  Operation python name
            kwrd:    Keyword, None for same as cpo name
            prio:    Priority
            signs:   Array of possible signatures
        """
        super(CpoOperation, self).__init__()
        
        # Store attributes
        self.cpname   = cpname
        self.pyname   = pyname
        self.priority = prio
        self.isoptim  = cpname.startswith("minimize") or cpname.startswith("maximize")
        if kwrd:
            self.keyword = kwrd
        else:
            self.keyword = cpname
        self.signatures = signs
        
        # Set pointer back on operation on each signature
        for s in signs:
            s.operation = self
        
    def get_cpo_name(self):
        """ Get the CPO name of this operation
        
        Returns:
            Operation CPO name
        """
        return self.cpname
        
    def get_py_name(self):
        """ Get the Python name of this operation

        Returns:
            Operation Python name
        """
        return self.pyname

    def is_optim(self):
        """ Check if this function requests an optimization.

        Returns:
            True if this operation requests an optimization
        """
        return self.isoptim

    def get_keyword(self):
        """ Get the operation keyword.
        
        Returns:
            Operation keyword
        """
        return self.keyword
        
    def get_priority(self):
        """ Get the operation priority
        
        Returns:
            Operation priority
        """
        return self.priority
        
    def get_signatures(self):
        """ Get the list of available signatures for this operation
        
        Returns:
            List of operation signatures
        """
        return self.signatures
        
    def __str__(self):
        return str(self.cpname) + "(" + ", ".join(map(str, self.signatures)) + ")"

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.cpname == other.cpname))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


def compute_all_type_links(ltypes):
    """ Compute all links between the different data types.

    Args:
        ltypes: List of all types
    """
    # Allocate id to each each type
    nbtypes = len(ltypes)
    for i, tp in enumerate(ltypes):
        tp.id = i

    # Compute kind of for each type
    for tp1 in ltypes:
        tp1.kind_of_types = tuple(map(lambda tp2: (tp2.base_type in tp1.higher_types), ltypes))

    # Compute common type
    for tp1 in ltypes:
        ctypes = [None] * nbtypes
        for tp2 in ltypes:
            ct = tp1._compute_common_type(tp2)
            if ct is not None:
                ctypes[tp2.id] = ct
        tp1.common_types = tuple(ctypes)

