# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Miscellaneous utility functions. Some of theme are here to prevent possible
port problems between the different versions of Python.
"""

import os
import time
import logging
import sys


###############################################################################
## Constants
###############################################################################

# Constant used to indicate to set a parameter to its default value
# Useful if default value is not static
DEFAULT = "default"

# Determine list of types representing the different python scalar types
BOOL_TYPES = {bool}
INTEGER_TYPES = {int}
FLOAT_TYPES = {float}

# Add Python2 long if any
try:
    if type(long) is type:
        INTEGER_TYPES.add(long)
except:
    pass

# Numpy available indicator
IS_NUMPY_AVAILABLE = False

# Add numpy types if any
try:
    import numpy

    BOOL_TYPES.add(numpy.bool_)
    BOOL_TYPES.add(numpy.bool)

    INTEGER_TYPES.add(numpy.int_)
    INTEGER_TYPES.add(numpy.intc)
    INTEGER_TYPES.add(numpy.intp)

    INTEGER_TYPES.add(numpy.int8)
    INTEGER_TYPES.add(numpy.int16)
    INTEGER_TYPES.add(numpy.int32)
    INTEGER_TYPES.add(numpy.int64)

    INTEGER_TYPES.add(numpy.uint8)
    INTEGER_TYPES.add(numpy.uint16)
    INTEGER_TYPES.add(numpy.uint32)
    INTEGER_TYPES.add(numpy.uint64)

    FLOAT_TYPES.add(numpy.float_)
    FLOAT_TYPES.add(numpy.float16)
    FLOAT_TYPES.add(numpy.float32)
    FLOAT_TYPES.add(numpy.float64)

    IS_NUMPY_AVAILABLE = True
except:
    pass

# Build all number type sets
NUMBER_TYPES  = frozenset(INTEGER_TYPES.union(FLOAT_TYPES))
INTEGER_TYPES = frozenset(INTEGER_TYPES)
FLOAT_TYPES   = frozenset(FLOAT_TYPES)
BOOL_TYPES    = frozenset(BOOL_TYPES)


###############################################################################
## Public classes
###############################################################################

class CpoException(Exception):
    """ Exception thrown in case of CPO errors
    """
    def __init__(self, msg):
        """ Create a new exception

        Args:
            msg: Error message
        """
        super(Exception, self).__init__(msg)


class Context(dict):
    """ Class handling miscellaneous list of parameters """
    def __init__(self, **kwargs):
        """ Create a new context

        Args:
            List of key=value to initialize context with.
        """
        super(Context, self).__init__()
        vars(self)['parent'] = None
        for k, v in kwargs.items():
            self.set_attribute(k, v)

    def __setattr__(self, name, value):
        """ Set a context parameter.

        Args:
            name:  Parameter name
            value: Parameter value
        """
        self.set_attribute(name, value)

    def __getattr__(self, name):
        """ Get a context parameter.

        Args:
            name:  Parameter name
        Return:
            Parameter value, None if not set
        """
        return self.get_attribute(name)

    def set_attribute(self, name, value):
        """ Set a context attribute.

        Args:
            name:  Attribute name
            value: Attribute value
        """
        self[name] = value
        if isinstance(value, Context):
            vars(value)['parent'] = self

    def get_attribute(self, name, default=None):
        """ Get a context attribute.

        Args:
            name:    Attribute name
            default: Optional, default value if attribute is not found
        Return:
            Attribute value, default value if not found
        """
        if name.startswith('__'):
            raise AttributeError
        ctx = self
        while True:
            res = ctx.get(name, None)
            if res is not None:
                return res
            ctx = ctx.get_parent()
            if ctx is None:
                return default

    def get_attribute_by_path(self, path, default=None):
        """ Get a context attribute using its path.

        Attribute path is a sequence of attribute names separated by dots.

        Args:
            path:    Attribute path
            default: Optional, default value if attribute is not found
        Return:
            Attribute value, default value if not found
        """
        res = self
        for k in path.split('.'):
            if k:
                res = res.get_attribute(k)
                if res is None:
                    return None
        return res

    def search_and_replace_attribute(self, name, value, path=""):
        """ Replace an existing attribute.

        The attribute is searched recursively in children contexts if any.

        Args:
            name:  Attribute name
            value: Attribute value
        Return:
            Full path of the attribute that has been found and replaced, None if not found
        """
        for k, v in self.items():
            npath = path + "." + k
            if k == name:
                ov = self.get_attribute(name)
                if (ov is not None):
                    if isinstance(value, Context):
                        if not isinstance(ov, Context):
                            raise Exception("Attribute '" + npath + "' is a Context and can only be replaced by a Context")
                    else:
                        if isinstance(ov, Context):
                            raise Exception("Attribute '" + npath + "' is a simple value and can not be replaced by a Context")
                self.set_attribute(name, value)
                return npath
            elif isinstance(v, Context):
                apth = v.search_and_replace_attribute(name, value, path=npath)
                if apth:
                    return apth
        return None

    def get_parent(self):
        """ Get the parent context.

        Each time a context attribute is set to a context, its parent is assigned to the context where it is stored.

        Return:
            Parent context, None if this context is root
        """
        return vars(self)['parent']

    def clone(self):
        """ Clone this context and all sub-contexts recursively.

        Return:
            Cloned copy of this context.
        """
        res = type(self)()
        vars(res)['parent'] = vars(self)['parent']
        for k, v in self.items():
            if isinstance(v, Context):
                v = v.clone()
            res.set_attribute(k, v)
        return res

    def is_log_enabled(self, vrb):
        """ Check if log is enabled for a given verbosity

        This method get this context 'log_output' attribute to retrieve the log output, and the
        attribute 'verbose' to retrieve the current verbosity level.

        Args:
            vrb:  Required verbosity level, None for always
        """
        return self.log_output and ((vrb is None) or (self.verbose and (self.verbose >= vrb)))

    def log(self, vrb, *msg):
        """ Log a message if log is enabled with enough verbosity

        This method get this context 'log_output' attribute to retrieve the log output, and the
        attribute 'verbose' to retrieve the current verbosity level.

        Args:
            vrb:  Required verbosity level, None for always
            msg:  Message elements to log (concatenated on one line)
        """
        if self.is_log_enabled(vrb):
            out = self.log_output
            prfx = self.log_prefix
            if prfx:
                out.write(str(prfx))
            out.write(''.join([str(m) for m in msg]) + "\n")
            out.flush()

    def print_context(self, out=None, indent=""):
        """ Print this context.

        At each level, atomic values are printed first, then sub-contexts, in alphabetical order.

        Args:
            out:    Print output. stdout by default.
            indent: Start line indentation. Default is empty
        """
        if out is None:
            out = sys.stdout
        sctxs = []  # List of subcontexts
        # Print atomic values
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, Context):
                sctxs.append((k, v))
            else:
                vstr = str(v)
                if isinstance(v, str):
                    vstr = '"' + vstr + '"'
                out.write(indent + str(k) + " = " + vstr + "\n")
        # Print sub-contexts
        for (k, v) in sctxs:
            out.write(indent + str(k) + ' =\n')
            v.print_context(out, indent + "   ")
        out.flush()


class IdAllocator(object):
    """ Allocator of identifiers """
    __slots__ = ('prefix',  # Id prefix
                 'count'    # Allocated id count
                 )
    def __init__(self, prefix):
        """ Create a new id allocator

        Args:
            prefix:  Prefix of all ids
        """
        super(IdAllocator, self).__init__()
        self.prefix = prefix
        self.count = 0

    def get_count(self):
        """ Get the number of id that has been allocated by this allocator.

        Returns:
            Number of id that has been allocated by this allocator.
        """
        return self.count

    def allocate(self):
        """ Allocate a new id

        Returns:
            Next id for this allocator
        """
        self.count += 1
        return(self.prefix + str(self.count))


class IdentityAccessor(object):
    """ Object implementing a __getitem__ that returns the key as value """
    def __getitem__(self, key):
        return(key)


class Chrono(object):
    """ Chronometer """
    __slots__ = ('startTime',  # Chrono start time
                 )
    def __init__(self):
        """ Create a new chronometer initialized with current time
        """
        super(Chrono, self).__init__()
        self.restart()

    def get_start(self):
        """ Get the chrono start time

        Returns:
            Time when chronometer has been started
        """
        return self.startTime

    def get_elapsed(self):
        """ Get the chrono elapsed time

        Returns:
            Time spent from chronometer start time (float), in seconds
        """
        return time.time() - self.startTime

    def restart(self):
        """ Restart chrono to current time
        """
        self.startTime = time.time()

    def __str__(self):
        """ Convert this chronometer into a string

        Returns:
            String of the chrono elapsed time
        """
        return str(self.get_elapsed())


###############################################################################
## Public functions
###############################################################################

def check_default(val, default):
    """ Check that an argument value is DEFAULT and returns the default value if so.

    This method has to be used in conjunction with usage of the DEFAULT constant as
    default value of a parameter. It allows to assign a parameter to a default value
    that can be computed dynamically.

    Args:
        val      Value to check
        default: Default value to return if val is DEFAULT
    Returns:
        val if val is different from DEFAULT, default otherwise
    """
    if (val is DEFAULT):
        return default
    return val


def is_bool(val):
    """ Check if a value is a boolean, including numpy variants if any

    Args:
        val: Value to check
    Returns:
        True if value is a boolean.
    """
    return type(val) in BOOL_TYPES


def is_int(val):
    """ Check if a value is an integer, including numpy variants if any

    Args:
        val: Value to check
    Returns:
        True if value is an integer.
    """
    return type(val) in INTEGER_TYPES


def is_float(val):
    """ Check if a value is a float, including numpy variants if any

    Args:
        val: Value to check
    Returns:
        True if value is a float
    """
    return type(val) in FLOAT_TYPES


def is_number(val):
    """ Check if a value is a number, including numpy variants if any

    Args:
        val: Value to check
    Returns:
        True if value is a number
    """
    return type(val) in NUMBER_TYPES


def is_array(val):
    """ Check if a value is an array (list or tuple)

    Args:
        val: Value to check
    Returns:
        True if value is an array (list or tuple)
    """
    return isinstance(val, (list, tuple))


def is_array_of_type(val, typ):
    """ Check that a value is an array with all elements instances of a given type

    Args:
        val: Value to check
        typ: Expected element type
    Returns:
        True if value is an array with all elements with expected type
    """
    return isinstance(val, (list, tuple)) and (all(isinstance(x, typ) for x in val))


def assert_arg_int_interval(val, mn, mx, name=None):
    """ Check that an argument is an integer in a given interval

    Args:
        val:  Argument value
        mn:   Minimal possible value (included)
        mx:   Maximal possible value (excluded)
        name: Name of the parameter (optional), used in raised exception.
    Raises:
      TypeError exception if wrong argument type
    """
    assert is_int(val) and (val >= mn) and (val < mx), \
           "Argument '" + name + "' should be an integer in [" + str(mn) + ".." + str(mx) + ")"


def to_string(val):
    """ Convert a value into a string, recursively for lists and tuples

    Args:
        val: Value to convert value
    Returns:
        String representation of the value
    """
    # Check tuple
    if (isinstance(val, tuple)):
        if (len(val) == 1):
            return "(" + to_string(val[0]) + ",)"
        return "(" + ", ".join(map(to_string, val)) + ")"
    
    # Check list
    if (isinstance(val, list)):
        return "[" + ", ".join(map(to_string, val)) + "]"
    
    # Default
    return str(val)


def equals(v1, v2):
    """ Check that two values are logically equal, i.e. with the same attributes with the same values, recursively

    This method does NOT call __eq__ and is then proof to overloading of '=='

    Args:
       val1: First value
       val2: Second value
    Returns:
        True if both values are identical, false otherwise
    """
    # Check same value (also covers some primitive types as int, float and strings, but not guarantee)
    if v1 is v2:
        return True
    
    # Check same type
    t = type(v1)
    if not (t is type(v2)):
        return False
    
    # Check basic types
    if (t in (int, float, str)):
        return (v1 == v2)
    
    # Check list or tuple
    if isinstance(v1, (list, tuple)):
        l = len(v1)
        if not (l is len(v2)):
            return False
        for i in range(l):
            if not equals(v1[i], v2[i]):
                return False
        return True
    
    # Check dictionary
    if isinstance(v1, dict):
        if not (len(v1) is len(v2)):
            return False
        # Compare keys
        k1 = sorted(tuple(v1.keys()))
        k2 = sorted(tuple(v2.keys()))
        if not equals(k1, k2):
            return False
        # Compare values
        for k in k1:
            if not equals(v1.get(k), v2.get(k)):
                return False
        return True

    # Check sets
    if isinstance(v1, set):
        if not (len(v1) is len(v2)):
            return False
        # Compare values
        return equals(sorted(tuple(v1)), sorted(tuple(v2)))

    # Check object values with slots
    if (hasattr(v1, '__slots__')):
        for k in v1.__slots__:
            if not equals(getattr(v1, k), getattr(v2, k)):
                return False
        return True
    
    try:
        # Compare dictionaries
        return equals(v1.__dict__, v2.__dict__)
    except:
        return False


def make_directories(path):
    """ Ensure a directory path exists

    Args:
        path: Directory path to check or create 
    Raises:
        Any IO exception if directory creation is not possible
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def create_stdout_logger(name):
    """ Create a default logger on stdout with default formatter printing time at the beginning
        of the line.

    Args:
        name:  Name of the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
    logger.addHandler(ch)
    return logger


def get_file_name_only(file):
    """ Get the name of a file, without directory or extension
    Args:
        file:  File name
    Returns:
        Name of the file, without directory or extension
    """
    return os.path.splitext(os.path.basename(file))[0]


def read_string_file(file):
    """ Read a file as a string
    Args:
        file:  File name
    Returns:
        File content as a string
    """
    with open(file, "r") as f:
        str = f.read()
    return str


def write_string_file(file, str):
    """ Write a string into a file.
    Args:
        file:  File name
        str:   String to write
    """
    with open(file, "w") as f:
        f.write(str)


def format_text(txt, size):
    """ Format a given text in multiple lines
    Args:
        txt:  Text to format
        size: Line size
    Returns:
        List of lines.
    """
    res = []
    sepchars = ' \t\n\r'
    txt = txt.strip(sepchars)
    while (len(txt) > size):
        # Search end of line
        enx = size
        while (enx > 0) and (txt[enx] not in sepchars):
            enx -= 1
        # Check no separator in the line
        if (enx == 0):
            enx = size
        # Check for a end of line in the line
        x = txt.find('\n', 0, enx)
        if (x > 0):
            enx = x
        # Append line
        res.append(txt[:enx])
        # Remove line from source
        txt = txt[enx:].strip(sepchars)
    # Add last line
    if txt != "":
        res.append(txt)
        return res


#-----------------------------------------------------------------------------
# String conversion functions
#-----------------------------------------------------------------------------

# Dictionary of special characters conversion
_FROM_SPECIAL_CHARS = {'n': "\n", 't': "\t", 'r': "\r", 'f': "\f", 'b': "\b", '\\': "\\", '"': "\""}

# Dictionary of special characters conversion
_TO_SPECIAL_CHARS = {'\n': "\\n", '\t': "\\t", '\r': "\\r", '\f': "\\f", '\b': "\\b", '\\': "\\\\", '\"': "\\\""}

# Set of symbol characters
_SYMBOL_CHARS = set(x for x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def is_symbol_char(c):
    """ Check whether a character can be used in a symbol

    Args:
        c: Character
    Returns:
        True if character in 0..9, a..z, A..Z, _ or .
    """
    # return ((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')) or ((c >= '0') and (c <= '9')) or (c == '_')
    # Following is 25% faster
    return (c in _SYMBOL_CHARS)


def to_printable_string(str):
    """ Build a printable string from raw string (add escape sequences and quotes if necessary)

    Args:
        str: Identifier string
    Returns:
        CPO identifier string, including double quotes and escape sequences if needed if not only chars and integers
    """
    # Check is string can be used as it is
    if (all(is_symbol_char(c) for c in str)):
        return(str)
    # Build result string
    res = ['"']
    for c in str:
        res.append(_TO_SPECIAL_CHARS.get(c, c))
    res.append('"')
    return(''.join(res))


def to_internal_string(str):
    """ Convert string (with enclosing quotes) into internal string (interpret escape sequences)

    Args:
        str: String to convert
    Returns:
        Raw string corresponding to source
    """
    res = []
    i = 1
    slen = len(str) - 1
    while i < slen:
        c = str[i]
        if (c == '\\'):
            i += 1
            c = _FROM_SPECIAL_CHARS.get(str[i], None)
            if c is None:
                raise CpoException("Unknown special character '\\" + str[i] + "'")
        res.append(c)
        i += 1
    return ''.join(res)
    
    
#-----------------------------------------------------------------------------
# Logging
#-----------------------------------------------------------------------------


def log(*msg):
    """ Log a message on default log output
    Args:
        msg: List of elements to print
    """
    for m in msg:
        sys.stdout.write(str(m))
    sys.stdout.write('\n')

