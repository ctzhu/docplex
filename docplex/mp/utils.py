# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

import logging
import os
import tempfile

from six import iteritems

# copy_reg is copyreg in Py3
try:
    import copy_reg as copyreg
except ImportError:  # pragma: no cover
    import copyreg

# we want StringIO to process strings in Py2 and Py3
try:
    from cStringIO import StringIO
except ImportError:  # pragma: no cover
    from io import StringIO  # pragma: no cover

try:
    from string import maketrans as mktrans  # Python 2
except ImportError:  # pragma: no cover
    def mktrans(a, b):  # pragma: no cover
        return str.maketrans(a, b)  # pragma: no cover

__int_types = {int}
__float_types = {float}
__numpy_ndslot_types = set()

try:
    type(long)
    # long is indeed a type we are in Python2,
    __int_types.add(long)
except NameError:  # pragma: no cover
    # long is not a type, do nothing
    pass  # pragma: no cover

try:
    from numpy import int32, float32, int64, float64, int16, uint16, uint32, uint64, float32, int_, float_, bool_

    _numpy_is_available = True
    __int_types.add(int64)
    __int_types.add(int32)
    __int_types.add(int16)
    __int_types.add(uint64)
    __int_types.add(uint32)
    __int_types.add(uint16)
    __int_types.add(int_)
    __int_types.add(bool_)
    __float_types.add(float64)
    __float_types.add(float32)
    __float_types.add(float_)

    from numpy import ndarray
    import numpy as npcplex

    __numpy_ndslot_types.add(ndarray)
except ImportError:  # pragma: no cover
    _numpy_is_available = False  # pragma: no cover


def is_int(s):
    type_of_s = type(s)
    return type_of_s in __int_types


__all_num_types = __int_types.union(__float_types)


def is_number(s):
    type_of_s = type(s)
    if type_of_s in __all_num_types:
        return True
    else:
        return _numpy_is_available and _is_numpy_ndslot(s)


def _is_numpy_ndslot(s):
    # returns True if the argument is a numpy number
    # wrapped in a fake ndarray
    # all the following conditions must be satisfied:
    # 1. numpy is present
    # 2. type is ndarray
    # 3. shape is () empty tuple
    # 4. wrapped type in ndarray is numeric.
    return type(s) in __numpy_ndslot_types and s.shape == () and s.dtype.type in __all_num_types


_all_zeros = frozenset({0, 0.0})


def is_zero(x):
    return is_number(x) and x in _all_zeros


def is_numpy_ndarray(s):
    return type(s) in __numpy_ndslot_types


def is_string(e):
    if e is None:
        return False
    try:
        return e == str(e)
    except:
        return False


def has_len(e):
    try:
        len(e)
        return True
    except TypeError:
        return False


def is_indexable(e):
    ''' Returns true if it is indexable
    '''
    return hasattr(e, "__getitem__")


def is_iterable(e):
    ''' Returns true if we can extract an iterator from it
    '''
    try:
        iter(e)
        return True
    except TypeError:
        return False


def is_iterator(e):
    ''' Returns true if e is its own iterator.
    '''
    try:
        # some numpy iterators just fail on == but are ok with "is"
        return e is iter(e)
    except TypeError:
        return False


def is_function(e):
    from collections import Callable

    return isinstance(e, Callable)


def fix_format_string(fmt, dimen=1, key_format='_%s'):
    ''' Fixes a format string so that it contains dimen slots with %s inside
        arguments are:
         --- dimen is th enumber of slots we need
         --- meta-format is the format in which the %s is embedded. By default '_%s'
             for example if each item has to be surrounded by {} set key_format to _{%s}
    '''
    assert (dimen >= 1)
    actual_nb_slots = 0
    curpos = 0
    str_size = len(fmt)
    while curpos < str_size and actual_nb_slots < dimen:
        new_pos = fmt.find('%', curpos)
        if new_pos < 0:
            break
        actual_nb_slots += 1
        if actual_nb_slots >= dimen:
            break
        curpos = new_pos + 2
    # how much slots do we need to add to the end of the string??
    nb_missing = max(0, dimen - actual_nb_slots)
    return fmt + nb_missing * (key_format % '%s')


class DOcplexException(Exception):
    """ Base class for modeling exceptions 
    """
    DEFAULT_MSG = 'CplexPythonModeling exception raised'

    def __init__(self, msg, *args):
        Exception.__init__(self, msg)
        self.__msg = msg or self.__class__.DEFAULT_MSG
        self.__edited_message = None
        self.__args = args
        self._resolve_message()

    def _resolve_message(self):
        self.__edited_message = None
        if self.__args:
            if self.__msg.find('%') >= 0:
                self.__edited_message = self.__msg % self.__args
            elif self.__msg.find('{') >= 0:
                self.__edited_message = self.__msg.format(*self.__args)

    @property
    def message(self):
        return self.__edited_message or self.__msg


class DOCplexSolutionValueError(DOcplexException):
    def __init__(self, vartype, raw_value, tolerance):
        msg = "Cannot process value: %g to type: %s, tolerance: %f" % (raw_value, vartype.short_name, tolerance)
        DOcplexException.__init__(self, msg)


class DOCplexQuadraticNotImplementedError(DOcplexException):
    def __init__(self, first, second):
        msg = "Cannot multiply {0} by {1}: quadratic programming not supported".format(first, second)
        DOcplexException.__init__(self, msg)


_default_key_format = '_%s'

def __map_to_str(key_tuple):
    return tuple((str(z) for z in key_tuple))


def ensure_naming_function(keys, naming_rule, default_fn, dimen=1, key_format=None):
    ''' builds a naming rule from an input , a dimension, and an optional meta-format'''


    '''
    Makes sure the format string does contain the right number of format slots'''
    assert key_format is None or isinstance(key_format, str)
    key_format = key_format or _default_key_format

    if isinstance(naming_rule, str):
        fixed_naming_rule = fix_format_string(naming_rule, dimen, key_format)
        if 1 == dimen:
            return lambda _obj: fixed_naming_rule % str(_obj)
        else:
            return lambda _tuple: fixed_naming_rule % __map_to_str(_tuple)

    elif is_function(naming_rule):
        return naming_rule

    elif is_iterable(naming_rule):
        # use a closure
        key_to_names_dict = dict(zip(keys, naming_rule))
        return lambda k: key_to_names_dict[k] if k in key_to_names_dict else default_fn()
    else:
        raise DOcplexException('Cannot use this for naming variables: {0!s} -expecting string, function or iterable'
                               .format(naming_rule))


def normalize(s, force_lowercase=True):
    l = s.lower() if force_lowercase else s
    table = mktrans(" -+/\\<>", "_mpd___")
    return l.translate(table)


def make_path(error_handler, basename, extension, output_dir=None, name_transformer=None):
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    elif not os.path.exists(output_dir):
        if not os.makedirs(output_dir):
            error_handler.error("directory not found and not created: {0:s}", output_dir)
            return None

    norm_name = normalize(basename)
    basename = norm_name if not name_transformer else name_transformer % norm_name
    filename = basename + extension
    full_path = '/'.join([output_dir, filename])
    return full_path


def generate_constant(the_constant, count_max=9999):
    loop_counter = 0
    while loop_counter <= count_max:
        yield the_constant
        loop_counter += 1


def resolve_pattern(pattern, args):
    """
    returns a string in which slots have been resolved with args, iff the string has slots anyway,
    else returns the strng itself (no copy, should we??)
    :param pattern:
    :param args:
    :return:
    """
    if args is None:
        return pattern
    elif pattern.find('%') >= 0:
        return pattern % args
    elif pattern.find("{") >= 0:
        # star magic does not work for single args
        return pattern.format(*args)
    else:
        # fixed pattern, no placeholders
        return pattern


DOCPLEX_CONSOLE_HANDLER = None


def get_logger(name, verbose=False):
    logging_level = logging.WARNING
    if verbose:
        logging_level = logging.DEBUG

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    global DOCPLEX_CONSOLE_HANDLER

    if DOCPLEX_CONSOLE_HANDLER is None:
        DOCPLEX_CONSOLE_HANDLER = logging.StreamHandler()
        DOCPLEX_CONSOLE_HANDLER.setLevel(logging_level)

    if DOCPLEX_CONSOLE_HANDLER not in logger.handlers:
        logger.addHandler(DOCPLEX_CONSOLE_HANDLER)

    return logger


try:
    xrange(2)
    fast_range = xrange
except NameError:  # pragma: no cover
    fast_range = range  # pragma: no cover

import math

from collections import Counter


class ExprCounter(Counter):
    """
    A subclass of Counter which does not require a dictionary to be updated
    Can be updated from an item (assumed to be a key)
    or from a key and a value
    SEE how to remember the order in which objects are added.
    """

    @classmethod
    def fromkeys(cls, iterable, v=None):
        raise NotImplementedError()  # pragma: no cover

    def update_from_item(self, item):
        """
        Adds one item occurence
        :param item:
        :return:
        """
        self.update_from_item_value(item, value=1)

    def update_from_item_value(self, item, value):
        """
        This differs from standard Counter when a dict instance is required.
        :param item: the key to be updated
        :param value: the associated value
        :return:
        """
        if value:
            self[item] = self.get(item, 0) + value

    def update_from_scaled_dict(self, other_dict, factor):
        """
        Updates counter from a dict instance, but with an inflation factor.
        Does nothing if factor is 0
        """
        self_get = self.get
        if factor is 0:
            # nothin to do
            pass
        elif factor is 1:
            # standard update
            self.update(other_dict)
        else:
            # update by scaled value
            for item, value in iteritems(other_dict):
                if value:
                    self[item] = self_get(item, 0) + value * factor

    def normalize(self):
        """
        Removes all entries with zero value
        :return:
        """
        self_get = self.get
        doomed_keys = [k for k in self if self_get(k) is 0]
        for dk in doomed_keys:
            del self[dk]
        return self


import sys


class RedirectedOutputContext(object):

    def __init__(self, new_out, error_handler=None):
        if new_out is not None:
            self._of = new_out
        else:
            self._of = sys.stdout
        self._saved_out = sys.stdout
        self.error_handler = error_handler

    def __enter__(self):
        self._saved_out = sys.stdout
        sys.stdout = self._of
        if self.error_handler:
            self.error_handler.suspend()
        return self._of

    # noinspection PyUnusedLocal
    def __exit__(self, atype, avalue, atraceback):
        sys.stdout = self._saved_out
        if self.error_handler:
            self.error_handler.flush()


class RedirectedOutputToStringContext(RedirectedOutputContext):

    def __init__(self, error_handler=None):
        self._oss = StringIO()
        RedirectedOutputContext.__init__(self, new_out=self._oss, error_handler=error_handler)

    def __enter__(self):
        RedirectedOutputContext.__enter__(self)
        # return self as we need to extract the string after exit
        return self

    def get_str(self):
        return self._oss.getvalue()

    def __del__(self):
        # kill the stringio on deletion
        self._oss = None


# numeric utilities
def round_nearest_halfway_from_zero(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For values like 1.5 the intetger with greater absolute value is returned.
    This treats positive and negative values in a symmetric manner.
    This is called "round half away from zero"


    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = my_round_even(x)  # math.floor(x + 0.5)
        return int(raw_nearest)


def my_round_even(number):
    """
    Simplified version from future
    """
    from decimal import Decimal, ROUND_HALF_EVEN

    d = Decimal.from_float(number).quantize(1, rounding=ROUND_HALF_EVEN)
    return int(d)


def round_nearest_towards_infinity(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For ties like 1.5 the ceiling integer is returned.
    This is called "round towards infinity"

    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = math.floor(x + 0.5)
        return int(raw_nearest)


def open_universal_newline(filename, mode):
    try:
        # try the python 3 syntax
        return open(filename, mode=mode, newline=None)
    except TypeError as te:
        if "'newline'" in te.message:
            # so open does not have a newline parameter -> python 2, use "U"
            # mode
            return open(filename, mode=mode + "U")
        else:
            # for other errors, just raise them
            raise
