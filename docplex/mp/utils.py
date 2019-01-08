# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore
import logging
import os
import sched
import tempfile
import threading
import time
import sys

from six import PY2 as six_py2
from six import itervalues

from docplex.mp.compat23 import Queue
from docplex.mp.compat23 import izip

__int_types = {int}
__float_types = {float}
__numpy_ndslot_type = None
__pandas_series_type = None
__pandas_dataframe_type = None

try:
    type(long)
    # long is indeed a type we are in Python2,
    __int_types.add(long)
except NameError:  # pragma: no cover
    # long is not a type, do nothing
    pass  # pragma: no cover

try:
    import numpy
    _numpy_is_available = True


    __int_types.add(numpy.bool_)
    __int_types.add(numpy.bool)

    __int_types.add(numpy.int_)
    __int_types.add(numpy.intc)
    __int_types.add(numpy.intp)

    __int_types.add(numpy.int8)
    __int_types.add(numpy.int16)
    __int_types.add(numpy.int32)
    __int_types.add(numpy.int64)

    __int_types.add(numpy.uint8)
    __int_types.add(numpy.uint16)
    __int_types.add(numpy.uint32)
    __float_types.add(numpy.uint64)

    __float_types.add(numpy.float_)
    __float_types.add(numpy.float16)
    __float_types.add(numpy.float32)
    __float_types.add(numpy.float64)


    from numpy import ndarray
    __numpy_ndslot_type = ndarray
except ImportError:  # pragma: no cover
    _numpy_is_available = False  # pragma: no cover
    numpy_is_numeric = None
    numpy_is_integer = None

try:
    from pandas import Series, DataFrame

    __pandas_series_type = Series
    __pandas_dataframe_type = DataFrame
except ImportError:
    __pandas_series_type = None
    __pandas_dataframe_type = None

__int_types = frozenset(__int_types)
def is_int(s):
    type_of_s = type(s)
    return type_of_s in __int_types or (_numpy_is_available and numpy_is_integer(type(s)))


__all_python_num_types = frozenset(__float_types.union(__int_types))

if _numpy_is_available:
    def numpy_is_numeric(t):
        # returns True if the specified type is numeric
        try:
            flag =  numpy.issubdtype(t, numpy.number)
            global __all_python_num_types
            if flag is True:
                tmp = set(__all_python_num_types)
                tmp.add(type(t))
                __all_python_num_types = frozenset(tmp)
            return flag
        except TypeError:
            return False

    def numpy_is_integer(t):
        # returns True if the specified type is integer
        try:
            flag = numpy.issubdtype(t, numpy.integer)
            if flag is True:
                global __all_python_num_types
                global __int_types
                tmp = set(__all_python_num_types)
                tmp.add(type(t))
                __all_python_num_types = frozenset(tmp)

                tmp = set(__int_types)
                tmp.add(type(t))
                __int_types = frozenset(tmp)
            return flag
        except TypeError:
            return False

def is_number(s):
    type_of_s = type(s)
    return type_of_s in __all_python_num_types or (_numpy_is_available and (numpy_is_numeric(type_of_s) or _is_numpy_ndslot(s)))


def _is_numpy_ndslot(s):
    # returns True if the argument is a numpy number
    # wrapped in a fake ndarray
    # all the following conditions must be satisfied:
    # 1. numpy is present
    # 2. type is ndarray
    # 3. shape is () empty tuple
    # 4. wrapped type in ndarray is numeric.
    try:
        retval = is_numpy_ndarray(s) and s.shape == () and (s.dtype.type in __all_python_num_types or numpy_is_numeric(s.dtype))
        return retval
    except AttributeError:  # if s is not a numpy type, s.dtype triggers this
        return False


def is_pandas_series(s):
    return __pandas_series_type is not None and type(s) is __pandas_series_type

def is_pandas_dataframe(s):
    return __pandas_dataframe_type and isinstance(s, __pandas_dataframe_type)


def is_numpy_ndarray(s):
    return __numpy_ndslot_type and type(s) is __numpy_ndslot_type

string_types  = {str}
if six_py2:
    string_types.add(unicode)
string_types  = frozenset(string_types)

def is_string(e):
    return type(e) in string_types
    #if e is None:
    #    return False
    #elif
    #isinstance(e, str):
    #    return True
    #elif six_py2:
    #    return isinstance(e, unicode)
    #else:
    #    return False

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


def is_iterable(e, accept_string=True):
    ''' Returns true if we can extract an iterator from it
    '''
    try:
        iter(e)
        return accept_string or not is_string(e)
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


def _to_list(arg):
    # INTERNAL:
    # 1. checks the argument is either a sequence or iterator,;
    # if sequence, returns the sequence, else converts to a list by exhsuating the iterator
    # BEWARE of the infinite generator!
    if is_iterator(arg):
        return list(arg)
    elif is_iterable(arg):
        return arg
    else:
        # an atom: wrap it into a list
        return list(arg)

def _build_ordered_sequence_types():
    if __pandas_series_type and __numpy_ndslot_type:
        return (list, __pandas_series_type, __numpy_ndslot_type)
    elif __pandas_series_type:
        return (list, __pandas_series_type)
    elif __numpy_ndslot_type:
        return (list, __numpy_ndslot_type)
    else:
        return (list,)

def is_ordered_sequence(arg, type_tuple=_build_ordered_sequence_types()):
    return isinstance(arg, type_tuple)


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
        msg = "Cannot process value: {0:s} to type: {1!s}, tolerance: {2:g}".format(raw_value, vartype, tolerance)
        DOcplexException.__init__(self, msg)




def normalize_basename(s, force_lowercase=True):
    # replace all whietspaces by _
    l = s.lower() if force_lowercase else s
    # table = mktrans(" ", "_")
    # return l.translate(table)
    return l.replace(" ", "_")


def make_output_path2(actual_name, extension, basename_arg, path=None):
    # INTERNAL
    raw_basename = resolve_pattern(basename_arg, actual_name) if basename_arg else actual_name
    if raw_basename.find(" ") > 0:
        actual_basename = raw_basename.replace(" ", "_")
    else:
        actual_basename = raw_basename
    output_dir = path or tempfile.gettempdir()
    if not actual_basename.endswith(extension):
        actual_basename = actual_basename + extension
    path = os.path.join(output_dir, actual_basename)
    return path


def make_path(error_handler, basename, extension, output_dir=None, name_transformer=None):
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    elif not os.path.exists(output_dir):
        if not os.makedirs(output_dir):
            error_handler.error("directory not found and not created: {0:s}", output_dir)
            return None

    norm_name = normalize_basename(basename)
    basename = norm_name if not name_transformer else name_transformer % norm_name
    filename = basename + extension
    full_path = '/'.join([output_dir, filename])
    return full_path


def generate_constant(the_constant, count_max):
    if count_max is None:
        count_max = sys.maxsize
    loop_counter = 0
    while loop_counter <= count_max:
        yield the_constant
        loop_counter += 1


def iter_emptyset():
    return iter([])


def resolve_pattern(pattern, args):
    """
    returns a string in which slots have been resolved with args, if the string has slots anyway,
    else returns the strng itself (no copy, should we??)
    :param pattern:
    :param args:
    :return:
    """
    if args is None or len(args) == 0:
        return pattern
    elif pattern.find('%') >= 0:
        return pattern % args
    elif pattern.find("{") >= 0:
        # star magic does not work for single args
        return pattern.format(*args)
    else:
        # fixed pattern, no placeholders
        return pattern


def str_holo(arg, maxlen):
    """ Returns a truncated string representation of arg

    If maxlen is positive (or null), returns str(arg) up to maxlen chars.

    :param arg:
    :param maxlen:
    :return:
    """
    s = str(arg)
    if maxlen < 0 or len(s) <= maxlen:
        return s
    else:
        return "{}..".format(s[:maxlen])


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


def open_universal_newline(filename, mode):
    """Opens a file in universal new line mode, in a python 2 and python 3
    compatible way.
    """
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


class CyclicLoop(object):
    """ A cyclic loop executes actions at specified intervals, until
    ``stop()`` is called.

    This loop is based on sched.scheduler

    Attributes:
        stopped: True if the loop is stopped.
    """

    class Task(object):
        """This class stores information needed to manage tasks.

        Attributes:
            id: The id of the task (automatically generated)
            interval: The interval on which that task is called
            action: The action function to call at ``interval``
            argument: The arguments for the action function
        """
        id = 0
        idgen_lock = threading.Lock()

        def __init__(self, interval, priority, action, argument=()):
            self.interval = interval
            self.priority = priority
            self.action = action
            self.argument = argument
            with self.idgen_lock:
                self.id = CyclicLoop.Task.id
                CyclicLoop.Task.id += 1

    def __init__(self):
        """Initialize a new empty CyclicLoop
        """
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.scheduler = sched.scheduler(time.time, time.sleep)
        # maps task id -> ev
        self.events_by_id = {}
        self.tasks_by_id = {}  # task id -> task

    def enter(self, interval, priority, action, argument=()):
        """Schedule a new event.

        Works like sched.scheduler.enter(), but instead of a ``delay``, the
        first argument is the ``interval`` the action must be performed.
        """
        with self.stop_lock:
            if not self.stopped:
                task = CyclicLoop.Task(interval, priority, action, argument)
                self.tasks_by_id[task.id] = task
                self._queue(task)

    def _queue(self, task):
        ev = self.scheduler.enter(task.interval, task.priority,
                                  lambda a: self._process_task(a), (task.id,))
        self.events_by_id[task.id] = ev

    def _process_task(self, task_id):
        task = self.tasks_by_id[task_id]
        task.action(*task.argument)
        del self.events_by_id[task_id]
        # do not reschedule if we are shutting down
        with self.stop_lock:
            if not self.stopped:
                self._queue(task)

    def start(self):
        """Starts the loop. The loop stops only when ``stop()`` is called.
        """
        self.scheduler.run()

    def stop(self):
        """Stops the Loop.

        When the loop is stopped, its ``stopped`` attribute is set immediately,
        then all tasks in the scheduler are canceled.
        """
        with self.stop_lock:
            self.stopped = True
            for ev in itervalues(self.events_by_id):
                try:
                    self.scheduler.cancel(ev)
                except ValueError:
                    # if stop() is called from an event, the event has already
                    # been triggered and poped'
                    pass


class ClosableQueue(Queue):
    LAST = object()

    def close(self):
        self.put(self.LAST)

    def __iter__(self):
        while True:
            item = self.get()
            try:
                if item is self.LAST:
                    return
                yield item
            finally:
                self.task_done()


class ThreadedCyclicLoop(object):
    """ A cyclic loop executes actions at specified intervals, until
    ``stop()`` is called.

    This loop is based on threads.

    Attributes:
        stopped: True if the loop is stopped.
    """

    class Task(threading.Thread):
        """
        Attributes:
            id: The id of the task (automatically generated)
            interval: The interval on which that task is called
            action: The action function to call at ``interval``
            argument: The arguments for the action function
        """

        def __init__(self, loop, interval, priority, action, argument=()):
            super(ThreadedCyclicLoop.Task, self).__init__()
            self.loop = loop
            self.interval = interval
            self.priority = priority
            self.action = action
            self.argument = argument
            self.stopped = False

        def run(self):
            while not self.stopped:
                # instead of one big sleep, do some smaller sleeps so that
                # we can stop the thread with smaller granularity
                for _ in range(self.interval):
                    time.sleep(1)
                    if self.stopped:
                        break
                if not self.stopped:
                    self.perform()

        def stop(self):
            self.stopped = True

        def perform(self):
            self.action(*self.argument)

    def __init__(self):
        """Initialize a new empty ThreadedCyclicLoop
        """
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.threads = set()
        self.event_queue = ClosableQueue()

    def enter(self, interval, priority, action, argument=()):
        """Schedule a new event.

        Works like sched.scheduler.enter(), but instead of a ``delay``, the
        first argument is the ``interval`` the action must be performed.
        """
        with self.stop_lock:
            if not self.stopped:
                task = ThreadedCyclicLoop.Task(self, interval, priority,
                                               action, argument)
                self.threads.add(task)

    def start(self, mt_worker=None, mt_arg=()):
        """Starts the loop. The loop stops only when ``stop()`` is called.
        """
        for t in self.threads:
            t.start()
        if mt_worker:
            while not self.stopped:
                for task in self.event_queue:
                    mt_worker(*((task,) + mt_arg))
        for t in self.threads:
            t.join()

    def stop(self):
        """Stops the Loop.

        When the loop is stopped, its ``stopped`` attribute is set immediately,
        then all tasks in the scheduler are canceled.
        """
        with self.stop_lock:
            self.stopped = True
        self.event_queue.close()
        for t in self.threads:
            t.stop()


class _SymbolGenerator(object):
    """
    INTERNAL class
    """

    def __init__(self, pattern, offset=1):
        ''' Initialize the counter and the pattern.
            Fixes the pattern by suffixing '%d' if necessary.
        '''
        self.__pattern = pattern
        # add offset to counter.
        self.__offset = offset
        self._last_index = -1
        self._set_pattern(pattern)

    def _set_pattern(self, pattern):
        if pattern.endswith('%d'):
            self.__pattern = pattern
        else:
            self.__pattern = pattern + '%d'

    def _get_pattern(self):
        return self.__pattern

    pattern = property(_get_pattern, _set_pattern)

    def reset(self):
        self._last_index = -1

    def notify_new_index(self, new_index):
        # INTERNAL
        if new_index > self._last_index:
            self._last_index = new_index

    def new_symbol(self):
        """
        Generates and returns a new symbol.
        Guess a new (yet) unallocated index, then use the pattern.
        Note that we use the offset of 1 to generate the name so x1 has index 0, x3 has index 2, etc.
        :return: A symbol string, suposedly not yet allocated.
        """
        guessed_index = self._last_index + 1
        coined_symbol = self.__pattern % (guessed_index + self.__offset)
        self._last_index = guessed_index
        return coined_symbol


class _IndexScope(_SymbolGenerator):
    # INTERNAL: full scope of indices.

    def __init__(self, obj_iter, pattern, offset=1):
        _SymbolGenerator.__init__(self, pattern, offset)
        self._obj_iter = obj_iter
        self._index_map = None

    def _make_index_map(self):
        return {m.get_index(): m for m in self._obj_iter()}

    @property
    def iter(self):
        return self._obj_iter()

    def get_object_by_index(self, idx):
        if self._index_map is None:
            self._index_map = self._make_index_map()
        # do not raise when not found, return None.
        return self._index_map.get(idx)

    def reset(self):
        _SymbolGenerator.reset(self)
        self._index_map = None

    def notify_obj_index(self, obj, index):
        _SymbolGenerator.notify_new_index(self, index)
        if self._index_map is not None:
            self._index_map[index] = obj

    def notify_obj_indices(self, objs, indices):
        # take the last one??
        if indices:
            _SymbolGenerator.notify_new_index(self, max(indices))
            idxmap = self._index_map
            if idxmap is not None:
                for obj, idx in izip(objs, indices):
                    idxmap[idx] = obj

    def update_indices(self):
        if self._index_map is not None:
            self._index_map = self._make_index_map()

    def reindex_one(self, reindexed_index, indexer):
        for ct in self._obj_iter():
            old_model_index = ct.unchecked_index
            if old_model_index > reindexed_index:
                updated_index = indexer.get_ct_index(ct)
                if updated_index != old_model_index:
                    ct.set_index(updated_index)

    def reindex_all(self, indexer):
        for ct in self._obj_iter:
            old_model_index = ct.unchecked_index
            updated_index = indexer.get_ct_index(ct)
            if updated_index != old_model_index:
                ct.set_index(updated_index)


class CplexParameterHandler(object):
    # INTERNAL: util function to handle Cplex parameters appropriately

    def __init__(self, parameters):
        self._parameters = parameters

    def get_updated_parameters(self, solver):
        # --- limit threads if needed
        if getattr(solver, 'max_threads', None) is not None:
            if self._parameters.threads.get() == 0:
                max_threads = solver.max_threads
            else:
                max_threads = min(solver.max_threads,
                                  self._parameters.threads.get())
            # we don't want to duplicate parameters unnecessary
            if max_threads != self._parameters.threads.get():
                self._parameters = self._parameters.copy()
                self._parameters.threads = max_threads
                out_stream = solver.log_output_as_stream
                if out_stream:
                    out_stream.write(
                        "WARNING: Number of workers has been reduced to %s to comply with platform limitations.\n" % max_threads)
                    # ---
        return self._parameters
