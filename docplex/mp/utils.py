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

from six import itervalues

from docplex.mp.compat23 import Queue, StringIO, has_unicode_type


__int_types = {int}
__float_types = {float}
__numpy_ndslot_types = set()
__pandas_series_type = None

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

try:
    from pandas import Series

    __pandas_series_type = Series
except ImportError:
    __pandas_series_type = None


def is_int(s):
    type_of_s = type(s)
    return type_of_s in __int_types


__all_num_types = __float_types.union(__int_types)


def is_number(s):
    return type(s) in __all_num_types or (_numpy_is_available and _is_numpy_ndslot(s))


def _is_numpy_ndslot(s):
    # returns True if the argument is a numpy number
    # wrapped in a fake ndarray
    # all the following conditions must be satisfied:
    # 1. numpy is present
    # 2. type is ndarray
    # 3. shape is () empty tuple
    # 4. wrapped type in ndarray is numeric.
    return type(s) in __numpy_ndslot_types and s.shape == () and s.dtype.type in __all_num_types


def is_pandas_series(s):
    return __pandas_series_type is not None and type(s) is __pandas_series_type


def is_numpy_ndarray(s):
    return type(s) in __numpy_ndslot_types


def is_string(e):
    if e is None:
        return False
    elif isinstance(e, str):
        return True
    elif has_unicode_type():
        return isinstance(e, unicode)
    else:
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


class DOCplexQuadraticNotImplementedError(DOcplexException):
    def __init__(self, first, second):
        msg = "Cannot multiply {0!s} by {1!s}: quadratic programming not supported".format(first, second)
        DOcplexException.__init__(self, msg)


class DOCPlexQuadraticArithException(Exception):
    pass


# def normalize(s, force_lowercase=True):
#     l = s.lower() if force_lowercase else s
#     table = mktrans(" -+/\\<>", "_mpd___")
#     return l.translate(table)


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
    loop_counter = 0
    while loop_counter <= count_max:
        yield the_constant
        loop_counter += 1


def resolve_pattern(pattern, args):
    """
    returns a string in which slots have been resolved with args, if the string has slots anyway,
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
