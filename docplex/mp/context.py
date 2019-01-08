# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import os

from copy import deepcopy
import six
from six import iteritems
import socket
import sys
import warnings
from os.path import isfile, isabs

from docplex.util.environment import get_environment

from docplex.mp.format import ExchangeFormat
from docplex.mp.utils import is_string, open_universal_newline
from docplex.mp.environment import Environment
from docplex.mp.params.cplex_params import get_params_from_cplex_version
from docplex.mp.utils import DOcplexException

try:
    from docplex.worker.solvehook import get_solve_hook
except ImportError:
    get_solve_hook = None

# some utility methods
def _get_value_as_int(d, option):
    try:
        value = int(d[option])
    except Exception:
        value = None
    return value


def _convert_to_int(value):
    if str(value).lower() == 'none':
        return None
    try:
        value = int(value)
    except Exception:
        value = None
    return value


def _get_value_as_string(d, option):
    return d.get(option, None)


def _get_value_as_boolean(d, option):
    try:
        value = _convert_to_bool(d[option])
    except Exception:
        value = None
    return value

_BOOLEAN_STATES = {'1': True, 'yes': True, 'true': True, 'on': True,
                   '0': False, 'no': False, 'false': False, 'off': False}


def _convert_to_bool(value):
    if value is None:
        return None
    svalue = str(value).lower()
    if svalue == "none":
        return None
    if svalue not in _BOOLEAN_STATES:
        raise ValueError('Not a boolean: %s' % value)
    return _BOOLEAN_STATES[svalue]


class open_filename_universal(object):
    def __init__(self, filename, *args, **kwargs):
        self.closing = kwargs.pop('closing', False)
        if isinstance(filename, six.string_types):
            self.fh = open_universal_newline(filename, "r")
            self.closing = True
        else:
            self.fh = filename

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closing:
            self.fh.close()
        return False


def is_ignored(what, key):
    try:
        # if string, allow comma separated form
        if isinstance(what, six.string_types):
            values = what.split(",")
            if key in values:
                return True
        elif key in what:
            return True
    except AttributeError:
        # no ignored_keys, just pass
        pass
    return False


def is_key_ignored(context, key):
    return is_ignored(context.solver.docloud.ignored_keys, key)


def is_url_ignored(context, url):
    return is_ignored(context.solver.docloud.ignored_urls, url)


def is_auto_publishing_solve_details(context):
    if get_solve_hook == None:
        return False  # not in a worker
    try:
        auto_publish_details = context.solver.auto_publish.solve_details
    except AttributeError:
        try:
            auto_publish_details = context.solver.auto_publish
        except AttributeError:
            auto_publish_details = False
    return auto_publish_details


def is_auto_publishing_json_solution(context):
    if get_solve_hook == None:
        return False  # not in a worker
    try:
        auto_publish = context.solver.auto_publish.json_solution
    except AttributeError:
        try:
            auto_publish = context.solver.auto_publish
        except AttributeError:
            auto_publish = False
    return auto_publish

class BaseContext(dict):
    # Class for handling the list of parameters.
    def __init__(self, **kwargs):
        """ Create a new context.

        Args:
            List of ``key=value`` to initialize context with.
        """
        super(BaseContext, self).__init__()
        for k, v in kwargs.items():
            self.set_attribute(k, v)

    def __setattr__(self, name, value):
        self.set_attribute(name, value)

    def __getattr__(self, name):
        return self.get_attribute(name)

    def set_attribute(self, name, value):
        self[name] = value

    def get_attribute(self, name, default=None):
        if name.startswith('__'):
            raise AttributeError
        res = self.get(name, None)
        if res is not None:
            return res
        raise AttributeError("'{0}' object has no attribute '{1}'".format(type(self).__name__, name))


class SolverContext(BaseContext):
    # for internal use
    def __init__(self, **kwargs):
        super(SolverContext, self).__init__(**kwargs)
        self.log_output = False
        self.max_threads = get_environment().get_available_core_count()
        self.auto_publish = BaseContext()
        self.auto_publish.solve_details = True
        self.auto_publish.json_solution = True

    def __deepcopy__(self, memo):
        # We override deepcopy here just to make sure that we don't deepcopy
        # file descriptors...
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in iteritems(self):
            # do not duplicate those (io like objects)
            if k == "log_output" and hasattr(v, "write"):
                value = v
            else:
                value = deepcopy(v, memo)
            setattr(result, k, value)
        return result

    def get_log_output_as_stream(self):
        log_output = None
        try:
            log_output = self.log_output
        except AttributeError:
            return None

        output_stream = None
        # if log_output is an object with a lower attribute, let's use it
        # as string.lower and check for some known string values
        if hasattr(log_output, "lower"):
            k = log_output.lower()
            if k in _BOOLEAN_STATES:
                if _convert_to_bool(k):
                    output_stream = sys.stdout
                else:
                    output_stream = None
            if k in ["stdout", "sys.stdout"]:
                output_stream = sys.stdout
            if k in ["stderr", "sys.stderr"]:
                output_stream = sys.stderr
        # if log_output is a boolean, do the direct mapping to default streams
        if isinstance(log_output, bool):
            if log_output:
                output_stream = sys.stdout
        # if it has a write() attribute, just return it
        if hasattr(log_output, "write"):
            output_stream = log_output

        return output_stream

    log_output_as_stream = property(get_log_output_as_stream)


class Context(BaseContext):
    """ The context used to control the behavior of solve engine.

    Attributes:
        cplex_parameters: A
           :class:`docplex.mp.params.parameters.RootParameterGroup` to store
           CPLEX parameters.
        solver.auto_publish: If ``True``, a model being solved will automatically
            publish all publishable items (``solve_details``,
            ``json_solution``).
        solver.auto_publish.solve_details: If ``True``, solve details are
            automatically published when :meth:`docplex.mp.model.Model.solve` is called.
            The default value is ``True``.
        solver.auto_publish.json_solution: If ``True``, the solution is automatically
            published when :meth:`docplex.mp.model.Model.solve` is called.
            The solution is saved as an output attachment whose name is
            ``solution.json``. The default value is ``False``, unless the Python script is run on the
            DOcplexcloud service.
        solver.log_output: This attribute can have the following values:

            * True: When True, logs are printed to sys.out.
            * False: When False, logs are not printed.
            * A file-type object: Logs are printed to that file-type object.

        solver.docloud: The parent node for attributes controlling the solve on Decision Optimization on Cloud.
        solver.docloud.url: The DOcplexcloud service URL.
        solver.docloud.key: The DOcplexcloud service API key.
        solver.docloud.ignored_keys: A collection or a string that is a comma-separated list of
            values to ignore. If any ``key`` passed has a value in this list,
            the key is ignored.
        solver.docloud.ignored_urls: A collection or a string that is a comma-separated list of
            values to ignore. If any ``url`` passed has a value in this list,
            the url is ignored.
        solver.docloud.run_deterministic: Specific engine parameters are uploaded to keep the
            run deterministic.
        solver.docloud.verbose: Makes the connector verbose.
        solver.docloud.timeout: The timeout for requests.
        solver.docloud.waittime: The wait time to wait for jobs to finish.
        solver.docloud.verify: If True, verifies SSL certificates.
        solver.docloud.log_requests: If True, the REST requests are logged.
        solver.docloud.log_poll_interval: The interval for log polling.
        solver.docloud.progress_poll_interval: The interval for progress polling.
        solver.docloud.exchange_format: The exchange format to use.
            When setting the format, you can use the following strings: "lp".
            When getting the format, the property type is
            `docplex.mp.format.ExchangeFormat`.
    """
    def __init__(self, **kwargs):
        # initialize default env
        cplex_parameters = kwargs.get('cplex_parameters', None)
        if cplex_parameters is None:
            local_env = kwargs.get('_env') or Environment.make_new_configured_env()
            cplex_version = local_env.cplex_version
            cplex_parameters = get_params_from_cplex_version(cplex_version)
        # initialize fields of this
        super(Context, self).__init__(solver=SolverContext(docloud=DOcloudContext()),
                                      cplex_parameters=cplex_parameters,
                                      docplex_tests=BaseContext())
        # update will also ensure compatibility with older kwargs like
        # 'url' and 'api_key'
        self.update(kwargs, create_missing_nodes=True)


    @staticmethod
    def make_default_context(file_list=None, logger=None, **kwargs):
        """Creates a default context.

        If `file_list` is a string, then it is considered to be the name
        of a config file to be read.

        If `file_list` is a list, it is considered to be a list of names
        of a config files to be read.

        if `file_list` is None or not specified, the following files are
        read if they exist:

            * the PYTHONPATH is searched for the following files:

                * cplex_config.py
                * cplex_config_<hostname>.py
                * docloud_config.py

            * if a ``.docplexrc`` file exists in ``~``
              (``os.path.expanduser("~")`` in Python), that file is parsed, and
              the properties set into the context.

        :deprecated:
           As of V1.0, reading ``.docplexrc`` is deprecated, and ``.py`` files for
           configuration should be used instead.

        A ``.docplexrc`` file is similar to a Java properties file (``=`` or ``:`` separated
        pairs of `key,values`) or a Python file if it ends with ``.py``.
        Python files are evaluated with a `context` object in the current
        scope, and you set values from this context::

            context.solver.docloud.url = 'http://testing.blabla.ibm.com'
            context.solver.docloud.key = 'This is an api_key'
            context.cplex_parameters.emphasis.memory = 1
            context.cplex_parameters.emphasis.mip = 2

        Args:
            file_list: The list of config files to read.
            kwargs: context parameters to override. See :func:`docplex.mp.context.Context.update`
        """
        context = Context()
        context.read_settings(file_list=file_list, logger=logger)
        context.update(kwargs)
        return context

    def copy(self):
        # Makes a deep copy of the context.
        #
        # Returns:
        #   A deep copy of the context.
        return deepcopy(self)

    def update_from_list(self, values, logger=None):
        # For each pair of `(name, value)` in values, try to set the
        # attribute.
        for name, value in values:
            try:
                if name in DOcloudContext.LEGACY_PROPERTIES:
                    mapping = DOcloudContext.LEGACY_PROPERTIES[name]
                    if mapping is None:
                        mapping = "solver.docloud.%s" % name
                    self._set_value(self, mapping, value)
                else:
                    self._set_value(self, name, value)
            except AttributeError:
                if logger is not None:
                    logger.warning("Ignoring undefined attribute : {0}".format(name))

    def _set_value(self, root, property_spec, property_value):
        property_list = property_spec.split('.')
        property_chain = property_list[:-1]
        to_be_set = property_list[-1]
        o = root
        for c in property_chain:
            o = getattr(o, c)
        try:
            target_attribute = getattr(o, to_be_set)
        except AttributeError:
            target_attribute = None
        if target_attribute is None:
            # Simply set the attribute
            try:
                setattr(o, to_be_set, property_value)
            except DOcplexException:
                pass  # ignore this
        else:
            # try a set_converted_value if it's a Parameter
            try:
                target_attribute.set(property_value)
            except AttributeError:
                # no set(), just setattr
                setattr(o, to_be_set, property_value)

    def update(self, kwargs, create_missing_nodes=False):
        """ Updates this context from child parameters specified in ``kwargs``.

        The following keys are recognized:
        
            - cplex_parameters: A set of CPLEX parameters to use instead of the parameters defined as ``context.cplex_parameters``.
            - agent: Changes the ``context.solver.agent`` parameter.
                Supported agents include:

                - ``docloud``: forces the solve operation to use DOcplexcloud
                - ``local``: forces the solve operation to use native CPLEX

            - url: Overwrites the URL of the DOcplexcloud service defined by ``context.solver.docloud.url``.
            - key: Overwrites the authentication key of the DOcplexcloud service defined by ``context.solver.docloud.key``.
            - log_output: if ``True``, solver logs are output to stdout.
                If this is a stream, solver logs are output to that stream object.
                Overwrites the ``context.solver.log_output`` parameter.
        
        Args:
            kwargs: A ``dict`` containing keyword args to use to update this context.
            create_missing_nodes: When a keyword arg specify a parameter that is not already member of this context,
                creates the parameter if ``create_missing_nodes`` is True.
        
        """
        for k in kwargs:
            value = kwargs.get(k)
            if value is not None:
                self.update_key_value(k, value,
                                      create_missing_nodes=create_missing_nodes)

    def update_key_value(self, k, value, create_missing_nodes=False, warn=True):
        if k is 'docloud_context':
            warnings.warn('docloud_context is deprecated, use context.solver.docloud instead')
            self.solver.docloud = value
        elif k is 'cplex_parameters':
            self.cplex_parameters = value
        elif k is 'url':
            self.solver.docloud.url = value
        elif k is 'api_key' or k is 'key':
            self.solver.docloud.key = value
        elif k is 'log_output':
            self.solver.log_output = value
        elif k is 'override':
            self.update_from_list(iteritems(value))
        elif k is 'proxies':
            self.solver.docloud.proxies = value
        elif k is '_env':
            # do nothing this is just here to avoid creating too many envs
            pass
        elif k is 'agent':
            self.solver.agent = value
        else:
            if create_missing_nodes:
                self[k] = value
            elif warn:
                warnings.warn("Unknown quick-setting in Context: {0:s}, value: {1!s}".format(k, value),
                              stacklevel=2)

    def read_settings(self, file_list=None, logger=None):
        """Reads settings for a list of files.

        If `file_list` is a string, then it is considered to be the name
        of a config file to be read.

        If `file_list` is a list, it is considered to be a list of names
        of config files to be read.

        if `file_list` is None or not specified, the following files are
        read if they exist:

            * the PYTHONPATH is searched for the following files:

                * cplex_config.py
                * cplex_config_<hostname>.py
                * docloud_config.py

            * if a ``.docplexrc`` file exists in ``~``
              (``os.path.expanduser("~")`` in Python), that file is parsed, and
              the properties set into the context.

        :deprecated:
           As of V1.0, reading ``.docplexrc`` is deprecated, and ``.py`` files for
           configuration should be used instead.
        
        A ``.docplexrc`` file is similar to a Java properties file (``=`` or ``:`` separated
        pairs of `key,values`) or a Python file if it ends with ``.py``.
        Python files are evaluated with a `context` object in the current
        scope, and you set values from this context::

            context.solver.docloud.url = 'http://testing.blabla.ibm.com'
            context.solver.docloud.key = 'This is an api_key'
            context.cplex_parameters.emphasis.memory = 1
            context.cplex_parameters.emphasis.mip = 2

        Args:
            file_list: The list of config files to read.
        """
        if file_list is None:
            file_list = []
            targets = ['cplex_config.py',
                       'cplex_config_{0}.py'.format(socket.gethostname()),
                       'docloud_config.py',
                       os.path.expanduser("~") + os.path.sep + ".docplexrc",
                       ]
            for target in targets:
                if isabs(target) and isfile(target) and target not in file_list:
                    file_list.append(target)
                else:
                    for d in sys.path:
                        f = os.path.join(d, target)
                        if os.path.isfile(f):
                            abs_name = os.path.abspath(f)
                            if abs_name not in file_list:
                                file_list.append(f)

            if len(file_list) == 0:
                file_list = None  # let read_settings use its default behavior

        if isinstance(file_list, six.string_types):
            file_list = [file_list]

        if file_list is not None:
            for f in file_list:
                if os.path.isfile(f):
                    if logger:
                        logger.info("Reading settings from %s" % f)
                    if f.endswith(".py"):
                        self.read_from_python_file(f)
                    else:
                        self.read_from_rcfile(f, logger)

    def read_from_python_file(self, filename):
        # Evaluates the content of a Python file containing code to set up a
        # context.
        #
        # Args:
        #    filename (str): The name of the file to evaluate.
        if os.path.isfile(filename):
            with open_universal_newline(filename, 'r') as f:
                # This is so that there is a context in the scope of the exec 
                context = self
                exec(f.read())
        return self

    def read_from_rcfile(self, filename, logger=None):
        # Reads this context from the given resource file.
        #
        # The specified resource file contains `name=value` pairs. For example::
        #
        #   docloud.url = "https://docloud.service.com/job_manager/rest/v1"
        #   docloud.key = "example api_key"
        #
        # Args:
        #    filename (str) : The name of the file to evaluate.
        list_of_properties = []
        with open_filename_universal(filename) as f:
            current_name = None
            current_value = None
            lineno = 0
            for l in f:
                lineno += 1
                l = l.strip()
                if len(l) == 0:
                    continue
                if l[-1] is '\\':
                    s = l[:-1]
                else:
                    s = l
                # process case of continuation
                if current_value is not None:
                    current_value += s
                else:
                    hash_rindex = s.rfind("#")
                    if hash_rindex != -1:
                        s = s[:hash_rindex]
                    if len(s) == 0:
                        continue
                    if (s.find("=") != -1):
                        spl = s.split("=", 1)
                    elif (s.find(":") != -1):
                        spl = s.split(":", 1)
                    else:
                        raise ValueError("Syntax error in line %s" % lineno)
                    if len(spl) == 2:
                        current_name = spl[0].strip()
                        current_value = spl[1].strip()
                        if current_value == '\\':
                            current_value = ""
                    else:
                        raise ValueError("Syntax error in line %s" % lineno)
                # continuation ?
                if l[-1] != '\\':
                    list_of_properties.append((current_name, current_value))
                    current_name = None
                    current_value = None
            # cases where last line of file ends with '\'
            if current_name is not None:
                list_of_properties.append((current_name, current_value))
        self.update_from_list(list_of_properties, logger=logger)


class DOcloudContext(object):
    # for internal use only
    def __init__(self, url=None, api_key=None):
        """ Creates a new DOcplexcloud context.
        """
        # There'se a bunch of properties defined so that we can set
        # any values
        self.url = url
        self.key = api_key
        self._run_deterministic = False
        self._verbose = False
        self._timeout = None
        self._waittime = None
        self._verify = None  # default is None so that we use defaults
        self._log_requests = None
        self._exchange_format = None
        self._debug_dump = None
        self.debug_dump_dir = None
        self._log_poll_interval = None
        self._progress_poll_interval = None
        self.ignored_keys = "ENTER YOUR KEY HERE"
        self.ignored_urls = "ENTER YOUR URL HERE"
        self.verbose_progress_logger = None
        self.delete_job = True
        # if true, download job info after solve() has finished and fire
        # the last details as a progress_info. Mostly for debug.
        self.fire_last_progress = False
        # Mostly for debug: This callback is called when the solve is finished.
        # It should be a method taking **kwargs. It will be called with those kwargs:
        # - jobid: the jobid
        # - client: the docloud client used to connect to docloud
        # - connector: the DOcloudConnector
        self.on_solve_finished_cb = None
        # The proxies
        self.proxies = None


    # This maps "old property names" to the corresponding new qualified name.
    # if the qualified name is None, then it's supposed to be:
    # solver.docloud.old_name
    LEGACY_PROPERTIES = {"url": None,
                         "api_key": "solver.docloud.key",
                         "run_deterministic": None,
                         "verbose": None,
                         "timeout": None,
                         "waittime": None,
                         "verify": None,
                         "log_requests": None,
                         "exchange_format": None,
                         "debug_dump": None,
                         "debug_dump_dir": None
                         }

    @property
    def api_key(self):
        return self.key

    # The waittime property
    def get_waittime(self):
        return self._waittime

    def set_waittime(self, value):
        self._waittime = _convert_to_int(value)

    waittime = property(get_waittime, set_waittime)

    # The timeout property
    def get_timeout(self):
        return self._timeout

    def set_timeout(self, value):
        self._timeout = _convert_to_int(value)

    timeout = property(get_timeout, set_timeout)

    # The log poll interval
    def get_log_poll_interval(self):
        return self._log_poll_interval

    def set_log_poll_interval(self, value):
        self._log_poll_interval = _convert_to_int(value)

    log_poll_interval = property(get_log_poll_interval, set_log_poll_interval)

    # The progress poll interval
    def get_progress_poll_interval(self):
        return self._progress_poll_interval

    def set_progress_poll_interval(self, value):
        self._progress_poll_interval = _convert_to_int(value)

    progress_poll_interval = property(get_progress_poll_interval,
                                      set_progress_poll_interval)

    # The run_deterministic property
    def get_run_deterministic(self):
        return self._run_deterministic

    def set_run_deterministic(self, value):
        self._run_deterministic = _convert_to_bool(value)

    run_deterministic = property(get_run_deterministic, set_run_deterministic)


    # The verbose property
    def get_verbose(self):
        return self._verbose

    def set_verbose(self, value):
        self._verbose = _convert_to_bool(value)

    verbose = property(get_verbose, set_verbose)

    # The debug_dump property
    def get_debug_dump(self):
        return self._debug_dump

    def set_debug_dump(self, value):
        self._debug_dump = _convert_to_bool(value)

    debug_dump = property(get_debug_dump, set_debug_dump)

    # The verify property
    def get_verify(self):
        return self._verify

    def set_verify(self, value):
        self._verify = _convert_to_bool(value)

    verify = property(get_verify, set_verify)

    # The log_requests property
    def get_log_requests(self):
        return self._log_requests

    def set_log_requests(self, value):
        self._log_requests = _convert_to_bool(value)

    log_requests = property(get_log_requests, set_log_requests)

    # The exchange format property
    def get_exchange_format(self):
        return self._exchange_format

    def set_exchange_format(self, exchange_format):
        if exchange_format is not None:
            self._exchange_format = ExchangeFormat.fromstring(exchange_format)
        else:
            self._exchange_format = None

    exchange_format = property(get_exchange_format, set_exchange_format)

    def clone(self):
        # Makes a deep copy of this.
        #
        # Returns:
        #    A deep copy of this.
        return deepcopy(self)

    def copy(self):
        return self.clone()

    def check_credentials(self):
        """Checks if this context has syntactically valid credentials.

        This method uses `warnings.warn()` to issue a warning if the credentials
        are not valid.

        Returns:
            Boolean: `has_credentials` - True if this context contains syntactical credentials.
        Returns:
            string: `message`  - contains a message if applicable.
        """
        has_credentials = True
        message = None
        if not self.url or not self.key:
            has_credentials = False
        elif not is_string(self.url):
            message = "DOcplexcloud: URL is not a string: {0!s}".format(self.url)
            has_credentials = False
        elif not is_string(self.key):
            message = "API key is not a string: {0!s}".format(self.key)
            has_credentials = False
        # process ignored_keys
        if self.key and has_credentials:
            has_credentials = not is_ignored(self.ignored_keys, self.key)
        return has_credentials, message


    def has_credentials(self):
        """Checks if this context has valid credentials.

        Returns:
            True if this context has valid credentials.
        """
        has_credentials, message = self.check_credentials()
        return has_credentials

    def print_information(self):
        print(self.to_string())

    def to_string(self):
        quoted_url = "\"{}\"".format(self.url) if isinstance(self.url, str) else str(self.url)
        quoted_key = "\"{}\"".format(self.key[:4]+"*******"+self.key[len(self.key)-4:]) if isinstance(self.key, str) else str(self.key)
        t_out = str(self.timeout)
        return "context<url={0}, auth={1}, timeout={2}>".format(quoted_url,
                                                                quoted_key,
                                                                t_out)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()



