# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint: disable=too-many-lines
from __future__ import print_function

from copy import deepcopy
from itertools import product
from collections import Counter, OrderedDict
import warnings
import os

import six

from docplex.mp.context import Context, is_key_ignored, is_url_ignored, is_auto_publishing_solve_details, \
    is_auto_publishing_json_solution
from docplex.mp.environment import Environment
from docplex.mp.error_handler import DefaultErrorHandler
from docplex.mp.constants import ComparisonType, SolveAttribute

from docplex.mp.docloud_engine import DOcloudEngine
from docplex.mp.vartype import BinaryVarType, IntegerVarType, ContinuousVarType, SemiContinuousVarType
from docplex.mp.constr import AbstractConstraint, LinearConstraint, RangeConstraint, \
    IndicatorConstraint, QuadraticConstraint

from docplex.mp.basic import ObjectiveSense
from docplex.mp.mfactory import ModelFactory
from docplex.mp.aggregator import ModelAggregator
from docplex.mp.compat23 import StringIO, izip
from docplex.mp.utils import is_indexable, is_iterable, is_iterator, has_len, is_int, is_number, is_string, \
    make_output_path2, generate_constant, is_ordered_sequence, _SymbolGenerator, _IndexScope, _to_list
from docplex.mp.numutils import round_nearest_towards_infinity
from docplex.mp.utils import DOcplexException
from docplex.mp.engine_factory import EngineFactory
from docplex.mp.printer_factory import ModelPrinterFactory
from docplex.mp.format import LP_format, SAV_format
from docplex.util.environment import get_environment
from docplex.mp.constants import RelaxationMode, SOSType

from docplex.mp.tck import get_typechecker

try:
    from docplex.worker.solvehook import get_solve_hook
except ImportError:  # pragma: no cover
    get_solve_hook = None  # pragma: no cover


def stringify_tuple(tuple_key, sep='_'):
    """ Generates variable names from a set of key objects.

    This method is used as a default name argument for all containers of dimension two or higher.
    From a tupl eof key objects, it generates the concatenation of string representation sof keys,
    separated by 'sep'.

    Args:
        tuple_key: a tuple of key objects
        sep: a string, default is '_'

    Returns:
        a string, the name of the variable index by the tuple key

    Example:
        If the tuple is ("foo", 1), this function returns the string "foo_1".

    Note:
        If you wish to change the separator string, use a lambda function that
        redefines the sep argument; for example, if you wish to use '___' (three underscores)
        as separator, use the following:

        m.binary_var_matrix(name=lambda tk: stringify_tuple(tk, "___")
    """
    return sep.join(str(k) for k in tuple_key)


class _ToleranceScheme(object):
    """
        INTERNAL
        """

    def __init__(self, absolute=1e-6, relative=1e-4):
        assert absolute >= 0
        assert relative >= 0
        self._absolute_tolerance = absolute
        self._relative_tolerance = relative

    def compute_tolerance(self, obj):
        abs_obj = abs(obj)
        return max(self._absolute_tolerance, self._relative_tolerance * abs_obj)

    def to_string(self):
        return "tolerance(abs=%f,rel=%f".format(self._absolute_tolerance, self._relative_tolerance)

    def __str__(self):
        return self.to_string()  # pragma: no cover


class _VariableContainer(object):
    def __init__(self, vartype, key_seq, lb, ub, name):
        self._vartype = vartype
        self._keys = key_seq
        self._lb = lb
        self._ub = ub
        self._namer = name

    def copy(self, target_model):
        copied_ctn = _VariableContainer(self.vartype, self._keys, self.lb, self.ub, self._namer)
        return copied_ctn

    @property
    def keys(self):
        return self._keys

    @property
    def vartype(self):
        return self._vartype

    @property
    def nb_dimensions(self):
        return len(self._keys)

    @property
    def namer(self):
        return self._namer

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def name(self):
        """
        Try to extract a name string from a naming function.
        :return: A string or None.
        """
        if isinstance(self._namer, str):
            raw_name = self._namer
            # drop opl-style formats
            raw_name = raw_name.replace("({%s})", "")
            # purge fields
            pos_pct = raw_name.find('%')
            if pos_pct >= 0:
                return raw_name[:pos_pct - 1]
            elif raw_name.find('{') > 0:
                pos = raw_name.find('{')
                return raw_name[:pos - 1]
            else:
                return raw_name

    def size(self, dim_index):
        return len(self._keys[dim_index]) if dim_index < self.nb_dimensions else 0

    def shape(self):
        return tuple(len(k) for k in self._keys)

    @property
    def dimension_string(self):
        dim_string = "".join(["[%d]" % self.size(d) for d in range(self.nb_dimensions)])
        return dim_string

    def to_string(self):
        # dvar xxx
        dim_string = self.dimension_string
        ctname = self._namer or 'x'
        return "dvar {0} {1} {2}".format(self.vartype.short_name, ctname, dim_string)

    def __str__(self):
        return self.to_string()


class ModelStatistics(object):
    """ModelStatistics()

    This class gathers statistics from the model.

    Instances of this class are returned by the method :func:`docplex.mp.model.Model.get_statistics`.

    The class contains counters on the various types of variables and constraints
    in the model.

    """

    def __init__(self):
        self._number_of_binary_variables = 0
        self._number_of_integer_variables = 0
        self._number_of_continuous_variables = 0
        self._number_of_semicontinuous_variables = 0
        self._number_of_le_constraints = 0
        self._number_of_ge_constraints = 0
        self._number_of_eq_constraints = 0
        self._number_of_range_constraints = 0
        self._number_of_indicator_constraints = 0
        self._number_of_quadratic_constraints = 0

    def as_tuple(self):
        return (self._number_of_binary_variables,
                self._number_of_integer_variables,
                self._number_of_continuous_variables,
                self._number_of_semicontinuous_variables,
                self._number_of_le_constraints,
                self._number_of_ge_constraints,
                self._number_of_eq_constraints,
                self._number_of_range_constraints,
                self._number_of_indicator_constraints,
                self._number_of_quadratic_constraints)

    def equal_stats(self, other):
        return isinstance(other, ModelStatistics) and (self.as_tuple() == other.as_tuple())

    def __eq__(self, other):
        return self.equal_stats(other)

    def __sub__(self, other):
        if not isinstance(other, ModelStatistics):
            raise TypeError
        diffstats = ModelStatistics()
        for attr in ["_number_of_le_constraints", "_number_of_ge_constraints", "_number_of_eq_constraints"]:
            setattr(diffstats, attr, getattr(self, attr) - getattr(other, attr))
        return diffstats

    @staticmethod
    def _make_new_stats(mdl):
        # INTERNAL
        stats = ModelStatistics()
        vartype_count = Counter(type(dv.vartype) for dv in mdl.iter_variables())
        stats._number_of_binary_variables = vartype_count[BinaryVarType]
        stats._number_of_integer_variables = vartype_count[IntegerVarType]
        stats._number_of_continuous_variables = vartype_count[ContinuousVarType]
        stats._number_of_semicontinuous_variables = vartype_count[SemiContinuousVarType]

        linct_count = Counter(ct.type for ct in mdl.iter_binary_constraints())
        stats._number_of_le_constraints = linct_count[ComparisonType.LE]
        stats._number_of_eq_constraints = linct_count[ComparisonType.EQ]
        stats._number_of_ge_constraints = linct_count[ComparisonType.GE]
        stats._number_of_range_constraints = mdl.number_of_range_constraints
        stats._number_of_indicator_constraints = mdl.number_of_indicator_constraints
        stats._number_of_quadratic_constraints = mdl.number_of_quadratic_constraints
        return stats

    @property
    def number_of_variables(self):
        """ This property returns the total number of variables in the model.

        """
        return self._number_of_binary_variables \
               + self._number_of_integer_variables \
               + self._number_of_continuous_variables \
               + self._number_of_semicontinuous_variables

    @property
    def number_of_binary_variables(self):
        """ This property returns the number of binary decision variables in the model.

        """
        return self._number_of_binary_variables

    @property
    def number_of_integer_variables(self):
        """ This property returns the number of integer decision variables in the model.

        """
        return self._number_of_integer_variables

    @property
    def number_of_continuous_variables(self):
        """ This property returns the number of continuous decision variables in the model.

        """
        return self._number_of_continuous_variables

    @property
    def number_of_semicontinuous_variables(self):
        """ This property returns the number of semicontinuous decision variables in the model.

        """
        return self._number_of_semicontinuous_variables



    @property
    def number_of_linear_constraints(self):
        """ This property returns the total number of linear constraints in the model.

        This number comprises all relational constraints: <=, ==, and >=
        and also range constraints.

        """
        return self._number_of_eq_constraints + \
               self._number_of_le_constraints + \
               self._number_of_ge_constraints + \
               self._number_of_range_constraints

    @property
    def number_of_le_constraints(self):
        """ This property returns the number of <= constraints

        """
        return self._number_of_le_constraints

    @property
    def number_of_eq_constraints(self):
        """ This property returns the number of == constraints

        """
        return self._number_of_eq_constraints

    @property
    def number_of_ge_constraints(self):
        """ This property returns the number of >= constraints

        """
        return self._number_of_ge_constraints

    @property
    def number_of_range_constraints(self):
        """ This property returns the number of range constraints.

        Range constraints are of the form L <= expression <= U.

        See Also:
            :class:`docplex.mp.linear.RangeConstraint`

        """
        return self._number_of_range_constraints

    @property
    def number_of_indicator_constraints(self):
        """ This property returns the number of indicator constraints.

        See Also:
            :class:`docplex.mp.linear.IndicatorConstraint`

        """
        return self._number_of_indicator_constraints

    @property
    def number_of_quadratic_constraints(self):
        """ This property returns the number of quadratic constraints.

        See Also:
            :class:`docplex.mp.linear.QuadraticConstraint`

        """
        return self._number_of_quadratic_constraints

    @property
    def number_of_constraints(self):
        return self.number_of_linear_constraints +\
               self.number_of_quadratic_constraints +\
               self.number_of_indicator_constraints


    def print_information(self):
        """ Prints model statistics in readable format.

        """
        print(' - number of variables: {0}'.format(self.number_of_variables))
        var_fmt = '   - binary={0}, integer={1}, continuous={2}'
        if 0 != self._number_of_semicontinuous_variables:
            var_fmt += ', semi-continuous={3}'
        print(var_fmt.format(self.number_of_binary_variables,
                             self.number_of_integer_variables,
                             self.number_of_continuous_variables,
                             self._number_of_semicontinuous_variables
                             ))

        print(' - number of constraints: {0}'.format(self.number_of_constraints))
        ct_fmt = '   - linear={0}'
        if 0 != self._number_of_indicator_constraints:
            ct_fmt += ', indicator={1}'
        if 0 != self._number_of_quadratic_constraints:
            ct_fmt +=', quadratic={2}'
        print(ct_fmt.format(self.number_of_linear_constraints,
                            self.number_of_indicator_constraints,
                            self.number_of_quadratic_constraints))

    def to_string(self):
        oss = StringIO()
        oss.write(" - number of variables: %d\n" % self.number_of_variables)
        var_fmt = '   - binary={0}, integer={1}, continuous={2}'
        if 0 != self._number_of_semicontinuous_variables:
            var_fmt += ', semi-continuous={3}'
        oss.write(var_fmt.format(self.number_of_binary_variables,
                                self.number_of_integer_variables,
                                self.number_of_continuous_variables,
                                self._number_of_semicontinuous_variables
                                ))
        nb_constraints = self.number_of_constraints
        oss.write(' - number of constraints: {0}'.format(nb_constraints))
        if nb_constraints:
            ct_fmt = '   - linear={0}'
            if 0 != self._number_of_indicator_constraints:
                ct_fmt += ', indicator={1}'
            if 0 != self._number_of_quadratic_constraints:
                ct_fmt +=', quadratic={2}'
            oss.write(ct_fmt.format(self.number_of_linear_constraints,
                                    self.number_of_indicator_constraints,
                                    self.number_of_quadratic_constraints))
        return oss.getvalue()

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "docplex.mp.Model.ModelStatistics()"


# noinspection PyProtectedMember
class Model(object):
    """ This is the main class to embed modeling objects.

    The :class:`Model` class acts as a factory to create optimization objects,
    decision variables, and constraints.
    It provides various accessors and iterators to the modeling objects.
    It also manages solving operations and solution management.

    The Model class is the context manager and can be used with the Python `with` statement:

    .. code-block:: python

       with Model() as mdl:
         # start modeling...

    When the `with` block is finished, the :func:`end` method is called automatically, and all resources
    allocated by the model are destroyed.

    When a model is created without a specified ``context``, a default
    ``Context`` is created and initialized as described in :func:`docplex.mp.context.Context.read_settings`.

    Example::

        # Creates a model named 'my model' with default context
        model = Model('my model')

    In the following example, credentials for the solve service are looked up as
    described in :func:`docplex.mp.context.Context.read_settings`::

        # Creates a model with default ``context``, and override the ``agent``
        # to solve on Decision Optimization on Cloud.
        model = Model(agent='docloud')

    In this example, we create a model to solve with just 2 threads::

        context = Context.make_default_context()
        context.cplex_parameters.threads = 2
        model = Model(context=context)

    Alternatively, this can be coded as::

        model = Model()
        model.context.cplex_parameters.threads = 2

    Args:
        name (optional): The name of the model.
        context (optional): The solve context to be used. If no ``context`` is
            passed, a default context is created.
        agent (optional): The ``context.solver.agent`` is initialized with
            this string.
        log_output (optional): If ``True``, solver logs are output to
            stdout. If this is a stream, solver logs are output to that
            stream object.
        checker (optional): If ``None``, then checking is disabled everywhere. Turning off checking
            may improve performance but should be done only with extreme caution.
    """

    _name_generator = _SymbolGenerator(pattern="docplex_model", offset=1)

    @property
    def binary_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.BinaryVarType`.

        This type instance is used to build all binary decision variable collections of the model.
        """
        return self._binary_vartype

    @property
    def integer_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.IntegerVarType`.

        This type instance is used to build all integer variable collections of the model.
        """
        return self._integer_vartype

    @property
    def continuous_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.ContinuousVarType`.

        This type instance is used to build all continuous variable collections of the model.
        """
        return self._continuous_vartype

    @property
    def semicontinuous_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.SemiContinuousVarType`.

        This type instance is used to build all semi-continuous variable collections of the model.
        """
        return self._semicontinuous_vartype

    def _make_environment(self):
        env = Environment.make_new_configured_env()
        # rtc-28869
        env.numpy_hook = Model.init_numpy
        return env

    def _lazy_get_environment(self):
        if self._environment is None:
            self._environment = self._make_environment()  # pragma: no cover
        return self._environment

    _saved_numpy_options = None

    _unknown_status = None

    @staticmethod
    def init_numpy():
        """ Static method to customize `numpy` for DOcplex.

        This method makes `numpy` aware of DOcplex.
        All `numpy` arrays with DOcplex objects will be printed by their string representations
        as returned by `str(`) instead of `repr()` as with standard numpy behavior.

        All customizations can be removed by calling the :func:`restore_numpy` method.

        Note:
            This method does nothing if `numpy` is not present.

        See Also:
            :func:`restore_numpy`
        """
        try:
            # noinspection PyUnresolvedReferences
            import numpy as np

            Model._saved_numpy_options = np.get_printoptions()
            np.set_printoptions(formatter={'numpystr': lambda f: str(f) if ModelFactory._is_operand(f) else repr(f)})
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

    @staticmethod
    def restore_numpy(self):
        """ Static method to restore `numpy` to its default state.

        This method is a companion method to :func:`init_numpy`. It restores `numpy` to its original state,
        undoing all customizations that were done for DOcplex.

        Note:
            This method does nothing if numpy is not present.

        See Also:
            :func:`init_numpy`
        """
        try:
            # noinspection PyUnresolvedReferences
            import numpy as np

            if Model._saved_numpy_options is not None:
                np.set_printoptions(Model._saved_numpy_options)
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

    @property
    def environment(self):
        # for a closed model with no CPLEX, numpy, etc return ClosedEnvironment
        # return get_no_graphics_env()
        # from docplex.environment import ClosedEnvironment
        # return ClosedEnvironment
        return self._lazy_get_environment()

    # ---- type checking

    def typecheck_var(self, obj):
        self._checker.typecheck_var(obj)

    def typecheck_num(self, arg, caller=None):
        self._checker.typecheck_num(arg, caller)

    def unsupported_relational_operator_error(self, left_arg, op, right_arg):
        # INTERNAL
        self.fatal("Unsupported relational operator:  {0!s} {1!s} {2!s}, only <=, ==, >= are allowed", left_arg, op,
                   right_arg)

    def unsupported_neq_error(self, left_arg, right_arg):
        self.fatal("The \"not_equal\" logical constraint is not supported: {0!s} != {1!s}", left_arg, right_arg)

    def cannot_be_used_as_denominator_error(self, denominator, numerator):
        self.fatal("{1!s} / {0!s} : operation not supported, only numbers can be denominators", denominator, numerator)

    def typecheck_as_denominator(self, denominator, numerator):
        if not is_number(denominator):
            self.cannot_be_used_as_denominator_error(denominator, numerator)
        else:
            float_e = float(denominator)
            if 0 == float_e:
                self.fatal("Zero divide on {0!s}", numerator)
            else:
                # ok
                pass

    def typecheck_as_power(self, e, power):
        # INTERNAL: checks <power> is 0,1,2
        if power < 0 or power > 2:
            self.fatal("Cannot raise {0!s} to the power {1}. A variable's exponent must be 0, 1 or 2.", e, power)

    def _check_cplex_version(self, verbose=True):
        # checks that env's cplex version is greater or equal to parameters'
        # otherwise the get() operation may fail.
        self_env = self.environment
        ok = True
        if self_env.has_cplex:
            env_cplex_version = self_env.cplex_version
            assert env_cplex_version
            if env_cplex_version < self.parameters.cplex_version:
                if verbose:
                    self.error(
                        "Found CPLEX DLL with version: {0}, older than parameter version: {1}. Parameter setting is disabled.",
                        env_cplex_version, self.parameters.cplex_version)
                ok = False
        return ok

    def round_nearest(self, x):
        # INTERNAL
        return round_nearest_towards_infinity(x, self.infinity)

    def _parse_kwargs(self, kwargs):
        # parse some arguments from kwargs
        for arg_name, arg_val in six.iteritems(kwargs):
            if arg_name == "float_precision":
                self.float_precision = arg_val
            elif arg_name in frozenset({"keep_ordering"}):
                self.keep_ordering = bool(arg_val)
            elif arg_name in frozenset({"info_level", "output_level"}):
                self.output_level = arg_val
            elif arg_name in {"agent", "solver_agent"}:
                self.__solver_agent = arg_val
            elif arg_name == "log_output":
                self.context.solver.log_output = arg_val
            elif arg_name == "warn_trivial":
                self._trivial_cts_message_level = arg_val
            elif arg_name == "max_repr_len":
                self._max_repr_len = int(arg_val)
            elif arg_name == "clone_transients":
                self._clone_transient_exprs = bool(arg_val)
            elif arg_name == 'checker':
                self._checker_key = arg_val.lower() if is_string(arg_val) else 'default'
            elif arg_name == 'name_all_cts':
                self._name_all_cts = arg_val
            else:
                self.info("argument: {0:s}:{1!s} - is not recognized (ignored)", arg_name, arg_val)

    def _get_kwargs(self):
        kwargs_map = {'float_precision': self.float_precision,
                      'keep_ordering': self.keep_ordering,
                      'output_level': self.output_level,
                      'solver_agent': self.solver_agent,
                      'log_output': self.log_output,
                      'warn_trivial': self._trivial_cts_message_level,
                      'max_repr_len': self._max_repr_len,
                      'clone_transients': self._clone_transient_exprs,
                      'checker': self._checker_key}
        return kwargs_map

    _default_varname_pattern = "_x"
    _default_ctname_pattern = "_c"
    _default_indicator_pattern = "_ic"
    _default_quadct_pattern = "_qc"

    warn_trivial_feasible = 0
    warn_trivial_infeasible = 1
    warn_trivial_none = 2

    def __init__(self, name=None, context=None, **kwargs):
        """Init a new Model.

        Args:
            name (optional): The name of the model
            context (optional): The solve context to be used. If no ``context`` is
                passed, a default context is created.
            agent (optional): The ``context.solver.agent`` is initialized with
                this string.
            log_output (optional): if ``True``, solver logs are output to
                stdout. If this is a stream, solver logs are output to that
                stream object.
        """
        if name is None:
            name = Model._name_generator.new_symbol()
        self._name = name

        self.__solver_agent = None  # default
        self.__error_handler = DefaultErrorHandler("info")

        # type instances
        self._binary_vartype = BinaryVarType()
        self._integer_vartype = IntegerVarType()
        self._continuous_vartype = ContinuousVarType()
        self._semicontinuous_vartype = SemiContinuousVarType()

        #
        self.__allvarctns = []
        self.__allvars = []
        self.__vars_by_name = {}
        self.__allcts = []
        self.__cts_by_name = None

        self._allsos = []

        # -- kpis --
        self._allkpis = []

        self._progress_listeners = []
        self._solve_hooks = []  # debugSolveHook()
        self._mipstarts = []


        self._keep_ordering = False
        # -- float formats
        self._float_precision = 3
        self._continuous_var_format = "%.3f"

        self._environment = self._make_environment()
        self_env = self._environment
        # init context
        if context is None:
            # This is roughly equivalent to Context.make_default_context()
            # but save us a modification to Context.make_default_context()
            # to accept _env
            self.context = Context(_env=self_env)
            self.context.read_settings()  # read default settings
        else:
            self.context = context

        self._engine_factory = EngineFactory(env=self_env)

        if 'docloud_context' in kwargs:
            warnings.warn(
                "Model construction with DOcloudContext is deprecated, use initializer with docplex.mp.context.Context instead.",
                DeprecationWarning, stacklevel=2)

        # offset between generate dnames and zero based indices.
        self._name_offset = 1

        # maximum length for expression in repr strings...
        self._max_repr_len = -1

        # control whether to warn about trivial constraints
        self._trivial_cts_message_level = self.warn_trivial_infeasible

        # internal
        self._clone_transient_exprs = True  # use False to get fast clone...with the risk of side effects...

        self._checker_key = 'default'

        # if True, forces docplex ot generate internal names for anonymous constraints.
        self._name_all_cts = True

        # update from kwargs, before the actual inits.
        self._parse_kwargs(kwargs)

        self._checker = get_typechecker(arg=self._checker_key, logger=self.error_handler)

        # -- scopes
        name_offset = self._name_offset
        self._var_scope = _IndexScope(self.iter_variables, self._default_varname_pattern, offset=name_offset)
        self._linct_scope = _IndexScope(self.iter_linear_constraints, self._default_ctname_pattern, offset=name_offset)
        self._indicator_scope = _IndexScope(self.iter_indicator_constraints, self._default_indicator_pattern,
                                            offset=name_offset)
        self._quadct_scope = _IndexScope(self.iter_quadratic_constraints, self._default_quadct_pattern,
                                         offset=name_offset)
        self._scopes = [self._var_scope, self._linct_scope, self._indicator_scope, self._quadct_scope]
        self._ctscopes = [self._linct_scope, self._indicator_scope, self._quadct_scope]
        # a counter to generate unique numbers, incremented at each new request
        self._unique_counter = -1

        # init engine
        engine = self._make_new_engine(self.solver_agent, self.context)
        self.__engine = engine
        self._lfactory = ModelFactory(self, engine)
        from docplex.mp.quadfact import QuadFactory
        self._qfactory = QuadFactory(self)
        self._aggregator = ModelAggregator(self._lfactory, self._qfactory)
        self._solve_count = 0
        self.__solution = None
        self._solve_details = None

        # stats
        self._linexpr_instance_counter = 0
        self._linexpr_clone_counter = 0
        self._quadexpr_instance_counter = 0
        self._quadexpr_clone_counter = 0

        # all the following must be placed after an engine has been set.
        self._default_objective_expr = self._lfactory.constant_expr(cst=0)
        self.__objective_expr = self._default_objective_expr
        self.__objective_sense = self._lfactory.default_objective_sense()

        # parameters
        self_cplex_parameters_version = self.context.cplex_parameters.cplex_version

        if self_env.has_cplex:
            installed_cplex_version = self_env.cplex_version
            # installed version is different from parameters: reset all defaults
            if installed_cplex_version != self_cplex_parameters_version:
                # cplex is more recent than parameters. must update defaults.
                self.trace(
                    "reset parameter defaults, from parameter version: {0} to installed version: {1}"  # pragma: no cover
                        .format(self_cplex_parameters_version, installed_cplex_version))  # pragma: no cover
                resets = self._sync_parameter_defaults_from_engine()  # pragma: no cover
                if resets:  # pragma: no cover
                    for p, old, new in resets:
                        if p.short_name != 'randomseed':  # usual practice to change randomseed at each version
                            self.info('parameter changed, name: {0}, old default: {1}, new default: {2}',
                                      p.short_name, old, new)



    def _add_solve_hook(self, hook):
        if hook is not None:
            self._solve_hooks.append(hook)

    @property
    def infinity(self):
        """ This property returns the numerical value used as the upper bound for continuous decision variables.

        Note:
            CPLEX usually sets this limit to 1e+20.
        """
        return self.__engine.get_infinity()

    @property
    def progress_listeners(self):
        """ This property returns the list of progress listeners.
        """
        return self._progress_listeners

    def get_name(self):
        return self._name

    def _set_name(self, name):
        self._check_name(name)
        self._name = name

    def _check_name(self, new_name):
        self._checker.typecheck_string(arg=new_name, accept_empty=False, accept_none=False)
        if new_name.find(" ") >= 0:
            self.warning("Model name contains whitespaces: |{0:s}|", new_name)

    name = property(get_name, _set_name)

    # adjust the maximum length of repr.. strings
    def _get_max_repr_len(self):
        return self._max_repr_len

    def _set_max_repr_len(self, max_repr):
        self._max_repr_len = max_repr

    max_repr_len = property(_get_max_repr_len, _set_max_repr_len)

    # you can change the way variables are named by default, just define a string here.
    def get_automatic_var_name_pattern(self):
        """
        This property is used to configure how DOcplex
        generates an automatic names for decision variables with no name.

        The default naming scheme is _x<i> where <i> is the creation rank, starting at 1,
        so variables without user names will get automatic names _x1, _x2, ...

        You can change this by passing a nonempty string, for example "var",
        in which case variables without user names will be named var1, var2, ...
        """
        return self._var_scope.pattern

    def set_automatic_var_name_pattern(self, varname):
        self._checker.typecheck_string(varname, accept_empty=False)
        self._var_scope.pattern = varname

    automatic_var_name_pattern = property(get_automatic_var_name_pattern, set_automatic_var_name_pattern)

    def get_automatic_ct_name_pattern(self):
        return self._linct_scope.pattern

    def set_automatic_ct_name_pattern(self, ctname):
        self._checker.typecheck_string(ctname, accept_empty=False)
        self._linct_scope.pattern = ctname

    automatic_ct_name_pattern = property(get_automatic_ct_name_pattern, set_automatic_ct_name_pattern)

    def get_automatic_indicator_name_pattern(self):
        return self._indicator_scope.pattern

    def set_automatic_indicator_name_pattern(self, indname):
        self._checker.typecheck_string(indname, accept_empty=False)
        self._indicator_scope.pattern = indname

    automatic_indicator_name_pattern = property(get_automatic_indicator_name_pattern,
                                                set_automatic_indicator_name_pattern)

    def get_automatic_quadratic_ctname_pattern(self):
        return self._quadct_scope.pattern

    def set_automatic_quadratic_ctname_pattern(self, qctname):
        self._checker.typecheck_string(qctname, accept_empty=False)
        self._quadct_scope.pattern = qctname

    automatic_quadratic_ctname_pattern = property(get_automatic_quadratic_ctname_pattern,
                                                  set_automatic_quadratic_ctname_pattern)

    def _create_automatic_varname(self):
        return self._var_scope.new_symbol()

    def _create_automatic_ctname(self, ct):
        # need to find the proper scope: linct or indicators
        return self._get_ct_scope(ct).new_symbol()

    def _get_keep_ordering(self):
        return self._keep_ordering

    def _set_keep_ordering(self, is_sorted):
        self._keep_ordering = bool(is_sorted)

    keep_ordering = property(_get_keep_ordering, _set_keep_ordering)

    def get_float_precision(self):
        """ This property is used to get or set the float precision of the model.

        The float precision is an integer number of digits, used
        in printing the solution and objective.
        This number of digits is used for variables and expressions which are not discrete.
        Discrete variables and objectives are printed with no decimal digits.

        """
        return self._float_precision

    def set_float_precision(self, nb_digits):
        used_digits = nb_digits
        if nb_digits < 0:
            self.warning("Negative float precision given: {0}, using 0 instead", nb_digits)
            used_digits = 0
        else:
            max_digits = self.environment.max_nb_digits
            bitness = self.environment.bitness
            if nb_digits > max_digits:
                self.warning("Given precision of {0:d} goes beyond {1:d}-bit capability, using maximum: {2:d}".
                             format(nb_digits, bitness, max_digits))
                used_digits = max_digits
        self._float_precision = used_digits
        # recompute float format
        self._continuous_var_format = "%%.%df" % nb_digits

    float_precision = property(get_float_precision, set_float_precision)

    # generate a new , unique counter, in the scope of this model
    def _new_unique_counter(self):
        # INTERNAL
        self._unique_counter += 1
        return self._unique_counter

    @property
    def time_limit_parameter(self):
        # INTERNAL
        return self.parameters.timelimit

    def get_time_limit(self):
        """
        Returns:
            The time limit for the model.

        """
        return self.time_limit_parameter.get()

    def set_time_limit(self, time_limit):
        """ Set a time limit for solve operations.

        Args:
            time_limit: The new time limit; must be a positive number.

        """
        self._checker.typecheck_num(time_limit)
        if time_limit < 0:
            self.fatal("Negative time limit: {0}", time_limit)
        elif time_limit < 1:
            self.warning("Time limit too small: {0} - using 1 instead", time_limit)
            time_limit = 1
        else:
            pass

        self.time_limit_parameter.set(time_limit)

    @property
    def solver_agent(self):
        return self.__solver_agent

    @property
    def error_handler(self):
        return self.__error_handler

    @property
    def solution(self):
        """ This property returns the current solution of the model or None if the model has not yet been solved
        or if the last solve has failed.
        """
        return self.__solution

    def _get_solution(self):
        # INTERNAL
        return self.__solution

    @property
    def mip_starts(self):
        """ This property returns the list of MIP start solutions (a list of instances of :class:`docplex.mp.solution.SolveSolution`)
        attached to the model if MIP starts have been defined, possibly an empty list.
        """
        return self._mipstarts[:]

    def clear_mip_starts(self):
        """  Clears all MIP start solutions associated with the model.
        """
        self._mipstarts = []

    def remove_mip_start(self, mipstart):
        """ Removes one MIP start solution from the list of MIP starts associated with the model.

        Args:
            mipstart: A MIP start solution, an instance of :class:`docplex.mp.solution.SolveSolution`.
        """
        self._mipstarts.remove(mipstart)

    def fatal(self, msg, *args):
        self.__error_handler.fatal(msg, args)

    def error(self, msg, *args):
        self.__error_handler.error(msg, args)

    def warning(self, msg, *args):
        self.__error_handler.warning(msg, args)

    def info(self, msg, *args):
        self.__error_handler.info(msg, args)

    def trace(self, msg, *args):
        self.error_handler.trace(msg, args)

    def get_output_level(self):
        return self.__error_handler.get_output_level()

    def set_output_level(self, new_output_level):
        self.__error_handler.set_output_level(new_output_level)

    def set_quiet(self):
        self.error_handler.set_quiet()

    output_level = property(get_output_level, set_output_level)

    def set_log_output(self, out=None):
        self.context.solver.log_output = out
        outs = self.context.solver.log_output_as_stream
        self.__engine.notify_trace_output(outs)

    def get_log_output(self):
        return self.context.solver.log_output_as_stream

    log_output = property(get_log_output, set_log_output)

    def set_log_output_as_stream(self, outs):
        self.__engine.notify_trace_output(outs)

    def is_logged(self):
        return self.context.solver.log_output_as_stream is not None

    def clear(self):
        """ Clears the model of all modeling objects.
        """
        self._clear_internal()

    def _clear_internal(self):
        self.__allvars = []
        self.__allvarctns = []
        self.__vars_by_name = {}
        self.__allcts = []
        self.__cts_by_name = None
        self._allkpis = []
        self.clear_kpis()
        self._solve_count = 0
        self._last_solve_status = self._unknown_status
        self.__solution = None
        self._mipstarts = []
        self._clear_scopes()
        self._allsos = []

    def _clear_scopes(self):
        for a_scope in self._scopes:
            a_scope.reset()

    def _make_new_engine(self, solver_agent, context):
        new_engine = self._engine_factory.new_engine(solver_agent, self.environment, model=self, context=context)
        new_engine.notify_trace_output(self.context.solver.log_output_as_stream)
        return new_engine

    def _set_engine(self, e2):
        self.__engine = e2
        self._lfactory.update_engine(e2)

    def _clear_engine(self, restart):
        # INTERNAL
        old_engine = self.__engine
        if old_engine:
            # dispose of old engine.
            old_engine.end()
            # from Ryan
            del old_engine
            self.__engine = None
        if restart:
            new_engine = self._make_new_engine(self.solver_agent, self.context)
            self._set_engine(new_engine)

    def get_engine(self):
        # INTERNAL for testing
        return self.__engine

    def ensure_setup(self):
        # defined for compatibility with AbstractModel
        pass

    def print_information(self):
        """ Prints general informational statistics on the model.

        Prints the number of variables and their breakdown by type.
        Prints the number of constraints and their breakdown by type.

        """
        print("Model: %s" % self.name)
        self.get_statistics().print_information()

        self_params = self.parameters
        if self_params.has_nondefaults():
            print(" - parameters:")
            self_params.print_information(indent_level=5)  # 3 for " - " + 2 = 5
        else:
            print(" - parameters: defaults")

    def _is_empty(self):
        # INTERNAL
        return 0 == self.number_of_constraints and 0 == self.number_of_variables

    @property
    def number_of_linear_expr_instances(self):
        return self._linexpr_instance_counter

    # @profile
    def __notify_new_model_object(self, descr,
                                  mobj, mindex, mobj_name,
                                  name_dir, idx_scope,
                                  is_name_safe=False):
        """
        Notifies the return af an object being create on the engine.
        :param descr: A string describing the type of the object being created (e.g. Constraint, Variable).
        :param mobj:  The newly created modeling object.
        :param mindex: The index as returned by the engine.
        :param name_dir: The directory of objects by name (e.g. name -> constraitn directory).
        :param idx_scope:  The index scope
        """
        mobj.set_index(mindex)

        if name_dir is not None:
            mobj_name = mobj_name or mobj.get_name()
            if mobj_name:
                # in some cases, names are checked before register
                if not is_name_safe:
                    if mobj_name in name_dir:
                        old_name_value = name_dir[mobj_name]
                        # Duplicate constraint name: foo
                        self.fatal("Duplicate {0} name: {1} already used for {2!s}", descr, mobj_name, old_name_value)

                name_dir[mobj_name] = mobj

        # store in idx dir if any
        if idx_scope:
            idx_scope.notify_obj_index(mobj, mindex)

    def _register_one_var(self, var, var_index, var_name):
        self.__notify_new_var(var, var_index, var_name)
        self._var_scope.notify_new_index(var_index)
        #
        self.__allvars.append(var)

    # @profile
    def _register_block_vars(self, allvars, indices, allnames):
        varname_dict = self.__vars_by_name
        var_header = "variable"
        notifier = self.__notify_new_model_object
        safe_all_names = allnames or generate_constant(None, len(allvars))
        for var, var_index, var_name in izip(allvars, indices, safe_all_names):
            notifier(var_header, var, var_index, var_name, varname_dict, idx_scope=None)
        self.__allvars.extend(allvars)
        # update variable scope once
        self._var_scope.notify_obj_indices(objs=allvars, indices=indices)

    def __notify_new_var(self, var, var_index, var_name):
        self.__notify_new_model_object("variable", var, var_index, var_name, self.__vars_by_name, self._var_scope)

    def _get_ct_scope(self, ct):
        if ct.is_linear():
            return self._linct_scope
        elif ct.is_quadratic():
            return self._quadct_scope
        else:
            return self._indicator_scope

    def _register_one_constraint(self, ct, ct_index, is_ctname_safe=False):
        """
        INTERNAL
        :param ct: The new constraint to register.
        :param ct_index: The index as returned by the engine.
        :param is_ctname_safe: True if ct name has been checked for duplicates already.
        :return:
        """
        if ct_index < 0:
            self.warning('Negative index for constraint: {0!s}, index={1}', ct, ct_index)
        scope = self._get_ct_scope(ct)

        self.__notify_new_model_object(
            "constraint", ct, ct_index, None,
            self.__cts_by_name, scope,
            is_name_safe=is_ctname_safe)

        self.__allcts.append(ct)

    def _ensure_cts_by_name(self):
        if self.__cts_by_name is None:
            self.__cts_by_name = { ct.get_name() : ct for ct in self.iter_constraints() if ct.has_user_name()}
        return self.__cts_by_name

    def _register_block_cts(self, cts, indices, safe_names=False):
        # INTERNAL: assert len(cts) == len(indices)
        ct_name_map = self.__cts_by_name
        notifier = self.__notify_new_model_object
        ct_descr = "constraint"
        # --
        for ct, ct_index in izip(cts, indices):
            ct_scope = self._get_ct_scope(ct)
            notifier(ct_descr, mobj=ct, mindex=ct_index, mobj_name=None,
                     name_dir=ct_name_map, idx_scope=ct_scope, is_name_safe=True)
        # extend container faster than append()
        self.__allcts.extend(cts)

    # iterators
    def iter_var_containers(self):
        # INTERNAL
        return iter(self.__allvarctns)

    def _add_var_container(self, ctn):
        # INTERNAL
        self.__allvarctns.append(ctn)

    def _is_binary_var(self, dvar):
        return dvar.vartype.get_cplex_typecode() == 'B'

    def _is_integer_var(self, dvar):
        return dvar.vartype.get_cplex_typecode() == 'I'

    def _is_continuous_var(self, dvar):
        return dvar.vartype.get_cplex_typecode() == 'C'

    def _is_semicontinuous_var(self, dvar):
        return dvar.vartype.get_cplex_typecode() == 'S'

    def _count_variables_filtered(self, predicate):
        return sum(1 for _ in filter(predicate, self.__allvars))

    def _iter_variables_filtered(self, predicate):
        for v in self.iter_variables():
            if predicate(v):
                yield v

    @property
    def number_of_variables(self):
        """ This property returns the total number of decision variables, all types combined.

        """
        return len(self.__allvars)

    @property
    def number_of_binary_variables(self):
        """ This property returns the total number of binary decision variables added to the model.
        """
        return self._count_variables_filtered(lambda v: self._is_binary_var(v))

    @property
    def number_of_integer_variables(self):
        """ This property returns the total number of integer decision variables added to the model.
        """
        return self._count_variables_filtered(lambda v: self._is_integer_var(v))

    @property
    def number_of_continuous_variables(self):
        """ This property returns the total number of continuous decision variables added to the model.
        """
        return self._count_variables_filtered(lambda v: self._is_continuous_var(v))

    @property
    def number_of_semicontinuous_variables(self):
        """ This property returns the total number of semi-continuous decision variables added to the model.
        """
        return self._count_variables_filtered(lambda v: self._is_semicontinuous_var(v))

    def _has_discrete_var(self):
        # INTERNAL
        return any(v.is_discrete() for v in self.iter_variables())

    def _solves_as_mip(self):
        # INTERNAL: will the model solve as a MIP?
        # returns TRue if the model contains a discrete variable or if it has SOS
        return self._has_discrete_var() or self._allsos

    def get_statistics(self):
        """ Returns statistics on the model.

        :returns: A new instance of :class:`ModelStatistics`.
        """
        return ModelStatistics._make_new_stats(self)

    statistics = property(get_statistics)

    def iter_variables(self):
        """ Iterates over all the variables in the model.

        Returns the variables in the order they were added to the model,
        regardless of their type.

        Returns:
            An iterator object.
        """
        return iter(self.__allvars)

    def iter_binary_vars(self):
        """ Iterates over all binary decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_binary_var(v))

    def iter_integer_vars(self):
        """ Iterates over all integer decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_integer_var(v))

    def iter_continuous_vars(self):
        """ Iterates over all continuous decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_continuous_var(v))

    def iter_semicontinuous_vars(self):
        """ Iterates over all semi-continuous decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_semicontinuous_var(v))

    def get_var_by_name(self, name):
        """ Searches for a variable from a name.

        Returns a variable if it finds one with exactly this name, or None.

        Args:
            name (str): The name of the variable being searched for.

        :returns: A variable (instance of :class:`docplex.mp.linear.Var`) or None.
        """
        return self.__vars_by_name.get(name, None)

    # @staticmethod
    # def _build_name_dict(mobj_seq):
    # return {mobj.name: mobj for mobj in mobj_seq if mobj.name is not None}

    # index management
    def _build_index_dict(self, mobj_it, raise_on_invalid_index=False):
        #  INTERNAL
        idx_dict = {}
        for mobj in mobj_it:
            if mobj.has_valid_index():
                idx_dict[mobj.index] = mobj
            elif raise_on_invalid_index:  # pragma: no cover
                self.fatal("Object has invalid index: {0!s}", mobj)  # pragma: no cover
        return idx_dict

    def get_var_by_index(self, idx):
        # INTERNAL
        self._checker.typecheck_valid_index(idx)
        return self._var_scope.get_object_by_index(idx)

    def set_var_type(self, dvar, new_vartype):
        # INTERNAL
        self._checker.typecheck_vartype(new_vartype)
        if new_vartype != dvar.vartype:
            self.__engine.set_var_type(dvar, new_vartype)
            # change type in the Var object.
            dvar._set_vartype_internal(new_vartype)
        return self

    def set_var_name(self, dvar, new_name):
        # INTERNAL: use var.name to set variable names
        if new_name != dvar.name:
            self.__engine.rename_var(dvar, new_name)
            dvar._set_name(new_name)

    @staticmethod
    def _get_bound_resolver_fn(is_lb):
        return lambda vt: getattr(vt, 'resolve_lb' if is_lb else 'resolve_ub')

    def _zip_var_bounds(self, dvars, bounds, is_lb):
        # INTERNAL
        self._checker.typecheck_iterable(dvars)
        # transform bounds
        if bounds is None:
            if is_lb:
                return [(v, v.vartype.default_lb) for v in dvars]
            else:
                return [(v, v.vartype.default_ub) for v in dvars]
        elif is_number(bounds):
            resolver = self._get_bound_resolver_fn(is_lb)  # VarType.resolve_lb if is_lb else VarType.resolve_ub
            return [(v, resolver(v.vartype)) for v in dvars]

        elif is_iterable(bounds):
            # zip variables and bounds
            raw_zip = izip(dvars, bounds)
            resolver = self._get_bound_resolver_fn(is_lb)  # VarType.resolve_lb if is_lb else VarType.resolve_ub
            return [(v, resolver(v.vartype, b)) for v, b in raw_zip]
        else:
            header = "set_lbs" if is_lb else "set_ubs"
            self.fatal("Model.{0}: expecting either number or number_sequence as bounds, got: {1!s}",
                       header, bounds)

    def set_var_lb(self, var, candidate_lb):
        # INTERNAL: use var.lb to set lb
        new_lb = var.vartype.resolve_lb(candidate_lb)
        self.__engine.set_var_lb((var, new_lb))
        var._internal_set_lb(new_lb)

    def set_var_lbs(self, changed_vars, lbs):
        """ Change lower bounds of a collection of variables.

        `new_lbs` accepts either a single number (all lower bounds are set to this number)
        or an array of numbers. With an array, the lower bounds of the variables are set the corresponding bound in the array.
        If the bounds array is smaller than the variable array, the remaining variables are unchanged.


        Args:
            changed_vars: The collection of variables to be changed.
            lbs: Either a number or a collection of numbers.

        """
        warnings.warn("This function is deprecated, use var.lb in a loop", stacklevel=2)
        var_lbs = self._zip_var_bounds(changed_vars, lbs, is_lb=True)
        self.__engine.set_var_lb(var_lbs)
        for var, new_lb in var_lbs:
            var._internal_set_lb(new_lb)

    def set_var_ub(self, var, candidate_ub):
        # INTERNAL: use var.ub to set ub
        new_ub = var.vartype.resolve_ub(candidate_ub)
        self.__engine.set_var_ub((var, new_ub))
        var._internal_set_ub(new_ub)

    def set_var_ubs(self, changed_vars, ubs):
        """ Change upper bounds of a collection of variables.

        `new_ubs` accepts either a single number (all upper bounds are set to this number)
        or an array of numbers. With an array, the upper bounds of the variables are set the corresponding bound in the array.
        If the bounds array is smaller than the variable array, the remaining variable bounds are unchanged.

        Args:
            changed_vars: The collection of variables to be changed.
            new_ubs: Either a number or a collection of numbers.

        Note:
            This method is deprecated since docplex version > 545

        """
        warnings.warn("This function is deprecated, use var.ub in a loop", stacklevel=2)
        var_ubs = self._zip_var_bounds(changed_vars, ubs, is_lb=False)
        self.__engine.set_var_ub(var_ubs)
        for var, new_ub in var_ubs:
            var._internal_set_ub(new_ub)

    def get_constraint_by_name(self, name):
        """ Searches for a constraint from a name.

        Returns the constraint if it finds a constraint with exactly this name, or None
        if no constraint has this name.

        This function will not raise an exception if the named constraint is not found.

        Args:
            name (str): The name of the constraint being searched for.

        Returns:
            A constraint or None.
        """
        return self._ensure_cts_by_name().get(name)

    def get_constraint_by_index(self, idx):
        # INTERNAL
        self._checker.typecheck_valid_index(idx)
        return self._linct_scope.get_object_by_index(idx)

    def get_indicator_by_index(self, idx):
        self._checker.typecheck_valid_index(idx)
        return self._indicator_scope.get_object_by_index(idx)

    def get_quadratic_by_index(self, idx):
        # INTERNAL
        self._checker.typecheck_valid_index(idx)
        return self._quadct_scope.get_object_by_index(idx)

    @property
    def number_of_constraints(self):
        """ This property returns the total number of constraints that were added to the model.

        The number includes linear constraints, range constraints, and indicator constraints.
        """
        return len(self.__allcts)

    def iter_constraints(self):
        """ Iterates over all constraints (linear, ranges, indicators).

        Returns:
          An iterator object over all constraints in the model.
        """
        return iter(self.__allcts)

    def _count_constraints_with_type(self, cttype):
        return self._count_constraints_filtered(lambda ct: isinstance(ct, cttype))

    def _count_constraints_filtered(self, ct_filter):
        return sum(1 for _ in filter(ct_filter, self.__allcts))

    def gen_constraints_with_type(self, cttype):
        for ct in self.__allcts:
            if isinstance(ct, cttype):
                yield ct

    def gen_constraints_filtered(self, predicate1):
        for ct in self.__allcts:
            if predicate1(ct):
                yield ct

    @property
    def number_of_range_constraints(self):
        """ This property returns the total number of range constraints added to the model.

        """
        return self._count_constraints_with_type(RangeConstraint)

    @property
    def number_of_linear_constraints(self):
        """ This property returns the total number of linear constraints added to the model.

        This counts binary linear constraints (<=, >=, or ==) and
        range constraints.

        See Also:
            :func:`number_of_range_constraints`

        """
        return self._count_constraints_filtered(lambda c: c.is_linear())

    def iter_range_constraints(self):
        return self.gen_constraints_with_type(RangeConstraint)

    def iter_binary_constraints(self):
        return self.gen_constraints_with_type(LinearConstraint)

    def iter_linear_constraints(self):
        """
        Returns an iterator on the linear constraints of the model.
        This includes binary linear constraints and ranges but not indicators.

        Returns:
            An iterator object.
        """
        return self.gen_constraints_filtered(lambda c: c.is_linear())

    def iter_indicator_constraints(self):
        """ Returns an iterator on indicator constraints in the model.

        Returns:
            An iterator object.
        """
        return self.gen_constraints_with_type(IndicatorConstraint)

    @property
    def number_of_indicator_constraints(self):
        """ This property returns the number of indicator constraints in the model.
        """
        return self._count_constraints_with_type(IndicatorConstraint)

    def iter_quadratic_constraints(self):
        """
        Returns an iterator on the quadratic constraints of the model.

        Returns:
            An iterator object.
        """
        return self.gen_constraints_with_type(QuadraticConstraint)

    @property
    def number_of_quadratic_constraints(self):
        """ This property returns the number of quadratic constraints in the model.
        """
        return self._count_constraints_with_type(QuadraticConstraint)

    def has_quadratic_constraint(self):
        return any(isinstance(c, QuadraticConstraint) for c in self.__allcts)

    def var(self, vartype, lb=None, ub=None, name=None):
        """ Creates a decision variable and stores it in the model.

        Args:
            vartype: The type of the decision variable;
                This field expects a concrete instance of the abstract class
                :class:`docplex.mp.vartype.VarType`.

            lb: The lower bound of the variable; either a number or None, to use the default.
                 The default lower bound for all three variable types is 0.

            ub: The upper bound of the variable domain; expects either a number or None to use the type's default.
                The default upper bound for Binary is 1, otherwise positive infinity.

            name: An optional string to name the variable.

        :returns: The newly created decision variable.
        :rtype: :class:`docplex.mp.linear.Var`

        Note:
            The model holds local instances of BinaryVarType, IntegerVarType, ContinuousVarType which
            are accessible by properties (resp. binary_vartype, integer_vartype, continuous_vartype).

        See Also:
            :func:`infinity`,
            :func:`binary_vartype`,
            :func:`integer_vartype`,
            :func:`continuous_vartype`

        """
        self._checker.typecheck_vartype(vartype)
        return self._var(vartype, lb, ub, name)

    def _var(self, vartype, lb=None, ub=None, name=None):
        # INTERNAL
        if lb is not None:
            self._checker.typecheck_num(lb, caller='Var.lb')
        if ub is not None:
            self._checker.typecheck_num(ub, caller='Var.ub')
        return self._lfactory.new_var(vartype, lb, ub, name)

    def continuous_var(self, lb=None, ub=None, name=None):
        """ Creates a new continuous decision variable and stores it in the model.

        Args:
            lb: The lower bound of the variable, or None. The default is 0.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.ContinuousVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        return self._var(self.continuous_vartype, lb, ub, name)

    def integer_var(self, lb=None, ub=None, name=None):
        """ Creates a new integer variable and stores it in the model.

        Args:
            lb: The lower bound of the variable, or None. The default is 0.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name: An optional name for the variable.

        :returns: An instance of the :class:`docplex.mp.linear.Var` class with type `IntegerVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        return self._var(self.integer_vartype, lb, ub, name)

    def binary_var(self, name=None):
        """ Creates a new binary decision variable and stores it in the model.

        Args:
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.BinaryVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        return self._var(self.binary_vartype, name=name)

    def semicontinuous_var(self, lb, ub=None, name=None):
        """ Creates a new semi-continuous decision variable and stores it in the model.

        Args:
            lb: The lower bound of the variable.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.SemiContinuousVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        self._checker.typecheck_num(lb)  # lb cannot be None
        return self._var(self.semicontinuous_vartype, lb, ub, name)

    def var_list(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        self._checker.typecheck_vartype(vartype)
        actual_name, fixed_keys = self._lfactory._make_key_seq(keys, name)
        ctn = _VariableContainer(vartype, [fixed_keys], lb, ub, name)
        self._add_var_container(ctn)
        return self._lfactory.new_var_list(ctn, fixed_keys, vartype, lb, ub, actual_name, 1, key_format)

    def var_dict(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        self._checker.typecheck_vartype(vartype)
        return self._var_dict(keys, vartype, lb, ub, name, key_format)

    def _var_dict(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        # INTERNAL
        actual_name, key_seq = self._lfactory._make_key_seq(keys, name)
        ctn = _VariableContainer(vartype, [key_seq], lb, ub, name)
        self._add_var_container(ctn)
        var_list = self._lfactory.new_var_list(ctn, key_seq, vartype, lb, ub, actual_name, 1, key_format)
        _dict_type = OrderedDict if self._keep_ordering else dict
        return _dict_type(zip(key_seq, var_list))

    def binary_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """ Creates a list of binary decision variables and stores them in the model.

        Args:
            keys: Either a sequence of objects or an integer.
            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if keys is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
                        
        Example:
            If you want each key string to be surrounded by {}, use a special key_format: "_{%s}",
            the %s denotes where the key string will be formatted and appended to `name`.

        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`doc.mp.vartype.BinaryVarType`.

        Example:
            `mdl.binary_var_list(3, "z")` returns a list of size 3
            containing three binary decision variables with names `z_0`, `z_1`, `z_2`.

        """
        return self.var_list(keys, self.binary_vartype, name=name, lb=lb, ub=ub, key_format=key_format)

    def integer_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """ Creates a list of integer decision variables with type `IntegerVarType`, stores them in the model,
        and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.
            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers with the same size as keys,
                a function (which will be called on each key argument), or None.
            ub: Upper bounds of the variables.  Accepts either a floating-point number,
                a list of numbers with the same size as keys,
                a function (which will  be called on each key argument), or None.
            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
            key_format: A format string or None. This format string describes how keys contribute to variable names.
                The default is "_%s". For example if name is "x" and each key object is represented by a string
                like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        Note:
            Using None as the lower bound means the default lower bound (0) is used.
            Using None as the upper bound means the default upper bound (the model's positive infinity)
            is used.

        :returns: A list of :class:`doc.mp.linear.Var` objects with type `IntegerVarType`.

        """
        return self.var_list(keys, self.integer_vartype, lb, ub, name, key_format)

    def continuous_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """
        Creates a list of continuous decision variables with type :class:`docplex.mp.vartype.ContinuousVarType`,
        stores them in the model, and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers
                or use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means using the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers
                or use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str()` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
        Note:
            When `keys` is either an empty list or the integer 0, an empty list is returned.


        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`docplex.mp.vartype.ContinuousVarType`.

        See Also:
            :func:`infinity`

        """
        return self.var_list(keys, self.continuous_vartype, lb, ub, name, key_format)

    def semicontinuous_var_list(self, keys, lb, ub=None, name=str, key_format=None):
        """
        Creates a list of semi-continuous decision variables with type :class:`docplex.mp.vartype.SemiContinuousVarType`,
        stores them in the model, and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                Note that the lower bound of a semi-continuous variable must be strictly positive.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str()` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
        Note:
            When `keys` is either an empty list or the integer 0, an empty list is returned.


        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`docplex.mp.vartype.SemiContinuousVarType`.

        See Also:
            :func:`infinity`

        """
        return self.var_list(keys, self.semicontinuous_vartype, lb, ub, name, key_format)

    def continuous_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None):
        """ Creates a dictionary of continuous decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how `keys` contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type `ContinuousVarType`) indexed by
                  the objects in `keys`.

        See Also:
            :class:`docplex.mp.linear.Var`,
            :func:`infinity`
        """
        return self._var_dict(keys, self.continuous_vartype, lb=lb, ub=ub, name=name, key_format=key_format)

    def integer_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None):
        """  Creates a dictionary of integer decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.


            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns:  A dictionary of :class:`docplex.mp.linear.Var` objects (with type `IntegerVarType`) indexed by the
                   objects in `keys`.

        See Also:
            :func:`infinity`
        """
        return self._var_dict(keys, self.integer_vartype, lb=lb, ub=ub, name=name, key_format=key_format)

    def binary_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None):
        """ Creates a dictionary of binary decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            name (string): Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects with type
                  :class:`docplex.mp.vartype.BinaryVarType` indexed by the objects in `keys`.
        """
        return self._var_dict(keys, self.binary_vartype, lb=lb, ub=ub, name=name, key_format=key_format)

    def var_multidict(self, vartype, seq_of_key_seqs, lb=None, ub=None, name=stringify_tuple, key_format=None):
        # INTERNAL
        self._checker.typecheck_vartype(vartype)
        self._checker.typecheck_iterable(seq_of_key_seqs)
        fixed_keys = [self._lfactory._make_key_seq(ks, name)[1] for ks in seq_of_key_seqs]

        ctn = _VariableContainer(vartype, fixed_keys, lb, ub, name)
        self._add_var_container(ctn)
        # the sequence of keysets should answer to len(no generators here)
        dimension = len(fixed_keys)
        if dimension < 1:
            self.fatal("len of key sequence must be >= 1, got: {0}", dimension)  # pragma: no cover

        # create cartesian product of keys...
        all_key_tuples = list(product(*fixed_keys))
        # check empty list
        if not all_key_tuples:
            self.fatal('multidict has no keys to index the variables')
        cube_vars = self._lfactory.new_var_list(ctn, all_key_tuples, vartype, lb, ub, name, dimension, key_format)

        var_dict = dict(zip(all_key_tuples, cube_vars))
        return var_dict

    def var_matrix(self, vartype, keys1, keys2, lb=None, ub=None, name=stringify_tuple, key_format=None):
        return self.var_multidict(vartype, [keys1, keys2], lb, ub, name, key_format)

    def binary_var_matrix(self, keys1, keys2, name=stringify_tuple, key_format=None):
        """ Creates a dictionary of binary decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If either of `keys1` or `keys2` is empty, this method returns an empty dictionary.

        Args:
            keys1: Either a sequence of objects, an iterator, or a positive integer. If passed an integer N,
                    it is interpreted as a range from 0 to N-1.

            keys2: Same as `keys1`.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if keys is a sequence) or the
                index of the variable within the range, if an integer argument is passed.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects with type
                  :class:`docplex.mp.vartype.BinaryVarType` indexed by
                  all couples `(k1, k2)` with `k1` in `keys1` and `k2` in `keys2`.
        """
        return self.var_multidict(self.binary_vartype, [keys1, keys2], 0, 1, name=name, key_format=key_format)

    def integer_var_matrix(self, keys1, keys2, lb=None, ub=None, name=stringify_tuple, key_format=None):
        """ Creates a dictionary of integer decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted as in :func:`integer_var_dict`.
        """

        return self.var_multidict(self.integer_vartype, [keys1, keys2], lb, ub, name, key_format)

    def continuous_var_matrix(self, keys1, keys2, lb=None, ub=None, name=stringify_tuple, key_format=None):
        """ Creates a dictionary of continuous decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows retrieval of variables from a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted the same as in :func:`integer_var_dict`.

        """
        return self.var_multidict(self.continuous_vartype, [keys1, keys2], lb, ub, name, key_format)

    def continuous_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=stringify_tuple, key_format=None):
        """ Creates a dictionary of continuous decision variables, indexed by triplets of key objects.

        Same as :func:`continuous_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys1`, `k2` in `keys2`, `k3` in `keys3`.
        """
        return self.var_multidict(self.continuous_vartype, [keys1, keys2, keys3], lb, ub, name, key_format)

    def integer_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=str):
        """ Creates a dictionary of integer decision variables, indexed by triplets.

        Same as :func:`integer_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys1`, `k2` in `keys2`, `k3` in `keys3`.

        See Also:
            :func:`integer_var_matrix`
        """
        return self.var_multidict(self.integer_vartype, [keys1, keys2, keys3], lb, ub, name)

    def binary_var_cube(self, keys1, keys2, keys3, name=stringify_tuple, key_format=None):
        """Creates a dictionary of binary decision variables, indexed by triplets.

        Same as :func:`binary_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys1`, `k2` in `keys2`, `k3` in `keys3`.

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type :class:`docplex.mp.vartype.BinaryVarType`) indexed by
            triplets.

        """
        return self.var_multidict(self.binary_vartype, [keys1, keys2, keys3], name=name, key_format=key_format)

    def linear_expr(self, name=None):
        ''' Returns a new empty linear expression.

        Args:
            name: An optional string to name the expression.

        :returns: An instance of :class:`docplex.mp.linear.LinearExpr`.
        '''
        return self._lfactory.linear_expr(e=None, name=name)

    def quad_expr(self, name=None):
        ''' Returns a new empty quadratic expression.

        Args:
            name: An optional string to name the expression.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr`.
        '''
        return self._qfactory.new_quad(name=name)

    def _linear_expr(self, e=0, constant=0, name=None):
        # INTERNAL
        return self._lfactory.linear_expr(e, constant, name)

    def _quad_expr(self, quads, linexpr):
        # INTERNAL
        return self._qfactory.new_quad(quad_args=quads, linexpr=linexpr, safe=False)

    def monomial_expr(self, dvar, coef=1):
        # INTERNAL
        self._checker.typecheck_var(dvar)
        return self._lfactory.new_monomial_expr(dvar, coef)

    def abs(self, e):
        """ Builds an expression equal to the absolute value of its argument.

        Args:
            e: Accepts any object that can be transformed into an expression:
                decision variables, expressions, or numbers.

        Returns:
            An expression that can be used in arithmetic operators and constraints.

        Note:
            Building the expression generates auxiliary decision variables, including binary decision variables,
            and this may change the nature of the problem from a LP to a MIP.

        """
        self._checker.typecheck_operand(e, caller="Model.abs", accept_numbers=True)
        return self._lfactory.new_abs_expr(e)

    def min(self, *args):
        """ Builds an expression equal to the minimum value of its arguments.

        This method accepts a variable number of arguments.

        If no arguments are provided, returns positive infinity (see :func:`infinity`).

        Args:
            args: A variable list of arguments, each of which is either an expression, a variable,
                or a container.

        If passed a container or an iterator, this container or iterator must be the unique argument.

        If passed one dictionary, returns the minimum of the dictionary values.


        Returns:
            An expression that can be used in arithmetic operators and constraints.

        Example:
            `model.min()` -> returns `model.infinity`.

            `model.min(e)` -> returns `e`.

            `model.min(e1, e2, e3)` -> returns a new expression equal to the minimum of the values of `e1`, `e2`, `e3`.

            `model.min([x1,x2,x3])` where `x1`, `x2` .. are variables or expressions -> returns the minimum of these expressions.

            `model.min([])` -> returns `model.infinity`.

        Note:
            Building the expression generates auxiliary variables, including binary decision variables,
            and this may change the nature of the problem from a LP to a MIP.
        """
        min_args = args
        nb_args = len(args)
        if 0 == nb_args:
            pass
        elif 1 == nb_args:
            unique_arg = args[0]
            if is_iterable(unique_arg):
                if isinstance(unique_arg, dict):
                    min_args = unique_arg.values()
                else:
                    min_args = _to_list(unique_arg)
                for a in min_args:
                    self._checker.typecheck_operand(a, caller="Model.min()")
            else:
                self._checker.typecheck_operand(unique_arg, caller="Model.min()")

        else:
            for arg in args:
                self._checker.typecheck_operand(arg, caller="Model.min")

        return self._lfactory.new_min_expr(*min_args)

    def max(self, *args):
        """ Builds an expression equal to the maximum value of its arguments.

        This method accepts a variable number of arguments.

        Args:
            args: A variable list of arguments, each of which is either an expression, a variable,
                or a container.

        If passed a container or an iterator, this container or iterator must be the unique argument.

        If passed one dictionary, returns the maximum of the dictionary values.

        If no arguments are provided, returns negative infinity (see :func:`infinity`).


        Example:
            `model.max()` -> returns `-model.infinity`.

            `model.max(e)` -> returns `e`.

            `model.max(e1, e2, e3)` -> returns a new expression equal to the maximum of the values of `e1`, `e2`, `e3`.

            `model.max([x1,x2,x3])` where `x1`, `x2` .. are variables or expressions -> returns the maximum of these expressions.

            `model.max([])` -> returns `-model.infinity`.


        Note:
            Building the expression generates auxiliary variables, including binary decision variables,
            and this may change the nature of the problem from a LP to a MIP.
        """
        max_args = args
        nb_args = len(args)
        if 0 == nb_args:
            pass
        elif 1 == nb_args:
            unique_arg = args[0]
            if is_iterable(unique_arg):
                if isinstance(unique_arg, dict):
                    max_args = unique_arg.values()
                else:
                    max_args = _to_list(unique_arg)
                for a in max_args:
                    self._checker.typecheck_operand(a, caller="Model.max")
            else:
                self._checker.typecheck_operand(unique_arg, caller="Model.max")
        else:
            for arg in args:
                self._checker.typecheck_operand(arg, caller="Model.max")

        return self._lfactory.new_max_expr(*max_args)

    def _get_zero_expr(self):
        # INTERNAL
        return self._lfactory.zero_expr

    def _to_linear_expr(self, e, force_clone=False, context=None):
        # INTERNAL
        return self._lfactory._to_linear_expr(e, force_clone=force_clone, context=context)

    def scal_prod(self, terms, coefs):
        """
        Creates a linear expression equal to the scalar product of a list of decision variables and a sequence of coefficients.

        This method accepts different types of input for both arguments. The variable sequence can be either a list
        or an iterator of objects that can be converted to linear expressions, that is, variables, expressions, or numbers.
        The most usual case is variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.
        In this last case the scalar product reduces to a sum times this coefficient.


        :param terms: A list or an iterator on variables or expressions.
        :param coefs: A list or an iterator on numbers, or a number.

        Note: 
           If either list or iterator is empty, this method returns zero.

        :return: A linear expression or 0.
        """
        self._checker.check_ordered_sequence(arg=terms, header='Model.scal_prod() requires a list of expressions/variables')
        return self._aggregator.scal_prod(terms, coefs)

    def dot(self, terms, coefs):
        """ Synonym for :func:`scal_prod`.

        """
        return self.scal_prod(terms, coefs)

    def sum(self, args):
        """ Creates a linear expression summing over a sequence.


        Note:
           This method returns 0 if the argument is an empty list or iterator.
        
        :param args: A list of objects that can be converted to linear expressions, that is, linear expressions, decision variables, or numbers.

        :return: A linear expression or 0.
        """
        return self._aggregator.sum(args)

    def sumsq(self, args):
        """ Creates a quadratic expression summing squares over a sequence.

        Each element of the list is squared and added to the result. Quadratic expressions
        are not accepted, as they cannot be squared.

        Note:
           This method returns 0 if the argument is an empty list or iterator.

        :param args: A list of objects that can be converted to linear expressions, that is,
            linear expressions, decision variables, or numbers. Each item is squared and
            added to the result.

        :return: A quadratic expression (possibly constant).
        """
        return self._aggregator.sumsq(args)



    def le_constraint(self, lhs, rhs, name=None):
        """ Creates a "less than or equal to" linear constraint.

        Note:
            This method returns a constraint object, that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        Args:
            lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            rhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            name (string): An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_le_constraint(lhs, rhs, name)

    def ge_constraint(self, lhs, rhs, name=None):
        """ Creates a "greater than or equal to" linear constraint.

        Note:
            This method returns a constraint object that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        Args:
            lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            rhs: An object that can be converted to a linear expression, typically a variable,
                    a number of an expression.
            name (string): An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_ge_constraint(lhs, rhs, name)

    def eq_constraint(self, lhs, rhs, name=None):
        """ Creates an equality constraint.

        Note:
            This method returns a constraint object that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        :param lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param rhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param name: An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_eq_constraint(lhs, rhs, name)

    def _post_constraint(self, ct):
        # INTERNAL
        eng = self.__engine
        if isinstance(ct, LinearConstraint):
            return eng.create_binary_linear_constraint(ct)
        elif isinstance(ct, RangeConstraint):
            return eng.create_range_constraint(ct)
        elif isinstance(ct, IndicatorConstraint):
            # here check whether linear ct is trivial. if yes do not send to CPLEX
            indicator = ct
            linct = indicator.linear_constraint
            if linct.is_trivial():
                is_feasible = linct._is_trivially_feasible()
                if is_feasible:
                    self.warning("Indicator constraint {0!s} has a trivially feasible constraint (no effect)",
                                 indicator)
                    return -2
                else:
                    self.warning(
                        "indicator constraint {0!s} has a trivially infeasible constraint; variable invalidated",
                        indicator)
                    indicator.invalidate()
                    return -4
            return eng.create_indicator_constraint(ct)
        elif isinstance(ct, QuadraticConstraint):
            return eng.create_quadratic_constraint(ct)
        else:
            self.fatal("Expecting binary constraint, indicator or range, got: {0!s}", ct)  # pragma: no cover

    def _notify_trivial_constraint(self, ct, ctname, is_feasible):
        self_trivial_warn_level = self._trivial_cts_message_level
        if is_feasible and self_trivial_warn_level > self.warn_trivial_feasible:
            return
        elif self_trivial_warn_level > self.warn_trivial_infeasible:
            return
        # ---
        # herefater we are sure to warn
        if ct is None:
            arg = None
        elif ctname and not ctname.startswith("_"):
            arg = ctname
        elif ct.has_user_name():
            arg = ct.name
        else:
            arg = str(ct)
        # ---
        ct_typename = ct.short_typename() if ct is not None else "constraint"
        ct_rank = self.number_of_constraints + 1
        # BEWARE: do not use if arg here
        # because if arg is a constraint, boolean conversion won't work.
        if arg is not None:
            if is_feasible:
                self.info("Adding trivial feasible {2}: {0!s}, rank: {1}", arg, ct_rank, ct_typename)
            else:
                self.error("Adding trivial infeasible {2}: {0!s}, rank: {1}", arg, ct_rank, ct_typename)
        else:
            if is_feasible:
                self.info("Adding trivial feasible {1}, rank: {0}", ct_rank, ct_typename)
            else:
                self.error("Adding trivial infeasible {1}, rank: {0}", ct_rank, ct_typename)

    def _prepare_constraint(self, ct, ctname, do_check=True):
        # INTERNAL
        if ct is True:
            # sum([]) == 0
            self._notify_trivial_constraint(None, ctname, is_feasible=True)
            return False, None

        elif ct is False:
            # happens with sum([]) and constant e.g. sum([]) == 2
            ct = self._lfactory.new_trivial_infeasible_ct()
            self._notify_trivial_constraint(None, ctname, is_feasible=False)
            if ctname:
                self.fatal("Adding a trivially infeasible constraint with name: {0}", ctname)
            else:
                # analogous to 0 == 1, model is sure to fail
                self.fatal("Adding trivially infeasible constraint")
        else:
            if do_check:
                self._checker.typecheck_constraint(ct)
                self._checker.typecheck_in_model(model=self, mobj=ct)
            # -- watch for trivial cts e.g. linexpr(0) <= linexpr(1)
            if ct.is_trivial():
                if ct._is_trivially_feasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=True)
                elif ct._is_trivially_infeasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=False)

        # --- name management ---
        if ctname:
            ct_name_map = self.__cts_by_name
            if ct_name_map:
                if ctname in ct_name_map:
                    self.warning("Duplicate constraint name: {0!s}, used for: {1}", ctname, ct_name_map[ctname])
                ct_name_map[ctname] = ct
            ct.name = ctname
        else:
            if self._name_all_cts and not ct.has_name():
                ct_auto_name = self._create_automatic_ctname(ct)
                ct._set_automatic_name(ct_auto_name)

        # ---

        # check for already posted cts.
        if do_check and ct._index >= 0:
            self.warning("constraint has already been posted: {0!s}, index is: {1}", ct, ct.index)  # pragma: no cover
            return False, ct  # pragma: no cover
        return True, ct

    # @profile
    def _add_constraint_internal(self, ct, ctname, do_check=True):
        do_post, posted_ct = self._prepare_constraint(ct, ctname, do_check)
        if do_post:
            ct_engine_index = self._post_constraint(ct)
            self._register_one_constraint(ct, ct_engine_index, is_ctname_safe=True)

        return posted_ct

    def _remove_constraint_internal(self, ct):
        """
        No typechecking for this internal version.
        :param ct:
        :return:
        """
        ct_name = ct.name
        if ct_name and self.__cts_by_name:
            try:
                del self.__cts_by_name[ct_name]
            except KeyError:
                # already removed
                pass

        ct_index = ct.unchecked_index
        if ct.is_valid_index(ct_index):
            # remove from model.
            try:
                self.__allcts.remove(ct)
            except ValueError:
                # not been added yet
                pass

            # remove from engine.
            self.__engine.remove_constraint(ct)
            cscope = self._get_ct_scope(ct)
            cscope.reindex_one(ct_index, self.__engine)
            self._sync_constraint_indices(cscope.iter)
            cscope.update_indices()
        # reset index to negative
        ct.notify_deleted()

    def _resolve_ct(self, ct_arg, silent=False, context=None):
        verbose = not silent
        if context:
            printed_context = context + ": "
        else:
            printed_context = ""
        if isinstance(ct_arg, AbstractConstraint):
            return ct_arg
        elif is_string(ct_arg):
            ct = self.get_constraint_by_name(ct_arg)
            if ct is None and verbose:
                self.error("{0}no constraint with name: \"{1}\" - ignored", printed_context, ct_arg)
            return ct
        elif is_int(ct_arg):
            if ct_arg >= 0:
                ct_index = ct_arg
                ct = self.get_constraint_by_index(ct_index)
                if ct is None and verbose:
                    self.error("{0}no constraint with index: \"{1}\" - ignored", printed_context, ct_arg)
                return ct
            else:
                self.error("{0}not a valid index: \"{1}\" - ignored", printed_context, ct_arg)
                return None

        else:
            if verbose:
                self.error("{0}unexpected argument {1!s}, expecting string or constraint", printed_context, ct_arg)
            return None

    def remove_constraint(self, ct_arg):
        """ Removes a constraint from the model.

        Args:
            ct_arg: The constraint to remove. Accepts either a constraint object or a string.
                If passed a string, looks for a constraint with that name.

        """
        ct = self._resolve_ct(ct_arg, silent=False, context="remove_constraint")
        if ct is not None:
            self._checker.typecheck_in_model(self, ct, header="constraint")
            self._remove_constraint_internal(ct)

    def clear_constraints(self):
        self.__engine.remove_constraints(cts=None)  # special case to denote all
        # clear containers
        self.__allcts = []
        self.__cts_by_name = None
        # clear constraint index scopes.
        for ctscope in self._ctscopes:
            ctscope.reset()

    def remove_constraints(self, *args):
        if 0 == len(args):
            self.clear_constraints()
        else:
            doomed = list(args)
            for d in doomed:
                self._checker.typecheck_constraint(d)
            self.__engine.remove_constraints(doomed)
            self._remove_constraints_internal(doomed)

    def _remove_constraints_internal(self, doomed):
        for d in doomed:
            dname = d.get_name()
            if dname:
                del self.__cts_by_name[dname]
            d.notify_deleted()
        # update container
        self.__allcts = [c for c in self.__allcts if c not in doomed]
        # TODO: handle reindexing
        doomed_scopes = set(self._get_ct_scope(c) for c in doomed)
        for ds in doomed_scopes:
            ds.reindex_all(self.__engine)
            ds.update_indices()

    def add_range(self, lb, expr, ub, rng_name=None):
        """ Adds a new range constraint to the model.

        A range constraint states that a linear
        expression has to stay within an interval `[lb..ub]`.
        Both `lb` and `ub` have to be float numbers with `lb` smaller than `ub`.

        The method creates a new range constraint and adds it to the model.

        Args:
            lb (float): A floating-point number.
            expr: A linear expression, e.g. X+Y+Z.
            ub (float): A floating-point number, which should be greater than `lb`.
            rng_name (string): An optional name for the range constraint.

        :returns: The newly created range constraint.
        :rtype: :class:`docplex.mp.constr.RangeConstraint`

        Raises:
            An exception if `lb` is greater than `ub`.

        """
        rng = self.range_constraint(lb, expr, ub, rng_name)
        ct = self._add_constraint_internal(rng, rng_name)
        return ct

    def indicator_constraint(self, binary_var, linear_ct, active_value=1):
        self._checker.typecheck_var(binary_var)
        self._checker.typecheck_linear_constraint(linear_ct)
        self._checker.typecheck_zero_or_one(active_value)
        self._checker.typecheck_in_model(self, binary_var, header="binary variable")
        self._checker.typecheck_in_model(self, linear_ct, header="linear_constraint")
        return self._lfactory.new_indicator_constraint(binary_var, linear_ct, active_value)

    def add_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        """ Adds a new indicator constraint to the model.

        An indicator constraint links (one-way) the value of a binary variable to
        the satisfaction of a linear constraint.
        If the binary variable equals the active value, then the constraint is satisfied, but
        otherwise the constraint may or may not be satisfied.

        Args:
             binary_var: The binary variable used to control the satisfaction of the linear constraint.
             linear_ct: A linear constraint (EQ, LE, GE).
             active_value: 0 or 1. The value used to trigger the satisfaction of the constraint. The default is 1.
             name (string): An optional name for the indicator constraint.

        Returns:
            The newly created indicator constraint.
        """
        self._checker.typecheck_var(binary_var)
        self._checker.typecheck_constraint(linear_ct)
        self._checker.typecheck_zero_or_one(active_value)
        self._checker.typecheck_in_model(self, binary_var, header="binary variable")
        self._checker.typecheck_in_model(self, linear_ct, header="linear_constraint")
        return self._add_indicator(binary_var, linear_ct, active_value, name, do_check=False)

    _indicator_trivial_feasible_idx = -2
    _indicator_trivial_infeasible_idx = -4

    def _add_indicator(self, binary_var, linear_ct, active_value=1, name=None, do_check=True):
        # INTERNAL
        indicator = self._lfactory.new_indicator_constraint(binary_var, linear_ct, active_value=active_value)
        if linear_ct.is_trivial():
            is_feasible = linear_ct._is_trivially_feasible()
            if is_feasible:
                self.warning("Indicator constraint {0!s} has a trivial feasible linear constraint (has no effect)",
                             indicator)
                return self._indicator_trivial_feasible_idx
            else:
                self.warning("indicator constraint {0!s} has a trivial infeasible linear constraint - invalidated",
                             indicator)
                indicator.invalidate()
                return self._indicator_trivial_infeasible_idx
        else:
            return self._add_constraint_internal(indicator, name, do_check=do_check)

    def range_constraint(self, lb, expr, ub, rng_name=None):
        """ Creates a new range constraint but does not add it to the model.

        A range constraint states that a linear
        expression has to stay within an interval `[lb..ub]`.
        Both `lb` and `ub` have to be floating-point numbers with `lb` smaller than `ub`.

        The method creates a new range constraint but does not add it to the model.

        Args:
            lb: A floating-point number.
            expr: A linear expression, e.g. X+Y+Z.
            ub: A floating-point number, which should be greater than `lb`.
            rng_name: An optional name for the range constraint.

        Returns:
            The newly created range constraint.
        Raises:
             An exception if `lb` is greater than `ub`.

        """
        self._checker.typecheck_num(lb, 'Model.range_constraint')
        self._checker.typecheck_num(ub, 'Model.range_constraint')
        if not lb <= ub:
            self.error("infeasible range constraint, lb={0}, ub={1}, expr={2}", lb, ub, expr)

        expr1 = self._to_linear_expr(expr)
        rng = self._lfactory.new_range_constraint(lb, expr1, ub, rng_name)
        return rng

    def add_constraint(self, ct, ctname=None):
        """ Adds a new linear constraint to the model.

        Args:
            ct: A linear constraint of the form <expr1> <op> <expr2>, where both expr1 and expr2 are
                 linear expressions built from variables in the model, and <op> is a relational operator
                 among <= (less than or equal), == (equal), and >= (greater than or equal).
            ctname (string): An optional string used to name the constraint.

        Returns:
            The newly added constraint.
        """
        ct = self._add_constraint_internal(ct, ctname)
        return ct

    def add(self, ct, ctname=None):
        return self._add_constraint_internal(ct, ctname)

    def _add_constraints(self, cts, names=None, do_check=True):
        # INTERNAL
        posted_cts = []
        if not names:
            names = generate_constant(the_constant=None, count_max=len(cts))
        for ct, ctname in izip(cts, names):
            do_post, posted = self._prepare_constraint(ct, ctname, do_check=do_check)
            if do_post:
                posted_cts.append(posted)

        if posted_cts:
            ct_indices = self.__engine.create_block_linear_constraints(posted_cts)
            self._register_block_cts(posted_cts, ct_indices, safe_names=True)
        return posted_cts

    def add_constraints(self, cts, names=None, do_check=True):
        """ Adds a collection of linear constraints to the model in one operation.

        Each constraint in the collection is added to the model, if it was not already added.
        If present, the `names` collection is used to set names to the constraints.

        Note: The `cts` argument can be a collection of (constraint, name) tuples such
        as `mdl.add_constraints([(x >=3, 'ct0'), (x + y == 1, 'ct1')])`.

        Args:
            cts: A collection of linear constraints or an iterator over a collection of linear constraints.
            names: An optional collection or iterator on strings.

        Returns:
            A list of those constraints added to the model.
        """
        # build a sequence as we need to traverse it more than once.
        self._checker.typecheck_iterable(cts)
        cts_seq = _to_list(cts)
        # typecheck
        if 0 == len(cts_seq):
            return []
        else:
            ctnames = []
            first_item = cts_seq[0]
            if isinstance(first_item, LinearConstraint):
                if do_check:
                    for ct in cts_seq:
                        self._checker.typecheck_linear_constraint(ct)
                cts = cts_seq
                ctnames = names
            elif isinstance(first_item, tuple) and 2 == len(first_item):
                if names:
                    self.fatal("Model.add_constraints(): cannot mix tuples and explicit name sequence")
                if do_check:
                    for ct, ctn in cts_seq:
                        self._checker.typecheck_linear_constraint(ct)
                        self._checker.typecheck_string(ctn, accept_empty=True)
                zipped = list(izip(*cts_seq))
                cts = zipped[0]
                ctnames = zipped[1]
            else:
                self.fatal(
                    "Model.add-constraints expects either a sequence of constraints or (constraint, name) tuples, got: {0!s}",
                    first_item)

            return self._add_constraints(cts, ctnames, do_check=do_check)

    # ----------------------------------------------------
    # objective
    # ----------------------------------------------------

    def round_objective_if_discrete(self, raw_obj):
        # INTERNAL
        if self.__objective_expr.is_discrete():
            return self.round_nearest(raw_obj)
        else:
            return raw_obj

    @property
    def objective_expr(self):
        """ This property returns the current expression used as the model objective.
        """
        return self.__objective_expr

    @property
    def objective_sense(self):
        """ This property returns the direction of the optimization as an instance of
        :class:`docplex.mp.basic.ObjectiveSense`, either Minimize or Maximize.
        """
        return self.__objective_sense

    def minimize(self, expr):
        """ Sets an expression as the expression to be minimized.

        The argument is converted to a linear expression. Accepted types are variables (instances of
        :class:`docplex.mp.linear.Var` class), linear expressions (instances of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        :param expr: A linear expression or a variable.
        """
        self.set_objective(ObjectiveSense.Minimize, expr)

    def maximize(self, expr):
        """
        Sets an expression as the expression to be maximized.

        The argument is converted to a linear expression. Accepted types are variables (instances of
        :class:`docplex.mp.linear.Var` class), linear expressions (instances of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        :param expr: A linear expression or a variable.
        """
        self.set_objective(ObjectiveSense.Maximize, expr)

    def is_minimize(self):
        """ Checks whether the model is a minimization model.

        Note:
            This returns True even if the expression to minimize is a constant.
            To check whether the model has a non-constant objective, use :func:`is_optimized`.

        Returns:
           Boolean: True if the model is a minimization model.
        """
        return self.__objective_sense is ObjectiveSense.Minimize

    def is_maximize(self):
        """ Checks whether the model is a maximization model.

        Note:
           This returns True even if the expression to maximize is a constant.
           To check whether the model has a non-constant objective, use :func:`is_optimized`.

        Returns:
            Boolean: True if the model is a maximization model.
        """
        return self.__objective_sense is ObjectiveSense.Maximize

    def objective_coef(self, dvar):
        """ Returns the objective coefficient of a variable.

        The objective coefficient is the coefficient of the given variable in
        the model's objective expression. If the variable is not explicitly
        mentioned in the objective, it returns 0.

        :param dvar: The decision variable for which to compute the objective coefficient.

        Returns:
            float: The objective coefficient of the variable.
        """
        self._checker.typecheck_var(dvar)
        return self._objective_coef(dvar)

    def _objective_coef(self, dvar):
        return self.__objective_expr[dvar]

    def remove_objective(self):
        """ Clears the current objective.

        This is equivalent to setting "minimize 0".
        Any subsequent solve will look only for a feasible solution.
        You can detect this state by calling :func:`has_objective` on the model.

        """
        self.set_objective(self._lfactory.default_objective_sense(), self._default_objective_expr)

    def is_optimized(self):
        """ Checks whether the model has a non-constant objective expression.

        A model with a constant objective will only search for a feasible solution when solved.
        This happens either if no objective has been assigned to the model,
        or if the objective has been removed with :func:`remove_objective`.

        Returns:
            Boolean: True, if the model has a non-constant objective expression.

        """
        return not self.__objective_expr.is_constant()

    def has_objective(self):
        # INTERNAL
        return self.__objective_expr is not self._default_objective_expr

    def set_objective(self, sense, expr):
        """ Sets a new objective.

        Args:
            sense: Either an instance of :class:`docplex.mp.basic.ObjectiveSense` (Minimize or Maximize),
                or a string: "min" or "max".
            expr: Is converted to an expression. Accepted types are variables,
                linear expressions, quadratic expressions or numbers.

        Note:
            When using a number, the search will not optimize but only look for a feasible solution.

        """
        actual_sense = self._resolve_sense(sense)
        if expr is None:
            expr = self._default_objective_expr
        else:
            expr = self._lfactory._to_expr(expr)
            expr.notify_used(self)

        self.__engine.set_objective(actual_sense, expr)

        self.__objective_sense = actual_sense
        self.__objective_expr = expr

    def _can_solve(self):
        return self.__engine.can_solve()

    def _make_start_infodict(self):
        # INTERNAL
        infodict = {}
        stats = self.get_statistics()
        infodict.update(zip(["number_of_%s_vars" % tn for tn in ["binary", "integer", "continuous"]],
                            [stats.number_of_binary_variables,
                             stats.number_of_integer_variables,
                             stats.number_of_continuous_variables]))
        infodict["number_of_constraints"] = self.number_of_constraints
        return infodict

    def _make_end_infodict(self):
        return self.solution.as_dict(keep_zeros=False) if self.solution is not None else dict()

    def _get_key_in_kwargs(self, context, kwargs_dict):
        """Returns the overloaded value of api_key in the specified dict.

        If a 'key'  is found, it is returned. If 'key' is not found, this
        looks up 'api_key' (compatibility mode with versions < 1.0)
        """
        key = kwargs_dict.get('key')
        if not key:
            key = kwargs_dict.get('api_key')
        if key:
            try:
                ignored_keys = context.solver.docloud.ignored_keys
                # if string, allow comma separated form
                if isinstance(ignored_keys, six.string_types):
                    values = ignored_keys.split(",")
                    if key in values:
                        return None
                elif key in ignored_keys:
                    return None
            except AttributeError:
                # no ignored_keys, just pass
                pass
        return key

    def _get_url_in_kwargs(self, context, kwargs_dict):
        """Returns the overloaded value of url in the specified dict.
        """
        url = kwargs_dict.get('url')
        if url:
            try:
                ignored_urls = context.solver.docloud.ignored_urls
                # if string, allow comma separated form
                if isinstance(ignored_urls, six.string_types):
                    values = ignored_urls.split(",")
                    if url in values:
                        return None
                elif url in ignored_urls:
                    return None
            except AttributeError:
                # no ignored_urls, just pass
                pass
        return url

    def _must_use_docloud(self, __context, **kwargs):
        # returns True if context + kwargs require an execution on cloud
        # this happens in the following cases:
        # (i)  kwargs contains a "docloud_context" key (compat??)
        # (ii) both an explicit url and api_key appear in kwargs
        # (iv) the context's "solver.agent" is "docloud"
        # (v)  kwargs override agent to be "docloud"
        docloud_agent_name = "docloud"  # this might change
        have_docloud_context = kwargs.get('docloud_context') is not None
        have_api_key = self._get_key_in_kwargs(__context, kwargs)
        have_url = self._get_url_in_kwargs(__context, kwargs)
        context_agent_is_docloud = __context.solver.get('agent') == docloud_agent_name
        kwargs_agent_is_docloud = kwargs.get('agent') == docloud_agent_name
        return have_docloud_context \
               or (have_api_key and have_url) \
               or context_agent_is_docloud \
               or kwargs_agent_is_docloud

    def prepare_actual_context(self, **kwargs):
        # prepares the actual context that will be used for a solve

        # use the provided context if any, or the self.context otherwise
        if not kwargs:
            return self.context

        if kwargs.get('context') is not None:
            context = deepcopy(kwargs['context'])
        else:
            # we are sure kwargs is not empty
            context = deepcopy(self.context)

        # update the context with provided kwargs
        for argname, argval in six.iteritems(kwargs):
            # skip context argname if any
            if argname == "url" and is_url_ignored(context, argval) and context.solver.docloud.url:
                pass
            elif argname == "key" and is_key_ignored(context, argval) and context.solver.docloud.key:
                pass
            elif argname != "context" and argval is not None:
                context.update_key_value(argname, argval)

        return context

    def solve(self, **kwargs):
        """ Starts a solve operation on the model.

        If CPLEX is available, the solve operation will be performed using the native CPLEX.
        If CPLEX is not available, the solve operation will be started on DOcplexcloud.
        The DOcplexcloud connection parameters are looked up in the following order:

            - if ``kwargs`` contains valid ``url`` and ``key`` values, they are used.
            - if ``kwargs`` contains a ``context`` and that context contains
                valid ``solver.docloud.url`` and ``solver.docloud.key`` values,
                those values are used. Other attributes of ``solver.docloud``
                can also be used. See :class:`docplex.mp.context.Context`
            - finally, the model's attribute ``context`` is used. This ``context``
                is set at model creation time.

        If CPLEX is not available and the model has no valid credentials, an error is raised, as there is
        no way to perform the solve.

        Note that if ``url`` and ``key`` parameters are present and the
        values of the parameters are not in the ignored url or key list,
        the solve operation will be started on DOcplexcloud even if CPLEX is
        available.

        Example::

            # forces the solve on DOcplexcloud with the specified url and keys
            model.solve(url='https://foo.com', key='bar')

        Example::

            # set some DOcplexcloud credentials, but depend on another
            # method to decide if solve is local or not
            ctx.solver.docloud.url = 'https://foo.com'
            ctx.solver.docloud.key = 'bar'
            agent = 'local' if method_that_decides_if_solve_is_local() or 'docloud'
            model.solve(context=ctx, agent=agent)

        Args:
            context (optional): An instance of context to be used in instead of
                the context this model was built with.
            cplex_parameters (optional): A set of CPLEX parameters to use
                instead of the parameters defined as
                ``context.cplex_parameters``.
            agent (optional): Changes the ``context.solver.agent`` parameter.
                Supported agents include:

                - ``docloud``: forces the solve operation to use DOcplexcloud
                - ``local``: forces the solve operation to use native CPLEX

            url (optional): Overwrites the URL of the
                DOcplexcloud service defined by ``context.solver.docloud.url``.
            key (optional): Overwrites the
                authentication key of the DOcplexcloud service defined by
                ``context.solver.docloud.key``.
            log_output (optional): if ``True``, solver logs are output to
                stdout. If this is a stream, solver logs are output to that
                stream object. Overwrites the ``context.solver.log_output``
                parameter.
            proxies (optional): a dict with the proxies mapping.
        Returns:
            A :class:`docplex.mp.solution.SolveSolution` object if the solve operation succeeded.
            None if the solve operation failed.
        """
        if not self.is_optimized():
            self.info("No objective to optimize - searching for a feasible solution")

        context = self.prepare_actual_context(**kwargs)

        # log stuff
        saved_context_log_output = self.context.solver.log_output
        saved_log_output_stream = self.get_log_output()

        try:
            self.set_log_output(context.solver.log_output)

            forced_docloud = self._must_use_docloud(context, **kwargs)

            have_credentials = False
            if context.solver.docloud:
                have_credentials, error_message = context.solver.docloud.check_credentials()
                if error_message is not None:
                    warnings.warn(error_message, stacklevel=2)
            if forced_docloud:
                if have_credentials:
                    return self._solve_cloud(context)
                else:
                    self.fatal("DOcplexcloud context has no valid credentials: {0!s}",
                               context.solver.docloud)
            # from now on docloud_context is None
            elif self.environment.has_cplex:
                # if CPLEX is installed go for it
                return self._solve_local(context)
            elif have_credentials:
                # no context passed as argument, no Cplex installed, try model's own context
                return self._solve_cloud(context)
            else:
                # no way to solve.. really
                return self.fatal("CPLEX DLL not found: please provide DOcplexcloud credentials")
        finally:
            if saved_log_output_stream != self.get_log_output():
                self.set_log_output_as_stream(saved_log_output_stream)
            if saved_context_log_output != self.context.solver.log_output:
                self.context.solver.log_output = saved_context_log_output

    def _connect_progress_listeners(self):
        # INTERNAL... connect progress listeners (if any) if problem is mip
        if self._progress_listeners:
            if self._solves_as_mip():
                self.__engine.connect_progress_listeners(self._progress_listeners)
            else:
                self.info("Model: \"{}\" is not a MIP problem, progress listeners are disabled", self.name)

    def _disconnect_progress_listeners(self):
        for pl in self._progress_listeners:
            pl.disconnect()

    def _notify_solve_hit_limit(self, solve_details):
        # INTERNAL
        if solve_details.has_hit_limit():
            self.info("solve: {0}".format(solve_details.status))

    def _solve_local(self, context):
        """ Starts a solve operation on the local machine.

        Note:
        This method will never try to solve on DOcplexcloud, regardless of whether the model
        has an attached DOcplexcloud context.
        If CPLEX is not available, an error is raised.

        Args:
            context: a (possibly new) context whose parameters override those of the modle
                during this solve.

        Returns:
            A Solution object if the solve operation succeeded, None otherwise.

        """
        parameters = context.cplex_parameters

        # threads limitation
        if getattr(context.solver, 'max_threads', None) is not None:
            if parameters.threads.get() == 0:
                max_threads = context.solver.max_threads
            else:
                max_threads = min(context.solver.max_threads,
                                  parameters.threads.get())
            # we don't want to duplicate parameters unnecessary
            if max_threads != parameters.threads.get():
                parameters = parameters.copy()
                parameters.threads = max_threads
                out_stream = context.solver.log_output_as_stream
                if out_stream:
                    out_stream.write(
                        "WARNING: Number of workers has been reduced to %s to comply with platform limitations.\n" % max_threads)

        self.notify_start_solve()

        self_solve_hooks = self._solve_hooks
        auto_publish_details = is_auto_publishing_solve_details(context)
        auto_publish_solution = is_auto_publishing_json_solution(context)
        if auto_publish_details:
            # don't bother adding the solve hook if not in auto publish mode
            wk_hook = get_solve_hook() if get_solve_hook else None
            if wk_hook is not None:
                self._add_solve_hook(wk_hook)
        else:
            wk_hook = None

        # connect progress listeners (if any) if problem is mip
        self._connect_progress_listeners()

        # call notifyStart on progress listeners
        self._fire_start_solve_listeners()
        # notify hooks only on solve_local

        if auto_publish_details and self_solve_hooks:
            self_stats = self.get_statistics()
            for h in self_solve_hooks:
                h.notify_start_solve(self, self_stats)  # pragma : no cover

        # --- solve is protected in try/except block
        has_solution = False
        reported_obj = 0
        engine_status = self._unknown_status
        self_engine = self.__engine

        if parameters is not self.parameters:
            saved_params = {p: p.get() for p in self.parameters}
        else:
            saved_params = {}

        new_solution = None
        try:
            used_parameters = parameters or self.parameters
            assert used_parameters is not None
            self._apply_parameters_to_engine(used_parameters)

            new_solution = self_engine.solve(self, parameters=parameters)
            has_solution = new_solution is not None
            self._set_solution(new_solution)
            reported_obj = self._reported_objective_value()

            # store solve status as returned by the engine.
            engine_status = self_engine.get_solve_status()
            self._last_solve_status = engine_status

        except DOcplexException as docpx_e:
            self._set_solution(None)
            raise docpx_e

        finally:
            solve_details = self_engine.get_solve_details()
            self._notify_solve_hit_limit(solve_details)
            self._solve_details = solve_details
            self._fire_end_solve_listeners(has_solution, reported_obj)
            self._disconnect_progress_listeners()

            # call hooks
            if auto_publish_details:
                for h in self_solve_hooks:
                    h.notify_end_solve(self, has_solution, engine_status, reported_obj,
                                       self._make_end_infodict())  # pragma : no cover
                    h.update_solve_details(solve_details.as_worker_dict())  # pragma: no cover

            # save solution
            if auto_publish_solution and new_solution is not None:
                with get_environment().get_output_stream("solution.json") as output:
                    output.write(new_solution.export_as_string(format="json").encode('utf-8'))

            # unplug worker hook if any
            if wk_hook:
                self_solve_hooks.remove(wk_hook)  # pragma : no cover

            # restore parameters in sync with model, if necessary
            if saved_params:
                for p, v in six.iteritems(saved_params):
                    self_engine.set_parameter(p, v)

        return new_solution

    def get_solve_status(self):
        """ Returns the solve status of the last successful solve.

        If the model has been solved successfully, returns the status stored in the
        model solution. Otherwise returns None`.

        :returns: The solve status of the last successful solve, a string, or None.
        """
        return self._last_solve_status

    def _solve_cloud(self, context):
        docloud_context = context.solver.docloud
        parameters = context.cplex_parameters
        # see if we can reuse the local docloud engine if any?
        docloud_engine = self._engine_factory.new_docloud_engine(model=self,
                                                                 docloud_context=docloud_context,
                                                                 log_output=context.solver.log_output_as_stream)

        self.notify_start_solve()
        self._fire_start_solve_listeners()
        new_solution = docloud_engine.solve(self, parameters=parameters)
        self._set_solution(new_solution)
        self._solve_details = docloud_engine.get_solve_details()

        # store solve status as returned by the engine.
        self._last_solve_status = docloud_engine.get_solve_status()
        reported_obj = self._reported_objective_value()

        solve_details = docloud_engine.get_solve_details()
        self._notify_solve_hit_limit(solve_details)
        self._solve_details = solve_details
        self._fire_end_solve_listeners(new_solution is not None, reported_obj)
        # return new_solution in all cases: either None or a solution instance
        return new_solution

    def solve_cloud(self, context=None):
        # Starts execution of the model on the cloud.
        #
        # This method accepts a context (an instance of Context) to be used when
        # solving on the cloud. If the context argument is None or invalid, then it will
        # use the model's own instance of Context, set at model creation time.
        #
        # Note:
        #    This method will always solve the model on the cloud, whether or not CPLEX
        #    is available on the local machine.
        #
        # Args:
        #    context: An optional context to use on the cloud. If None, uses the model's Context instance, if any.
        #
        # :returns: A :class:`docplex.mp.solution.SolveSolution` object if the solve operation succeeded, else None.
        if not context:
            if self.context.solver.docloud:
                if isinstance(self.__engine, DOcloudEngine):
                    return self.solve()
                else:
                    return self._solve_cloud(self.context)
            else:
                self.fatal("context is None: cannot solve on the cloud")
        elif context.solver.docloud.has_credentials():
            return self._solve_cloud(context)
        else:
            self.fatal("DOcplexcloud context has no valid credentials: {0!s}", context.solver.docloud)

    def _solve_anywhere(self, context, force_cloud):
        # INTERNAL
        if force_cloud:
            return self._solve_cloud(context)
        else:
            return self._solve_local(context)

    def notify_start_solve(self):
        # INTERNAL
        self._solve_count += 1

    def notify_solve_failed(self):
        pass

    def get_solve_details(self):
        """
        This property returns detailed information about the last solve,  an instance of :class:`docplex.mp.solution.SolveDetails`.

        Note:
            Detailed information is returned whether or not the solve succeeded.
        """
        from copy import copy as shallow_copy

        return shallow_copy(self._solve_details)

    solve_details = property(get_solve_details)

    def notify_solve_relaxed(self, relaxed_solution, solve_details):
        # INTERNAL: used by relaxer
        self._solve_details = solve_details
        self._set_solution(relaxed_solution)
        if relaxed_solution is not None:
            self.notify_start_solve()
        else:
            self.notify_solve_failed()

    def _resolve_sense(self, sense_arg):
        """
        INTERNAL
        :param sense_arg:
        :return:
        """
        return ObjectiveSense.parse(sense_arg, self.error_handler, default_sense=None)  # raise if invalid

    def solve_lexicographic(self, goals,
                            senses=ObjectiveSense.Minimize,
                            abs_tolerance=1e-5,
                            relative_tolerance=1e-4,
                            dump_pass_files=False,
                            **kwargs):
        """ Performs a lexicographic solve from an ordered collection of goals.

        :param goals: An ordered collection of linear expressions.

        :param senses: Either an ordered collection of senses, one sense, or None. The default is None,
         in which case the solve uses a Minimize sense. Each sense can be either a sense object,
         that is either `ObjectiveSense.Minimize` or `Maximize`, or a string "min" or "max".

        :param abs_tolerance: A floating-point number (default is 1e-5) used as the absolute slack in
         intermediate constraints set to keep the previous passes' objectives.

        :param relative_tolerance: A floating-point number (default is 1e-5) used as the relative slack in
         intermediate constraints set to keep the previous passes' objectives. The slack used is the maximum
         of the absolute slack and the relative slack times the objective of the pass.

        Return:
            Boolean: True if all passes ran successfully.
        """

        old_objective_expr = self.__objective_expr
        old_objective_sense = self.__objective_sense
        if not goals:
            self.error("solve_lexicographic requires a non-empty list of goals")
            return False

        if not is_indexable(goals):
            self.fatal("solve_lexicographic requires an indexable collection of goals, got: {0!s}", goals)

        pass_count = 0
        m = self
        current_sol = None
        # currentObjective = -1
        nested_kpi_format = '   - %s ='
        results = []
        actual_goals = []
        tolerance_scheme = _ToleranceScheme(abs_tolerance, relative_tolerance)
        # keep extra constraints, in order to remove them at the end.
        extra_cts = []
        for gi, g in enumerate(goals):
            if isinstance(g, tuple):
                goal_expr = self._to_linear_expr(g[0])
                goal_name = g[1] or goal_expr.name
            else:
                try:
                    goal_expr = self._lfactory._to_expr(g)
                except AttributeError:
                    goal_expr = None
                    self.fatal("Cannot interpret this as a goal: {0!s}", g)
                try:
                    goal_name = g.name
                except AttributeError:
                    goal_name = None
                if not goal_name:
                    goal_name = "pass%d" % (gi + 1)
            actual_goals.append((goal_name, goal_expr))

        nb_goals = len(actual_goals)
        # --- senses ---
        if senses is None:
            senses = generate_constant(ObjectiveSense.Minimize, count_max=nb_goals)
        elif isinstance(senses, ObjectiveSense):
            senses = generate_constant(senses, count_max=nb_goals)
        elif is_string(senses):
            senses = generate_constant(senses, count_max=nb_goals)
        elif is_iterable(senses):
            pass
        else:
            self.fatal("solve_lexicographic expects as senses: None, min/max or iterable, got: {0!s}", senses)
        iter_senses = iter(senses)
        # --- senses ---
        prev_step = (None, None, None)

        try:

            for goal_name, goal_expr in actual_goals:
                if goal_expr.is_constant() and pass_count > 1:
                    self.warning("Constant expression in lexicographic solve: {0!s}, skipped", goal_expr)
                    continue
                pass_count += 1

                if pass_count > 1:
                    prev_goal, prev_obj, prev_sense = prev_step
                    tolerance = tolerance_scheme.compute_tolerance(prev_obj)
                    if prev_sense.is_minimize():
                        pass_ct = m.add_constraint(prev_goal <= prev_obj + tolerance,
                                                   '_ctlex_le_pass_%d' % (pass_count - 1))
                    else:
                        pass_ct = m.add_constraint(prev_goal >= prev_obj - tolerance,
                                                   '_ctlex_ge_pass_%d' % (pass_count - 1))

                    extra_cts.append(pass_ct)

                next_sense = next(iter_senses)
                sense = self._resolve_sense(next_sense)

                self.trace("lexicographic: starting pass %d, %s: %s", pass_count, sense.action().lower(), goal_name)
                m.set_objective(sense, goal_expr)
                # print("-- current objective is: {0!s}".format(goal_expr))
                if dump_pass_files:
                    pass_basename = 'lexico_%s_pass%d' % (self.name, pass_count)
                    self.dump_as_lp(basename=pass_basename)
                current_sol = m.solve(**kwargs)
                if current_sol is not None:
                    current_obj = m.objective_value
                    results.append(current_obj)
                    if m.error_handler.prints_trace():
                        self.trace("lexicographic: pass #%d ok with objective=%.4f" % (pass_count, current_obj))
                        m._report_lexicographic_goals(actual_goals, kpi_header_format=nested_kpi_format)

                    prev_step = (goal_expr, current_obj, sense)
                    # tolerance = tolerance_scheme.compute_tolerance(current_obj)
                    # prev_step = (goal_expr, current_obj, sense)
                    # if sense.is_minimize():
                    #     pass_ct = m.add_constraint(goal_expr <= current_obj + tolerance,
                    #                                '_ctlex_le_pass_%d' % pass_count)
                    # else:
                    #     pass_ct = m.add_constraint(goal_expr >= current_obj - tolerance,
                    #                                '_ctlex_ge_pass_%d' % pass_count)
                    #
                    # # print(">>>> new pass ct is {0!s}".format(pass_ct))
                    # extra_cts.append(pass_ct)
                else:
                    self.error("lexicographic fails. pass #%d, stop!", pass_count)
                    break
        finally:
            # print("-> start restoring model at end of lexicographic")
            while extra_cts:
                # using LIFO logic to avoid holes in indices.
                ct_to_remove = extra_cts.pop()
                # print("* removing constraint: name: {0}, idx: {1}".format(ct_to_remove.name, ct_to_remove.index))
                self._remove_constraint_internal(ct_to_remove)
            # restore objective whatsove
            self.set_objective(old_objective_sense, old_objective_expr)
            # print("<- end restoring model at end of lexicographic")

        if current_sol:
            self.info("lexicographic ok, #passes={0}, results={1!s}", pass_count, results)
            if m.error_handler.prints_info():
                m._report_lexicographic_goals(actual_goals, kpi_header_format=nested_kpi_format)
        else:
            self.warning("lexicographic failed at pass {0}", pass_count)

        # return a solution or None
        return current_sol

    def _has_solution(self):
        # INTERNAL
        return self.__solution is not None

    def _set_solution(self, new_solution):
        """
        INTERNAL: Sets this solution as the model's current solution.
        Copies values to variables (for now, let's think more about this)
        :param new_solution:
        :return:
        """
        self.__solution = new_solution

    def check_has_solution(self):
        # see if we can refine messages here...
        if not self._has_solution():
            self.fatal("Model<{0}> did not solve successfully", self.name)

    def add_mip_start(self, mip_start_sol):
        """  Adds a (possibly partial) solution to use as a starting point for a MIP.

        This is valid only for models with binary or integer decision variables.
        The given solution must contain the value for at least one binary or integer variable.

        This feature is also known as warm start.

        Args:
            mip_start_sol (:class:`docplex.mp.solution.SolveSolution`): The solution object to use as a starting point.

        """
        if not self._solves_as_mip():
            self.error("Problem is not a MIP, cannot use MIP start")
            return
        else:
            try:
                mip_start_sol.check_as_mip_start()
                self._mipstarts.append(mip_start_sol)
            except AttributeError:
                self.fatal("add_mip_starts expects solution, got: {0!r}", mip_start_sol)

    @property
    def objective_value(self):
        """ This property returns the value of the objective expression in the solution of the last solve.

        Raises an exception if the model has not been solved successfully.

        """
        self.check_has_solution()
        return self._objective_value()

    def _objective_value(self):
        return self.solution.objective_value

    def _reported_objective_value(self, failure_obj=0):
        return self.solution.objective_value if self.solution else failure_obj

    def _resolve_path(self, path_arg, basename_arg, extension):
        # INTERNAL
        if is_string(path_arg):
            if os.path.isdir(path_arg):
                if path_arg == ".":
                    path_arg = os.getcwd()
                return self._make_output_path(extension, basename_arg, path_arg)
            else:
                # add extension if not present (but not twice!)
                return path_arg if path_arg.endswith(extension) else path_arg + extension
        else:
            assert path_arg is None
            return self._make_output_path(extension, basename_arg, path_arg)

    def _make_output_path(self, extension, basename, path=None):
        return make_output_path2(self.name, extension, basename, path)

    lp_format = LP_format

    def _get_printer(self, exchange_format, do_raise=False):
        """
        :param exchange_format: The format to be used.
        :param do_raise:  A Boolean, raise exception if format is unuspported.
        :return: printer or None.
        """
        printer = ModelPrinterFactory.new_printer(exchange_format, do_raise=False)
        if not printer:
            if do_raise:
                self.fatal("Unsupported output format: {0!s}", exchange_format)
            else:
                self.error("Unsupported output format: {0!s}", exchange_format)
        return printer

    def dump_as_lp(self, path=None, basename=None):
        return self.dump(path, basename, exchange_format=LP_format)

    def export_as_lp(self, path=None, basename=None, hide_user_names=False):
        """ Exports a model in LP format.

        Args:
            basename: Controls the basename with which the model is printed.
                Accepts None, a plain string, or a string format.
                if None, uses the model's name;
                if passed a plain string, the string is used in place of the model's name;
                if passed a string format (either with %s or {0}, it is used to format the
                model name to produce the basename of the written file.

            path: A path to write file, expects a string path or None.
                can be either a directory, in which case the basename
                that was computed with the basename argument, is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempfile.gettempdir()``.

            hide_user_names: A Boolean indicating whether or not to keep user names for
                variables and constraints. If True, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ... for constraints.

        Examples:
            Assuming the model's name is `mymodel`:
            
            >>> m.export_as_lp()
            
            will write ``mymodel.lp`` in ``gettempdir()``.
            
            >>> m.export_as_lp(basename="foo")
            
            will write ``foo.lp`` in ``gettempdir()``.
            
            >>> m.export_as_lp(basename="foo", path="e:/home/docplex")
            
            will write file ``e:/home/docplex/foo.lp``.
            
            >>> m.export_as_lp("e/home/docplex/bar.lp")
            
            will write file ``e:/home/docplex/bar.lp``.
            
            >>> m.export_as_lp(basename="docplex_%s", path="e/home/") 
            
            will write file ``e:/home/docplex/docplex_mymodel.lp``.
        """
        return self.export(path, basename, hide_user_names, exchange_format=LP_format)

    def export_as_sav(self, path=None, basename=None):
        """ Exports a model in CPLEX SAV format.

        Exporting to SAV format requires that CPLEX is installed and
        available in PYTHONPATH. If the CPLEX DLL cannot be found, an exception is raised.

        Args:
            basename: Controls the basename with which the model is printed.
                Accepts None, a plain string, or a string format.
                If None, the model's name is used.
                If passed a plain string, the string is used in place of the model's name.
                If passed a string format (either with %s or {0}), it is used to format the
                model name to produce the basename of the written file.

            path: A path to write the file, expects a string path or None.
                Can be a directory, in which case the basename
                that was computed with the basename argument, is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempfile.gettempdir()``.

        Examples:
            See the documentation of  :func:`export_as_lp` for examples of pathname generation.
            The logic is identical for both methods.

        """
        return self._export(path, basename, use_engine=True,
                            hide_user_names=False,
                            exchange_format=SAV_format)

    def dump(self, path=None, basename=None, hide_user_names=False, exchange_format=LP_format):
        return self._export(path, basename,
                            use_engine=True,
                            hide_user_names=hide_user_names,
                            exchange_format=exchange_format)

    def export(self, path=None, basename=None,
               hide_user_names=False, exchange_format=LP_format):
        # INTERNAL
        return self._export(path, basename,
                            hide_user_names=hide_user_names,
                            exchange_format=exchange_format)

    def _export(self, path=None, basename=None,
                use_engine=False, hide_user_names=False,
                exchange_format=LP_format):
        # INTERNAL
        # path is either a nonempty path string or None
        self._checker.typecheck_string(path, accept_none=True, accept_empty=False)
        self._checker.typecheck_string(basename, accept_none=True, accept_empty=False)
        # INTERNAL
        extension = ""
        try:
            extension = exchange_format.extension
        except AttributeError:
            self.fatal("Not a supported exchange format: {0!s}", exchange_format)

        # combination of path/directory and basename resolution are done in resolve_path
        path = self._resolve_path(path, basename, extension)
        ret = self._export_to_path(path, hide_user_names, use_engine, exchange_format)
        if ret:
            self.trace("model file: {0} overwritten", path)
        return ret

    def _export_to_path(self, path, hide_user_names=False, use_engine=False, exchange_format=LP_format):
        # INTERNAL
        self.ensure_setup()
        try:
            if use_engine:
                # rely on engine for the dump
                self_engine = self.__engine
                if self_engine.has_cplex():
                    self_engine.dump(path)
                else:
                    self.fatal(
                        "Format: {0} requires CPLEX, but a local CPLEX installation could not be found, file: {1} could not be written",
                        exchange_format.name, path)
                    return None
            else:
                # a path is not a stream but anyway it will work
                self._export_to_stream(stream=path, hide_user_names=hide_user_names, exchange_format=exchange_format)
            return path

        except IOError:
            self.error("Cannot open file: \"{0}\", model: {1} not exported".format(path, self.name))
            raise

    def _export_to_stream(self, stream, hide_user_names=False, exchange_format=LP_format):
        printer = self._get_printer(exchange_format, do_raise=True)
        if printer:
            printer.forget_user_names = hide_user_names
            printer.printModel(self, stream)

    def export_to_stream(self, stream, hide_user_names=False, exchange_format=LP_format):
        """ Export the model to an output stream in LP format.

        A stream can be one of:
            - a string, interpreted as a system path,
            - None, interpreted as `stdout`, or
            - a Python file-type object (e.g. a StringIO() instance).
                
        Args:
            stream: An object defining where the output will be sent.
            
            hide_user_names: An optional Boolean indicating whether or not to keep user names for
                variables and constraints. If True, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ,... for constraints. Default is to keep user names.

        """
        self._export_to_stream(stream, hide_user_names, exchange_format)

    def export_as_lp_string(self, hide_user_names=False):
        """ Exports the model to a string in LP format.

        The output string contains the model in LP format.

        Args:
            hide_user_names: An optional Boolean indicating whether or not to keep user names for
                variables and constraints. If True, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ... for constraints. Default is to keep user names.

        Returns:
            A string, containing the model exported in LP format.
        """
        return self.export_to_string(hide_user_names, LP_format)

    def export_to_string(self, hide_user_names=False, exchange_format=LP_format):
        # INTERNAL
        oss = StringIO()
        self._export_to_stream(oss, hide_user_names, exchange_format)
        return oss.getvalue()

    def export_parameters_as_prm(self, path=None, basename=None):
        # path is either a nonempty path string or None
        self._checker.typecheck_string(path, accept_none=True, accept_empty=False)
        self._checker.typecheck_string(basename, accept_none=True, accept_empty=False)

        # combination of path/directory and basename resolution are done in resolve_path
        prm_path = self._resolve_path(path, basename, extension='.prm')
        self.parameters.export_prm_to_path(path=prm_path)
        return prm_path

    # advanced values
    def _get_engine_attribute(self, arg, attr_name):
        """
        Queries and returns some solve attribute from the last solve.
        :param arg: Object or iterable.
        :param attr_name: The name of the attribute.
        :return:
        """
        attribute = SolveAttribute.parse(attr_name, do_raise=True)
        if is_iterable(arg):
            if not arg:
                return []
            else:
                return self._get_engine_attributes_internal(arg, attribute)
        else:
            # defer checking to the checker
            if attribute.requires_vars:
                self._checker.typecheck_var(arg)
            else:
                self._checker.typecheck_constraint(arg)
            attrs = self._get_engine_attributes_internal([arg], attribute)
            return attrs[0]

    def _get_engine_attributes_internal(self, mobjs, attribute):
        attr_name = attribute.name
        is_for_vars = attribute.requires_vars
        if is_for_vars:
            indices = [v.index for v in self.iter_variables()]
        else:
            indices = [ct.index for ct in self.iter_constraints()]
        if not self.solution.is_attributes_fetched(attribute.name):
            # get index to value from engine
            attr_idx_map = self.__engine.get_solve_attribute(attr_name, indices)
            self.solution._store_attribute_result(attr_name, attr_idx_map, is_for_vars)
        return self.solution.get_attribute(mobjs, attr_name)

    def dual_values(self, cts):
        self.check_has_solution()
        duals = self._get_engine_attribute(cts, 'duals')
        return duals

    def slack_values(self, cts):
        self.check_has_solution()
        return self._get_engine_attribute(cts, 'slacks')

    def reduced_costs(self, dvars):
        self.check_has_solution()
        if self._solves_as_mip():
            self.fatal('reduced costs are only available for LP problems')
        return self._get_engine_attribute(dvars, 'reduced_costs')

    DEFAULT_VAR_VALUE_QUOTED_SOLUTION_FMT = '  \"{varname}\"={value:.{prec}f}'
    DEFAULT_VAR_VALUE_UNQUOTED_SOLUTION_FMT = '  {varname}={value:.{prec}f}'
    DEFAULT_OBJECTIVE_FMT = "objective: {0:.{prec}f}"

    def _has_username_with_spaces(self):
        for v in self.iter_variables():
            if v.has_user_name() and v.name.find(" ") >= 0:
                return True
        else:
            return False

    def print_solution(self, print_zeros=False,
                       objective_fmt=DEFAULT_OBJECTIVE_FMT,
                       var_value_fmt=None,
                       **kwargs):
        """  Prints the values of the model variables after a solve.

        Only valid after a successful solve. If the model has not been solved successfully, an
        exception is raised.

        Args:
            print_zeros (Boolean): If False, only non-zero values are printed. Default is False.

            objective_fmt : A format string in format syntax. The default printout is objective: xx where xx is formatted as a float with prec digits. The value of prec is computed automatically by DOcplex, either 0 if the objective expression is discrete or the model's float precision.

            var_value_fmt : A format string to format the variable name and value. Again, the default uses the automatically computed precision.

        """
        self.check_has_solution()
        if var_value_fmt is None:
            if self._has_username_with_spaces():
                var_value_fmt = self.DEFAULT_VAR_VALUE_QUOTED_SOLUTION_FMT
            else:
                var_value_fmt = self.DEFAULT_VAR_VALUE_UNQUOTED_SOLUTION_FMT
        if not self.has_objective():
            var_value_fmt = var_value_fmt[2:]
        # scope of variables.
        iter_vars = self.iter_variables() if print_zeros else None
        # if some username has a whitespace, use quoted format
        self.solution.display(print_zeros=print_zeros,
                              header_fmt=None,
                              value_fmt=var_value_fmt,
                              iter_vars=iter_vars, **kwargs)

    def report(self):
        """  Prints the value of the objective and the KPIs.
        Only valid after a successful solve, otherwise states that the model is not solved.
        """
        if self._has_solution():
            used_prec = 0 if self.objective_expr.is_discrete() else self.get_float_precision()
            self.info("model solved with objective: {0:.{prec}f}".format(self._objective_value(), prec=used_prec))
            self.report_kpis()
        else:
            self.info("Model has not been solved successfully, no reporting done.")

    def report_kpis(self, selected_kpis=None, kpi_header_format='* KPI: %s='):
        """  Prints the values of the KPIs.
        Only valid after a successful solve.
        """
        kpi_format = kpi_header_format + self._continuous_var_format  # be safe even integer KPIs might yield floats
        printed_kpis = selected_kpis if is_iterable(selected_kpis) else self.iter_kpis()
        for kpi in printed_kpis:
            kpi_value = kpi.compute()
            print(kpi_format % (kpi.name, kpi_value))

    def _report_lexicographic_goals(self, goal_name_values, kpi_header_format):
        # INTERNAL
        kpi_format = kpi_header_format + self._continuous_var_format  # be safe even integer KPIs might yield floats
        printed_kpis = goal_name_values if is_iterable(goal_name_values) else self.iter_kpis()
        for goal_name, goal_expr in printed_kpis:
            goal_value = goal_expr.solution_value
            print(kpi_format % (goal_name, goal_value))

    def iter_kpis(self):
        """ Returns an iterator over all KPIs in the model.

        Returns:
           An iterator object.
        """
        return iter(self._allkpis)

    def kpi_by_name(self, name, try_match=True, match_case=False, do_raise=True):
        """ Fetches a KPI from a string.

        This method fetches a KPI from a string, using either exact naming or trying
        to match a substring of the KPI name.

        Args:
            name (string): The string to be matched.
            try_match (Boolean): If True, returns KPI whose name is not equal to the
                argument, but contains it. Default is True.
            match_case: If True, looks for a case-exact match, else ignores case. Default is False.
            do_raise: If True, raise an exception when no KPI is found.

        Example:
            If the KPI name is "Total CO2 Cost" then fetching with argument `co2` and `match_case` to False
            will succeed. If `match_case` is True, then no KPI will be returned.

        Returns:
            The KPI expression if found. If the search fails, either raises an exception or returns a dummy
            constant expression with 0.
        """
        for kpi in iter(reversed(self._allkpis)):
            kpi_name = kpi.name
            ok = False
            if kpi_name == name:
                ok = True
            elif try_match:
                if match_case:
                    ok = kpi_name.find(name) >= 0
                else:
                    ok = kpi_name.lower().find(name.lower()) >= 0
            if ok:
                return kpi
        else:
            if do_raise:
                self.fatal('Cannot find any KPI matching: "{0:s}"', name)
            else:
                return self._lfactory.zero_expr

    def kpi_value_by_name(self, name, try_match=True, match_case=False, do_raise=True):
        """ Returns a KPI value from a KPI name.

        This method fetches a KPI value from a string, using either exact naming or trying
        to match a substring of the KPI name.

        Args:
            name (str): The string to be matched.
            try_match (Bool): If True, returns KPI whose name is not equal to the
                argument, but contains it. Default is True.
            match_case: If True, looks for a case-exact match, else ignores case. Default is False.
            do_raise: If True, raise an exception when no KPI is found.

        Example:
            If the KPI name is "Total CO2 Cost" then fetching with argument `co2` and `match_case` to False
            will succeed. If `match_case` is True, then no KPI will be returned.

        Note:
            Expression KPIs require a valid solution to be computed. Make sure that the model contains
            a valid solution before computing an expression-based KPI, otherwise an exception will be raised.
            Functional KPIs do not require a valid solution.

        Returns:
            float: The KPI value.
        """
        kpi = self.kpi_by_name(name, try_match, match_case=match_case, do_raise=do_raise)
        return kpi.compute()

    def add_kpi(self, kpi_arg, publish_name=None):
        """ Adds a Key Performance Indicator to the model.

        Key Performance Indicators (KPIs) are objects that can be evaluated after a solve().
        Typical use is with decision expressions, the evaluation of which return the expression's solution value.

        KPI values are displayed with the method :func:`report_kpis`.

        Args:
            kpi_arg:  Accepted arguments are either an expression, a lambda function with one argument or
                an instance of a subclass of abstract class KPI.

            publish_name (string, optional): The published name of the KPI.

        Note:
            If no publish_name is provided, DOcplex will use the name of the argument. If none,
            it will use the string representation of the argument.

        Example:
            `model.add_kpi(x+y+z, "Total Profit")` adds the expression `(x+y+z)` as a KPI with the name "Total Profit".

            `model.add_kpi(x+y+z)` adds the expression `(x+y+z)` as a KPI with
            the name "x+y+z", assumng variables x,y,z have names 'x', 'y', 'z' (resp.)

        Returns:
            The newly added KPI instance.

        See Also:
            :class:`docplex.mp.kpi.KPI`,
            :class:`docplex.mp.kpi.DecisionKPI`
        """
        new_kpi = self._lfactory.new_kpi(kpi_arg, publish_name)
        new_kpi_name = new_kpi.get_name()
        for kp in self.iter_kpis():
            if kp.name == new_kpi_name:
                self.warning("Duplicate KPI name \"{0!s}\" ", new_kpi_name)
        self._allkpis.append(new_kpi)
        return new_kpi

    def remove_kpi(self, kpi_arg):
        """ Removes a Key Performance Indicator from the model.

        Args:
            kpi_arg:  A KPI instance that was previously added to the model. Accepts either a KPI object or a string.
                If passed a string, looks for a KPI with that name.

        See Also:
            :func:`add_kpi`
            :class:`docplex.mp.kpi.KPI`,
            :class:`docplex.mp.kpi.DecisionKPI`
        """
        if is_string(kpi_arg):
            kpi = self.kpi_by_name(kpi_arg)
            if kpi:
                self._allkpis.remove(kpi)
        else:
            for k, kp  in enumerate(self._allkpis):
                if kp is kpi_arg:
                    kx = k
                    break
            else:
                kx = -1
            if kx >= 0:
                self._allkpis.pop(kx)
            else:
                self.warning('Model.remove_kpi() cannot interpret this either as a string or as a KPI: {0!r}', kpi_arg)


    def clear_kpis(self):
        ''' Clears all KPIs defined in the model.

        
        '''
        self._allkpis = []

    @property
    def number_of_kpis(self):
        return len(self._allkpis)

    def add_progress_listener(self, listener):
        self._checker.typecheck_progress_listener(listener)
        self._progress_listeners.append(listener)

    def remove_progress_listener(self, listener):
        self._progress_listeners.remove(listener)

    def _fire_start_solve_listeners(self):
        for l in self._progress_listeners:
            l.notify_start()

    def _fire_end_solve_listeners(self, has_solution, objective_value):
        for l in self._progress_listeners:
            l.notify_end(has_solution, objective_value)

    def fire_jobid(self, jobid):
        for l in self._progress_listeners:
            l.notify_jobid(jobid)

    def fire_progress(self, progress_data):
        for l in self._progress_listeners:
            l.notify_progress(progress_data)

    def clear_progress_listeners(self):
        self._progress_listeners = []

    def prettyprint(self):
        ModelPrinterFactory.new_pretty_printer().printModel(self)

    def clone(self, new_name=None):
        """ Makes a deep copy of the model, possibly with a new name.
        Decision variables, constraints, and objective are copied.

        Args:
            new_name (string): The new name to use. If None is provided, returns a "Copy of xxx" where xxx is the original model name.

        :returns: A new model.

        :rtype: :class:`docplex.mp.model.Model`
        """
        return self.copy(new_name)

    def copy(self, copy_name=None, removed_cts=None):
        # INTERNAL
        actual_copy_name = copy_name or "Copy of %s" % self.name
        # copy kwargs
        copy_kwargs = self._get_kwargs()
        copy_model = Model(name=actual_copy_name, **copy_kwargs)

        # clone variable containers
        for ctn in self.iter_var_containers():
            copy_model._add_var_container(ctn.copy(copy_model))

        # clone variables
        var_mapping = {}
        for v in self.iter_variables():
            copied_var = copy_model.var(v.vartype, v.lb, v.ub, v.name)
            var_mapping[v] = copied_var

        # clone constraints
        setof_removed_cts = set(removed_cts) if removed_cts else {}
        for ct in self.iter_constraints():
            if ct not in setof_removed_cts:
                copied_ct = ct.copy(copy_model, var_mapping)
                # names have been copied already
                copy_model.add_constraint(copied_ct)

        # clone objective
        if self.is_optimized():
            copy_model.set_objective(self.objective_sense, self.objective_expr.copy(copy_model, var_mapping))

        # clone kpis
        for kpi in self.iter_kpis():
            copy_model.add_kpi(kpi.copy(copy_model, var_mapping))

        if self.context:
            copy_model.context = self.context.copy()

        # clone sos
        for sos in self.iter_sos():
            copy_model._register_sos(sos.copy(copy_model, var_mapping))

        # clone params (some day)
        return copy_model

    def _is_fully_indexed(self):
        """
        INTERNAL: Returns true if all objects have valid index, else None.
        :return: True if all variables and constraints have a valid index, else False.
        """
        justifier = None
        for dv in self.iter_variables():
            if not dv.has_valid_index():
                justifier = dv
                break
        for ct in self.iter_constraints():
            if not ct.has_valid_index():
                justifier = ct
                break
        return justifier is None

    def refresh_model(self, do_setup=True):
        # compatibility with AbtractModel
        pass  # pragma : no cover

    def setup(self):
        # compatibility with AbtractModel
        pass  # pragma : no cover

    def is_free_lb(self, var_lb):
        return self._lfactory.is_free_lb(var_lb)

    def is_free_ub(self, var_ub):
        return self._lfactory.is_free_ub(var_ub)

    def _sync_constraint_indices(self, ct_iter=None):
        # INTERNAL: check only when CPLEX is present.
        self_engine = self.__engine
        if self_engine.has_cplex():
            self_engine._sync_constraint_indices(ct_iter or self.iter_constraints())

    def print_var_indices(self):
        for dvar in self.iter_variables():
            print("name: {0}, index={1}".format(dvar.name, dvar.get_index()))

    def _sync_var_indices(self):
        self_engine = self.__engine
        if self_engine.has_cplex():
            self_engine._sync_var_indices(self.iter_variables())

    def end(self):
        """ Terminates a model instance.

        Since this method destroys the objects associated with the model, you must not use the model
        after you call this member function.

        """
        self.clear()
        self._clear_engine(restart=False)

    @property
    def parameters(self):
        """ This property returns the root parameter group of the model.

        The root parameter group models the parameter hirerachy.
        It is the way to access any CPLEX parameter and get or set its value.

        Examples:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap

            Returns the parameter itself, an instance of the `Parameter` class.

            To get the value of the parameter, use the `get()` method, as in:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap.get()
               >>> 0.0001

            To change the value of the parameter, use a standard Python assignment:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap = 0.05
               model.parameters.mip.tolerances.mipgap.get()
              >>> 0.05

            Assignment is equivalent to the `set()` method:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap.set(0.02)
               model.parameters.mip.tolerances.mipgap.get()
               >>> 0.02

        Returns:
            The root parameter group, an instance of the `ParameterGroup` class.

        """
        return self.context.cplex_parameters

    def set_parameter(self, param, value):
        warnings.warn("This method is deprecated, use the parameters field to get/set parameter values.")
        param.set(value)

    def get_parameter_from_id(self, parameter_cpx_id):
        """ Finds a parameter from a CPLEX id code.

        Args:
            parameter_cpx_id: A CPLEX parameter id (positive integer, for example, 2009 is mipgap).

        :returns: An instance of :class:`docplex.mp.params.parameters.Parameter` if found, else None.
        """
        assert parameter_cpx_id >= 0
        for p in self.parameters.generate_params():
            if p.cpx_id == parameter_cpx_id:
                return p
        else:
            return None

    def apply_parameters(self):
        self._apply_parameters_to_engine(self.parameters)

    def _sync_parameters_to_engine(self, parameters):
        self._apply_parameters_to_engine(parameters)

    def _apply_parameters_to_engine(self, parameters_to_use):
        if parameters_to_use is not None:
            self_engine = self.__engine
            for param in parameters_to_use:
                self_engine.set_parameter(param, param.current_value)

    def _sync_parameters_from_engine(self, parameters):
        # INTERNAL.
        self_engine = self.__engine
        parameters_to_use = parameters or self.parameters
        for param in parameters_to_use:
            param_engine_value = self_engine.get_parameter(param)
            param.set(param_engine_value)

    def _sync_parameter_defaults_from_engine(self):
        # used when a more recent CPLEX DLL is present
        self_engine = self.__engine
        resets = []
        for param in self.parameters:
            engine_value = self_engine.get_parameter(param)
            if engine_value != param.default_value:
                resets.append((param, param.default_value, engine_value))
                param.reset_default_value(engine_value)
        return resets

    # with protocol
    def __enter__(self):
        return self

    def __exit__(self, atype, avalue, atraceback):
        # terminate the model upon exiting a 'with' block.
        self.end()

    def __iadd__(self, e):
        # implements the "+=" dialect a la PulP
        if not isinstance(e, AbstractConstraint):
            self.fatal("Model += can only be used with constraints, not with: {0!s}", e)
        else:
            self.add_constraint(ct=e, ctname=None)
            return self

    def _resync(self):
        # INTERNAL
        self._lfactory.resync_whole_model()

    def refresh_engine(self):
        self._clear_engine(restart=True)
        self._resync()

    def run_feasopt(self, relaxables, relax_mode):
        relaxable_list = _to_list(relaxables)
        groups = []
        try:
            for pref, ctseq in relaxable_list:
                self._checker.typecheck_num(pref)
                cts = _to_list(ctseq)
                for ct in cts:
                    self._checker.typecheck_constraint(ct)
                groups.append((pref, cts))
        except ValueError:
            self.fatal("expecting container with (preference, constraints), got: {0!s}", relaxable_list)

        feasible = self.__engine.solve_relaxed(mdl=self, relaxable_groups=groups,
                                               prio_name='',
                                               relax_mode=relax_mode)
        return feasible

    def add_sos1(self, dvars, name=None):
        ''' Adds  an SOS of type 1 to the model.

        Args:
            dvars: The variables in the special ordered set.
                This method only accepts ordered sequences of variables or iterators.
                Unordered collections (dictionaries, sets) are not accepted.

            name: An optional name.

        Returns:
            The newly added SOS.
        '''
        return self.add_sos(dvars, sos_arg=SOSType.SOS1, name=name)

    def add_sos2(self, dvars, name=None):
        ''' Adds  an SOS of type 2 to the model.

        Args:
           dvars: The variables in the specially ordered set.
                This method only accepts ordered sequences of variables or iterators.
                Unordered collections (dictionaries, sets) are not accepted.
           name: An optional name.

        Returns:
            The newly added SOS.
        '''
        return self.add_sos(dvars, sos_arg=SOSType.SOS2, name=name)

    def add_sos(self, dvars, sos_arg, name=None):
        ''' Adds  an SOS to the model.

        Args:
           sos_arg: The SOS type. Valid values are numerical (1 and 2) or enumerated (`SOSType.SOS1` and
              `SOSType.SOS2`).
           dvars: The variables in the special ordered set.
                This method only accepts ordered sequences of variables or iterators.
                Unordered collections (dictionaries, sets) are not accepted.
           name: An optional name.

        Returns:
            The newly added SOS.
        '''
        sos_type = SOSType.parse(sos_arg)
        msg = 'Model.add_%s() expects an ordered sequence (or iterator) of variables' % sos_type.lower()
        self._checker.check_ordered_sequence(arg=dvars, header=msg)
        var_list = _to_list(dvars)
        self._checker.typecheck_var_seq(var_list)
        if len(var_list) < sos_type.min_size():
            self.fatal("A {0:s} variable set must contain at least {1:d} variables, got: {2:d}",
                       sos_type.name, sos_type.min_size(), len(var_list))
        # creates a new sos object
        return self._add_sos(dvars, sos_type, name)

    def _add_sos(self, dvars, sos_type, name):
        # INTERNAL
        new_sos = self._lfactory.new_sos(dvars, sos_type=sos_type, name=name)
        self._register_sos(new_sos)
        return new_sos

    def _register_sos(self, new_sos):
        self._allsos.append(new_sos)

    def iter_sos(self):
        ''' Iterates over all SOS sets in the model.

        Returns:
            An iterator object.
        '''
        return iter(self._allsos)

    @property
    def number_of_sos(self):
        ''' This property returns the total number of SOS sets in the model.

        '''
        return len(self._allsos)

    def clear_sos(self):
        ''' Clears all SOS sets in the model.
        '''
        self._allsos = []

    def _generate_sos(self, sos_type):
        # INTERNAL
        for sos_set in self.iter_sos():
            if sos_set.sos_type == sos_type:
                yield sos_set

    def iter_sos1(self):
        ''' Iterates over all SOS1 sets in the model.

        Returns:
            An iterator object.
        '''
        return self._generate_sos(SOSType.SOS1)

    def iter_sos2(self):
        ''' Iterates over all SOS2 sets in the model.

        Returns:
            An iterator object.
        '''
        return self._generate_sos(SOSType.SOS2)

    @property
    def number_of_sos1(self):
        ''' This property returns the total number of SOS1 sets in the model.

        '''
        return sum(1 for _ in self.iter_sos1())

    @property
    def number_of_sos2(self):
        ''' This property returns the total number of SOS2 sets in the model.

        '''
        return sum(1 for _ in self.iter_sos2())


class AbstractModel(Model):
    def __init__(self, name, context=None, **kwargs):
        Model.__init__(self, name=name, context=context, **kwargs)

    def is_valid(self):
        """ Redefine this function to return False if some data is invalid.
        """
        return True

    def setup_variables(self):
        raise NotImplementedError

    def setup_constraints(self):
        raise NotImplementedError

    def setup_objective(self):
        ''' Redefine this method to set the objective.
        This is not mandatory as a model might not have any objective.
        '''
        pass  # pragma: no cover

    # noinspection PyMethodMayBeStatic
    def check(self):
        ''' Redefine this method to check the model before solve.
        '''
        pass

    def setup_data(self):
        pass

    def post_process(self):
        pass

    def setup(self):
        """ Setup the model artifacts, raise exception if data are not correct.
        """
        self.setup_data()
        if self.is_valid():
            self.setup_variables()
            self.setup_constraints()
            self.setup_objective()
        else:
            self.error("model is invalid, setup stops")

    def ensure_setup(self):
        if not self.is_valid():
            self.fatal("Model is not valid: {0}".format(self.name))
        if self._is_empty():
            self.setup()

    def restart(self):
        """ Called to restart the model in an empty state.

        The underlying model is also restarted to a clean and empty state.
        All modeling objects previsouly defined and stored in the model are discarded.
        """
        self.clear()
        # if the superclass does not call the parent class, make sure...
        self._clear_internal()
        self._clear_engine(restart=True)

    def refresh_model(self, do_setup=True):
        ''' Clears all model elements plus sets a new engine.'''
        self.restart()
        if do_setup:
            self.ensure_setup()

    def before_solve_hook(self):
        """ This method is called just before solve inside a run.
        Redefine to get some particular behavior.
        """
        if self.error_handler.prints_info():
            self.print_information()

    def export(self, path=None, basename=None, hide_user_names=False, exchange_format=LP_format):
        # INTERNAL: redefine export at this stage to ensure model is setup
        self.ensure_setup()
        return Model.export(self, path, basename, hide_user_names, exchange_format)

    def run_silent(self, **kwargs):
        # make sure the model is setup
        self.ensure_setup()
        # check data and model if necessary (ddefault is do nothing)
        self.check()
        # insert some last minute code before solve.
        self.before_solve_hook()
        # call solve_run which by default calls solve
        s = self.solve_run(**kwargs)
        if s:
            self.post_process()
        return s

    def solve_run(self, **kwargs):
        return self.solve(**kwargs)

    def run(self, **kwargs):
        s = self.run_silent(**kwargs)
        if s:
            self.report()
        return s
