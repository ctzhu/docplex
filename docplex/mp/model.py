# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint: disable=too-many-lines
from __future__ import print_function

from itertools import product

import warnings

from docplex.mp.context import Context
from docplex.mp.lp_printer import LPModelPrinter
from docplex.mp.mprinter import ModelPrettyPrinter
from docplex.mp.environment import Environment
from docplex.mp.error_handler import DefaultErrorHandler, InfoLevel
from docplex.mp.progress import ProgressListener
from docloud.status import JobSolveStatus
from docplex.mp.docloud_engine import DOcloudEngine
# from docplex.Utils import MyTimer


from docplex.mp.linear import *
from docplex.mp.utils import fast_range

from docplex.mp.engine_factory import EngineFactory
from docplex.mp.format import LP_format, SAV_format

from docplex.mp.kpi import KPI, DecisionKPI

import docplex.mp.worker.worker_env as wk_env


class _SolveHook(object):
    # INTERNAL
    def __init__(self):
        pass  # pragma: no cover

    def notify_start_solve(self, mdl, model_statistics):
        """ Notifies the start of a solve.

        Args:
            attributes: A dictionary of string->values with various data attributes of the model.

        """
        pass  # pragma: no cover

    def notify_end_solve(self, mdl, has_solution, status, obj, var_value_dict):
        """ Notifies the end of a solve.

        Args:
            has_solution: Boolean, True if solve returned a solution.
            status: An enumerated value of type JobSolveStatus.
            obj: The objective value if solved ok, else irrelevant.
            attributes: A dictionary of variable names to values in the solution.
        """
        pass  # pragma: no cover


class TraceSolveHook(_SolveHook):
    def notify_start_solve(self, mdl, stats):
        print("-> start solve")
        stats.print_information()

    def notify_end_solve(self, mdl, has_solution, status, obj, var_value_dict):
        if has_solution:
            print("<- solve succeeds, status={0}, obj={1}".format(status, obj))
            for vn, vv in iteritems(var_value_dict):
                print("  - var \"{0:s}\" = {1!s}".format(vn, vv))
        else:
            print("<- solve fails, status={0}".format(status))


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


class _SolveAttribute(object):
    # A generic descriptor class for engine solve attributes (e.g. dual values, reduced costs).

    def __init__(self, name, is_for_vars, requires_solve=True):
        self._name = name
        self._is_for_vars = is_for_vars
        self._requires_solved = requires_solve

    @property
    def name(self):
        return self._name

    @property
    def is_for_vars(self):
        return self._is_for_vars

    @property
    def requires_solved(self):
        return self._requires_solved


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

    @property
    def dimension_string(self):
        dim_sizes = [self.size(d) for d in range(self.nb_dimensions)]
        dim_string = "".join(["[%d]" % sz for sz in dim_sizes])
        return dim_string

    def to_string(self):
        # dvar xxx
        dim_string = self.dimension_string
        ctname = self._namer or 'x'
        return "dvar {0} {1} {2}".format(self.vartype.short_name, ctname, dim_string)

    def __str__(self):
        return self.to_string()


class _SymbolGenerator(object):
    """
    INTERNAL class
    """

    def __init__(self, pattern, initial_index=-1, offset=1):
        ''' Initialize the counter and the pattern.
            Fixes the pattern by suffixing '%d' if necessary.
        '''
        self.__pattern = pattern
        # add offset to counter. this is to avoid 0 in names
        self.__offset = offset
        self._last_index = initial_index
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
        self.notify_new_index(guessed_index)
        return coined_symbol


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
        self._number_of_le_constraints = 0
        self._number_of_ge_constraints = 0
        self._number_of_eq_constraints = 0
        self._number_of_range_constraints = 0
        self._number_of_indicator_constraints = 0

    def equal_stats(self, other):
        if not isinstance(other, ModelStatistics):
            return False

        if self.number_of_binary_variables != other.number_of_binary_variables:
            return False
        if self.number_of_integer_variables != other.number_of_integer_variables:
            return False
        if self.number_of_continuous_variables != other.number_of_continuous_variables:
            return False
        if self.number_of_linear_constraints != other.number_of_linear_constraints:
            return False
        if self.number_of_le_constraints != other.number_of_le_constraints:
            return False
        if self.number_of_eq_constraints != other.number_of_eq_constraints:
            return False
        if self.number_of_ge_constraints != other.number_of_ge_constraints:
            return False
        if self.number_of_range_constraints != other.number_of_range_constraints:
            return False
        if self._number_of_indicator_constraints != other._number_of_indicator_constraints:
            return False
        # ok all counts are ok.
        return True

    def __eq__(self, other):
        return self.equal_stats(other)

    @staticmethod
    def _make_new_stats(mdl):
        # INTERNAL
        stats = ModelStatistics()
        vartype_count = Counter(type(dv.vartype) for dv in mdl.iter_variables())
        stats._number_of_binary_variables = vartype_count[BinaryVarType]
        stats._number_of_integer_variables = vartype_count[IntegerVarType]
        stats._number_of_continuous_variables = vartype_count[ContinuousVarType]

        linct_count = Counter(ct.type for ct in mdl.iter_binary_constraints())
        stats._number_of_le_constraints = linct_count[LinearConstraintType.LE]
        stats._number_of_eq_constraints = linct_count[LinearConstraintType.EQ]
        stats._number_of_ge_constraints = linct_count[LinearConstraintType.GE]
        stats._number_of_range_constraints = mdl.number_of_range_constraints
        stats._number_of_indicator_constraints = mdl.number_of_indicator_constraints
        return stats

    @property
    def number_of_variables(self):
        """ This property returns the total number of variables in the model.

        """
        return self._number_of_binary_variables + \
               self._number_of_integer_variables + \
               self._number_of_continuous_variables

    @property
    def number_of_binary_variables(self):
        """ This property returns the number of binary variables in the model.

        """
        return self._number_of_binary_variables

    @property
    def number_of_integer_variables(self):
        """ This property returns the number of integer variables in the model.

        """
        return self._number_of_integer_variables

    @property
    def number_of_continuous_variables(self):
        """ this property returns the number of continuous variables in the model.

        """
        return self._number_of_continuous_variables

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
        """ Returns the number of indicator constraints.

        See Also:
            :class:`docplex.mp.linear.IndicatorConstraint`

        """
        return self._number_of_indicator_constraints

    def print_information(self):
        """ Prints model statistics in readable format.

        """
        print(" - number of variables: %d" % self.number_of_variables)
        print("   - binary={0}, integer={1}, continuous={2}".
              format(self.number_of_binary_variables,
                     self.number_of_integer_variables,
                     self.number_of_continuous_variables
                     ))

        print(" - number of constraints: %d" % self.number_of_linear_constraints)
        print(" -   LE={0}, EQ={1}, GE={2}, RNG={3}"
              .format(self._number_of_le_constraints,
                      self._number_of_eq_constraints,
                      self._number_of_ge_constraints,
                      self._number_of_range_constraints))
        if self._number_of_indicator_constraints:
            print(" - number of indicator constraints: %d" % self._number_of_indicator_constraints)

    def __repr__(self):
        return "docplex.mp.Model.ModelStatistics()"


# noinspection PyProtectedMember
class Model(object):
    """ This is the main class to embed modeling objects.

    The :class:`Model` class acts as a factory to create optimization objects,
    variables, and constraints.
    It provides various accessors and iterators to the modeling objects.
    It also manages solving operations and solution management.


    Args:
        name (str): The name of the model.
        output_level: An instance of enum :class:`docplex.mp.error_handler.InfoLevel`; the default is `INFO`.
        docloud_context: An instance of :class:`docplex.mp.context.DOcloudContext` to provide credentials for solving on DOcloud.

    Attributes:
        parameters: Contains the list of CPLEX parameters with their current values.

    """

    _name_generator = _SymbolGenerator(pattern="docplex_model", initial_index=0, offset=0)

    @property
    def vartype_binary(self):
        self_binary_vartype = self._binary_vartype
        if self_binary_vartype is None:
            self_binary_vartype = BinaryVarType()
            self._binary_vartype = self_binary_vartype
        return self_binary_vartype

    @property
    def vartype_integer(self):
        self_integer_vartype = self._integer_vartype
        if self_integer_vartype is None:
            self_integer_vartype = IntegerVarType()
            self._integer_vartype = self_integer_vartype
        return self_integer_vartype

    @property
    def vartype_continuous(self):
        self_continuous_vartype = self._continuous_vartype
        if self_continuous_vartype is None:
            self_continuous_vartype = ContinuousVarType()
            self._continuous_vartype = self_continuous_vartype
        return self_continuous_vartype

    def _make_environment(self):
        env = Environment.make_new_configured_env()
        # rtc-28869
        # env.numpy_hook = Model.init_numpy
        return env

    def _lazy_get_environment(self):
        if self._environment is None:
            self._environment = self._make_environment()
        return self._environment

    _saved_numpy_options = None

    @staticmethod
    def init_numpy():
        """ Static method to customize numpy for DOcplex.

        This method makes numpy aware of DOcplex.
        All numpy arrays with DOcplex objects will be printed by their string representations
        as returned by str() instead of repr() as with standard numpy behavior.

        All customizations can be removed by calling the restore_numpy method.

        Note:
            This method does nothing if numpy is not present.

        See Also:
            :func:`restore_numpy`
        """
        try:
            import numpy as np

            Model._saved_numpy_options = np.get_printoptions()
            np.set_printoptions(formatter={'numpystr': lambda f: str(f) if Model._is_operand(f) else repr(f)})
        except ImportError:
            pass

    @staticmethod
    def restore_numpy(self):
        """ Static method to restore numpy to its default state.

        This method is acompanoion method to init_numpy. It restores numpy to its original state,
        undoing all customizations that were done for DOcplex.

        Note:
            This method does nothing if numpy is not present.

        See Also:
            :func:`init_numpy`
        """
        try:
            import numpy as np

            if Model._saved_numpy_options is not None:
                np.set_printoptions(Model._saved_numpy_options)
        except ImportError:
            pass

    @property
    def environment(self):
        # for a closed model with no CPLEX, numpy, etc return ClosedEnvironment
        # return get_no_graphics_env()
        # from docplex.environment import ClosedEnvironment
        # return ClosedEnvironment
        return self._lazy_get_environment()

    def _make_key_seq(self, keys, name):
        # INTERNAL Takes as input a candidate keys input and returns a valid key sequence
        if is_iterable(keys):
            if has_len(keys):
                return name, keys
            elif is_iterator(keys):
                return name, list(keys)
            else:
                # TODO: make a test for this case.
                self.fatal("Cannot handle iterable var keys: {0!s} : no len() and not an iterator",
                           keys)  # pragma: no cover

        elif is_int(keys) and keys >= 0:
            # if name is str and we have a size, trigger automatic names
            used_name = None if name is str else name
            return used_name, range(0, keys)
        else:
            self.fatal("Unexpected var keys: {0!s}, expecting iterable or integer", keys)  # pragma: no cover

    # ---- type checking
    def typecheck_iterable(self, arg):
        # INTERNAL: checks for an iterable
        self.error_handler.ensure(is_iterable(arg), "Expecting iterable, got: {0!s}", arg)

    # safe checks.
    def typecheck_valid_index(self, arg):
        self.__error_handler.ensure(ModelingObject.is_valid_index(arg), "Not an index: {0!s}", arg)

    def typecheck_vartype(self, arg):
        # INTERNAL: check for a valid vartype
        if not isinstance(arg, VarType):
            self.error_handler.fatal("Not a variable type: {0!s}, type: {1!s}", (arg, type(arg)))
        return True

    def typecheck_var(self, obj):
        # INTERNAL: check for Var instance
        checked_is_var = isinstance(obj, Var)
        if not checked_is_var:
            self.fatal("Expecting decision variable, got: {0!s} type: {1!s}", obj, type(obj))

    def typecheck_expr(self, arg):
        self.error_handler.ensure(isinstance(arg, Expr), "Expected expression, got: {0!s}", arg)

    @staticmethod
    def _is_operand(arg, accept_number=True):
        if isinstance(arg, Expr):
            return True
        elif isinstance(arg, Var):
            return True
        elif accept_number:
            return is_number(arg)
        else:
            return False

    def typecheck_constraint(self, obj, ct_subtype=None):
        checked_type = ct_subtype or AbstractConstraint
        checked = isinstance(obj, checked_type)
        self.__error_handler.ensure(checked, "Expecting constraint of type: {0!s}, got: {1!s} with type: {2!s}",
                                    checked_type, obj, type(obj))

    def typecheck_zero_or_one(self, arg):
        if arg != 0 and arg != 1:
            self.fatal("expecting 0 or 1, got: {0!s}", arg)

    def typecheck_int(self, arg):
        self.error_handler.ensure(is_int(arg), "Expecting integer, got: {0!s}", arg)

    def typecheck_num(self, arg, caller=None):
        if caller:
            self.error_handler.ensure(is_number(arg), "{0}: Expecting number, got: {1!s}", caller, arg)
        else:
            self.error_handler.ensure(is_number(arg), "Expecting number, got: {0!s}", arg)

    def typecheck_string(self, arg, enforce_nonempty=True):
        if not is_string(arg):
            self.fatal("expecting string, got: {0!s}", arg)
        elif enforce_nonempty and 0 == len(arg):
            self.fatal("A nonempty string is not allowed here")
        else:
            pass

    # --- other checks
    def _check_in_model(self, mobj, typename):
        # produces message of the type: "constraint ... does not belong to model
        if mobj.model != self:
            self.error_handler.fatal("{0} does not belong to model {1}: {2!s}".format(typename, self.name, mobj))

    def _check_both_in_selfmodel(self, mobj1, mobj2, ctx_msg):
        # INTERNAL
        mobj1_model = mobj1._get_model()
        mobj2_model = mobj2._get_model()
        if mobj1_model != mobj2_model:
            self.error_handler.fatal("Cannot mix objects from different models in {0}. obj1={1!s}, obj2={2!s}"
                                     .format(ctx_msg, mobj1, mobj2))
        elif mobj1_model != self:
            self.error_handler.fatal("Objects do not belong to model {0}. obj1={1!s}, obj2={2!s}"
                                     .format(self, mobj1, mobj2))
        else:
            pass

    def unsupported_operator_error(self, left_arg, op, right_arg):
        self.fatal("Unsupported operation {0!s} {1!s} {2!s}, only <=, ==, >= are allowed", left_arg, op, right_arg)

    def cannot_be_used_as_quotient_error(self, quotient, numerator):
        self.fatal("{1!s} / {0!s} : operation not supported, only numbers can be quotients", quotient, numerator)

    def typecheck_as_quotient(self, e, numerator):
        if not is_number(e):
            self.cannot_be_used_as_quotient_error(e, numerator)
        else:
            float_e = float(e)
            if 0 == float_e:
                self.fatal("Zero divide on {0!s}", numerator)
            else:
                pass

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
                        "Found PLEX DLL with version: {0}, older than parameter version: {1}. Parameter setting is disabled.",
                        env_cplex_version, self.parameters.cplex_version)
                ok = False
        return ok

    def round_nearest(self, x):
        # INTERNAL
        return round_nearest_towards_infinity(x, self.infinity)

    def _parse_kwargs(self, kwargs):
        # parse some arguments from kwargs
        for arg_name, arg_val in iteritems(kwargs):
            if arg_name == "float_precision":
                self.float_precision = arg_val
            elif arg_name in frozenset({"keep_ordering", "sort_expressions"}):
                self.keep_ordering = bool(arg_val)
            elif arg_name == "format_export":
                self._format_export = bool(arg_val)
            elif arg_name in frozenset({"info_level", "output_level"}):
                self.output_level = arg_val
            elif arg_name == "solver_agent":
                self.__solver_agent = arg_val
            elif arg_name == "log_output":
                self.context.solver.log_output = arg_val
            else:
                self.warning("argument: {0:s}:{1!s} - is not recognized (ignored)", arg_name, arg_val)

    _default_varname_pattern = "_x"
    _default_ctname_pattern = "_c"
    _default_indicator_pattern = "_ic"

    def __init__(self, name=None,
                 output_level=InfoLevel.INFO,
                 context=None,
                 docloud_context=None,
                 **kwargs):
        """__init__(name='docplex_model', output_level=InfoLevel.INFO, docloud_context=None)
        """
        if name is None:
            name = Model._name_generator.new_symbol()
        self._name = name

        self.__solver_agent = None  # default
        self.__error_handler = DefaultErrorHandler(output_level)

        # type instances
        self._binary_vartype = None
        self._integer_vartype = None
        self._continuous_vartype = None

        #
        self.__allvarctns = []
        self.__single_vars = []
        self.__allvars = []
        self.__vars_by_name = {}
        self.__vars_by_index = None
        self.__allcts = []
        self.__cts_by_name = {}
        self.__cts_by_index = None
        self._kpis_by_name = OrderedDict()
        self._progress_listeners = []
        self._solve_hooks = []  # debugSolveHook()
        self.__mip_start = None

        # -- float formats
        self._keep_ordering = False
        self._float_precision = 3
        self._continuous_var_format = "%.3f"

        # -- scopes
        self._var_scope = _SymbolGenerator(self._default_varname_pattern)
        self._ct_scope = _SymbolGenerator(self._default_ctname_pattern)
        self._indicator_scope = _SymbolGenerator(self._default_indicator_pattern)
        self._scopes = [self._var_scope, self._ct_scope, self._indicator_scope]

        self._environment = self._make_environment()
        self_env = self._environment
        # init context
        if context is None:
            self.context = Context(env=self_env)
        else:
            self.context = context

        if docloud_context is not None:
            self.context.solver.docloud = docloud_context
            warnings.warn(
                "Model construction with DOcloudContext is deprecated, use initializer with docplex.mp.context.Context instead.",
                DeprecationWarning, stacklevel=2)

        # this flag controls the column-wise formatting of text exports.
        self._format_export = True

        # update from kwargs,, before the actual inits.
        self._parse_kwargs(kwargs)

        # init engine
        engine = self._make_new_engine(self.solver_agent, self.context)
        self.__engine = engine
        self.__factory = ModelFactory(self, engine)
        self._solve_count = 0
        self.__solution = None
        self._solve_details = None

        # all the following must be placed after an engine has been set.
        self.__factory.init()
        self._default_objective_expr = self.linear_expr(constant=1)
        self.__objective_expr = self._default_objective_expr
        self.__objective_sense = ObjectiveSense.default_sense()

        # parameters

        self_cplex_parameters_version = self.context.cplex_parameters.cplex_version

        if self_env.has_cplex:
            installed_cplex_version = self_env.cplex_version
            # installed version is different from parameters: reset all defaults
            if installed_cplex_version != self_cplex_parameters_version:
                # cplex is more recent than parameters. must update defaults.
                self.info(
                    "reset parameter defaults, from parameter version: {0} to installed version: {1}"  # pragma: no cover
                        .format(self_cplex_parameters_version, installed_cplex_version))  # pragma: no cover
                nb = self._sync_parameter_defaults_from_engine()  # pragma: no cover
                if nb:  # pragma: no cover
                    self.trace("#parameter defaults have been reset: %d" % nb)  # pragma: no cover

    @property
    def infinity(self):
        """ This property returns the numerical value used as the upper bound for continuous variables.

        Note:
            CPLEX usually sets this limit to 1e+20.
        """
        return self.__engine.get_infinity()

    def get_name(self):
        return self._name

    def _set_name(self, name):
        self._check_name(name)
        self._name = name

    def _check_name(self, new_name):
        if not is_string(new_name):
            self.fatal("model.name requires a valid string, got: {0!s}", new_name)
        elif not new_name:
            self.fatal("model.name requires a non-empty string, got: {0!s}", new_name)
        elif new_name.find(" ") >= 0:
            self.warning("Model name contains whitespaces: |{0:s}|", new_name)
        else:
            pass

    name = property(get_name, _set_name)

    def _get_format_export(self):
        """
        This property is used to enable or disable the formatting of
        the exported LP model.
        By formatting, we mean the limitation to 80 columns of the generated LP.
        """
        return self._format_export

    def _set_format_export(self, do_format):
        self._format_export = bool(do_format)

    format_export = property(_get_format_export, _set_format_export)

    # you can change the way variables are named by default, just define a string here.
    def get_automatic_var_name_pattern(self):
        """
        This property is used to configure how DOcplex
        generates an automatic names for variables with no name.

        The default naming scheme is _x<i> where <i> is the creation rank, starting at 1,
        so variables without user name will get automatic names _x1, _x2, ...

        You can change this by passing a nonempty string, for example "var",
        in which case variables without user name will be named var1, var2, ...
        """
        return self._var_scope.pattern

    def set_automatic_var_name_pattern(self, varname):
        self.typecheck_string(varname, enforce_nonempty=True)
        self._var_scope.pattern = varname

    automatic_var_name_pattern = property(get_automatic_var_name_pattern, set_automatic_var_name_pattern)

    def get_automatic_ct_name_pattern(self):
        return self._ct_scope.pattern

    def set_automatic_ct_name_pattern(self, ctname):
        self.typecheck_string(ctname, enforce_nonempty=True)
        self._ct_scope.pattern = ctname

    automatic_ct_name_pattern = property(get_automatic_ct_name_pattern, set_automatic_ct_name_pattern)

    def get_automatic_indicator_name_pattern(self):
        return self._indicator_scope.pattern

    def set_automatic_indicator_name_pattern(self, indname):
        self.typecheck_string(indname, enforce_nonempty=True)
        self._indicator_scope.pattern = indname

    automatic_indicator_name_pattern = property(get_automatic_indicator_name_pattern,
                                                set_automatic_indicator_name_pattern)

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
    sort_expressions = property(_get_keep_ordering, _set_keep_ordering)

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
        self.typecheck_num(time_limit)
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
    def docloud_context(self):
        """ This property returns the DOcloud context attached to the model, possibly None.
        """
        warnings.warn("Model.docloud_context property is deprecated, use Model.context.solver.docloud instead",
                      DeprecationWarning, stacklevel=2)
        return self.context.docloud

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
    def mip_start(self):
        """ This property returns the MIP start solution (an instance of :class:`docplex.mp.solution.SolveSolution`)
        attached to the model if a MIP start has been defined, else None.
        """
        return self.__mip_start

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
        return self.__error_handler.outputLevel

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
        self.__single_vars = []
        self.__vars_by_name = {}
        self.__allcts = []
        self.__cts_by_name = {}
        self._kpis_by_name = OrderedDict()  # kpis must be ordered
        self._solve_count = 0
        self._last_solve_status = JobSolveStatus.UNKNOWN  # initial status
        self.__solution = None
        self.__mip_start = None
        self._clear_scopes()

    def _clear_scopes(self):
        for a_scope in self._scopes:
            a_scope.reset()

    def _make_new_engine(self, solver_agent, context):
        new_engine = EngineFactory.new_engine(solver_agent, self.environment, self, context=context)
        new_engine.notify_trace_output(self.context.solver.log_output_as_stream)
        return new_engine

    def _set_engine(self, e2):
        self.__engine = e2
        self.__factory.update_engine(e2)

    def set_engine_from_agent(self, agent):
        # INTERNAL
        new_engine = self._make_new_engine(agent, self.context)
        self._set_engine(new_engine)

    def _clear_engine(self, restart):
        # INTERNAL
        old_engine = self.__engine
        if old_engine:
            # dispose of old engine.
            old_engine.end()
            # from Ryan InstanceCounter
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
        if self.error_handler.prints_info():
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

    # @profile
    def __notify_new_model_object(self, descr, mobj, mindex,
                                  name_dir, idx_dir,
                                  is_name_safe=False):
        """
        Notifies the return af an object being create on the engine.
        :param descr: A string describing the type of the object being created (e.g. Constraint, Variable).
        :param mobj:  The newly created modeling object.
        :param mindex: The index as returned by the engine.
        :param name_dir: The directory of objects by name (e.g. name -> constraitn directory).
        :param idx_dir:  The directory from idices to objects.
        """
        mobj.set_index(mindex)

        if name_dir is not None:
            mobj_name = mobj._get_name()
            if mobj_name:
                # in some cases, names are checked before register
                if not is_name_safe:
                    if mobj_name in name_dir:
                        old_name_value = name_dir[mobj_name]
                        # Duplicate constraint name: foo
                        self.fatal("Duplicate {0} name: {1} already used for {2!s}", descr, mobj_name, old_name_value)

                name_dir[mobj_name] = mobj

        # store in idx dir if any
        if idx_dir is not None:  # do not use if idx_dir because an empty dir is False
            idx_dir[mindex] = mobj

    def _register_one_var(self, var, var_index):
        self.__notify_new_var(var, var_index)
        self._var_scope.notify_new_index(var_index)
        #
        self.__allvars.append(var)

    # @profile
    def _register_block_vars(self, allvars, indices):
        # with MyTimer("register_vars", verbose=True) as t:
        for k in fast_range(len(allvars)):
            self.__notify_new_var(allvars[k], indices[k])
        # increment global sequence container
        self.__allvars.extend(allvars)
        max_var_index = indices[-1]  # max index is last index
        self._var_scope.notify_new_index(max_var_index)

    def __notify_new_var(self, var, var_index):
        self.__notify_new_model_object("variable", var, var_index,
                                       self.__vars_by_name, self.__vars_by_index)

    def _get_ct_scope(self, ct):
        return self._indicator_scope if isinstance(ct, IndicatorConstraint) else self._ct_scope

    def _register_one_constraint(self, ct, ct_index, is_ctname_safe=False):
        """
        INTERNAL
        :param ct: The new constraint to register.
        :param ct_index: The index as returned by the engine.
        :param is_ctname_safe: True if ct name has been checked for duplicates already.
        :return:
        """
        self._get_ct_scope(ct).notify_new_index(ct_index)

        self.__notify_new_model_object(
            "constraint", ct, ct_index,
            self.__cts_by_name, self.__cts_by_index,
            is_name_safe=is_ctname_safe)

        self.__allcts.append(ct)

    def _register_block_cts(self, cts, indices, safe_names=False):
        nb_cts = len(cts)
        max_ct_index = -1
        for c in fast_range(nb_cts):
            ct_idx = indices[c]
            self.__notify_new_model_object("constraint", cts[c], ct_idx,
                                           self.__cts_by_name, self.__cts_by_index,
                                           is_name_safe=safe_names)
            if ct_idx > max_ct_index:
                max_ct_index = ct_idx
        self._ct_scope.notify_new_index(max_ct_index)
        # extend container faster than append()
        self.__allcts.extend(cts)

    # iterators
    def iter_var_containers(self):
        # INTERNAL
        return iter(self.__allvarctns)

    def _iter_single_vars(self):
        # INTERNAL
        return iter(self.__single_vars)

    def _add_var_container(self, ctn):
        # INTERNAL
        self.__allvarctns.append(ctn)

    def _is_binary_vartype(self, type_arg):
        # INTERNAL
        return type_arg is self._binary_vartype or isinstance(type_arg, BinaryVarType)

    def _is_integer_vartype(self, type_arg):
        # INTERNAL
        return type_arg is self._integer_vartype or isinstance(type_arg, IntegerVarType)

    def _is_continuous_vartype(self, type_arg):
        # INTERNAL
        return type_arg is self._continuous_vartype or isinstance(type_arg, ContinuousVarType)

    def _count_variables_filtered(self, predicate):
        return sum(1 for _ in filter(predicate, self.__allvars))

    def _iter_variables_filtered(self, predicate):
        for v in self.iter_variables():
            if predicate(v):
                yield v

    @property
    def number_of_variables(self):
        """ Returns the total number of decision variables, all types combined.

        """
        return len(self.__allvars)

    @property
    def number_of_binary_variables(self):
        """ This property returns the total number of binary variables added to the model.
        """
        binary_vartype_filter = lambda v: self._is_binary_vartype(v._vartype)
        return self._count_variables_filtered(binary_vartype_filter)

    @property
    def number_of_integer_variables(self):
        """ This property returns the total number of integer variables added to the model.
        """
        integer_vartype_filter = lambda v: self._is_integer_vartype(v._vartype)
        return self._count_variables_filtered(integer_vartype_filter)

    @property
    def number_of_continuous_variables(self):
        """ This property returns the total number of continuous variables added to the model.
        """
        continuous_vartype_filter = lambda v: self._is_continuous_vartype(v._vartype)
        return self._count_variables_filtered(continuous_vartype_filter)

    def is_MIP(self):
        return self._has_discrete_var()

    def _has_discrete_var(self):
        # INTERNAL
        for dv in self.iter_variables():
            if dv.is_discrete():
                return True
        return False

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
        """ Iterates over all binary variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_binary_vartype(v._vartype))

    def iter_integer_vars(self):
        """ Iterates over all integer variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_integer_vartype(v._vartype))

    def iter_continuous_vars(self):
        """ Iterates over all continuous variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_filtered(lambda v: self._is_continuous_vartype(v._vartype))

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
    # """
    # INTERNAL: builds a dictionary of names to objects, given a sequence
    # Objects with no name (that is name is None) are omitted
    # :param mobj_seq: the sequence of objects (assumed to have a "name" attribute
    # :return: a dictionary of names to objects
    # """
    # return {mobj.name: mobj for mobj in mobj_seq if mobj.name is not None}

    # index management
    def _build_index_dict(self, mobj_seq, raise_on_invalid_index=False):
        """
        Lazily creates a dict from indices to objects (variable or constraints)
        :param mobj_seq:
        :param raise_on_invalid_index:
        :return:
        """
        idx_dict = {}
        for mobj in mobj_seq:
            if mobj.has_valid_index():
                idx_dict[mobj.index] = mobj
            elif raise_on_invalid_index:  # pragma: no cover
                self.fatal("Object has invalid index: {0!s}", mobj)  # pragma: no cover
        return idx_dict

    def get_var_by_index(self, idx):
        # INTERNAL
        self.typecheck_valid_index(idx)
        if not self.__vars_by_index:
            self.__vars_by_index = self._build_index_dict(self.__allvars)
        return self.__vars_by_index.get(idx)

    def set_var_ub(self, var, new_ub):
        # INTERNAL: use var.ub property to set ub
        if new_ub is not None:
            self.typecheck_num(new_ub, "set_var_ub")
            if new_ub != var.ub:
                self.__engine.set_var_attribute(var, "ub", new_ub)
                var._internal_set_ub(new_ub)

    def set_var_lb(self, var, new_lb):
        # INTERNAL: use var.lb to set lb
        if new_lb is not None:
            self.typecheck_num(new_lb, "set_var_ub")
            if new_lb != var.lb:
                self.__engine.set_var_attribute(var, "lb", new_lb)
                var._internal_set_lb(new_lb)

    def set_var_name(self, dvar, new_arg_name):
        # INTERNAL: use var.name to set variable names
        new_name = str(new_arg_name)
        if not new_name:
            self.fatal("Not a valid name: {0!s}", new_arg_name)
        elif new_name != dvar.name:
            self.__engine.set_var_attribute(dvar, "name", new_name)
            ModelingObjectBase.set_name(dvar, new_name)
        return self

    def _typecheck_bounds(self, header, candidate_bounds):
        check_ok = True
        if is_number(candidate_bounds):
            pass
        elif not is_iterable(candidate_bounds):
            check_ok = False
        else:
            check_ok = all(is_number(b) for b in candidate_bounds)
        if not check_ok:
            self.fatal("Model.{0}: expecting either number or number_sequence as bounds, got: {1!s}",
                       header, candidate_bounds)

    def set_var_lbs(self, changed_vars, new_lbs):
        """ Change lower bounds of a collection of variables.

        `new_lbs` accepts either a single number (all lower bounds are set to this number)
        or an array of numbers. With an array, the lower bounds of the variables are set the corresponding bound in the array.
        If the bounds array is smaller than the variable array, the remaining variables are unchanged.


        Args:
            changed_vars: The collection of variables to be changed.
            new_lbs: Either a number or a collection of numbers.

        """
        self.typecheck_iterable(changed_vars)
        self._typecheck_bounds("set_var_lbs", new_lbs)

        if is_iterable(new_lbs):
            engine_new_lbs = new_lbs
        else:
            engine_new_lbs = [new_lbs] * len(changed_vars)
        actual_lbs = self.__engine.set_var_attribute(changed_vars, "lb", engine_new_lbs)
        for var, new_lb in zip(changed_vars, actual_lbs):
            var._internal_set_lb(new_lb)

    def set_var_ubs(self, changed_vars, new_ubs):
        """ Change upper bounds of a collection of variables.

        `new_ubs` accepts either a single number (all upper bounds are set to this number)
        or an array of numbers. With an array, the upper bounds of the variables are set the corresponding bound in the array.
        If the bounds array is smaller than the variable array, the remaining variable bounds are unchanged.

        Args:
            changed_vars: The collection of variables to be changed.
            new_ubs: Either a number or a collection of numbers.

        """
        self.typecheck_iterable(changed_vars)
        self._typecheck_bounds("set_var_ubs", new_ubs)

        new_lbs = [b for b in new_ubs] if is_iterable(new_ubs) else [new_ubs] * len(changed_vars)
        actual_ubs = self.__engine.set_var_attribute(changed_vars, "ub", new_lbs)
        for var, new_ub in zip(changed_vars, actual_ubs):
            var._internal_set_ub(new_ub)

    def get_constraint_by_name(self, name):
        """ Searches a for a constraint from a name.

        Returns constraint if it finds a constraint with exactly this name, or None
        if no constraint has this name.

        Will not raise an exception if not found

        Args:
            name (str): The name of the constraint being searched for.

        Returns:
            A constraint or None.
        """
        return self.__cts_by_name.get(name)

    def get_constraint_by_index(self, idx):
        # INTERNAL
        self.typecheck_valid_index(idx)
        if not self.__cts_by_index:
            self.__cts_by_index = self._build_index_dict(self.__allcts)
        return self.__cts_by_index.get(idx)

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
        type_filter = lambda ct: isinstance(ct, cttype)
        return sum(1 for _ in filter(type_filter, self.__allcts))

    def gen_constraints_with_type(self, cttype):
        for ct in self.__allcts:
            if isinstance(ct, cttype):
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
        return self._count_constraints_with_type(AbstractLinearConstraint)

    def iter_range_constraints(self):
        return self.gen_constraints_with_type(RangeConstraint)

    def iter_binary_constraints(self):
        return self.gen_constraints_with_type(LinearConstraint)

    def iter_linear_constraints(self):
        """
        Returns an iterator on the linear constraints of the model.
        This includes binary linear constraints and ranges, but not indicators.

        Returns:
            An iterator object.
        """
        return self.gen_constraints_with_type(AbstractLinearConstraint)

    def iter_indicator_constraints(self):
        """ Returns an iterator on idicator constraints in the model.

        Returns:
            An iterator object.
        """
        return self.gen_constraints_with_type(IndicatorConstraint)

    @property
    def number_of_indicator_constraints(self):
        """ This property returns the number of indicator constraints in the model.
        """
        return self._count_constraints_with_type(IndicatorConstraint)

    def var(self, vartype, lb=None, ub=None, name=None):
        """ Creates a decision variable and stores it in the model.

        Args:
            vartype: The type of the decision variable;
                this can take three values: Binary, Integer or Continuous.

            lb: The lower bound of the variable; either a number or None, to use the default.
                 The default lower bound for all three variable types is 0.

            ub: The upper bound of the variable domain; expectes either a number or None to use the type's default.
                The default upper bound for Binary is 1, otherwise positive infinity.

            name: An optional string to name the variable.

        :returns: The newly created decision variable, an instance of :class:`docplex.mp.linear.Var`.

        See Also:
            :class:`docplex.mp.linear.Var`,
            :func:`infinity`
        """
        self.typecheck_vartype(vartype)
        return self._var(vartype, lb, ub, name)

    def _var(self, vartype, lb=None, ub=None, name=None):
        # INTERNAL
        new_var = self.__factory.var(vartype, lb, ub, name)
        self.__single_vars.append(new_var)
        return new_var

    def continuous_var(self, lb=None, ub=None, name=None):
        """ Creates a new continuous decision variable and stores it in the model.

        Args:
            lb: The lower bound of the variable, or None. The default is 0.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name: An optional name for the variable.

        :returns: An instance of the :class:`docplex.mp.linear.Var` class with type `VarType.Continuous`.
        """
        return self._var(self.vartype_continuous, lb, ub, name)

    def integer_var(self, lb=None, ub=None, name=None):
        """ Creates a new integer variable and stores it in the model.

        Args:
            lb: The lower bound of the variable, or None. The default is 0.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name: An optional name for the variable.

        :returns: An instance of the :class:`docplex.mp.linear.Var` class with type `VarType.Integer`.
        """
        return self._var(self.vartype_integer, lb, ub, name)

    def binary_var(self, name=None):
        """ Creates a new binary variable and stores it in the model.

        Args:
            name: An optional name for the variable.

        :returns: An instance of the :class:`docplex.mp.linear.Var` class with type `VarType.Binary`.
        """
        return self._var(self.vartype_binary, name=name)

    def var_list(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        self.typecheck_vartype(vartype)
        actual_name, fixed_keys = self._make_key_seq(keys, name)
        ctn = _VariableContainer(vartype, [fixed_keys], lb, ub, name)
        self._add_var_container(ctn)
        return self.__factory.var_list(fixed_keys, vartype, lb, ub, actual_name, 1, key_format)

    def var_dict(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        self.typecheck_vartype(vartype)
        return self._var_dict(keys, vartype, lb, ub, name, key_format)

    def _var_dict(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        # INTERNAL
        actual_name, key_seq = self._make_key_seq(keys, name)
        ctn = _VariableContainer(vartype, [key_seq], lb, ub, name)
        self._add_var_container(ctn)
        var_list = self.__factory.var_list(key_seq, vartype, lb, ub, actual_name, 1, key_format)
        _dict_type = OrderedDict if self._keep_ordering else dict
        return _dict_type(zip(key_seq, var_list))

    def binary_var_list(self, keys, name=str, key_format=None):
        """binary_var_list(self, keys, name=str, key_format=None)

        Creates a list of binary decision variables and stores them in the model.

        Args:
            keys: Either a sequence of objects or an integer.
            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if keys is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", k2", ... then variables will be named "x_k1", "x_k2",...
                        
        Example:
            If you want each key string to be surrounded by {}, use a special key_format: "_{%s}",
            the %s denotes where the key string will be formatted and appended to `name`.

        :returns: A list of :class:`doc.mp.linear.Var` objects with type `VarType.Binary`.

        Example:
            `mdl.binary_var_list(3, "z")` returns a list of size 3
            containing three binary variables with name `z_0`, `z_1`, `z_2`.

        """
        return self.var_list(keys, self.vartype_binary, name=name, key_format=key_format)

    def integer_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """integer_var_list(self, keys, lb=None, ub=None, name=str, key_format=None)

        Creates a list of integer decision variables with type `VarType.Integer`, stores them in the model,
        and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.
            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a function (which will be called on each key argument), or None.
            ub: Upper bounds of the variables.  Accepts either a floating-point number,
                a function (which will  be called on each key argument), or None.
            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
            key_format: A format string or None. This format string describes how keys contribute to variable names.
                The default is "_%s". For example if name is "x" and each key object is represented by a string
                like "k1", k2", ... then variables will be named "x_k1", "x_k2",...

        Note:
            Using None as the lower bound means the default lower bound (0) is used.
            Using None as the upper bound means the default upper bound (the model's positive infinity)
            is used.

        :returns: A list of :class:`doc.mp.linear.Var` objects with type `VarType.Integer`.

        """
        return self.var_list(keys, self.vartype_integer, lb, ub, name, key_format)

    def continuous_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """continuous_var_list(self, keys, lb=None, ub=None, name=str, key_format=None)

        Creates a list of continuous decision variables with type `VarType.Integer`, stores them in the model,
        and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a function,  or None. Use a number if all variables share the same lower bound,
                or use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means using the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a function, or None. Use a number if all variables share the same upper bound,
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
                        like "k1", k2", ... then variables will be named "x_k1", "x_k2",...
        Note:
            When `keys` is either an empty list or the integer 0, an empty list is returned.


        :returns: A list of :class:`docplex.mp.linear.Var` objects with type `VarType.Continuous`.

        See Also:
            :class:`docplex.mp.linear.Var`,
            :func:`infinity`

        """
        return self.var_list(keys, self.vartype_continuous, lb, ub, name, key_format)

    def continuous_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None):
        """continuous_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None)

        Creates a dictionary of continuous variables, indexed by key objects.

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
                a function, or None. Use a number if all variables share the same lower bound,
                or use a function if lower bounds vary depending on the `key`, in which case,
                the function will be called on each `key` in `keys`.
                None means that the default lower bound (0) is used.
            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a function,  or None. Use a number if all variables share the same upper bound,
                or use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means that the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how `keys` contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type `VarType.Continuous`) indexed by
                  the objects in `keys`.

        See Also:
            :class:`docplex.mp.linear.Var`,
            :func:`infinity`
        """
        return self._var_dict(keys, self.vartype_continuous, lb=lb, ub=ub, name=name, key_format=key_format)

    def integer_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None):
        """integer_var_dict(self, keys, lb=None, ub=None, name=str, key_format=None)

        Creates a dictionary of integer variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either an integer number,
                a function,  or None. Use a number if all variables share the same lower bound,
                or use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either an integer number,
                a function,  or None. Use a number if all variables share the same upper bound,
                or use a function if upper bounds vary depending on the key, in which case,
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
                        like "k1", k2", ... then variables will be named "x_k1", "x_k2",...

        :returns:  A dictionary of :class:`docplex.mp.linear.Var` objects (with type `VarType.Integer`) indexed by the
                   objects in `keys`.

        See Also:
            :func:`infinity`
        """
        return self._var_dict(keys, self.vartype_integer, lb=lb, ub=ub, name=name, key_format=key_format)

    def binary_var_dict(self, keys, name=str, key_format=None):
        """binary_var_dict(self, keys, name=str, key_format=None)

        Creates a dictionary of binary variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            if keys is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type `VarType.Binary`) indexed by the
                  objects in `keys`.

        See Also:
            :class:`docplex.mp.linear.Var`
        """
        return self._var_dict(keys, self.vartype_binary, name=name, key_format=key_format)

    def var_multidict(self, vartype, seq_of_key_seqs, lb=None, ub=None, name=str, key_format=None):
        # INTERNAL
        self.typecheck_vartype(vartype)
        self.typecheck_iterable(seq_of_key_seqs)
        fixed_keys = [self._make_key_seq(ks, name)[1] for ks in seq_of_key_seqs]

        ctn = _VariableContainer(vartype, fixed_keys, lb, ub, name)
        self._add_var_container(ctn)
        # the sequence of keysets should answer to len(no generators here)
        dimension = len(fixed_keys)
        if dimension < 1:
            self.fatal("len of key sequence must be >= 1, got: {0}", dimension)  # pragma: no cover

        # create cartesian product of keys...
        all_key_tuples = list(product(*fixed_keys))
        cube_vars = self.__factory.var_list(all_key_tuples, vartype, lb, ub, name, dimension, key_format, False)

        var_dict = dict(zip(all_key_tuples, cube_vars))
        return var_dict

    def var_matrix(self, vartype, keys1, keys2, lb=None, ub=None, name=str, key_format=None):
        return self.var_multidict(vartype, [keys1, keys2], lb, ub, name, key_format)

    def binary_var_matrix(self, keys1, keys2, name=str, key_format=None):
        """binary_var_matrix(self, keys1, keys2, name=str, key_format=None)

        Creates a dictionary of binary variables, indexed by pairs of key objects.

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
                        like "k1", k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type `VarType.Binary`) indexed by
            all couples `(k1, k2)` with `k1` in `keys1` and `k2` in `keys2`.

        See Also:
            :class:`docplex.mp.linear.Var`
        """
        return self.var_multidict(self.vartype_binary, [keys1, keys2], 0, 1, name=name, key_format=key_format)

    def integer_var_matrix(self, keys1, keys2, lb=None, ub=None, name=str, key_format=None):
        """integer_var_matrix(self, keys1, keys2, lb=None, ub=None, name=str, key_format=None)

        Creates a dictionary of integer variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted as in :func:`integer_var_dict`.
        """

        return self.var_multidict(self.vartype_integer, [keys1, keys2], lb, ub, name, key_format)

    def continuous_var_matrix(self, keys1, keys2, lb=None, ub=None, name=str, key_format=None):
        """continuous_var_matrix(self, keys1, keys2, lb=None, ub=None, name=str, key_format=None)

        Creates a dictionary of continuous variables, indexed by pairs of key objects.

        Creates a dictionary that allows retrieval of variables from a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted the same as in :func:`integer_var_dict`.

        """
        return self.var_multidict(self.vartype_continuous, [keys1, keys2], lb, ub, name, key_format)

    def continuous_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=str, key_format=None):
        """continuous_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=str, key_format=None)

        Creates a dictionary of continuous variables, indexed by triplets.

        Same as :func:`continuous_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys`, `k2` in `keys2`, `k3` in `keys3`.

        See Also:
            :func:`continuous_var_matrix`
        """
        return self.var_multidict(self.vartype_continuous, [keys1, keys2, keys3], lb, ub, name, key_format)

    def integer_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=str):
        return self.var_multidict(self.vartype_integer, [keys1, keys2, keys3], lb, ub, name)

    def binary_var_cube(self, keys1, keys2, keys3, name=str):
        return self.var_multidict(self.vartype_binary, [keys1, keys2, keys3], name=name)

    def _get_zero_expr(self):
        # INTERNAL
        return self.__factory.zero_expr

    def linear_expr(self, e=0, constant=0, name=None):
        return self.__factory.linear_expr(e, constant, name)

    def monomial_expr(self, dvar, coef=1):
        # INTERNAL
        self.typecheck_var(dvar)
        return self._monomial_expr(dvar, coef)

    def _monomial_expr(self, dvar, coef):
        return MonomialExpr(self, dvar, coef)

    def abs(self, e):
        # NOT DOCUMENTED for now
        from docplex.mp.functional import AbsExpr

        expr = self._to_linear_expr(e)
        return AbsExpr(self, expr)

    def _to_linear_expr(self, e):
        # INTERNAL
        if isinstance(e, LinearExpr):
            return e
        # elif e is 0:
        # return self.__factory.unique_zero_expr
        else:
            try:
                return e.to_linear_expr()
            except AttributeError:
                # delegate to the factory
                return self.__factory.linear_expr(e)

    def scal_prod(self, dvars, coefs):
        """
        Creates a linear expression equal to the scalar product of a list of variables and a sequence of coefficients.

        This method allows different types of input for both arguments. The variable sequence can be either a list
        or an iterator of objects that can be converted to linear expressions, that is, variables, expressions, or numbers.
        The most usual case is variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or even a plain number.
        In this last case the scalar product reduces to a sum times this coefficient.


        :param dvars: A list or an iterator on variables or expressions.
        :param coefs: A list or an iterator on numbers, or a number.

        Note: 
           If either list or iterator is empty, this method returns zero.

        :return: A linear expression or 0.
        """
        return self.__factory.scal_prod(dvars, coefs)

    def dot(self, dvars, coefs):
        """ Synonym for  scal_prod.

        """
        return self.scal_prod(dvars, coefs)

    def sum(self, args):
        """ Creates a linear expression summing over a sequence.


        Note:
           This method returns 0 if the argument is an empty list or iterator.
        
        :param args: A list of objects that can be converted to linear expressions, that is, linear expressions, decision variables, or numbers.

        :return: A linear expression or 0.
        """
        return self.__factory.sum(args)

    def le_constraint(self, lhs, rhs, name=None):
        """ Creates a "less_than_or_equal" linear constraint.

        Note:
            This method returns a constraint object, that is not added to the model.
            To add it to the model, use the add_constraint method.

        :param lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param rhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param name: An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self.__factory.le_constraint(lhs, rhs, name)

    def ge_constraint(self, lhs, rhs, name=None):
        """ Creates a "greater than or equal" linear constraint.

        Note:
            This method returns a constraint object that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        :param lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param rhs: An object that can be converted to a linear expression, typically a variable,
                    a number of an expression.
        :param name: An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self.__factory.ge_constraint(lhs, rhs, name)

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
        return self.__factory.eq_constraint(lhs, rhs, name)

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
        else:
            self.fatal("Expecting binary constraint, indicator or range, got: {0!s}", ct)  # pragma: no cover

    def _notify_trivial_constraint(self, ct, ctname, is_feasible):
        if ct is None:
            arg = None
        elif ctname and not ctname.startswith("_"):
            arg = ctname
        elif ct.name and not ct.name.startswith("_"):
            arg = ct.name
        else:
            arg = str(ct)
        ct_typename = ct.short_typename() if ct is not None else "constraint"
        ct_rank = self.number_of_constraints + 1
        if arg is not None:
            if is_feasible:
                self.info("Adding trivial feasible {2}: {0!s}, rank: {1}", arg, ct_rank, ct_typename)
            else:
                self.error("Adding trivially infeasible {2}: {0!s}, rank: {1}", arg, ct_rank, ct_typename)
        else:
            if is_feasible:
                self.info("Adding trivial feasible {1}, rank: {0}", ct_rank, ct_typename)
            elif ct:
                self.error("Adding trivially infeasible {1}, rank: {0}", ct_rank, ct_typename)
            else:
                self.fatal("Adding trivially infeasible {1}, rank: {0}", ct_rank, ct_typename)

    def _prepare_constraint(self, ct, ctname, do_check=True):
        # INTERNAL
        if ct is True:
            # sum([]) == 0
            self._notify_trivial_constraint(None, ctname, is_feasible=True)
            return False
        elif ct is False:
            # happens with sum([]) and constant e.g. sum([]) == 2
            ct = self.__factory.new_trivial_infeasible_ct()
            self._notify_trivial_constraint(None, ctname, is_feasible=False)
            if ctname:
                self.fatal("Adding a trivially infeasible constraint with name: {0}", ctname)
            else:
                # analogous to 0 == 1, model is sure to fail
                self.fatal("Adding trivially infeasible constraint")
        else:
            if do_check:
                self.typecheck_constraint(ct)
            # -- watch for trivial cts e.g. linexpr(0) <= linexpr(1)
            if ct.is_trivial():
                if ct._is_trivially_feasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=True)
                elif ct._is_trivially_infeasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=False)

        # is this necessary if we're sure where we are sure of the calling context??
        if do_check and not ct.is_in_model(self):
            self.fatal("Constraint {0!s} does not belong to model {1}", ct, self.name)
        # --- name management ---
        if not ctname:
            if not ct.name:
                ct.name = self._create_automatic_ctname(ct)
                ct.notify_automatic_name()
        elif ctname in self.__cts_by_name:
            self.fatal("Duplicate constraint name: {0!s}, used for: {1}", ctname,
                       self.get_constraint_by_name(ctname).to_string())
        else:
            # might be an issue if both ctname and ct.name exist
            ct.name = ctname
        # ---
        # at this stage we are sure the ctname is valid whatsoever.
        assert ct.has_name()

        # check for already posted cts.
        if do_check and ct.has_valid_index():
            self.warning("constraint has already been posted: {0!s}, index is: {1}", ct, ct.index)  # pragma: no cover
            return False  # pragma: no cover
        return True

    # @profile
    def _add_constraint_internal(self, ct, ctname, do_check=True):
        if ct is True:
            # sum([]) == 0
            ct = self.__factory.new_trivial_feasible_ct()
            self._notify_trivial_constraint(None, ctname, is_feasible=True)
        elif ct is False:
            # happens with sum([]) and constant e.g. sum([]) == 2
            ct = self.__factory.new_trivial_infeasible_ct()
            self._notify_trivial_constraint(None, ctname, is_feasible=False)
            if ctname:
                self.fatal("Adding a trivially infeasible constraint with name: {0}", ctname)
            else:
                # analogous to 0 == 1, model is sure to fail
                self.fatal("Adding trivially infeasible constraint")
        else:
            self.typecheck_constraint(ct)
            # -- watch for trivial cts e.g. linexpr(0) <= linexpr(1)
            if ct.is_trivial():
                if ct._is_trivially_feasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=True)
                elif ct._is_trivially_infeasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=False)

        # is this necessary if we're sure where we are sure of the calling context??
        if do_check and not ct.is_in_model(self):
            self.fatal("Constraint {0!s} does not belong to model {1}", ct, self.name)
        # --- name management ---
        if not ctname:
            if not ct.name:
                ct.name = self._create_automatic_ctname(ct)
                ct.notify_automatic_name()
        elif ctname in self.__cts_by_name:
            self.fatal("Duplicate constraint name: {0!s}, used for: {1}", ctname,
                       self.get_constraint_by_name(ctname).to_string())
        else:
            # might be an issue if both ctname and ct.name exist
            ct.name = ctname

        assert ct.has_name()
        # ---
        # at this stage we are sure the ctname is valid whatsoever.
        if do_check and ct.has_valid_index():
            self.info("constraint has already been added: {0!s}", ct)
            return ct

        ct_engine_index = self._post_constraint(ct)
        self._register_one_constraint(ct, ct_engine_index, is_ctname_safe=True)
        return ct

    def _remove_constraint_internal(self, ct):
        """
        No typechecking for this internal version.
        :param ct:
        :return:
        """
        ct_name = ct.name
        ct_index = ct.unchecked_index
        # remove from engine.
        self.__engine.remove_constraint(ct)
        # remove from model.
        self.__allcts.remove(ct)
        if ct_name:
            del self.__cts_by_name[ct_name]
        if self.__cts_by_index is not None:
            del self.__cts_by_index[ct_index]
        ct.notify_deleted()
        if ModelingObject.is_valid_index(ct_index):
            # resync indices from doomed up
            for ct in self.iter_linear_constraints():
                old_model_index = ct.unchecked_index
                if old_model_index > ct_index:
                    updated_index = self.__engine.get_ct_index(ct)
                    if updated_index != old_model_index:
                        ct.set_index(updated_index)
        self._sync_constraint_indices(self.iter_linear_constraints())

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
            self._check_in_model(ct, "constraint")
            self._remove_constraint_internal(ct)

    def add_range(self, lb, expr, ub, rng_name=None):
        """ Adds a new range constraint to the model.

        A range constraint states that a linear
        expression has to stay within an interval `[lb..ub]`.
        Both `lb` and `ub` have to be float numbers with `lb` smaller than `ub`.
        Raises an exception if `lb` is greater than `ub`.

        The method creates a new range constraint and adds it to the model.

        Args:
            lb: A floating-point number.
            expr: A linear expression, e.g. X+Y+Z.
            ub: A floating-point number, which should be greater than lb.
            rng_name: An optional name for the range constraint.

        Returns:
            The newly created range constraint, an instance of :class:`docplex.mp.linear.RangeConstraint`.

        """
        rng = self.range_constraint(lb, expr, ub, rng_name)
        ct = self._add_constraint_internal(rng, rng_name)
        return ct

    def add_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        """ Adds a new indicator constraint to the model.

        An indicator constraint links (one-way) the value of a binary variable to
        the satisfaction of a linear constraint.
        If the binary variable equals the active value, then the constraint is satisfied, but
        otherwise the constraint may or may not be satisfied.

        :param binary_var: The binary variable used to control the satisfaction of the linear constraint.
        :param linear_ct: A linear constraint (EQ, LE, GE).
        :param active_value: 0 or 1. The value used to trigger the satisfaction of the constraint. The default is 1.
        :param name: An optional name for the indicator constraint.

        :return: The newly created indicator constraint.
        """
        # noinspection PyPep8
        indicator_trivial_feasible_idx = -2
        indicator_trivial_infeasible_idx = -4
        self.typecheck_var(binary_var)
        self.typecheck_constraint(linear_ct)
        self.typecheck_zero_or_one(active_value)
        self._check_in_model(binary_var, typename="binary variable")
        self._check_in_model(linear_ct, typename="linear_constraint")

        indicator = self.indicator_constraint(binary_var=binary_var, linear_ct=linear_ct, active_value=active_value)

        if linear_ct.is_trivial():
            is_feasible = linear_ct._is_trivially_feasible()
            if is_feasible:
                self.warning("Indicator constraint {0!s} has a trivial feasible linear constraint (has no effect)",
                             indicator)
                return indicator_trivial_feasible_idx
            else:
                self.warning("indicator constraint {0!s} has a trivial infeasible linear constraint - invalidated",
                             indicator)
                indicator.invalidate()
                return indicator_trivial_infeasible_idx
        else:
            return self._add_constraint_internal(indicator, name)

    def range_constraint(self, lb, expr, ub, rng_name=None):
        """ Creates a new range constraint but does not add it to the model.

        A range constraint states that a linear
        expression has to stay within an interval `[lb..ub]`.
        Both `lb` and `ub` have to be floating-point numbers with `lb` smaller than `ub`.
        Raises an exception if `lb` is greater than `ub`.

        The method creates a new range constraint but does not add it to the model.

        Args:
            lb: A floating-point number.
            expr: A linear expression, e.g. X+Y+Z.
            ub: A floating-point number, which should be greater than `lb`.
            rng_name: An optional name for the range constraint.

        Returns:
            The newly created range constraint.

        """
        self.typecheck_num(lb, 'Model.range_constraint')
        self.typecheck_num(ub, 'Model.range_constraint')
        if not lb <= ub:
            self.error("infeasible range constraint, lb={0}, ub={1}, expr={2}", lb, ub, expr)

        expr1 = self._to_linear_expr(expr)
        rng = self.__factory.new_range_constraint(lb, expr1, ub, rng_name)
        return rng

    def indicator_constraint(self, binary_var, linear_ct, active_value=1):
        self.typecheck_var(binary_var)
        self.typecheck_constraint(linear_ct, AbstractLinearConstraint)
        actual_active_value = 1 if bool(active_value) else 0
        return self.__factory.new_indicator_constraint(binary_var, linear_ct, actual_active_value)

    def add_constraint(self, ct, ctname=None):
        """ Adds a new linear constraint to the model.

        :param ct: A linear constraint of the form <expr1> <op> <expr2>, where both expr1 and expr2 are
         linear expressions built from variables in the model, and <op> is a relational operator
         among <= (less than or equal), == (equal), and >= (greater than or equal).


        :param ctname: An optional string used to name the constraint.

        :return: The newly added constraint.
        """
        ct = self._add_constraint_internal(ct, ctname)
        return ct

    def _add_constraints(self, cts, names=None, do_check=True):
        # INTERNAL
        if not names:
            posted_cts = [ct for ct in cts if self._prepare_constraint(ct, ctname=None, do_check=do_check)]
        else:
            zipped = zip(cts, names)
            posted_cts = [ct for ct, ctname in zipped if self._prepare_constraint(ct, ctname, do_check=do_check)]

        if posted_cts:
            ct_indices = self.__engine.create_block_linear_constraints(posted_cts)
            self._register_block_cts(posted_cts, ct_indices)
        return posted_cts

    def add_constraints(self, cts, names=None):
        """ Adds a collection of constraints to the model in one operation.

        Each constraint in the collection is added to the model, if it was not already added.
        If present, the names collection is used to set names to the constraints.

        Args:
            cts: a collection of constraints or an iterator over a collection of constraints.
            names: an optional collection or iterator on strings.

        Returns:
            a list of those constraints added to the model.
        """
        self.typecheck_iterable(cts)
        # build a sequence as we need to traverse it more than once.
        if is_iterator(cts):
            ct_seq = list(cts)
        else:
            ct_seq = cts
        # typecheck
        for ct in ct_seq:
            self.typecheck_constraint(ct)
        return self._add_constraints(cts, names, do_check=True)

    # ----------------------------------------------------
    # objective
    # ----------------------------------------------------

    def _clear_engine_objective(self):
        # INTERNAL
        self.__engine.clear_objective(self.__objective_expr)

    @property
    def objective_expr(self):
        """ This property returns the current expression used as the model objective.
        """
        return self.__objective_expr

    @property
    def objective_sense(self):
        """ This property returns the direction of the optimization as an instance of
        ObjectiveSense, either Minimize or Maximize.
        """
        return self.__objective_sense

    def minimize(self, expr):
        """ Sets an expression as the expression to be minimized.

        The argument is converted to a linear expression. Accepted types are variable (instances of
        :class:`docplex.mp.linear.Var` class), linear expressions (instance of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        :param expr: A linear expression or a variable.
        """
        self.set_objective(ObjectiveSense.Minimize, expr)

    def maximize(self, expr):
        """
        Sets an expression as the expression to be maximized.

        The argument is converted to a linear expression. Accepted types are variable (instances of
        :class:`docplex.mp.linear.Var` class), linear expressions (instance of
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
           True if the model is a minimization model.
        """
        return self.__objective_sense is ObjectiveSense.Minimize

    def is_maximize(self):
        """ Checks whether the model is a maximization model.

        Note:
           This returns True even if the expression to maximize is a constant.
        To check whether the model has a non-constant objective, use :func:`is_optimized`.

        Returns:
            True if the model is a maximization model.
        """
        return self.__objective_sense is ObjectiveSense.Maximize

    def objective_coef(self, dvar):
        """ Returns the objective coefficient of a variable.

        The objective coefficient is the coefficient of the given variable in
        the Model's objective expression. If the variable is not explicitly
        mentioned in the objective, it returns 0.

        :param dvar: The decision variable for which to compute the objective coefficient.

        :return: A floating-point number, the objective coefficient of the variable.
        """
        self.typecheck_var(dvar)
        return self._objective_coef(dvar)

    def _objective_coef(self, dvar):
        return self.__objective_expr[dvar]

    def remove_objective(self):
        """ Clears the current objective.

        This is equivalent to setting "minimize 1".
        Any subsequent solve will look only for a feasible solution.
        You can detect this state by calling :func:`has_objective` on the model.

        """
        self.set_objective(ObjectiveSense.Minimize, self._default_objective_expr)

    def is_optimized(self):
        """ Checks whether the model has a non-constant objective expression.

        A model with a constant objective will only search for a feasible solution when solved.
        This happens either if no objective has been assigned to the model,
        or if the objective has been removed with :func:`remove_objective`.

        Returns:
            True, if the model has a non-constant objective expression.

        """
        return not self.__objective_expr.is_constant()

    def has_objective(self):
        # INTERNAL
        return self.__objective_expr is not self._default_objective_expr

    def set_objective(self, sense, expr):
        """ Sets a new objective.

        Args:
            sense: Either an instance of ObjectiveSense (Minimize or Maximize),
                or a string: "min" or "max".
            expr: Is converted to a linear expression. Acceptable types are Var, LinearExpr, or plain number

        Note:
            When using a number, the search will not optimize but only look for a feasible solution.

        """
        if expr is None:
            expr = self._default_objective_expr
        else:
            expr = self._to_linear_expr(expr)

        if not self.objective_expr.is_constant():
            # do not try to clear a constant list as CPLEX/wrapper will crash!
            self._clear_engine_objective()

        actual_sense = self._resolve_sense(sense)

        # minor issue: we reset coefficients even if they are identical.
        self.__engine.set_objective(actual_sense, expr)

        self.__objective_sense = actual_sense
        self.__objective_expr = expr

    def _can_solve(self):
        return self.__engine.can_solve()

    def _make_start_infodict(self):
        # INTERNAL
        infodict = {}
        stats = self.get_statistics()
        infodict.update(zip(["number_of_%s_vars" % tn for tn in ["binary", "integer", "ccontinuous"]],
                            [stats.number_of_binary_variables,
                             stats.number_of_integer_variables,
                             stats.number_of_continuous_variables]))
        infodict["number_of_constraints"] = self.number_of_constraints
        return infodict

    def _make_end_infodict(self):
        return self.solution.as_dict(keep_zeros=False) if self.solution is not None else dict()

    def _force_docloud(self, __context, **kwargs):
        # returns True if the kwargs or this context forces a solve on docloud
        have_docloud_context = kwargs.get('docloud_context') is not None
        have_api_key = kwargs.get('api_key') is not None
        have_url = kwargs.get('url') is not None
        solver_agent_is_docloud = __context.solver.get('agent') == "docloud"
        return have_docloud_context or (have_api_key and have_url) or solver_agent_is_docloud

    def solve(self, **kwargs):
        """ Starts a solve operation on the model.

        If passed a valid DOcloud context, the solve operation will be started on DOcloud.
        If the argument is None, or does not contain valid DOcloud credentials,
        the solve operation will try to use CPLEX if CPLEX is available (it uses the Environment class
        to determine this).

        If CPLEX is available, the the solve operation will be performed using the native CPLEX.
        If CPLEX is not available but if the model has its own DOcloud credentials (set at model creation time),
        the the solve operation will be started on DOcloud, using the model's own DOcloudContex instance.

        If CPLEX is not available and the model has no valid credentials, an error is raised, as there is
        no way to perform the solve.

        Args:
            context: An optional instance of context to be used in instead of
                the context this model was built with.
            docloud_context: An optional instance of DOcloudContext that
                overwrites the ``context.docloud`` DOcloud context service.
            cplex_parameters: An optional set of CPLEX parameters to use
                instead of the parameters defined as
                ``context.cplex_parameters``.
            url: An optional parameter that overwrites the URL of the
                DOcloud service defined by ``context.solver.docloud.url``.
            key: An optional parameter that overwrites the
                authentication key of the DOcloud service defined by
                ``context.solver.docloud.key``.
        :returns: A :class:`docplex.mp.solution.SolveSolution` object if the solve operation succeeded.
            None if the solve operation failed.
        """
        if not self.is_optimized():
            self.info("No objective to optimize - searching for a feasible solution")

        from copy import deepcopy
        # use the provided context if any, or the self.context otherwise
        if kwargs.get('context', None) is not None:
            context = deepcopy(kwargs['context'])
        elif kwargs:
            context = deepcopy(self.context)
            # update the context with provided kwargs
            context.update(kwargs)
        else:
            # no need to copy anything ?
            context = self.context

        # log stuff
        saved_context_log_output = self.context.solver.log_output
        saved_log_output_stream = self.get_log_output()

        try:
            self.set_log_output(context.solver.log_output)

            forced_docloud = self._force_docloud(context, **kwargs)

            have_credentials = False
            error_message = None
            if context.solver.docloud:
                have_credentials, error_message = context.solver.docloud.check_credentials()
                if error_message is not None:
                    warnings.warn(error_message, stacklevel=2)

            if forced_docloud:
                if have_credentials:
                    return self._solve_cloud(docloud_context=context.solver.docloud,
                                             parameters=context.cplex_parameters)
                else:
                    self.fatal("DOcloud context has no valid credentials: {0!s}",
                               context.solver.docloud)
            # from now on docloud_context is None
            elif self.environment.has_cplex:
                # if CPLEX is installed go for it
                return self._solve_local(parameters=context.cplex_parameters)
            elif have_credentials:
                # no context passed as argument, no Cplex installed, try model's own context
                return self._solve_cloud(docloud_context=context.solver.docloud,
                                         parameters=context.cplex_parameters)
            else:
                # no way to solve.. really
                return self.fatal("CPLEX DLL not found: please provide DOcloud credentials")
        finally:
            if saved_log_output_stream != self.get_log_output():
                self.set_log_output_as_stream(saved_log_output_stream)
            if saved_context_log_output != self.context.solver.log_output:
                self.context.solver.log_output = saved_context_log_output

    def _connect_progress_listeners(self):
        # INTERNAL... connect progress listeners (if any) iff problem is mip
        if self._progress_listeners:
            if self.is_MIP():
                self.__engine.connect_progress_listeners(self._progress_listeners)
            else:
                self.info("Model: \"{}\" is not a MIP problem, progress listeners are disabled", self.name)

    def _solve_local(self, parameters):
        """ Starts a solve operation on the local machine.

        Note:
        This method will never try to solve on DOcloud, regardless of whether the model
        has an attached DOcloud context.
        If CPLEX is not available, an error is raised.

        Args:
            parameters: are provided, those parameters are used instead of the
                parameters built-in this model.

        Returns:
            A Solution object if the solve operation succeeded, None otherwise.

        """
        self.notify_start_solve()

        self_solve_hooks = self._solve_hooks
        wk_hook = wk_env.solve_hook
        if wk_hook is not None:
            self_solve_hooks.append(wk_hook)

        # connect progress listeners (if any) iff problem is mip
        self._connect_progress_listeners()

        # call notifyStart on progress listeners
        self._fire_start_solve_listeners()
        # notify hooks only on solve_local
        self_stats = self.get_statistics()
        for h in self_solve_hooks:
            h.notify_start_solve(self, self_stats)  # pragma : no cover

        # --- solve is protected in try/except block
        has_solution = False
        reported_obj = 0
        engine_status = JobSolveStatus.UNKNOWN
        self_engine = self.__engine

        if parameters is not self.parameters:
            saved_params = {p: p.get() for p in self.parameters}
        else:
            saved_params = {}
        try:
            # sync parameters
            self._sync_parameters_to_engine(parameters)

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
            self._solve_details = self_engine.get_solve_details()
            # call notifyEnd on progress listeners
            self._fire_end_solve_listeners(has_solution, reported_obj)

            # call hooks
            for h in self_solve_hooks:
                h.notify_end_solve\
                    (self, has_solution, engine_status, reported_obj, self._make_end_infodict())   # pragma : no cover

            # unplug worker hook if any
            if wk_hook:
                self_solve_hooks.remove(wk_hook)   # pragma : no cover

            # restore parameters in sync with model, if necessary
            if saved_params:
                for p, v in iteritems(saved_params):
                    self_engine.set_parameter(p, v)

        return new_solution

    def get_solve_status(self):
        """ Returns the solve status of the last successful solve.

        If the model has been solved successfully, returns the status stored in the
        model solution. Otherwise returns `JobSolveStatus.UNKNOWN`.

        :returns: the solve status of the last successful solve, an instance of :class:`docloud.status.JobSolveStatus`, or None.
        """
        return self._last_solve_status

    def _solve_cloud(self, docloud_context, parameters=None):
        # see if we can reuse the local docloud engine if any?
        docloud_engine = DOcloudEngine(self, docloud_context=docloud_context)

        self.notify_start_solve()
        self._fire_start_solve_listeners()
        new_solution = docloud_engine.solve(self, parameters=parameters)
        self._set_solution(new_solution)
        self._solve_details = docloud_engine.get_solve_details()

        # store solve status as returned by the engine.
        self._last_solve_status = docloud_engine.get_solve_status()
        ok = new_solution is not None
        objective_value = None
        if ok:
            objective_value = self.objective_value
        self._fire_end_solve_listeners(ok, objective_value)
        # return new_solution in all cases: either None or a solution instance
        return new_solution

    def solve_cloud(self, docloud_context=None, parameters=None):
        """ Starts execution of the model on the cloud.

        This method accepts a context (an instance of DOcloudContext) to be used when
        solving on the cloud. If the context argument is None or invalid, then it will
        use the model's own instance of DOcloudContext, set at model creation time.

        Note:
            This method will always solve the model on the cloud, whether or not CPLEX
            is available on the local machine.

        If docloud_context argument is None and the model has no context attached, an exception is raised.

        Args:
            docloud_context: An optional context to use on the cloud. If None, uses the model's DOcloudContext instance, if any.
            parameters: Optional parameters to be used instead of the
                model built-in parameters. This is a :class:`docplex.mp.params.parameters.ParameterGroup`.
        :returns: A :class:`docplex.mp.solution.SolveSolution` object if the solve operation succeeded, else None.

        """
        if not docloud_context:
            if self.context.solver.docloud:
                if isinstance(self.__engine, DOcloudEngine):
                    return self.solve()
                else:
                    return self._solve_cloud(self.context.solver.docloud,
                                             parameters=parameters)
            else:
                self.fatal("DOcloud context is None: cannot solve on the cloud")
        elif docloud_context.has_credentials():
            return self._solve_cloud(docloud_context, parameters=parameters)
        else:
            self.fatal("DOcloud context has no valid credentials: {0!s}", docloud_context)

    def notify_start_solve(self):
        # INTERNAL
        self._solve_count += 1

    def notify_solve_failed(self):
        pass

    def get_solve_details(self):
        return self._solve_details

    def notify_solve_relaxed(self, relaxed_solution):
        # INTERNAL: used by relaxer
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
        if isinstance(sense_arg, ObjectiveSense):
            return sense_arg
        elif is_string(sense_arg):
            return ObjectiveSense.parse(sense_arg, self.error_handler, do_raise=True)

    def resolve_objective_sense(self, sense_arg):
        return ObjectiveSense.parse(sense_arg, self.error_handler, do_raise=True)

    def solve_lexicographic(self, goals,
                            senses=ObjectiveSense.Minimize,
                            abs_tolerance=1e-5,
                            relative_tolerance=1e-4,
                            dump_pass_files=False,
                            docloud_context=None):
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

        :param docloud_context: An instance of a context or None. If not None, will force all solves
         to be ran on the cloud.

        :return: True if all passes ran successfully.
        """
        old_objective_expr = self.__objective_expr
        old_objective_sense = self.__objective_sense
        if not goals:
            self.error("solve_lexicographic requires a non-empty list of goals")
            return False

        if not is_indexable(goals):
            self.fatal("solve_lexicographic requires an indexable collection of goals, got: {0!s}", goals)
        # --- senses ---
        if senses is None:
            senses = generate_constant(ObjectiveSense.Minimize)
        elif isinstance(senses, ObjectiveSense):
            senses = generate_constant(senses)
        elif is_string(senses):
            senses = generate_constant(senses)
        elif is_iterable(senses):
            pass
        else:
            self.fatal("solve_lexicographic expects as senses: None, min/max or iterable, got: {0!s}", senses)
        iter_senses = iter(senses)
        # --- senses ---

        pass_count = 0
        m = self
        current_status = False
        # currentObjective = -1
        nested_kpi_format = '   - %s ='
        results = []
        actual_goals = []
        tolerance_scheme = _ToleranceScheme(abs_tolerance, relative_tolerance)
        # keep extra constraints, in order to remove them at the end.
        extra_cts = []
        for gi, g in enumerate(goals):
            if isinstance(g, DecisionKPI):
                goal_expr = self._to_linear_expr(g.as_expression())
                goal_name = g.name
            elif isinstance(g, tuple):
                goal_expr = self._to_linear_expr(g[0])
                goal_name = g[1] or goal_expr.name
            else:
                goal_expr = self._to_linear_expr(g)
                goal_name = goal_expr.name or "pass%d" % (gi + 1)
            actual_goals.append((goal_name, goal_expr))

        try:
            for goal_name, goal_expr in actual_goals:
                if goal_expr.is_constant() and pass_count > 1:
                    self.warning("Constant expression in lexicographoic solve: {0!s}, skipped", goal_expr)
                    continue
                pass_count += 1
                next_sense = next(iter_senses)
                sense = self._resolve_sense(next_sense)

                self.trace("lexicographic: starting pass %d, %s: %s", pass_count, sense.action().lower(), goal_name)
                m.set_objective(sense, goal_expr)
                # print("-- current objective is: {0!s}".format(goal_expr))
                if dump_pass_files:
                    pass_basename = 'lexico_%s_pass%d' % (self.name, pass_count)
                    self.dump_as_lp(basename=pass_basename)
                current_status = m.solve(docloud_context=docloud_context)
                if current_status:
                    current_obj = m.objective_value
                    results.append(current_obj)
                    if m.error_handler.prints_trace():
                        self.trace("lexicographic: pass #%d ok with objective=%.4f" % (pass_count, current_obj))
                        m._report_lexicographic_goals(actual_goals, kpi_header_format=nested_kpi_format)
                    tolerance = tolerance_scheme.compute_tolerance(current_obj)
                    if sense.is_minimize():
                        pass_ct = m.add_constraint(goal_expr <= current_obj + tolerance,
                                                   '_ctle_pass_%d' % pass_count)
                    else:
                        pass_ct = m.add_constraint(goal_expr >= current_obj - tolerance,
                                                   '_ctge_pass_%d' % pass_count)

                    # print(">>>> new pass ct is {0!s}".format(pass_ct))
                    extra_cts.append(pass_ct)
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
            #  print("<- end restoring model at end of lexicographic")

        if current_status:
            self.info("lexicographic ok, #passes={0}, results={1!s}", pass_count, results)
            if m.error_handler.prints_info():
                m._report_lexicographic_goals(actual_goals, kpi_header_format=nested_kpi_format)
        else:
            self.warning("lexicographic failed at pass {0}", pass_count)
        return current_status

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
            self.fatal("Model<{0}> has not been solved OK", self.name)

    def set_mip_start(self, mip_start_sol):
        """  Sets a (possibly partial) solution to use as a starting point for a MIP.

        This is valid only for models with binary or integer variables.

        Presently, this is also only valid for a native solve (when CPLEX DLL is installed) but not on DOcloud.

        :param mip_start_sol: The solution object to use as a starting point. This argument should be of type
         :class:`docplex.mp.solution.SolveSolution`.

        """
        if not mip_start_sol:
            self.fatal("Invalid MIP start: {0!s}", mip_start_sol)
        elif not self.is_MIP():
            self.error("Problem has no Integer or Binary variable, cannot use MIP start")
            return
        else:
            if mip_start_sol.check_as_mip_start(self.error_handler):
                self.__mip_start = mip_start_sol

    # def getObjectiveValue(self):
    # # PCO: not documented, to be replaced by objective_value property
    #     return self.objective_value

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

    def _make_output_path(self, extension, basename_arg, path=None):
        self_name = self.name
        raw_basename = resolve_pattern(basename_arg, self_name) if basename_arg else self_name
        if raw_basename.find(" ") > 0:
            basename = raw_basename.replace(" ", "_")
        else:
            basename = raw_basename
        output_dir = path or tempfile.gettempdir()
        if not basename.endswith(extension):
            basename = basename + extension
        path = os.path.join(output_dir, basename)
        return path

    __printer_ctor_dict = {LP_format: LPModelPrinter
                           }

    lp_format = LP_format

    def _get_printer_type(self, exchange_format, do_raise=False):
        """
        :param exchange_format: The format to be used.
        :param do_raise:  A Boolean, raise exception if format is unuspported.
        :return: printer or None.
        """
        printer = self.__printer_ctor_dict.get(exchange_format)
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
                If passed None, the output directory will be ``tempdir.gettempdir()``.

            hide_user_names: A Boolean indicating whether or not to keep user names for
                variables and constraints. If False, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ... for constraints.

        Examples:
            Assuming the model's name is `mymodel`:
            
            >>> m.export_as_lp()
            
            will write ``mymodel.lp`` in ``gettempdir()``
            
            >>> m.export_as_lp(basename="foo")
            
            will write ``foo.lp`` in ``gettempdir()``
            
            >>> m.export_as_lp(basename="foo", path="e:/home/docplex")
            
            will write file ``e:/home/docplex/foo.lp``
            
            >>> m.export_as_lp("e/home/docplex/bar.lp")
            
            will write file ``e:/home/docplex/bar.lp``
            
            >>> m.export_as_lp(basename="docplex_%s", path="e/home/") 
            
            will write file ``e:/home/docplex/docplex_mymodel.lp``.
        """
        return self.export(path, basename, hide_user_names, exchange_format=LP_format)

    def export_as_sav(self, basename=None, path=None):
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
        # path is either a path string or None
        if path is not None and not is_string(path):
            self.fatal("Expecting path string or None, got: {0!s}", path)
        if basename is not None and not is_string(basename):
            self.fatal("Expecting basename string or None, got: {0!s}", basename)
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
                    self.warning("No CPLEX DLl found, cannot write: {0}", path)
            else:
                # a path is not a stream but anyway it will work
                self._export_to_stream(stream=path, hide_user_names=hide_user_names, exchange_format=exchange_format)
            return path
        except IOError:
            self.error("Cannot open file: {0}, model not exported".format(path))
            return None

    def _export_to_stream(self, stream, hide_user_names=False, exchange_format=LP_format):
        printer_type = self._get_printer_type(exchange_format, do_raise=True)
        if printer_type:
            printer = printer_type()
            printer.wrap_lines = self.format_export
            if hide_user_names:
                printer.forget_user_names = True
            printer.printModel(self, stream)

    def export_to_stream(self, stream, hide_user_names=False, exchange_format=LP_format):
        """export_to_stream(self, stream, hide_user_names=False)

        Export the model to an output stream in LP format.

        A stream can be one of:
            - a string, interpreted as a system path,
            - None, interpreted as `stdout`, or
            - a Python file-type object (e.g. a StringIO() instance).
                
        Args:
            stream: An object defining where the output will be sent.
            
            hide_user_names: An optional Boolean indicating whether or not to keep user names for
                variables and constraints. If False, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ,... for constraints. Default is to keep user names.

        """
        self._export_to_stream(stream, hide_user_names, exchange_format)

    def export_as_lp_string(self, hide_user_names=False):
        return self.export_to_string(hide_user_names, LP_format)

    def export_to_string(self, hide_user_names=False, exchange_format=LP_format):
        """export_to_string(self, hide_user_names=False)

        Exports the model to a string.

        The output string contains the model in LP format.

        Args:
            hide_user_names: An optional Boolean indicating whether or not to keep user names for
                variables and constraints. If False, all names are replaced by `x`1, `x2`, ... for variables,
                and `c1`, `c2`, ... for constraints. Default is to keep user names.

        Returns:
            A string, containing the model exported in LP format.
        """
        oss = StringIO()
        self._export_to_stream(oss, hide_user_names, exchange_format)
        return oss.getvalue()

    # -- supported attributes
    _supported_attributes = {"duals": _SolveAttribute("duals", False),
                             "slacks": _SolveAttribute("slacks", False),
                             "reduced_costs": _SolveAttribute("reduced_costs", True)}

    # advanced values
    def _get_engine_attribute(self, arg, clazz, attr_name):
        """
        Queries and returns some solve attribute from the last solve.
        :param arg: Object or iterable.
        :param clazz: The type of the argument.
        :param attr_name: The name of the attribute.
        :return:
        """
        attr_data = self._supported_attributes.get(attr_name)
        if not attr_data:
            self.fatal("Unsupported solve attribute: {0:s}", attr_name)

        if attr_data.requires_solved and not self._can_solve():
            self.fatal('Cannot query attribute {0}, engine has no solve capability', attr_name, self.__engine.name())

        if isinstance(arg, clazz):
            attrs = self._get_engine_attributes_internal([arg], attr_data)
            return attrs[0]
        elif is_iterable(arg):
            if not arg:
                return []
            else:
                return self._get_engine_attributes_internal(arg, attr_data)
        else:
            self.fatal("Attribute {0:s} not available for: {1:s}, expecting object or iterable", attr_name, arg)

    def _get_engine_attributes_internal(self, mobjs, attr_data):
        attr_name = attr_data.name
        is_for_vars = attr_data.is_for_vars
        if is_for_vars:
            indices = [v.index for v in self.iter_variables()]
        else:
            indices = [ct.index for ct in self.iter_constraints()]
        if not self.solution.is_attributes_fetched(attr_data.name):
            # get index to value from engine
            attr_idx_map = self.__engine.get_solve_attribute(attr_name, indices)
            self.solution._store_attribute_result(attr_name, attr_idx_map, is_for_vars)
        return self.solution.get_attribute(mobjs, attr_name)

    def dual_values(self, cts):
        self.check_has_solution()
        duals = self._get_engine_attribute(cts, AbstractConstraint, 'duals')
        return duals

    def slack_values(self, cts):
        self.check_has_solution()
        return self._get_engine_attribute(cts, AbstractConstraint, 'slacks')

    def reduced_costs(self, dvars):
        self.check_has_solution()
        return self._get_engine_attribute(dvars, Var, 'reduced_costs')

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
                       var_value_fmt=None):
        """  Prints the values of the model variables after a solve.

        Only valid after a successful solve. If the model has not been solved successfully, an
        exception is raised.

        Args:
            print_zeros (bool): If False, only non-zero values are printed. Default is False.

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
                              iter_vars=iter_vars)

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
        return iter(self._kpis_by_name.values())

    def kpi_by_name(self, name, try_match=True, match_case=False, do_raise=True):
        """ Fetches a KPI from a string.

        This method fetches a KPI from a string, using either exact naming or trying
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

        Returns:
            The KPI expression if found. If the search fails, either raises an exception or returns a dummy
            constant expression with 0.
        """
        for kpi in self.iter_kpis():
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
                self.fatal("Cannot find any KPI matching: {0:s}", name)
            else:
                return self._get_zero_expr()

    def kpi_value_by_name(self, name, try_match=True, match_case=False, do_raise=True):
        """ Return a KPI value from a KPI name.

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
            The KPI value (a float).
        """
        kpi = self.kpi_by_name(name, try_match, match_case=match_case, do_raise=do_raise)
        return kpi.compute()

    def _make_kpi(self, kpi_arg, kpi_name):
        # return a concrete instance of KPI
        from docplex.mp.kpi import DecisionKPI, FunctionalKPI

        if isinstance(kpi_arg, Expr):
            return DecisionKPI(kpi_arg, kpi_name)
        elif isinstance(kpi_arg, Var):
            return DecisionKPI(kpi_arg, kpi_name)
        elif isinstance(kpi_arg, KPI):
            if kpi_name is None:
                return kpi_arg
            else:
                cloned = kpi_arg.clone()
                cloned.name = kpi_name
                return cloned
        elif is_function(kpi_arg):
            return FunctionalKPI(kpi_arg, self, kpi_name)
        else:
            self.fatal("Cannot interpret this as a KPI: {0!s}. expecting expression, variable or function", kpi_arg)

    def add_kpi(self, kpi_arg, publish_name=None):
        """ Adds a Key Performance Indicator to the model.

        Key Performance Indicators (KPIs) are objects that can be evaluated after a solve().
        Typical use is with decision expressions, the evaluation of which return the expression's solution value.

        KPI values are displayed with the method report_kpis()"

        Args:
            kpi_arg:  Accepted arguments are either an expression, a lambda function with one argument or
                an instance of a subclass of abstract class KPI.

            publish_name: The published name of the KPI: a nonempty, unique, string.
            KPIs are identified by their published name.

        Example:
            mdl.add_kpi(x+y+z, "Total Profit") adds the expression (x+y+z) as a KPI, with the name "Total Profit"

        Returns:
            The newly added KPI instance.

        See Also:
            :func:`report_kpis`,
            :class:`docplex.mp.kpi.KPI`,
            :class:`docplex.mp.kpi.DecisionKPI`
        """
        new_kpi = self._make_kpi(kpi_arg, publish_name)
        new_kpi_name = new_kpi.get_name()
        if new_kpi_name in self._kpis_by_name.keys():
            self.fatal("Duplicate KPI name: {0!s}. KPI names must be unique", new_kpi_name)
        self._kpis_by_name[new_kpi_name] = new_kpi
        return new_kpi

    def _check_progress_listener(self, e):
        if not isinstance(e, ProgressListener):
            self.error("Not a progress listener: {0!s}", e)
            return False
        else:
            return True

    def add_progress_listener(self, listener):
        if self._check_progress_listener(listener):
            self._progress_listeners.append(listener)

    def remove_progress_listener(self, listener):
        if self._check_progress_listener(listener):
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

    def clear_progress_listeners(self):
        self._progress_listeners = []

    def prettyprint(self):
        ModelPrettyPrinter().printModel(self)

    def clone(self, new_name=None):
        """ Makes a deep copy of the model, possibly with a new name.
        Variables, constraints, and objective are copied.

        Args:
            new_name: The new name to use. If None is provided, returns a "Copy of xxx" where xxx is the original model name.

        :returns: An instance of :class:`docplex.mp.model.Model`.
        """
        return self.copy(new_name)

    def copy(self, new_name=None, new_solver_agent=None):
        # INTERNAL
        copy_name = new_name or "Copy of %s" % self.name
        copy_solver_agent = new_solver_agent or self.solver_agent
        copy_model = Model(name=copy_name, solver_agent=copy_solver_agent)

        # clone variable containers
        for ctn in self.iter_var_containers():
            copy_model._add_var_container(ctn.copy(copy_model))

        # clone variables
        var_mapping = {}
        for v in self.iter_variables():
            copied_var = copy_model.var(v.vartype, v.lb, v.ub, v.name)
            var_mapping[v] = copied_var

        # clone constraints
        for ct in self.iter_constraints():
            copied_ct = ct.copy(copy_model, var_mapping)
            copy_model.add_constraint(copied_ct, ct.name)

        # clone objective
        if self.is_optimized():
            copy_model.set_objective(self.objective_sense, self.objective_expr.copy(copy_model, var_mapping))

        # clone kpis
        for kpi in self.iter_kpis():
            copy_model.add_kpi(kpi.copy(copy_model, var_mapping))

        if self.context:
            copy_model.context = self.context.copy()

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
        return self.__factory.is_free_lb(var_lb)

    def is_free_ub(self, var_ub):
        return self.__factory.is_free_ub(var_ub)

    def _sync_constraint_indices(self, ct_iter):
        # INTERNAL: check only when CPLEX is present.
        self_engine = self.__engine
        if self_engine.has_cplex():
            for ct in ct_iter:
                model_index = ct.get_index()
                cpx_index = self_engine.get_ct_index(ct)
                if model_index != cpx_index:
                    self.error \
                        ("indices differ, obj: {0!s}, docplex={1}, CPLEX={2}", ct, model_index,
                         cpx_index)  # pragma : nocover

    def _sync_var_indices(self):
        self_engine = self.__engine
        if self_engine.has_cplex():
            for dvar in self.iter_variables():
                model_index = dvar.get_index()
                cpx_index = self_engine.get_var_index(dvar)
                if model_index != cpx_index:
                    self.error \
                        ("indices differ, obj: {0!s}, docplex={1}, CPLEX={2}", dvar, model_index,
                         cpx_index)  # pragma : nocover

    def end(self):
        """ Terminates a model instance.

        Reclaims all memory consumed by the model.
        Reclaims memory allocated by CPLEX, if any.

        """
        self.clear()
        self._clear_engine(restart=False)

    @property
    def parameters(self):
        return self.context.cplex_parameters

    def set_parameter(self, param, value):
        # DEPRECATED, left only for compatibility....
        param.set(value)

    def get_parameter_from_id(self, parameter_cpx_id):
        """ Finds a parameter from a CPLEX id code.

        Args:
            parameter_cpx_id: A CPLEX parameter id (positive integer, e.g. 2009 is mipgap).

        :returns: An instance of :class:`docplex.mp.params.parameters.Parameter`, if found, else None
        """
        assert parameter_cpx_id >= 0
        for p in self.parameters.generate_params():
            if p.cpx_id == parameter_cpx_id:
                return p
        else:
            return None

    def _sync_parameters_to_engine(self, parameters=None):
        # INTERNAL.
        self_engine = self.__engine
        parameters_to_use = parameters or self.parameters
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
        nb_resets = 0
        for param in self.parameters:
            engine_value = self_engine.get_parameter(param)
            if engine_value != param.default_value:
                param.reset_default_value(engine_value)
                nb_resets += 1
        return nb_resets

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

    def resync(self):
        # INTERNAL
        self.__factory.resync_whole_model()


class AbstractModel(Model):
    def __init__(self, name, context=None, **kwargs):
        Model.__init__(self, name=name, context=context, **kwargs)

    @staticmethod
    def _check_data_args(args, expected_sizemin, do_raise=True):
        msg = None
        if not args:
            msg = "Empty data collection"
        elif len(args) < expected_sizemin:
            msg = "Missing data: expecting %d collections, got: %d" % (expected_sizemin, len(args))
        else:
            pass
        if msg:
            if do_raise:
                raise DOcplexException(msg)
            else:
                print(msg)
                return False
        else:
            return True

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
        self.error_handler.ensure(self.is_valid(), "Model is not valid")
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
        # make sure th emodel is setup
        self.ensure_setup()
        # check data and model if necessary (ddefault is do nothing)
        self.check()
        # insert some last minute code before solve.
        self.before_solve_hook()
        # call solve_run which by default calls solve
        ok = self.solve_run(**kwargs)
        if ok:
            self.post_process()
        return ok

    def solve_run(self, **kwargs):
        return self.solve(**kwargs)

    def run(self, **kwargs):
        ok = self.run_silent(**kwargs)
        if ok:
            self.report()
        return ok
