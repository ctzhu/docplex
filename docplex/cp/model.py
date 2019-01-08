# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains principally the class :class:`CpoModel` that handles all the
elements that compose a CPO model:

 * the variables of the domain (integer variables, interval variables, sequence variables and state functions),
 * the constraints of the model,
 * optional objective value(s),
 * optional search phases,
 * optional starting point (available for CPO solver release greater or equal to 12.7.0).

The different model expressions and elements are created using services provided by modules:

 * :mod:`docplex.cp.expression` for the simple expression elements,
 * :mod:`docplex.cp.modeler` to build complex expressions and constraints using the specialized CP Optimizer functions.

The solving of the model is handled by an object of class :class:`~docplex.cp.solver.solver.CpoSolver` that takes
this model as parameter.
However, most important solving functions are callable directly from this model to avoid explicit
creation of the *CpoSolver* object:

 * :meth:`~CpoModel.solve` solves the model and returns an object of class :class:`~docplex.cp.solution.CpoSolveResult`.
 * :meth:`~CpoModel.start_search` creates a solver that can iterate over multiple solutions of the model.
 * :meth:`~CpoModel.refine_conflict` identifies a minimal conflict for the infeasibility and return it as an object
   of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
 * :meth:`~CpoModel.propagate` invokes the propagation on the current model and returns a partial solution in an object
   of class :class:`~docplex.cp.solution.CpoSolveResult`.

All these methods are taking a variable number of optional parameters that allow to modify the solving context.
The list of arguments is not limited. Each named argument is used to replace the leaf attribute that has
the same name in the global *context* structure initialized in the module :mod:`docplex.cp.config` and its
customizations.

The most important of these parameters are:

 * **context** sets a complete customized context to be used instead of the default one defined in the module :mod:`docplex.cp.config`,
 * **params** overwrites the solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
   that are defined in the *context* object,
 * **url** and **key** modify access to *DOcplexcloud* (if it is the selected solving agent),
 * **agent** forces the selection of a particular solving agent,
 * **trace_cpo** activates the printing of the model in CPO format before its solve,
 * any CP Optimizer solving parameter, as defined in module :mod:`docplex.cp.parameters`, such as:

    * **TimeLimit** indicates a limit in seconds in the time spent in the solve,
      or **ConflictRefinerTimeLimit** that does the same for conflict refiner,
    * **LogVerbosity**, with values in ['Quiet', 'Terse', 'Normal', 'Verbose'],
    * **Workers** specifies the number of threads assigned to solve the model (default value is the number of cores),
    * **SearchType**, with value in ['DepthFirst', 'Restart', 'MultiPoint', 'Auto'], to select a particular solving algorithm,
    * **RandomSeed** changes the seed of the random generator,
    * and so on.

Detailed description
--------------------
"""

# Following imports required to allow modeling just importing this module
from docplex.cp.modeler import *
from docplex.cp.solution import *
from docplex.cp.expression import *
from docplex.cp.function import *
from docplex.cp.solver.solver_listener import CpoSolverListener

# Imports required locally
import docplex.cp.expression as expression
import docplex.cp.modeler as modeler
from docplex.cp.solver.solver import CpoSolver
from docplex.cp.cpo_compiler import CpoCompiler
import docplex.cp.config as config
import docplex.cp.utils as utils
import inspect
import sys
import time
import copy
import types
from collections import namedtuple
from collections import OrderedDict


###############################################################################
##  Constants
###############################################################################

# List of all modeler public functions
_MODELER_PUBLIC_FUNCTIONS = list_module_public_functions(modeler, ('maximize', 'minimize'))
#_MODELER_PUBLIC_FUNCTIONS = list_module_public_functions(modeler)


###############################################################################
##  Public classes
###############################################################################

# Model statistics
class CpoModelStatistics(object):
    """ This class represents model statistics informations.
    """

    def __init__(self, model):
        """ Initialize statisrtics

        Args:
            model: Source model
        """
        self.nb_root_exprs = len(model.expr_list) + len(model.search_phases) + 0 if model.objective is None else 1
        self.nb_integer_var   = 0     # Number of integer variables
        self.nb_interval_var  = 0     # Number of interval variables
        self.nb_expr_nodes    = 0     # Number of expression nodes
        self.operation_usage  = {}    # Map of operation usage count.
                                      # Key is the CPO name of the operation, value is the number of times it is used.

    def _add_expression(self, expr):
        """ Update statistics with an expression node.

        Args:
            expr:  Expression
        """
        self.nb_expr_nodes += 1
        if isinstance(expr, CpoIntVar):
            self.nb_integer_var += 1
        elif isinstance(expr, CpoIntervalVar):
            self.nb_interval_var += 1
        elif isinstance(expr, CpoFunctionCall):
            opname = expr.operation.cpo_name
            self.operation_usage[opname] = self.operation_usage.get(opname, 0) + 1

    def write(self, out=None, prefix=""):
        """ Write the solution

        Args:
            out (Optional):    Target output, as stream or file name. sys.stdout if not given
            prefix (Optional): Prefix added at the beginning of each line
        """
        # Check file
        if is_string(out):
            with open(os.path.abspath(out), mode='w') as f:
                self.write(f)
            return

        if out is None:
            out = sys.stdout

        # Write normal attributes
        out.write("{}number of integer variables:  {}\n".format(prefix, self.nb_integer_var))
        out.write("{}number of interval variables: {}\n".format(prefix, self.nb_interval_var))
        out.write("{}number of expressions:        {}\n".format(prefix, self.nb_root_exprs))
        out.write("{}number of expression nodes:   {}\n".format(prefix, self.nb_expr_nodes))
        out.write("{}operations:                   ".format(prefix))
        if self.operation_usage:
            for i, k in enumerate(sorted(self.operation_usage.keys())):
                if (i > 0):
                    out.write(", ")
                out.write("{}: {}".format(k, self.operation_usage[k]))
        else:
            out.write("None")
        out.write("\n")

    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)

    def __ne__(self, other):
        """ Overwrite inequality comparison """
        return not self.__eq__(other)


class CpoModel(object):
    """ This class is the Python container of a CPO model.
    """

    def __init__(self, name=None, sfile=None):
        """ Constructor.

        Args:
            name:  Model name, None for automatic (source file name).
            sfile: Source file, None for automatic.
        """
        ctx = config.get_default()
        super(CpoModel, self).__init__()
        self.expr_list        = []            # List of model root expressions as tuples (expression, location)
        self.parameters       = None          # Solving parameters
        self.search_phases    = []            # List of search phases
        self.starting_point   = None          # Starting point
        self.objective        = None          # Objective function
        self.kpis             = OrderedDict() # Dictionary of KPIs. Key is publish name.
        self.listeners        = []            # Solver listeners

        # Set version of the CPO format (default)
        self.format_version   = None

        # Indicate to set source location in the model information
        self.source_loc       = ctx.get_by_path("model.add_source_location", True)

        # Initialize times to compute modeling time
        self.create_time      = time.time()        # Model creation absolute time
        self.last_add_time    = self.create_time   # Last time something has been added to the model

        # Store filename of the calling Python source
        if sfile is None:
            mod = inspect.getmodule(inspect.stack()[1][0])
            if mod is not None:
                sfile = mod.__file__
        self.source_file = sfile

        # Store model name
        self.name = name

        # Duplicate constructor functions to make them callable from the model
        self.integer_var       = expression.integer_var
        self.integer_var_list  = expression.integer_var_list
        self.integer_var_dict  = expression.integer_var_dict
        self.binary_var        = expression.binary_var
        self.binary_var_list   = expression.binary_var_list
        self.binary_var_dict   = expression.binary_var_dict
        self.interval_var      = expression.interval_var
        self.interval_var_list = expression.interval_var_list
        self.interval_var_dict = expression.interval_var_dict
        self.sequence_var      = expression.sequence_var
        self.transition_matrix = expression.transition_matrix
        self.tuple_set         = expression.tuple_set
        self.state_function    = expression.state_function

        # Copy all modeler functions in the model object
        for f in _MODELER_PUBLIC_FUNCTIONS:
            setattr(self, f.__name__, f)

        # Special case for builtin functions
        self.min = modeler.min_of
        self.max = modeler.max_of
        self.sum = modeler.sum_of
        self.abs = modeler.abs_of
        self.range = modeler.in_range
        self.all = modeler.all_of
        self.any = modeler.any_of


    def __enter__(self):
        # Implemented for compatibility with cplex
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        # Implemented for compatibility with cplex
        return False  # No exception handling


    def add(self, expr):
        """ Adds a CP expression to the model.

        This method adds a CP expression to the model.
        All the variables that are used by this expression are automatically added to the model.

        The order in which expressions are added to the model is preserved when it is submitted for solving.

        Args:
            expr: Expression to add.
        Raises:
            CpoException in case of error.
        """

        # Check simple boolean expressions
        if is_bool(expr):
            # assert expr, "Try to add an expression which is already false"
            # return
            expr = build_cpo_expr(expr)

        # Check expression
        assert isinstance(expr, CpoExpr), "Argument 'expr' should be a CpoExpr instead of {} (type {})".format(expr, str(type(expr)))

        # Determine calling location
        if self.source_loc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Check type of expression
        etyp = expr.type
        if etyp is Type_SearchPhase:
            self.search_phases.append((expr, loc))
        elif etyp is Type_Objective:
            if self.objective is not None:
                # Remove previous objective from the model
                self.remove(self.objective)
            self.objective = expr
            self.expr_list.append((expr, loc))
        else:
            self.expr_list.append((expr, loc))

        # Update last add time
        self.last_add_time = time.time()


    def remove(self, expr):
        """ Remove an expression from the model.

        This method removes from the model the first occurrence of the expression given as parameter.
        It does not remove the expression if it used as sub-expression of another expression.

        Args:
            expr: Expression to remove.
        Returns:
            True if expression has been removed, False if not found
        """
        etyp = expr.type

        # Process case of search phase
        if etyp is Type_SearchPhase:
            return self._remove_from_expr_list(expr, self.search_phases)

        # Check if it is current objective expression
        if expr is self.objective:
            self.objective = None

        # Remove from list of expressions
        return self._remove_from_expr_list(expr, self.expr_list)


    def minimize(self, expr):
        """ Add an objective expression to minimize.

        DEPRECATED: use add(minimize()) instead.

        Calling this method is equivalent to add(minimize(expr)) except that, if exist,
        the previously defined objective expression is removed to be replaced by this new one.

        Args:
            expr: Expression to minimize.
        Returns:
            Minimization expression that has been added
        """
        # Add new minimization expression
        res = minimize(expr)
        self.add(res)
        return res


    def maximize(self, expr):
        """ Add an objective expression to maximize.

        DEPRECATED: use add(maximize()) instead.

        Calling this method is equivalent to add(maximize(expr)) except that, if exist,
        the previously defined objective expression is removed to be replaced by this new one.

        Args:
            expr: Expression to maximize.
        Returns:
            Maximization expression that has been added
        """
        # Add new maximization expression
        res = maximize(expr)
        self.add(res)
        return res


    def set_parameters(self, params):
        """ Set the solving parameters associated to this model.

        Args:
            params: Solving parameters, object of class :class:`~docplex.cp.parameters.CpoParameters`, or None.
        """
        assert isinstance(params, CpoParameters), "argument 'params' should be an object of class CpoParameters"
        self.parameters = params


    def get_parameters(self):
        """ Get the solving parameters associated to this model.

        Returns:
            Solving parameters, object of class :class:`~docplex.cp.parameters.CpoParameters`, or None if not defined.
        """
        return self.parameters


    def set_search_phases(self, phases):
        """ Set a list of search phases

        Args:
            phases: Array of search phases, or single phase
        """
        # Check arguments
        if not is_array(phases):
            phases = [phases]

        # Determine calling location
        if self.source_loc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Set list of phases
        self.search_phases = []
        for p in phases:
            if not p.is_type(Type_SearchPhase):
                raise AssertionError("Argument 'phases' should be an array of SearchPhases")
            self.search_phases.append((p, loc))


    def add_search_phase(self, phase):
        """ Add a search phase to the list of search phases

        This method is deprecated since release 2.3. Use :meth:`~CpoModel.set_search_phases` or
        :meth:`~CpoModel.add` instead.

        Args:
            phase: Phase to add to the list
        """
        warnings.warn("Method 'add_search_phase' is deprecated since release 2.4.", DeprecationWarning)

        # Check arguments
        assert isinstance(phase, CpoExpr) and phase.is_type(Type_SearchPhase), "Argument 'phase' should be a SearchPhases"

        # Determine calling location
        if self.source_loc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Append to list of phases
        self.search_phases.append((phase, loc))


    def get_search_phases(self):
        """ Get the list of search phases.

        Returns:
            List of search phases (pairs (expression, location)), [] if none.
        """
        return self.search_phases


    def set_starting_point(self, stpoint):
        """ Set a model starting point.

        A starting point specifies a (possibly partial) solution that could be used by CP Optimizer
        to start the search.

        Starting point is available for CPO solver release greater or equal to 12.7.0.

        Args:
            stpoint: Starting point, object of class :class:`~docplex.cp.solution.CpoModelSolution`
        """
        assert (stpoint is None) or isinstance(stpoint, CpoModelSolution), "Argument 'stpoint' should be None or an object of class CpoModelSolution"
        self.starting_point = stpoint


    def get_starting_point(self):
        """ Get the model starting point

        Returns:
            Model starting point, None if none
        """
        return self.starting_point


    def add_kpi(self, expr, name=None):
        # """ Add a Key Performance Indicator to the model.
        #
        # A KPI is an expression whose value is considered as representative of the global solution.
        #
        # The KPI expression can be:
        #
        #  * an integer model variable,
        #  * a Python lambda expression that computes the value of the KPI from the solve result given as parameter.
        #
        # Example of lambda expression used as KPI:
        # ::
        #
        #     mdl = CpoModel()
        #     a = integer_var(0, 3)
        #     b = integer_var(0, 3)
        #     mdl.add(a < b)
        #     mdl.add_kpi(lambda res: (res[a] + res[b]) / 2, "Average")
        #
        # If the model is solved in a cloud context, these KPIs are associated to the objective value in the
        # solve details that are sent periodically to the client.
        #
        # Args:
        #     expr:             Model variable to be used as KPI(s).
        #     name (optional):  Name used to publish this KPI. If absent anf if expression is a variable,
        #                       the variable name is used.
        # """
        assert isinstance(expr, (CpoVariable, types.FunctionType)), "Argument 'expr' should be a model variable or a lambda expression"
        if name is None:
            if isinstance(expr, CpoVariable):
                name = expr.get_name()
        assert name, "A KPI name is mandatory"
        assert not name in self.kpis, "Name '{}' is already used for another KPI.".format(name)
        self.kpis[name] = expr


    def get_kpis(self):
        # """ Returns the dictionary of this model KPIs.
        #
        # Returns:
        #     Ordered dictionary of KPIs. Key is publish name, value is kpi expression.
        #     Keys are sorted in the order the KPIs have been defined.
        # """
        return self.kpis


    def get_all_expressions(self):
        """ Gets the list of all model expressions

        Returns:
            List of model expressions
            Each expression is a tuple (expr, loc) where loc is a tuple (source_file, line).
        """
        return self.expr_list


    def get_all_variables(self):
        """ Gets the list of all model variables.

        This method goes across all model expressions to identify all variables that are pointed by them.
        Calling this method on a big model may be slow.

        Returns:
            List of model variables.
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        estack.extend([x for x, l in self.search_phases])
        if self.objective is not None:
            estack.append(self.objective)

        # Loop while expression stack is not empty
        varlist = []     # Result list
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                if e.type.is_variable:
                    varlist.append(e)
                # Stack children expressions
                estack.extend(e.children)

        return varlist


    def get_named_expressions_dict(self):
        """ Gets a dictionary of all named expressions.

        This method goes across all model expressions to identify all named expressions.
        Calling this method on a big model may be slow.

        Returns:
            Dictionary of all named expressions. Key is expression name, value is expression.
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        estack.extend([x for x, l in self.search_phases])
        if self.objective is not None:
            estack.append(self.objective)
        # Loop while expression stack is not empty
        result = {}      # Result dictionary
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                if e.name:
                    result[e.name] = e
                # Stack children expressions
                estack.extend(e.children)
        return result


    def get_objective_expression(self):
        """ Gets the objective expression (maximization or minimization).

        Returns:
            Objective expression, None if satisfaction problem.
        """
        return self.objective


    def is_minimization(self):
        """ Check if this model represents a minimization problem.

        Returns:
            True if this model represents a minimization problem.
        """
        return self.objective is not None and "min" in self.objective.operation.cpo_name


    def is_maximization(self):
        """ Check if this model represents a maximization problem.

        Returns:
            True if this model represents a maximization problem.
        """
        return self.objective is not None and "max" in self.objective.operation.cpo_name


    def is_satisfaction(self):
        """ Check if this model represents a satisfaction problem.

        Returns:
            True if this model represents a satisfaction problem.
        """
        return self.objective is None


    def get_optimization_expression(self):
        """ Gets the optimization expression (maximization or minimization).

        DEPRECATED. Use :meth:`~CpoModel.get_objective_expression` instead.

        Returns:
            Optimization expression, None if satisfaction problem.
        """
        return self.get_objective_expression()


    def replace_expression(self, oexpr, nexpr):
        """ In all model expressions, replace an expression by another.

        This method goes across all model expressions tree and replace each occurrence of the expression to
        replace by the new expression.
        The comparison of the expression to replace is done by reference (it must be the same object)

        Args:
            oexpr: Expression to replace
            nexpr: Expression to put instead
        Returns:
            Number of replacements done in the model
        """
        # Scan all expressions
        doneset = set()  # Set of expressions already processed
        nbrepl = 0
        for i, (x, l) in enumerate(self.expr_list):
            if x is oexpr:
                self.expr_list[i] = (nexpr, l)
                nbrepl += 1
            elif id(x) not in doneset:
                estack = [x]
                while estack:
                    e = estack.pop()
                    eid = id(e)
                    if eid not in doneset:
                        doneset.add(eid)
                        for cx, c in enumerate(e.children):
                            if c is oexpr:
                                e.children = replace_in_tuple(e.children, cx, nexpr)
                                nbrepl += 1
                            else:
                                estack.append(c)
        return nbrepl


    def get_name(self):
        """ Gets the name of the model.

        If the name is not explicitly defined, the name is the source file name without its extension.
        If source file name is also undefined, name is None.

        Returns:
            Name of the model, None if undefined.
        """
        if self.name is None and self.source_file:
            return utils.get_file_name_only(self.source_file)
        return self.name


    def set_format_version(self, ver):
        """ Set the expected version of the CPO format.

        Args:
            ver:  CPO format version
        """
        self.format_version = ver


    def get_format_version(self):
        """ Gets the version of the CPO format.

        This information is set only when parsing an existing CPO model that contains an explitly a version of the format.
        It is not set when creating a new model.
        It can be set explicitely using :meth:`set_format_version` if a specific CPO format is expected.

        Returns:
            String containing the version of the CPO format. None for default.
        """
        return self.format_version


    def get_source_file(self):
        """ Gets the name of the source file from which model has been created.

        Returns:
            Python source file name. None if undefined.
        """
        return self.source_file


    def get_modeling_duration(self):
        """ Get the time spent in modeling.

        The time is computes as difference between the last time an expression has been added
        and the model object creation time.

        Returns:
            Modeling duration in seconds
        """
        return self.last_add_time - self.create_time


    def get_statistics(self):
        """ Get statistics on the model

        This methods compute statistics on the model.

        Returns:
            Model statistics, object of class class :class:`CpoModelStatistics`.
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        estack.extend([x for x, l in self.search_phases])
        if self.objective is not None:
            estack.append(self.objective)
        result = CpoModelStatistics(self)

         # Loop while expression stack is not empty
        doneset = set()  # Set of ids of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                result._add_expression(e)
                # Stack children expressions
                estack.extend(e.children)

        return result


    def print_information(self, out=None):
        """ Prints model information.

        DEPRECATED. Use :meth:`write_information` instead.

        Args:
            out: Output stream or file name, default is sys.stdout.
        """
        self.write_information(out)


    def write_information(self, out=None):
        """ Write various information about the model.

        This method calls the method :meth:`get_statistics` to retrieve information on the model, and then
        print it with source file name and modeling time.

        Args:
            out: Output stream or file name, default is sys.stdout.
        """
        # Check output
        if is_string(out):
            with open(os.path.abspath(out), mode='w') as f:
                self.write_information(f)
            return

        if out is None:
            out = sys.stdout

        # Print information
        name = self.get_name()
        out.write("Model: {}\n".format(name if name else "Anonymous"))
        sfile = self.get_source_file()
        if sfile:
            out.write(" - source file: {}\n".format(sfile))
        out.write(" - modeling time: {0:.2f} sec\n".format(self.get_modeling_duration()))
        stats = self.get_statistics()
        stats.write(out, " - ")


    def solve(self, **kwargs):
        """ Solves the model.

        This method solves the model using the appropriate :class:`~docplex.cp.solver.solver.CpoSolver`
        created according to default solving context, possibly modified by the parameters of this method.

        The class :class:`~docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions.
        An advanced programming may require to explicitly create a CpoSolver instead of calling function at model level.
        Please refer to this class for more details.

        All necessary solving parameters are taken from the solving context that is constructed from the following list
        of sources, each one overwriting the previous:

           - the parameters that are set in the model itself,
           - the default solving context that is defined in the module :mod:`~docplex.cp.config`
           - the user-specific customizations of the context that may be defined (see :mod:`~docplex.cp.config` for details),
           - the optional arguments of this method.

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            url (Optional):     URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key (Optional):     Authentication key of the DOcplexcloud service that overwrites the one defined in
                                the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Model solve result (object of class :class:`~docplex.cp.solution.CpoSolveResult`).
        Raises:
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self._create_solver(**kwargs)
        msol = solver.solve()
        solver.end()
        return msol


    def start_search(self, **kwargs):
        """ Start a new search sequence to retrieve multiple solutions of the model.

        This method returns a new :class:`~docplex.cp.solver.solver.CpoSolver` object
        that acts as an iterator of the different solutions of the model.
        All solutions can be retrieved using a loop like:
        ::

           lsols = mdl.start_search()
           for sol in lsols:
               sol.write()

        A such solution iteration can be interrupted at any time by calling :meth:`~docplex.cp.solver.solver.CpoSolver.end_search`
        that returns a fail solution including the last solve status.

        Note that, to be sure to retrieve all solutions and only once each,
        recommended parameters are *start_search(SearchType='DepthFirst', Workers=1)*

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            url (Optional):     URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key (Optional):     Authentication key of the DOcplexcloud service that overwrites the one defined in
                                the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Object of class :class:`~docplex.cp.solver.solver.CpoSolver` allowing to iterate over the different solutions.
        Raises:
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self._create_solver(**kwargs)
        return solver


    def refine_conflict(self, **kwargs):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Conflict refiner is controlled by the following parameters, that can be set as parameters of this method:

         * ConflictRefinerBranchLimit
         * ConflictRefinerFailLimit
         * ConflictRefinerIterationLimit
         * ConflictRefinerOnVariables
         * ConflictRefinerTimeLimit

        that are described in module :mod:`docplex.cp.parameters`.

        Note that the general *TimeLimit* parameter is used as a limiter for each conflict refiner iteration, but the
        global limitation in time must be set using *ConflictRefinerTimeLimit* that is infinite by default.

        This method creates a new :class:`~docplex.cp.solver.solver.CpoSolver` with given arguments, and then call
        its method :meth:`~docplex.cp.solver.solver.CpoSolver.refine_conflict`.
        The class :class:`~docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions. An advanced programming may
        require to explicitly create a CpoSolver instead of calling function at model level.
        Please refer to this class for more details.

        This function is available on DOcplexcloud and with local CPO solver with release number greater or equal to 12.7.0.

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            url (Optional):     URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key (Optional):     Authentication key of the DOcplexcloud service that overwrites the one defined in
                                the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            List of constraints that cause the conflict (object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`)
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self._create_solver(**kwargs)
        rsol = solver.refine_conflict()
        solver.end()
        return rsol


    def propagate(self, cnstr=None, **kwargs):
        """ This method invokes the propagation on the current model.

        Constraint propagation is the process of communicating the domain reduction of a decision variable to
        all of the constraints that are stated over this variable.
        This process can result in more domain reductions.
        These domain reductions, in turn, are communicated to the appropriate constraints.
        This process continues until no more variable domains can be reduced or when a domain becomes empty
        and a failure occurs.
        An empty domain during the initial constraint propagation means that the model has no solution.

        The result is a object of class :class:`~docplex.cp.solution.CpoSolveResult`, the same than the one
        returned by the method :meth:`solve`.
        However, variable domains may not be completely defined.

        This method creates a new :class:`~docplex.cp.solver.solver.CpoSolver` with given arguments, and then call
        its method :meth:`~docplex.cp.solver.solver.CpoSolver.propagate`.
        The class :class:`~docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions. An advanced programming may
        require to explicitly create a CpoSolver instead of calling function at model level.
        Please refer to this class for more details.

        This function is available on DOcplexcloud and with local CPO solver with release number greater or equal to 12.7.0.

        Args:
            cnstr (Optional):   Optional constraint to be added to the model before invoking propagation.
                                If not given, solving context is the default one that is defined in the module
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            url (Optional):     URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key (Optional):     Authentication key of the DOcplexcloud service that overwrites the one defined in
                                the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Propagation result (object of class :class:`~docplex.cp.solution.CpoSolveResult`)
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        # Check if an optional constraint has been given
        if cnstr is None:
            mdl = self
        else:
            # Clone the model and add constraint
            mdl = self.clone()
            mdl.add(cnstr)
        # Call propagation
        solver = mdl._create_solver(**kwargs)
        psol = solver.propagate()
        solver.end()
        return psol


    def run_seeds(self, nbrun, **kwargs):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        Result statistics are displayed on the log output that should be activated.
        If the appropriate configuration variable *context.solver.add_log_to_solution* is set to True (default),
        log is also available in the *CpoRunResult* result object, accessible as a string using the method
        :meth:`~docplex.cp.solution.CpoRunResult.get_solver_log`

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        This function is available only with local CPO solver with release number greater or equal to 12.8.

        Args:
            nbrun:              Number of runs with different seeds.
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            url (Optional):     URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key (Optional):     Authentication key of the DOcplexcloud service that overwrites the one defined in
                                the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self._create_solver(**kwargs)
        rsol = solver.run_seeds(nbrun)
        solver.end()
        return rsol


    def add_solver_listener(self, lstnr):
        """ Add a solver listener.

        A solver listener is an object extending the class :class:`~docplex.cp.solver.solver_listener.CpoSolverListener`
        which provides multiple functions that are called to notify about the different solving steps.

        Args:
            lstnr:  Solver listener
        """
        assert isinstance(lstnr, CpoSolverListener), \
            "Listener should be an object of class docplex.cp.solver.solver_listener.CpoSolverListener"
        self.listeners.append(lstnr)


    def remove_solver_listener(self, lstnr):
        """ Remove a solver listener previously added with :meth:`~docplex.cp.model.CpoModel.add_listener`.

        Args:
            lstnr:  Listener to remove.
        """
        self.listeners.remove(lstnr)


    def export_model(self, out=None, **kwargs):
        """ Exports/prints the model in the standard CPO file format.

        Note that calling this method disables automatically all the settings that are set in the default configuration
        to change the format of the model:

         * *context.model.length_for_alias* that rename variables if name is too long,
         * *context.model.name_all_constraints* that force a name for each constraint.

        These options are however possible if explicitly given as parameter of this method, as in:
        ::

           mdl.export_model(length_for_alias=10)

        Args:
            out (Optional):     Target output, stream or file name. Default is sys.stdout.
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            add_source_location (Optional): Add source location into generated text
            length_for_alias (Optional): Minimum name length to use shorter alias instead
            (others) (Optional): Any leaf attribute with the same name in the solving context
        """
        # Remove all code transformations but respect those provided explicitly
        kwargs.setdefault('length_for_alias', None)
        kwargs.setdefault('name_all_constraints', False)

        CpoCompiler(self, **kwargs).write(out)


    def import_model(self, file):
        """ Import a model from a file containing a model expressed in CPO or FZN format.

        FZN format is sipported with restrictions to integer expressions.
        The full list of supported predicates is given in the documentation of module :mod:`~docplex.cp.fzn_parser`.

        Args:
            file: Input file, with extension ".cpo" of ".fzn".
        """
        ext = os.path.splitext(file)[1].lower()
        if ext == ".cpo":
            import docplex.cp.cpo_parser as cpo_parser
            prs = cpo_parser.CpoParser(self)
            prs.parse(file)
        elif ext == ".fzn":
            import docplex.cp.fzn_parser as fzn_parser
            prs = fzn_parser.FznParser(self)
            prs.parse(file)
            prs.get_model()
        else:
            raise CpoException("Unknown {} file format. Only .cpo and .fzn are supported.")


    def export_as_cpo(self, out=None, **kwargs):
        """ Deprecated form of method :meth:`export_model`.
        """
        self.export_model(out, **kwargs)


    def get_cpo_string(self, **kwargs):
        """ Compiles the model in CPO file format into a string.

        Note that calling this method disables automatically all the settings that are set in the default configuration
        to change the format of the model:

         * *context.model.length_for_alias* that rename variables if name is too long,
         * *context.model.name_all_constraints* that force a name for each constraint.

        These options are however possible if explicitly given as parameter of this method, as in:
        ::

           mstr = mdl.get_cpo_string(length_for_alias=10)

        Args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        Returns:
            String containing the model.
        """
        # Remove all code transformations but respect those provided explicitly
        kwargs.setdefault('length_for_alias', None)
        kwargs.setdefault('name_all_constraints', False)

        return CpoCompiler(self, **kwargs).get_as_string()


    def check_equivalence(self, other):
        """ Checks that this model is equivalent to another.

        Variables and expressions are compared, but not names that may differ because of automatic naming.

        Args:
            other:  Other model to compare with.
        Raises:
            Exception if models are not equivalent
        """

        # Check object types
        if not isinstance(other, CpoModel):
            raise CpoException("Other model is not an object of class CpoModel")

        # Compare expressions that are not variables
        lx1 = [x for x, l in self.expr_list if not isinstance(x, (CpoVariable, CpoValue, CpoAlias, CpoFunction))]
        lx2 = [x for x, l in other.expr_list if not isinstance(x, (CpoVariable, CpoValue, CpoAlias, CpoFunction))]
        if len(lx1) != len(lx2):
            raise CpoException("Different number of expressions, {} vs {}.".format(len(lx1), len(lx2)))
        for i, (x1, x2) in enumerate(zip(lx1, lx2)):
            #print("Check expression {}\n   and\n{}".format(lx1[i], lx2[i]))
            if not x1.equals(x2):
                print("X1 = {}".format(x1))
                print("X2 = {}".format(x2))
                raise CpoException("The expression {} differs: {} vs {}".format(i, x1, x2))

        # Compare search phases
        lx1 = self.search_phases
        lx2 = other.search_phases
        if len(lx1) != len(lx2):
            raise CpoException("Different number of search phases, {} vs {}.".format(len(lx1), len(lx2)))
        for i, (x1, x2) in enumerate(zip(lx1, lx2)):
            if not x1[0].equals(x2[0]):
                raise CpoException("The search phase {} differs: {} vs {}".format(i, x1[0], x2[0]))


    def equals(self, other):
        """ Checks if this model is equal to another.

        Args:
            other:  Other model to compare with.
        Returns:
            True if models are identical, False otherwise.
        """
        # Check object types
        if not isinstance(other, CpoModel):
            return False
        # Do not compare variables as there may me more with Python as all are named (for example SequenceVar)
        # Check list of expressions (will also compare variables)
        if len(self.expr_list) != len(other.expr_list):
            return False
        for x1, x2 in zip(self.expr_list, other.expr_list):
            if not x1[0].equals(x2[0]):
                # print("different expressions: \n1: {}\n2: {}".format(x1[0], x2[0]))
                return False
        return True


    def clone(self):
        """ Create a copy of this model """
        res = copy.copy(self)
        res.expr_list = list(self.expr_list)
        if self.parameters is not None:
            res.parameters = self.parameters.copy()
        res.search_phases = list(self.search_phases)
        return res


    def __eq__(self, other):
        """ Check if this model is equal to another

        Args:
            other:  Other model to compare with
        Returns:
            True if models are identical, False otherwise
        """
        return self.equals(other)


    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


    def __str__(self):
        """ Convert the model into string (returns model name) """
        return self.get_name()


    def _create_solver(self, **kwargs):
        """ Create a new solver instance attached to this model

        Args:
            kwargs: Parameters to pass to solver creation
        Returns:
            New solver properly initialized.
        """
        solver = CpoSolver(self, **kwargs)
        for l in self.listeners:
            solver.add_listener(l)
        return solver


    def _remove_from_expr_list(self, expr, elist):
        """ Remove an expression from a list of expressions (and map of names)
        Args:
            expr:  Expression to remove.
            elist: List of expressions where search
        Returns:
            True if expression has been removed, False if not found
        """
        for ix, (x, l) in enumerate(elist):
            if x is expr:
                del elist[ix]
                return True
        return False


    def _search_named_expression(self, name):
        """ Search in the model the first expression whose name is the given one.

        This method goes across all model expressions to search for named expression.
        Calling this method on a big model may be slow.

        Args:
            name:  Name of the expression to search
        Returns:
            Expression, None if not found
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        estack.extend([x for x, l in self.search_phases])
        if self.objective is not None:
            estack.append(self.objective)
        # Loop while expression stack is not empty
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                if e.name == name:
                    return e
                # Stack children expressions
                doneset.add(eid)
                estack.extend(e.children)
        return None



###############################################################################
##  Private Functions
###############################################################################

