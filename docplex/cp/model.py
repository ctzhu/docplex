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
 * optional starting point (not available for CPO solver release lower or equal to 12.6.3).

The different model expressions and elements are created using services provided by modules:

 * :mod:`docplex.cp.expression` for the simple expression elements,
 * :mod:`docplex.cp.modeler` to build complex expressions and constraints using the specialized CPO functions.
"""

# Following imports required to allow modeling just importing this module
from docplex.cp.expression import *
from docplex.cp.function import *
from docplex.cp.modeler import *
from docplex.cp.solution import *

# Imports required locally
import docplex.cp.expression as expression
from docplex.cp.solver.solver import CpoSolver
from docplex.cp.cpo_compiler import CpoCompiler
import docplex.cp.config as config
import docplex.cp.utils as utils
import inspect
import sys


###############################################################################
##  Public classes
###############################################################################

class CpoModel(object):
    """ This class is the Python container of a CPO model.
    """
    __slots__ = ('name',             # Name of the model
                 'format_version',   # Version of the CPO format
                 'source_file',      # Name of the python source file
                 'var_list',         # List of model variables, in declaration order
                 'var_set',          # Set of model variables
                 'expr_list',        # List of model root expressions as tuples (expression, location, root)
                 'search_phases',    # List of search phases
                 'starting_point',   # Starting point
                 'nb_expr_nodes',    # Number of expression nodes
                 'all_expr_set',     # Set of all expression ids already in the model
                 'source_loc',       # Indicate to set in the model information of source location
                 'name_constraints', # Indicate to always name added constraints (for conflict refiner)
                 'map_expr',         # Map of expressions by name
                )

    def __init__(self, name=None, sfile=None):
        """ Constructor.

        Args:
            name:  Model name, None for automatic (source file name).
            sfile: Source file, None for automatic.
        """
        ctx = config.get_default()
        super(CpoModel, self).__init__()
        self.format_version   = None
        self.var_list          = []
        self.var_set           = set()
        self.expr_list         = []
        self.search_phases    = []
        self.starting_point   = None
        self.nb_expr_nodes    = 0
        self.all_expr_set     = set()
        self.source_loc       = ctx.get_by_path("model.add_source_location", True)
        self.name_constraints = ctx.get_by_path("model.name_all_constraints", False)
        self.map_expr         = {}

        # Store filename of the calling Python source
        if sfile is None:
            mod = inspect.getmodule(inspect.stack()[1][0])
            if mod is not None:
                sfile = mod.__file__
            else:
                sfile = "UnknownSource.py"
        self.source_file = sfile

        # Store model name
        if name is None:
            name = utils.get_file_name_only(sfile)
        self.name = name


    def __enter__(self):
        # Implemented for compatibility with cplex
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        # Implemented for compatibility with cplex
        return False  # No exception handling


    def add(self, expr, root=True):
        """ Adds a CP expression to the model.

        This method adds a CP expression to the model. The expression is scanned to identify all variables
        that are referenced by the expression to automatically add them to the model.

        The order in which expressions are added to the model is preserved when it is submitted for solving.

        Args:
            expr: Expression to add.
        Raises:
            CpoException in case of error.
        """
        # Check expression
        assert isinstance(expr, CpoExpr), "Argument 'expr' should be a CpoExpr instead of " + str(type(expr))
        #assert expr.is_constraint_or_bool_expr(), "An expression added to the model should be a constraint or a boolean expression."
        #print("Adding expression: {}".format(expr))
        
        # Determine calling location
        if self.source_loc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Scan expression to check new variables
        self._scan_expression(expr, loc)
        
        # Append to the list of expressions
        if not expr.is_variable():
            # Check constraints
            if expr.is_constraint_or_bool_expr():
                # Check if expression is named
                if expr.has_name():
                    # Ignore if same than previous (for named constraints declared then added)
                    l = len(self.expr_list)
                    if (l > 0) and self.expr_list[l - 1][0] is expr:
                        self.expr_list[l - 1] = (expr, loc, True)
                        return
                else:
                    # Add name if required
                    if self.name_constraints:
                        expr.name = expression._CONSTRAINT_ID_ALLOCATOR.allocate()
                self.expr_list.append((expr, loc, root))
            elif expr.is_type(Type_SearchPhase):
                self.search_phases.append((expr, loc, False))
            else:
                self.expr_list.append((expr, loc, False))

            # Add to the map of named expressions
            if expr.has_name():
                self._add_named_expr(expr, expr.get_name())


    def remove(self, expr):
        """ Remove an expression from the model.

        This method removes from the model the first occurrence of the expression given as parameter.
        It does not remove the expression if it used as sub-expression of another expression.

        Args:
            expr: Expression to remove.
        Returns:
            True if expression has been removed, False if not found
        """
        for ix, (x, l, r) in enumerate(self.expr_list):
            if x is expr:
                del self.expr_list[ix]
                if expr.has_name():
                    del self.map_expr[expr.get_name()]
                return True
        return False


    def set_search_phases(self, phases):
        """ Set a list of search phases

        Args:
            phases: Array of search phases
        """
        # Check arguments
        assert is_array(phases), "Argument 'phases' should be a list of SearchPhases"

        # Determine calling location
        if self.source_loc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Set list of phases
        self.search_phases = []
        for p in phases:
            if p.get_type() is not Type_SearchPhase:
                raise AssertionError("Argument 'phases' should be an array of SearchPhases")
            self._scan_expression(p, loc)
            self.search_phases.append((p, loc, False))


    def add_search_phase(self, phase):
        """ Add a search phase to the list of search phases

        Args:
            phase: Phase to add to the list
        """
        # Check arguments
        assert isinstance(phase, CpoExpr) and phase.is_type(Type_SearchPhase), "Argument 'phase' should be a SearchPhases"

        # Determine calling location
        if self.source_loc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Append to list of phases
        self.search_phases.append((phase, loc, False))

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

        Starting point is not available for CPO solver release lower or equal to 12.6.3.

        Args:
            stpoint: Starting point, object of class CpoModelSolution
        """
        assert (stpoint is None) or isinstance(stpoint, CpoModelSolution), "Argument 'stpoint' should be None or an object of class CpoModelSolution"
        self.starting_point = stpoint


    def get_starting_point(self):
        """ Get the model starting point

        Returns:
            Model starting point, None if none
        """
        return self.starting_point


    def get_all_variables(self):
        """ Gets the list of all model variables.

        Returns:
            List of model variables. List elements are tuples (Variable, Location).
            Location is a tuple (filename, line number).
        """
        return self.var_list


    def get_all_expressions(self):
        """ Gets the list of all model expressions

        Returns:
            List of model expressions
            Each expression is a tuple (expr, loc, root) where loc is a tuple (source_file, line).
        """
        return self.expr_list


    def get_expression(self, name):
        """ Gets an expression from its name (expression or variable).

        Args:
            name: Name of the expression.
        Returns:
            Expression, None if not found.
        """
        return self.map_expr.get(name)


    def get_optimization_expression(self):
        """ Gets the optimization expression (maximization or minimization).

        Returns:
            Optimization expression, None if satisfaction problem.
        """
        # Search last optimization expression
        for (x, l, r) in reversed(self.get_all_expressions()):
            if isinstance(x, CpoFunctionCall) and x.get_operation().is_optim():
                return x
        return None


    def get_name(self):
        """ Gets the name of the model.

        Returns:
            Name of the model (file name with no path or extension).
        """
        return self.name


    def get_format_version(self):
        """ Gets the version of the CPO format

        Returns:
            String containing the version of the CPO format. None for default.
        """
        return self.format_version


    def get_source_file(self):
        """ Gets the name of the source file from which model has been created.

        Returns:
            Python source file name.
        """
        return self.source_file


    def print_information(self, out=None):
        """ Prints model information.

        Args:
            out: Output stream or file name, default is sys.stdout.
        """
        if out is None:
            out = sys.stdout
        if is_string(out, str):
            make_directories(os.path.dirname(out))
            with open(out, 'w') as f:
                self._write_information(f)
        else:
            self._write_information(out)


    def _write_information(self, out):
        """ Write model information on a output stream

        Args:
            out: Output stream
        """
        out.write("Model: " + self.get_name() + "\n")
        out.write(" - source file: " + str(self.get_source_file()) + "\n")
        nbintvar = 0
        nbintervvar = 0
        nbsequencevar = 0
        avars = self.get_all_variables()
        lexpr = self.get_all_expressions()
        for v in avars:
            if isinstance(v, CpoIntVar):
                nbintvar += 1
            elif isinstance(v, CpoIntervalVar):
                nbintervvar += 1
            elif isinstance(v, CpoSequenceVar):
                nbsequencevar += 1
        out.write(" - variables: " + str(len(avars)))
        out.write(" (integer: " + str(nbintvar))
        out.write(", interval: " + str(nbintervvar))
        out.write(", sequence: " + str(nbsequencevar) + ")\n")
        out.write(" - constraints: " + str(len(lexpr)) + ", expression nodes: " + str(self.nb_expr_nodes) + "\n")


    def solve(self, **kwargs):
        """ Solves the model.

        This method solves the model using the appropriate solver according to the optional parameters
        and/or configuration attributes.

        It is equivalent to call the method :meth:`docplex.cp.solver.solver.CpoSolver.solve` on a CpoSolver object
        created with this model as parameter.

        The class :class:`docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions such as
        conflict refiner, propagation, solution iteration, etc.
        Please refer to this class for more details (see :mod:`docplex.cp.solver.solver`).

        Args:
            context:   Complete solving context. If not given, context is the default context that is set in config.py.
            params:    Solving parameters (CpoParameters) that overwrite those in the solving context
            url:       URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key:       Authentication key of the DOcplexcloud service that overwrites the one defined in the solving context.
            (others):  All other context parameters that can be changed.
        Returns:
            Model solve result (type CpoSolveResult).
        """
        solver = CpoSolver(self, **kwargs)
        return solver.solve()


    def export_as_cpo(self, out=None, **kwargs):
        """ Exports/prints the model in the standard CPO file format.

        Args:
            out:    Target output, stream or file name. Default is sys.stdout.
        Optional args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        """
        CpoCompiler(self, **kwargs).print_model(out)


    def get_cpo_string(self, **kwargs):
        """ Compiles the model in CPO file format into a string.

        Args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        Returns:
            String containing the model.
        """
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

        # Compare variables
        lx1 = self.var_list
        lx2 = other.var_list
        if len(lx1) != len(lx2):
            raise CpoException("Different number of variables, {} vs {}.".format(len(lx1), len(lx2)))
        for i in range(len(lx1)):
            if not lx1[i].equals(lx2[i]):
                raise CpoException("The variable {} differs: {} vs {}".format(i, lx1[i], lx2[i]))

        # Compare expressions
        lx1 = self.expr_list
        lx2 = other.expr_list
        if len(lx1) != len(lx2):
            raise CpoException("Different number of expressions, {} vs {}.".format(len(lx1), len(lx2)))
        for i in range(len(lx1)):
            #print("Check expression {}\n   and\n{}".format(lx1[i][0], lx2[i][0]))
            if not lx1[i][0].equals(lx2[i][0]):
                raise CpoException("The expression {} differs: {} vs {}".format(i, lx1[i][0], lx2[i][0]))

        # Compare search phases
        lx1 = self.search_phases
        lx2 = other.search_phases
        if len(lx1) != len(lx2):
            raise CpoException("Different number of search phases, {} vs {}.".format(len(lx1), len(lx2)))
        for i in range(len(lx1)):
            if not lx1[i][0].equals(lx2[i][0]):
                raise CpoException("The search phase {} differs: {} vs {}".format(i, lx1[i][0], lx2[i][0]))


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


    def _add_named_expr(self, expr, name):
        """ Add an expression to the map of named expressions.

        Args:
            expr: Expression
            name: Expression name
        Raises:
            CpoException if name already used for another expression
        """
        #print("Add named expression: {}: {}".format(name, expr))
        oexpr = self.map_expr.get(name)
        if (oexpr is not None) and (oexpr is not expr):
            raise CpoException("Expression with name '{}' already exists: {}".format(name, oexpr))
        self.map_expr[name] = expr


    def _add_variable(self, var):
        """ Add a variable to the model

        Args:
            var: Variable expression to add
        """
        # Check if variable has a name
        name = var.get_name()
        if name is None:
            name = expression._allocate_var_name()
            var.set_name(name)
        else:
            # Check if variable already in expressions
            ov = self.map_expr.get(name)
            if ov is var:
                return
            elif ov is not None:
                raise CpoException("Variable name '" + str(name) + "' is already used.")
        # Add variable in structures
        self.map_expr[name] = var
        self.var_set.add(var)
        self.var_list.append(var)


    def _scan_expression(self, expr, loc):
        """ Scan an expression to add all referenced variables and update statistics

        Args:
            expr: Expression to scan
            loc:  Expression location
        Raises:
            CpoException if error detected
        """
        # Loop while expression stack is not empty
        estack = [expr]  
        eset = self.all_expr_set
        while estack:
            # Get expression to check
            e = estack.pop()
            t = e.get_type()
            if not t.is_constant():
                eid = id(e)
                if eid not in eset:
                    eset.add(eid)
                    self.nb_expr_nodes += 1
                    if (t.is_variable()):
                        self._add_variable(e)
                    else:
                        # Stack children expressions
                        oprnds = e._get_children()
                        if oprnds is not None:
                            estack.extend(oprnds)


    def _ensure_all_root_constraints_named(self):
        """ Check if all top-level constraints have a name.

        If not all constraints have a name, this method assign a name to them and returns False.

        If all are already named, this method returns True.

        Return:
            True if all constraints are already named, False otherwise.
        """
        # Check if constraints already named
        if self.name_constraints:
            return True
        # Loop on each top-level constraints
        for (expr, l, r) in self.expr_list:
            if r and (not expr.has_name()):
                name = expression._CONSTRAINT_ID_ALLOCATOR.allocate()
                expr.set_name(name)
                self._add_named_expr(expr, name)
        self.name_constraints = True
        return False


###############################################################################
##  Private Functions
###############################################################################

