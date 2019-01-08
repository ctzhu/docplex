# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
CPO Model representation
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

_CONSTRAINT_ID_ALLOCATOR = SafeIdAllocator('_CTRST_')
def _allocate_constraint_identifier():
    """ Allocate a new constraint identifier
    Returns:
        New identifier
    """
    return _CONSTRAINT_ID_ALLOCATOR.allocate()


class CpoModel(object):
    """ Root class for constraint programming models. """
    __slots__ = ('name',             # Name of the model
                 'sourcefile',       # Name of the python source file
                 'varList',          # List of model variables, in declaration order
                 'varSet',           # Set of model variables
                 'exprList',         # List of model root expressions (tuples (expression, location, root))
                 'search_phases',    # List of search phases
                 'starting_point',   # Starting point
                 'name_set',         # Set of variable names
                 'nb_expr_nodes',    # Number of expression nodes
                 'all_expr_set',     # Set of all expression ids already in the model
                 'source_loc',       # Indicate to set in the model information of source location
                 'name_constraints', # Indicate to always name added constraints (for conflict refiner)
                 'map_expr',         # Map of expressions having a name
                )

    def __init__(self, name=None, sfile=None):
        """ Creates a new model.

        Args:
            name:  Model name, None for automatic (source file name).
            sfile: Source file, None for automatic.
        """
        ctx = config.get_default()
        super(CpoModel, self).__init__()
        self.varList          = []
        self.varSet           = set()
        self.exprList         = []
        self.search_phases    = []
        self.starting_point   = None
        self.name_set         = set()
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
        self.sourcefile = sfile

        # Store model name
        if name is None:
            name = utils.get_file_name_only(sfile)
        self.name = name


    def add(self, expr):
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
            # Check named expression
            if expr.has_name():
                # Not a constraint
                self.exprList.append((expr, loc, False))
            else:
                # Add name if required
                if self.name_constraints:
                    expr.set_name(_allocate_constraint_identifier())
                self.exprList.append((expr, loc, True))

        # Append to the map of named expressions
        if expr.has_name():
            self._add_named_expr(expr)


    def remove(self, expr):
        """ Remove an expression from the model.

        This method removes from the model the first occurrence of the expression given as parameter.
        It does not remove the expression it it used as sub-expression of another expression.

        Args:
            expr: Expression to remove.
        Returns:
            True if expression has been removed, False if not found
        """
        for ix, (x, l, r) in enumerate(self.exprList):
            if x is expr:
                del self.exprList[ix]
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
            self.search_phases.append((p, loc, None))

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

        Starting point is available starting with CPO solver version 13.0.0.

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

    def set_source_location(self, sl):
        """ Sets the indicator allowing to attach Python source line
        to model expressions.

        Args:
            sl: Adds the source location indicator. Default value is True.
        """
        self.source_loc = sl
                
    def get_all_variables(self):
        """ Gets the list of all model variables.

        Returns:
            List of model variables. List elements are tuples (Variable, Location).
            Location is a tuple (filename, line number).
        """
        return self.varList

    def get_expressions(self):
        """ Gets the list of model expressions

        Returns:
            List of model expressions
            Each expression is a tuple (expr, loc, root) where loc is a tuple (source_file, line).
        """
        return self.exprList

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
        for (x, l, r) in reversed(self.get_expressions()):
            if isinstance(x, CpoFunctionCall) and x.get_operation().is_optim():
                return x
        return None

    def get_name(self):
        """ Gets the name of the model.

        Returns:
            Name of the model (file name with no path or extension).
        """
        return self.name

    def get_source_file(self):
        """ Gets the name of the source file from which model has been created.

        Returns:
            Python source file name.
        """
        return self.sourcefile

    def print_information(self, out=None):
        """ Prints model information.

        Args:
            out: Output stream or file name, default is sys.stdout.
        """
        if out is None:
            out = sys.stdout
        if isinstance(out, str):
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
        lexpr = self.get_expressions()
        for v, l in avars:
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

        Optional args:
            context:   Complete solving context. If not given, context is the default context that is set in config.py.
            params:    Solving parameters (CpoParameters) that overwrite those in the solving context
            url:       URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key:       Authentication key of the DOcplexcloud service that overwrites the one defined in the solving context.
            (others):  All other context parameters that can be changed.
        Returns:
            Model solution (type CpoModelSolution).
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

        Optional args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        Returns:
            String containing the model.
        """
        return CpoCompiler(self, **kwargs).get_as_string()

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
        # Check list of expressions (will also compare variables)
        if len(self.exprList) != len(other.exprList):
            return False
        for i in range(len(self.exprList)):
            #print("Compare expression " + str(i) + ": " + str(self.exprList[i]) + " with " + str(other.exprList[i]))
            if not self.exprList[i][0].equals(other.exprList[i][0]):
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

    def _add_var_name(self, name):
        """ Check that a variable name is not already used and add it to the namesSet.

        Args:
            name: Variable name to check/add
        Raises:
            CpoException if name already used.
        """
        if (name in self.name_set):
            raise CpoException("Variable name '" + str(name) + "' can not be used twice")
        self.name_set.add(name)

    def _add_named_expr(self, expr):
        """ Add an expression to the map of named expressions.

        Args:
            expr: Named expression
        Raises:
            CpoException if name already used.
        """
        name = expr.get_name()
        if name in self.map_expr:
            raise CpoException("Expression with name '" + name + "' already exists")
        self.map_expr[name] = expr

    def _add_variable(self, var, loc):
        """ Add a variable to the model

        Args:
            var: Variable expression to add
            loc: Variable location
        """
        # Check if variable already in the model
        if not(var in self.varSet):
            name = var.get_name()
            if name is None:
                name = expression._allocate_var_name()
                var.set_name(name)
            self._add_var_name(name)
            self.varSet.add(var)
            self.varList.append(var)
        return(var)
            
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
                        self._add_variable(e, loc)
                    else:
                        # Stack children expressions
                        oprnds = e._get_children()
                        if oprnds is not None:
                            estack.extend(oprnds)

###############################################################################
##  Private Functions
###############################################################################

