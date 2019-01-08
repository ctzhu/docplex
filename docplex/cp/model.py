# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
CPO Model representation
"""

from docplex.cp.expression import *
from docplex.cp.function import *
from docplex.cp.modeler import *
import docplex.cp.expression as expression
from docplex.cp.solver import CpoSolver
from docplex.cp.cpo_compiler import CpoCompiler
import docplex.cp.utils as utils
import inspect
import sys


###############################################################################
##  Public classes
###############################################################################

class CpoModel(object):
    """ Root class for constraint programming models. """
    __slots__ = ('name',         # Name of the model
                 'sourcefile',   # Name of the python source file
                 'varList',      # List of model variables, in declaration order
                 'varSet',       # Set of model variables
                 'exprList',     # List of model root expressions (tuples (expression, location))
                 'srch_phases',  # Search phases list
                 'nameSet',      # Set of variable names
                 'intIdCount',   # Counter for internal ids generated for sub-expressions
                 'nbExprNodes',  # Number of expression nodes
                 'allExprSet',   # Set of all expression ids already in the model
                 'sourceloc',    # Indicate to set in the model information of source location
                 'mapExpr',      # Map of expressions having a name
                )

    def __init__(self, name=None, sfile=None):
        """ Creates a new model.

        Args:
            name:  Model name, None for automatic (source file name).
            sfile: Source file, None for automatic.
        """
        super(CpoModel, self).__init__()
        self.varList         = []
        self.varSet          = set()
        self.exprList        = []
        self.srch_phases     = []
        self.nameSet         = set()
        self.intIdCount      = 0
        self.nbExprNodes     = 0
        self.allExprSet      = set()
        self.sourceloc       = True
        self.mapExpr         = {}
        
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
        """ Adds an expression to the model.

        Args:
            expr: Expression to add.
        Raises:
            CpoException in case of error.
        """
        # Check expression
        #if expr is True:
        #    return
        assert isinstance(expr, CpoExpr), "Argument 'expr' should be a CpoExpr instead of " + str(type(expr))
        
        # Determine source location
        if self.sourceloc:
            f = inspect.currentframe().f_back
            loc = (f.f_code.co_filename, f.f_lineno)
        else:
            loc = None

        # Scan expression to check new variables
        self._scan_expression(expr, loc)
        
        # Append to the map of named expressions    
        if expr.has_name():
            self._add_named_expr(expr)

        # Check search phase
        if expr.get_type() is Type_SearchPhase:
            self.srch_phases.append((expr, loc))
        else:
            # Append to the list of expressions    
            if not expr.is_variable():
                self.exprList.append((expr, loc))
            
    def set_search_phases(self, phases):
        """ Set a list of search phases

        Args:
            phases: Array of search phases
        """
        # Check arguments
        assert is_array_of_type(phases, CpoExpr), "Argument 'phases' should be an array of SearchPhases"
        for p in phases:
            if p.get_type() is not Type_SearchPhase:
                raise AssertionError("Argument 'phases' should be an array of SearchPhases")
            self.add(p)
        # self.srch_phases = phases
       
    def get_search_phases(self):
        """ Returns list of search phases.

        Returns:
            List of search phases (pairs (expression, location)), [] if none.
        """
        return self.srch_phases
        
    def set_source_location(self, sl):
        """ Sets the indicator allowing to attach Python source line
        to model expressions.

        Args:
            sl: Adds the source location indicator. Default value is True.
        """
        self.sourceloc = sl
                
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
            Each expression is a tuple (expr, loc) where loc is a tuple (source_file, line).
        """
        return self.exprList

    def get_expression(self, name):
        """ Gets an expression from its name (expression or variable).

        Args:
            name: Name of the expression.
        Returns:
            Expression, None if not found.
        """
        return self.mapExpr.get(name, None)

    def get_optimization_expression(self):
        """ Gets the optimization expression (maximization or minimization).

        Returns:
            Optimization expression, None if satisfaction problem.
        """
        # Search last optimization expression
        for (x, l) in reversed(self.get_expressions()):
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
        out.write(" - constraints: " + str(len(lexpr)) + ", expression nodes: " + str(self.nbExprNodes) + "\n")

    def solve(self, **kwargs):
        """ Solves the model.

        This method solves the model using the appropriate solver according to the optional parameters
        and/or configuration attributes.

        Args:
            context (optional):   Complete solving context. If not given, context is the default context that is set in config.py.
            params (optional):    Solving parameters (CpoParameters) that overwrite those in the solving context
            url (optional):       URL of the DOcloud service that overwrites the one defined in the solving context.
            key (optional):       Authentication key of the DOcloud service that overwrites the one defined in the solving context.
            (others) (optional):  All other context parameters that can be changed.
        Returns:
            Model solution (type CpoModelSolution).
        """
        solver = CpoSolver(self, **kwargs)
        return solver.solve()

    def export_as_cpo(self, out=None, srcloc=False):
        """ Exports/prints the model in the standard CPO file format.

        Args:
            out:    Target output, stream or file name. Default is sys.stdout.
            srcloc: Indicates to add the model source location information. Default is False.
        """
        cplr = CpoCompiler(self)
        cplr.set_source_location(srcloc)
        cplr.print_model(out)

    def get_cpo_string(self, srcloc=False):
        """ Compiles the model in CPO file format into a string.

        Args:
            srcloc: Indicates to add the model source location information. Default is False.
        Returns:
            String containing the model.
        """
        cplr = CpoCompiler(self)
        cplr.set_source_location(srcloc)
        return cplr.get_as_string()

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
        if (name in self.nameSet):
            raise CpoException("Variable name '" + str(name) + "' can not be used twice")
        self.nameSet.add(name) 

    def _add_named_expr(self, expr):
        """ Add an expression to the map of named expressions.

        Args:
            expr: Named expression
        Raises:
            CpoException if name already used.
        """
        name = expr.get_name()
        if name in self.mapExpr:
            raise CpoException("Expression with name '" + name + "' already exists")
        self.mapExpr[name] = expr

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
            self.varList.append((var, loc))
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
        eset = self.allExprSet
        while estack:
            # Get expression to check
            e = estack.pop()
            t = e.get_type()
            if not t.is_constant():
                eid = id(e)
                if eid not in eset:
                    eset.add(eid)
                    self.nbExprNodes += 1
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

