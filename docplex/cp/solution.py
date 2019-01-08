# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Representation of a solution of a constraint programming model.

This module implements one class per solution element:

 * CpoIntVarSolution: solution of an integer variable,
 * CpoIntervalVarSolution: solution of an interval variable,
 * CpoSequenceVarSolution: solution of a sequence variable,
 * CpoStateFunctionSolution: solution of a state function, and
 * CpoModelSolution: aggregation of all individual model element solutions,
   plus some other information regarding the solve.
"""

import os, sys, json

from docplex.cp.expression import CpoExpr, INT_MIN, INT_MAX, INTERVAL_MIN, INTERVAL_MAX, _ALIAS_ALLOCATOR
from docplex.cp.utils import make_directories, IdentityAccessor

###############################################################################
##  Public constants
###############################################################################

# Solve status: Unknown
SOLVE_STATUS_UNKNOWN = "Unknown"

# Solve status: Infeasible
SOLVE_STATUS_INFEASIBLE = "Infeasible"

# Solve status: Feasible
SOLVE_STATUS_FEASIBLE = "Feasible"

# Solve status: Optimal
SOLVE_STATUS_OPTIMAL = "Optimal"

# Solve status: Job aborted
SOLVE_STATUS_JOB_ABORTED = "JobAborted"

# Solve status: Job failed
SOLVE_STATUS_JOB_FAILED = "JobFailed"

# List of all possible search statuses
ALL_SOLVE_STATUSES = (SOLVE_STATUS_UNKNOWN,
                      SOLVE_STATUS_INFEASIBLE, SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL,
                      SOLVE_STATUS_JOB_ABORTED, SOLVE_STATUS_JOB_FAILED)

# Fail status: Unknown
FAIL_STATUS_UNKNOWN = "Unknown"

# Fail status: Failed normally
FAIL_STATUS_FAILED_NORMALLY = "SearchHasFailedNormally"

# Fail status: Not failed (success)
FAIL_STATUS_NOT_FAILED = "SearchHasNotFailed"

# Fail status: Stopped by abort
FAIL_STATUS_ABORT = "SearchStoppedByAbort"

# Fail status: Stopped by exception
FAIL_STATUS_EXCEPTION = "SearchStoppedByException"

# Fail status: Stopped by exit
FAIL_STATUS_EXIT = "SearchStoppedByExit"

# Fail status: Stopped by label
FAIL_STATUS_LABEL = "SearchStoppedByLabel"

# Fail status: Stopped by time limit
FAIL_STATUS_TIME_LIMIT = "SearchStoppedByLimit"

# List of all possible search statuses
ALL_FAIL_STATUSES = (FAIL_STATUS_UNKNOWN,
                     FAIL_STATUS_FAILED_NORMALLY, FAIL_STATUS_NOT_FAILED,
                     FAIL_STATUS_ABORT, FAIL_STATUS_EXCEPTION, FAIL_STATUS_EXIT, FAIL_STATUS_LABEL,
                     FAIL_STATUS_TIME_LIMIT)


###############################################################################
##  Public classes
###############################################################################

class CpoVarSolution(object):
    """ Super class for solution to a variable. """
    __slots__ = ('name',  # Variable name
                 )
    
    def __init__(self, name):
        """ Creates a new solution for a variable.
        """
        super(CpoVarSolution, self).__init__()
        self.name  = name
        
    def get_name(self):
        """ Gets the name of the variable.

        Returns:
            Variable name.
        """
        return self.name
    
    def get_value(self):
        """ Gets the variable value.
        This method is overloaded by each class extending this class.

        Returns:
            None.
        """
        return None

     
class CpoIntVarSolution(CpoVarSolution):
    """ Solution to an integer variable. """
    __slots__ = ('value'  # Variable value
                )
    
    def __init__(self, name, value):
        """ Creates a new int variable solution.
        """
        super(CpoIntVarSolution, self).__init__(name)
        self.value = value
        
    def get_value(self):
        """ Gets the variable value.

        Returns:
            Variable value (integer).
        """
        return self.value
    
    def __str__(self):
        """ Convert this expression into a string """
        return self.get_name() + ": " + str(self.get_value())
        

class CpoIntervalVarSolution(CpoVarSolution):
    """ Solution to an interval variable. """
    __slots__ = ('present',  # Presence indicator
                 'start',    # Interval start
                 'end',      # Interval end
                 'size',     # Interval size
                )
    
    def __init__(self, name, present, start=0, end=0, size=0):
        """ Creates a new interval variable solution.

        Args:
            name:    Name of the variable.
            present: Present interval indicator.
            start:   Value of start.
            end:     Value of end.
            size:    Value of size.
        """
        super(CpoIntervalVarSolution, self).__init__(name)
        self.present = present
        self.start = start
        self.end   = end
        self.size  = size
        
    def is_present(self):
        """ Gets whether the interval is present.

        Returns:
            True if interval is present.
        """
        return self.present

    def get_start(self):
        """ Gets the interval start.

        Returns:
            Interval start.
        """
        return self.start

    def get_end(self):
        """ Gets the interval end.

        Returns:
            Interval end.
        """
        return self.end

    def get_size(self):
        """ Gets the interval size.

        Returns:
            Interval size.
        """
        return self.size

    def get_value(self):
        """ Gets the interval variable value as a tuple (start, end, size), or () if absent.

        Returns:
            Interval variable value as tuple.
        """
        if (self.present):
            return (self.start, self.end, self.size)
        return ()
    
    def __str__(self):
        """ Convert this expression into a string """
        if (self.present):
            return self.get_name() + ": (start=" + str(self.get_start()) + ", end=" + str(self.get_end()) + ", size=" + str(self.get_size()) + ")"
        return self.get_name() + ": absent"
        
     
class CpoSequenceVarSolution(CpoVarSolution):
    """ Solution to a sequence variable. """
    __slots__ = ('lvars',  # List of interval variable solutions
                )
    
    def __init__(self, name, lvars):
        """ Creates a new sequence variable solution.

        Args:
            lvars: List of interval variables (objects CpoIntervalVarSolution).
        """
        super(CpoSequenceVarSolution, self).__init__(name)
        self.lvars = lvars
        
    def get_interval_variables(self):
        """ Gets the list of CpoIntervalVarSolution of this sequence.

        Returns:
            List of CpoIntervalVarSolution of this sequence.
        """
        return self.lvars

    def get_value(self):
        """ Gets the list of CpoIntervalVarSolution of this sequence.

        Returns:
            List of CpoIntervalVarSolution of this sequence.
        """
        return self.lvars
    
    def __str__(self):
        """ Convert this expression into a string """
        #print("List of variables: " + str(self.lvars))
        return self.get_name() + ": (" + ", ".join([v.get_name() for v in self.lvars]) + ")"
        
     
class CpoStateFunctionSolution(CpoVarSolution):
    """ Solution to a step function. """
    __slots__ = ('steps',  # List of function steps
                )
    
    def __init__(self, name, steps):
        """ Creates a new state function solution.

        Args:
            steps: List of function steps (tuples (start, end, value)).
        """
        super(CpoStateFunctionSolution, self).__init__(name)
        self.steps = steps
        
    def get_function_steps(self):
        """ Gets the list of function steps.

        Returns:
            List of function steps. Each step is a tuple (start, end, value).
        """
        return self.steps

    def get_value(self):
        """ Gets the list of function steps. Identical to `get_function_steps()`.

        Returns:
            List of function steps.
        """
        return self.steps
    
    def __str__(self):
        """ Convert this expression into a string """
        return self.get_name() + ": (" + ", ".join([str(s) for s in self.steps]) + ")"
        
     
class CpoModelSolution(object):
    """ Solution to a constraint programming model.

    The elements of the solutions are accessible through the different accessors that are
    provided by this class.
    For convenience reasons, more direct access is possible using the following attributes:

     * vars: Dictionary of all variables solutions. Key is variable name, value is
       either CpoIntVarSolution, CpoIntervalVarSolution, CpoSequenceVarSolution, CpoStateFunctionSolution.
     * parameters: Map of solving parameters. Key is parameter name, value is string, int or float.
     * infos: Map of solving information. Key is attribute name, value is string, int or float
    """
    
    def __init__(self):
        """ Creates a new empty solution.
        """
        super(CpoModelSolution, self).__init__()
        self.solve_status = SOLVE_STATUS_UNKNOWN   # Solve status
        self.fail_status = FAIL_STATUS_UNKNOWN     # Fail status
        self.nbintvars = 0                         # Number of integer variables
        self.nbintervalvars = 0                    # Number of interval variables
        self.nbsequencevars = 0                    # Number of sequence variables
        self.nbconstraints = 0                     # Number of constraints
        self.objvalues = None                      # List of objective values
        self.solveTime = 0                         # Solve time
        self.solverLog = None                      # Solver log
        self.vars = {}                             # Map of variable solutions
        self.parameters = {}                       # Solving parameters map
        self.infos = {}                            # Solving information attributes map
        
    def _set_solve_status(self, ssts):
        """ Set the solve status

        Args:
            ssts: Solve status
        """
        self.solve_status = ssts

    def get_solve_status(self):
        """ Gets the solve status.

        Returns:
            Solve status, values in ALL_SOLVE_STATUSES.
        """
        return self.solve_status

    def get_fail_status(self):
        """ Gets the solving fail status.

        Returns:
            Fail status, values in ALL_FAIL_STATUSES.
        """
        return self.fail_status

    def __nonzero__(self):
        """ Check if this descriptor contains a solution to the problem.
        Equivalent to is_solution()

        Returns:
            True if a solution is available (Search status is 'Feasible' or 'Optimal')
        """
        return self.is_solution()

    def __bool__(self):
        """ Check if this descriptor contains a solution to the problem.
        Equivalent to is_solution()

        Equivalent to __nonzero__ for Python 3

        Returns:
            True if a solution is available (Search status is 'Feasible' or 'Optimal')
        """
        return self.is_solution()

    def is_solution(self):
        """ Checks if this descriptor contains a valid solution to the problem.

        A solution is present if the solve status is 'Feasible' or 'Optimal'.
        Optimality of the solution should be tested with `is_solution_optimal()`.

        Returns:
            True if there is a solution.
        """
        return self.solve_status in (SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL)

    def is_solution_optimal(self):
        """ Checks if this descriptor contains an optimal solution to the problem.

        Returns:
            True if there is an optimal solution.
        """
        return self.solve_status is SOLVE_STATUS_OPTIMAL

    def _set_objective_values(self, ovals):
        """ Set the numeric values of all objectives.

        Args:
            ovals: Array of all objective values
        """
        self.objvalues = ovals

    def get_objective_values(self):
        """ Gets the numeric values of all objectives.

        Returns:
            Array of all objective values, None if none.
        """
        return self.objvalues

    def _set_parameters(self, params):
        """ Set the solving parameters.

        Args:
            params: Dictionary of parameters
        """
        self.parameters = params

    def get_parameters(self):
        """ Gets the complete dictionary of solving parameters.
        Identical to get the attribute `parameters` of this object.

        Returns:
            Dictionary of parameters.
        """
        return self.parameters

    def get_parameter(self, name):
        """ Get a particular solving parameter.
        Semantically equivalent to 'parameters[name]' on this object, except that it returns None if not found.

        Returns:
            Parameter value, None if not found.
        """
        return self.parameters.get(name, None)

    def _set_infos(self, infos):
        """ Set the solving information attributes.

        Args:
            infos: Dictionary of information attributes
        """
        self.infos = infos

    def get_infos(self):
        """ Gets the complete dictionary of information attributes.
        Identical to get the attribute 'info' of this object.

        Returns:
            Dictionary of information attributes.
        """
        return self.infos

    def get_info(self, name):
        """ Gets a particular information attribute.
        Semantically equivalent to `info[name]` on this object, except that it returns None if not found.

        Returns:
            Information attribute value, None if not found.
        """
        return self.infos.get(name, None)

    def _set_model_attributes(self, nbintvars=0, nbitvvars=0, nbseqvars=0, nbctrs=0):
        """ Set the numeric values of all objectives.

        Args:
            nbintvars: Number of integer variables
            nbitvvars: Number of interval variables
            nbseqvars: Number of sequence variables
            nbctrs:    Number of constraints
        """
        self.nbintvars = nbintvars
        self.nbintervalvars = nbitvvars
        self.nbconstraints = nbctrs
        self.nbsequencevars = nbseqvars

    def _set_solve_time(self, time):
        """ Set the solve time required for this solution.

        Args:
            time (float): Solve time in seconds
        """
        self.solveTime = time

    def get_solve_time(self):
        """ Gets the solve time required for this solution.

        Returns:
            (float) Solve time in seconds.
        """
        return self.solveTime

    def _set_solver_log(self, log):
        """ Set the solver log as a string.

        Args:
            log (str): Log of the solver
        """
        self.solverLog = log

    def get_solver_log(self):
        """ Gets the solver log.

        Returns:
            Solver log as a string, None if unknown.
        """
        return self.solverLog

    def get_number_of_integer_vars(self):
        """ Gets the number of integer variables in the model.

        Returns:
            Number of integer variables.
        """
        return self.nbintvars

    def get_number_of_interval_vars(self):
        """ Gets the number of interval variables in the model.

        Returns:
            Number of interval variables.
        """
        return self.nbintervalvars

    def get_number_of_sequence_vars(self):
        """ Gets the number of sequence variables in the model.

        Returns:
            Number of sequence variables.
        """
        return self.nbsequencevars

    def get_number_of_constraints(self):
        """ Gets the number of constraints in the model.

        Returns:
            Number of constraints.
        """
        return self.nbconstraints

    def _add_var_solution(self, vsol):
        """ Add a variable solution to this model solution.

        Args:
            vsol: Variable solution
        """
        self.vars[vsol.get_name()] = vsol

    def get_var_solution(self, name):
        """ Gets a variable solution from this model solution.

        Args:
            name: Variable name or variable expression.
        Returns:
            Variable solution (class CpoVarSolution), None if not found.
        """
        if isinstance(name, CpoExpr):
            name = name.get_name()
        return self.vars.get(name, None)

    def get_all_var_solutions(self):
        """ Gets the list of all variable solutions from this model solution.

        Returns:
            List of all variable solutions (class CpoVarSolution).
        """
        return self.vars.values()

    def get_value(self, name):
        """ Gets the value of a variable.

        For IntVar, value is an integer.
        For IntervalVar, value is a tuple (start, end, size), () if absent.
        For SequenceVar, value is list of interval variable solutions.
        For StateFunction, value is list of steps.

        Args:
            name: Variable name, or model variable descriptor.
        Returns:
            Variable value, None if variable not found.
        """
        var = self.get_var_solution(name)
        if (var is None):
            return None
        return var.get_value()

    def _add_json_solution(self, model, jsol):
        """ Add a json solution to this solution descriptor

        Args:
            model:  Source model
            jsol:   String containing model solution expressed as JSON document
        """
        # Parse json string
        jsol = json.loads(jsol, parse_constant=True)

        # Add solution status
        status = jsol.get('solutionStatus', ())
        self.solve_status = status.get('solveStatus', self.solve_status)
        self.fail_status = status.get('failStatus', self.fail_status)

        # Add objectives
        ovals = jsol.get('objectives', None)
        if ovals:
            self._set_objective_values(ovals)

        # In case of alias, build a map between alias and real name
        if (_ALIAS_ALLOCATOR.get_count() == 0):
            alias_map = IdentityAccessor()
        else:
            alias_map = {}
            for v, l in model.get_all_variables():
                alias_map[v._get_alias_then_name()] = v.get_name()

        # Add integer variables if any
        vars = jsol.get('intVars', ())
        for vname in vars:
            self._add_var_solution(CpoIntVarSolution(alias_map[vname], _get_num_value(vars[vname])))

        # Add interval variables if any
        vars = jsol.get('intervalVars', ())
        for vname in vars:
            v = vars[vname]
            if 'start' in v:
                vsol = CpoIntervalVarSolution(alias_map[vname], True, _get_num_value(v['start']), _get_num_value(v['end']), _get_num_value(v['size']))
            else:
                vsol = CpoIntervalVarSolution(alias_map[vname], False)
            self._add_var_solution(vsol)

        # Add sequence variables if any
        vars = jsol.get('sequenceVars', ())
        for vname in vars:
            vnlist = [alias_map[v] for v in vars[vname]]
            ivres = [self.get_var_solution(vn) for vn in vnlist]
            self._add_var_solution(CpoSequenceVarSolution(alias_map[vname], ivres))

        # Add state functions if any
        funs = jsol.get('stateFunctions', ())
        for fname in funs:
            lpts = [( _get_num_value(v['start']), _get_num_value(v['end']), _get_num_value(v['value'])) for v in funs[fname]]
            self._add_var_solution(CpoStateFunctionSolution(alias_map[fname], lpts))

        # Add parameters
        self._set_parameters(jsol.get('parameters', {}))

        # Add information attributes
        cpinf = jsol.get('cpInfo', {})
        self._set_infos(cpinf)

        # Retrieve critical solving information
        self.nbconstraints = cpinf.get('NumberOfConstraints', self.nbconstraints)
        self.nbintvars = cpinf.get('NumberOfIntegerVariables', self.nbintvars)
        self.nbintervalvars = cpinf.get('NumberOfIntervalVariables', self.nbintervalvars)
        self.nbsequencevars = cpinf.get('NumberOfSequenceVariables', self.nbsequencevars)


    def __getitem__(self, name):
        """ Overloading of [] to get a variable solution from this model solution

        Args:
            name: Variable name or CPO variable expression
        Returns:
            Variable solution (class CpoVarSolution)
        """
        return(self.get_value(name))

    def print_solution(self, out=None):
        """ Prints the solution on a given output.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        # Select appropriate output
        if out is None:
            out = sys.stdout

        # Check file
        if isinstance(out, str):
            make_directories(os.path.dirname(out))
            with open(out, 'w') as f:
                self._write_solution(f)
        else:
            self._write_solution(out)

    def _write_solution(self, out):
        """ Write the solution

        Args:
            out: Target output
        """
        # Print model attributes
        out.write("-------------------------------------------------------------------------------\n")
        out.write("Model constraints: " + str(self.get_number_of_constraints()))
        out.write(", variables: integer: " + str(self.get_number_of_integer_vars()))
        out.write(", interval: " + str(self.get_number_of_interval_vars()))
        out.write(", sequence: " + str(self.get_number_of_sequence_vars()))
        out.write('\n')
        # Print search status
        out.write("Solve status: " + str(self.get_solve_status()) + ", Fail status: " + str(self.get_fail_status()) + "\n")
        out.write("Solve time: " + str(round(self.get_solve_time(), 2)) + " sec\n")
        out.write("-------------------------------------------------------------------------------\n")
        
        if self.is_solution():
            # Print objective value
            ovals = self.get_objective_values()
            if ovals is not None:
                out.write("Objective values: " + str(ovals))
                out.write('\n')
            # Print all variables in name order
            lvars = sorted(self.vars.keys())
            for v in lvars:
                out.write(str(self.get_var_solution(v)))
                out.write('\n')

###############################################################################
##  Private functions
###############################################################################

# Constants conversion
_CONSTANTS_VALUES = {'intmin': INT_MIN, 'intmax': INT_MAX, 'intervalmin': INTERVAL_MIN, 'intervalmax': INTERVAL_MAX,
                     'NaN': float('nan'), 'Infinity': float('inf'), '-Infinity': -float('inf')}

def _get_num_value(val):
    """ Convert a solution value into number.
    Interpret intmin, intmax, intervalmin, intervalmax, NaN, Infinity if any.

    Args:
        val: Value to convert
    Returns:
        Converted value, itself if not found
    """
    return(_CONSTANTS_VALUES.get(val, val))


