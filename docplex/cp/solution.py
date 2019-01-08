# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the different elements that represent a solution resulting
from the solve of a model.

This module implements the following object classes to represent solution elements:

 * :class:`CpoIntVarSolution`: solution of an integer variable,
 * :class:`CpoIntervalVarSolution`: solution of an interval variable,
 * :class:`CpoSequenceVarSolution`: solution of a sequence variable,
 * :class:`CpoStateFunctionSolution`: solution of a state function, and
 * :class:`CpoModelSolution`: aggregation of all individual model element solutions,

and the following ones to represent results:

 * :class:`CpoSolveResult`: result of a model solve, including a solution to the model (if any)
   plus other technical information (solve details, log, etc)
 * :class:`CpoRefineConflictResult`: result of an invocation of the conflict refiner.

The solution objects (:class:`CpoModelSolution`, :class:`CpoIntVarSolution`, etc) can be used in multiple ways:

 * To represent a *complete* (fully instantiated) solution, where each model has a unique fixed value, as returned
   by a successful model solve.
 * To represent a *partial* model solution, that is proposed as a solve starting point
   (see :meth:`docplex.cp.model.CpoModel.set_starting_point`)
   In this case, not all variables are present in the solution, and some of them may be partially instantiated.
 * To represent a *partial* model solution that is returned by the solver as result of calling method
   :meth:`docplex.cp.solver.solver.CpoSolver.propagate`.
"""

import json

from docplex.cp.expression import CpoExpr, INT_MIN, INT_MAX, INTERVAL_MIN, INTERVAL_MAX
import docplex.cp.utils as utils
from docplex.cp.utils import *
from docplex.cp.parameters import CpoParameters

###############################################################################
##  Constants
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
""" List of all possible search statuses """

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

# Fail status: Search completed
FAIL_STATUS_SEARCH_COMPLETED = "SearchCompleted"

# List of all possible search statuses
ALL_FAIL_STATUSES = (FAIL_STATUS_UNKNOWN,
                     FAIL_STATUS_FAILED_NORMALLY, FAIL_STATUS_NOT_FAILED,
                     FAIL_STATUS_ABORT, FAIL_STATUS_EXCEPTION, FAIL_STATUS_EXIT, FAIL_STATUS_LABEL,
                     FAIL_STATUS_TIME_LIMIT, FAIL_STATUS_SEARCH_COMPLETED)
""" List of all possible fail statuses """

# Name of some useful CPO solver information attributes
INFO_NUMBER_OF_CONSTRAINTS        = 'NumberOfConstraints'
INFO_NUMBER_OF_INTEGER_VARIABLES  = 'NumberOfIntegerVariables'
INFO_NUMBER_OF_INTERVAL_VARIABLES = 'NumberOfIntervalVariables'
INFO_NUMBER_OF_SEQUENCE_VARIABLES = 'NumberOfSequenceVariables'


###############################################################################
##  Public classes
###############################################################################

class CpoVarSolution(object):
    """ This class is a super class of all classes representing a solution to a variable.
    """
    __slots__ = ('name',  # Variable name
                 )
    
    def __init__(self, name):
        """ Constructor:

        Args:
            name: Variable name, or object providing a name with a function get_name() or an attribute 'name'.
        """
        # Determine name
        if not is_string(name):
            try:
                name = name.get_name()
            except:
                try:
                    name = name.name
                except:
                    raise AssertionError("Argument 'name' should be a string, or an object with a method get_name() or an attribute 'name'")
        self.name = name


    def get_name(self):
        """ Gets the name of the variable.

        Returns:
            Name of the variable.
        """
        return self.name


    def get_value(self):
        """ Gets the variable value.
        This method is overloaded by each class extending this class.

        Returns:
            Value of the variable, represented according to its semantic (see specific variable documentation).
        """
        return None


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)


class CpoIntVarSolution(CpoVarSolution):
    """ This class represents a solution to an integer variable.

    The solution can be:
     * *complete* when the value is a single integer,
     * *partial* when the value is a domain, set of multiple values.

    A domain is a list of discrete integer values and/or intervals of values represented by a tuple containing
    interval min and max values (included).

    For example, following are valid domains for an integer variable:
     * 7 (complete solution)
     * (1, 2, 4, 9)
     * (2, 3, (5, 7), 9, (11, 13))
    """
    __slots__ = ('value',  # Variable value / domain
                )
    
    def __init__(self, name, value):
        """ Constructor:

        Args:
            name:  Variable name
            value: Variable value, or domain if not completely instantiated
        """
        super(CpoIntVarSolution, self).__init__(name)
        self.value = _check_arg_domain(value, 'value')


    def get_value(self):
        """ Gets the value of the variable.

        Returns:
            Variable value (integer), or domain (list of integers or intervals)
        """
        return self.value
    
    def __str__(self):
        """ Convert this expression into a string """
        return self.get_name() + ": " + str(self.get_value())
        

class CpoIntervalVarSolution(CpoVarSolution):
    """ This class represents a solution to an interval variable.

    The solution can be complete if all attribute values are integers, or partial if at least one
    of them is an interval expressed as a tuple.
    """
    __slots__ = ('start',    # Interval start
                 'end',      # Interval end
                 'size',     # Interval size
                 'length',   # Interval length
                 'presence', # Presence indicator
                )
    
    def __init__(self, name, presence=None, start=None, end=None, size=None):
        # """ Constructor:
        #
        # Args:
        #     name:     Name of the variable.
        #     presence: Presence indicator (true for present, false for absent, None for undetermined). Default is None.
        #     start:    Value of start, or tuple representing the start range. Default is None.
        #     end:      Value of end, or tuple representing the end range. Default is None.
        #     size:     Value of size, or tuple representing the size range. Default is None.
        # """
        super(CpoIntervalVarSolution, self).__init__(name)
        self.presence = presence
        self.start = start
        self.end   = end
        self.size  = size
        self.length = None


    def is_present(self):
        """ Check if the interval is present.

        Returns:
            True if interval is present.
        """
        return self.presence is True


    def is_absent(self):
        """ Check if the interval is absent.

        Returns:
            True if interval is absent.
        """
        return self.presence is False


    def is_optional(self):
        """ Check if the interval is optional.

        Returns:
            True if interval is optional.
        """
        return self.presence is None


    def get_start(self):
        """ Gets the interval start.

        Returns:
            Interval start value, or domain (tuple (min, max)) if not fully instantiated
        """
        return self.start


    def get_end(self):
        """ Gets the interval end.

        Returns:
            Interval end value, or domain (tuple (min, max)) if not fully instantiated
        """
        return self.end


    def get_size(self):
        """ Gets the interval size.

        Returns:
            Interval size value, or domain (tuple (min, max)) if not fully instantiated
        """
        return self.size


    def get_length(self):
        """ Gets the interval length.

        Returns:
            Interval length value, or domain (tuple (min, max)) if not fully instantiated
        """
        if self.length is None:
            return self.end - self.start
        return self.length


    def get_value(self):
        """ Gets the interval variable value as a tuple (start, end, size), or () if absent.

        If the variable is absent, then the result is an empty tuple.

        If the variable is fully instantiated, the result is a tuple of 3 integers (start, end, size).
        The variable length, easy to compute as end - start, can also be retrieved by calling :meth:`get_length`.

        If the variable is partially instantiated, the result is a tuple (start, end, size, length) where each
        individual value can be an integer or an interval expressed as a tuple.

        Returns:
            Interval variable value as a tuple.
        """
        if (self.is_present()):
            if self.length is None:
                return (self.start, self.end, self.size)
            else:
                return (self.start, self.end, self.size, self.length)
        return ()


    def __str__(self):
        """ Convert this expression into a string """
        res = [self.get_name(), ': ']
        if (self.is_absent()):
            res.append("absent")
        else:
            if (self.is_optional()):
                res.append("optional")
            res.append("(start=" + str(self.get_start()))
            res.append(", end=" + str(self.get_end()))
            res.append(", size=" + str(self.get_size()))
            res.append(", length=" + str(self.get_length()))
            res.append(")")
        return ''.join(res)

     
class CpoSequenceVarSolution(CpoVarSolution):
    """ This class represents a solution to a sequence variable.
    """
    __slots__ = ('lvars',  # List of interval variable solutions
                )
    
    def __init__(self, name, lvars):
        """ Constructor:

        Args:
            lvars: List of interval variable solutions that are in this sequence (objects CpoIntervalVarSolution).
        """
        super(CpoSequenceVarSolution, self).__init__(name)
        self.lvars = lvars


    def get_interval_variables(self):
        """ Gets the list of CpoIntervalVarSolution in this sequence.

        Returns:
            List of CpoIntervalVarSolution in this sequence.
        """
        return self.lvars


    def get_value(self):
        """ Gets the list of CpoIntervalVarSolution in this sequence.

        Returns:
            List of CpoIntervalVarSolution in this sequence.
        """
        return self.lvars


    def __str__(self):
        """ Convert this expression into a string """
        return self.get_name() + ": (" + ", ".join([v.get_name() for v in self.lvars]) + ")"
        
     
class CpoStateFunctionSolution(CpoVarSolution):
    """ This class represents a solution to a step function.

    A solution to a step function is represented by a list of steps.
    A step is a triplet (start, end, value) that gives the value of the function on the interval [start, end).
    """
    __slots__ = ('steps',  # List of function steps
                )
    
    def __init__(self, name, steps):
        """ Constructor:

        Args:
            steps: List of function steps represented as tuples (start, end, value).
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
    """ This class represents a solution to a model. It contains the solutions of model variables plus
    the value of objective, if any.

    Each variable solution is accessed with its name
    and its value is either :class:`CpoIntVarSolution`, :class:`CpoIntervalVarSolution`,
    :class:`CpoSequenceVarSolution` or :class:`CpoStateFunctionSolution` depending on the type of the variable.

    The solution can be:
      * *complete*, if each variable is assigned to a single value,
      * *partial* if not all variables are defined, or if some variables are defined with domains that are not
        restricted to a single value.

    An instance of this class may be created explicitly by the programmer of the model to express a *starting point*
    that can be passed to the model to optimize its solve
    (see :meth:`docplex.cp.model.CpoModel.set_starting_point` for details).
    """
    __slots__ = ('vars',          # Map of variable solutions
                 'objvalues',     # Objective values
                )

    def __init__(self):
        super(CpoModelSolution, self).__init__()
        self.vars = {}
        self.objvalues = None


    def _set_objective_values(self, ovals):
        """ Set the numeric values of all objectives.

        Args:
            ovals: Array of objective values
        """
        self.objvalues = ovals


    def get_objective_values(self):
        """ Gets the numeric values of all objectives.

        If the solution is partial, each objective value may be an interval expressed as a tuple (min, max)

        Returns:
            Array of objective values, None if none.
        """
        return self.objvalues


    def add_var_solution(self, vsol):
        """ Add a solution to a variable to this model solution.

        Args:
            vsol: Variable solution (object of a class extending :class:`CpoVarSolution`)
        """
        assert isinstance(vsol, CpoVarSolution)
        self.vars[vsol.get_name()] = vsol


    def add_integer_var_solution(self, name, value):
        """ Add a new integer variable solution from its name and value..

        The solution can be complete if the value is a single integer, or partial if the value
        is a domain, given as a list of integers or intervals expressed as tuples.

        Args:
            name:  Variable name
            value: Variable value, or domain if not completely instantiated
        """
        self.add_var_solution(CpoIntVarSolution(name, value))


    def add_interval_var_solution(self, name, presence=None, start=None, end=None, size=None):
        """ Add a new interval variable solution.

        The solution can be complete if all attribute values are integers, or partial if at least one
        of them is an interval expressed as a tuple.

        Args:
            name:     Name of the variable.
            presence: Presence indicator (true for present, false for absent, None for undetermined). Default is None.
            start:    Value of start, or tuple representing the start range
            end:      Value of end, or tuple representing the end range
            size:     Value of size, or tuple representing the size range
        """
        self.add_var_solution(CpoIntervalVarSolution(name, presence, start, end, size))


    def get_var_solution(self, name):
        """ Gets a variable solution from this model solution.

        Args:
            name: Variable name or variable expression.
        Returns:
            Variable solution (class extending :class:`CpoVarSolution`)
        Raises:
            CpoException if variable solution does not exists
        """
        if isinstance(name, CpoExpr):
            name = name.get_name()
        value = self.vars.get(name)
        if value is None:
            raise CpoException("Variable '{}' does not exists in this solution".format(name))
        return value


    def get_all_var_solutions(self):
        """ Gets the list of all variable solutions from this model solution.

        Returns:
            List of all variable solutions (class extending :class:`CpoVarSolution`).
        """
        return list(self.vars.values())


    def get_value(self, name):
        """ Gets the value of a variable.

        This method first find the variable with :meth:`get_var_solution` and, if exists,
        returns the result of a call to the method get_value() on this variable.

        The result depends on the type of the variable. For details, please consult documentation of methods:

         * :meth:`CpoIntVarSolution.get_value`
         * :meth:`CpoIntervalVarSolution.get_value`
         * :meth:`CpoSequenceVarSolution.get_value`
         * :meth:`CpoStateFunctionSolution.get_value`

        Args:
            name: Variable name, or model variable descriptor.
        Returns:
            Variable value, None if variable is not found.
        Raises:
            CpoException if variable solution does not exists
        """
        var = self.get_var_solution(name)
        return None if var is None else var.get_value()


    def _add_json_solution(self, jsol):
        """ Add a json solution to this solution descriptor

        Args:
            jsol: JSON document representing solution, or string containing its JSON representation.
        """
        # Parse json string if needed
        if not isinstance(jsol, dict):
            jsol = json.loads(jsol, parse_constant=True)

        # Add objectives
        ovals = jsol.get('objectives', None)
        if ovals:
            self._set_objective_values([tuple(v) if isinstance(v, list) else v for v in ovals])

        # Add integer variables
        vars = jsol.get('intVars', ())
        for vname in vars:
            self.add_var_solution(CpoIntVarSolution(vname, _get_domain(vars[vname])))

        # Add interval variables
        vars = jsol.get('intervalVars', ())
        for vname in vars:
            v = vars[vname]
            if 'start' in v:
                # Check partially instanciated
                if 'presence' in v:
                    vsol = CpoIntervalVarSolution(vname,  True if v['presence'] == 1 else None,
                                                  _get_domain(v['start']), _get_domain(v['end']), _get_domain(v['size']))
                    vsol.length = _get_domain(v['length'])
                else:
                    vsol = CpoIntervalVarSolution(vname, True, _get_num_value(v['start']), _get_num_value(v['end']), _get_num_value(v['size']))
            else:
                vsol = CpoIntervalVarSolution(vname, False)
            self.add_var_solution(vsol)

        # Add sequence variables
        vars = jsol.get('sequenceVars', ())
        for vname in vars:
            vnlist = [v for v in vars[vname]]
            ivres = [self.get_var_solution(vn) for vn in vnlist]
            self.add_var_solution(CpoSequenceVarSolution(vname, ivres))

        # Add state functions
        funs = jsol.get('stateFunctions', ())
        for fname in funs:
            lpts = [( _get_num_value(v['start']), _get_num_value(v['end']), _get_num_value(v['value'])) for v in funs[fname]]
            self.add_var_solution(CpoStateFunctionSolution(fname, lpts))


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

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        # Select appropriate output
        if out is None:
            out = sys.stdout

        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self._write_solution(f)
        else:
            self._write_solution(out)


    def _write_solution(self, out):
        """ Write the solution

        Args:
            out: Target output
        """
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


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)


class CpoRunResult(object):
    """ This class is an abstract class extended by classes representing the result of a call to the solver.
    """
    def __init__(self):
        super(CpoRunResult, self).__init__()
        self.solverLog = None                      # Solver log


    def _set_solver_log(self, log):
        """ Set the solver log as a string.

        Args:
            log (str): Log of the solver
        """
        self.solverLog = log


    def get_solver_log(self):
        """ Gets the log of the solver.

        Returns:
            Solver log as a string, None if unknown.
        """
        return self.solverLog


class CpoSolveResult(CpoRunResult):
    """ This class represents the result of a call to the solve of a model.

    It contains the following elements:
       * solve status,
       * solver parameters,
       * solver information
       * output log
       * solution, if any (class :class:`CpoModelSolution`)

    If this result contains a solution, the methods implemented in the class :class:`CpoModelSolution`
    to access solution elements are available directly from this class.
    """
    def __init__(self, model):
        """ Constructor:

        Args:
           model: Related model
        """
        super(CpoSolveResult, self).__init__()
        self.model = model
        self.solve_status = SOLVE_STATUS_UNKNOWN   # Solve status, with value in SOLVE_STATUS_*
        self.fail_status = FAIL_STATUS_UNKNOWN     # Fail status, with values in FAIL_STATUS_*
        self.solveTime = 0                         # Solve time
        self.parameters = None                     # Solving parameters
        self.infos = None                          # Solving information attributes map
        self.solution = CpoModelSolution()         # Solution


    def _set_solve_status(self, ssts):
        """ Set the solve status

        Args:
            ssts: Solve status
        """
        self.solve_status = ssts


    def get_solve_status(self):
        """ Gets the solve status.

        Returns:
            Solve status, element of the global list :const:`ALL_SOLVE_STATUSES`.
        """
        return self.solve_status


    def _set_fail_status(self, fsts):
        """ Set the fail status

        Args:
            fsts: Fail status
        """
        self.fail_status = fsts


    def get_fail_status(self):
        """ Gets the solving fail status.

        Returns:
            Fail status, element of the global list :const:`ALL_FAIL_STATUSES`.
        """
        return self.fail_status


    def is_solution(self):
        """ Checks if this descriptor contains a valid solution to the problem.

        A solution is present if the solve status is 'Feasible' or 'Optimal'.
        Optimality of the solution should be tested using method :meth:`is_solution_optimal()`.

        Returns:
            True if there is a solution.
        """
        return (self.solve_status in (SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL)) and (self.fail_status != FAIL_STATUS_SEARCH_COMPLETED)


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


    def is_solution_optimal(self):
        """ Checks if this descriptor contains an optimal solution to the problem.

        Returns:
            True if there is a solution that is optimal.
        """
        return self.solve_status is SOLVE_STATUS_OPTIMAL


    def get_objective_values(self):
        """ Gets the numeric values of all objectives.

        Returns:
            Array of all objective values, None if none.
        """
        return self.solution.get_objective_values()


    def get_parameters(self):
        """ Gets the complete dictionary of solving parameters.

        Returns:
            Solving parameters (object of class CpoParameters), None if undefined.
        """
        return self.parameters


    def get_parameter(self, name, default=None):
        """ Get a particular solving parameter.

        Args:
            name:    Name of the parameter to get
            default: (optional) Default value if not found. None by default.
        Returns:
            Parameter value, default value if not found.
        """
        if self.parameters is None:
            return default
        return self.parameters.get(name, default)


    def _add_infos(self, infos):
        """ Add solving information to existing ones.

        Args:
            infos: Dictionary of information attributes to append
        """
        if self.infos is None:
            self.infos = {}
        self.infos.update(infos)


    def get_infos(self):
        """ Gets the complete dictionary of information attributes.

        Returns:
            Dictionary of information attributes, None if undefined.
        """
        return self.infos


    def get_info(self, name, default=None):
        """ Gets a particular information attribute.

        Args:
            name:    Name of the information to get
            default: (optional) Default value if not found. None by default.
        Returns:
            Information attribute value, None if not found.
        """
        if self.infos is None:
            return default
        return self.infos.get(name, default)


    def _set_model_attributes(self, nbintvars=0, nbitvvars=0, nbseqvars=0, nbctrs=0):
        """ Set the general model attributes.

        This method is called when solve is done on the cloud, when not all information is available from the solver.

        Args:
            nbintvars: Number of integer variables
            nbitvvars: Number of interval variables
            nbseqvars: Number of sequence variables
            nbctrs:    Number of constraints
        """
        self._add_infos({INFO_NUMBER_OF_INTEGER_VARIABLES: nbintvars,
                         INFO_NUMBER_OF_INTERVAL_VARIABLES: nbitvvars,
                         INFO_NUMBER_OF_SEQUENCE_VARIABLES: nbseqvars,
                         INFO_NUMBER_OF_CONSTRAINTS: nbctrs})


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


    def get_number_of_integer_vars(self):
        """ Gets the number of integer variables in the model.

        Returns:
            Number of integer variables.
        """
        return self.get_info(INFO_NUMBER_OF_INTEGER_VARIABLES, 0)


    def get_number_of_interval_vars(self):
        """ Gets the number of interval variables in the model.

        Returns:
            Number of interval variables.
        """
        return self.get_info(INFO_NUMBER_OF_INTERVAL_VARIABLES, 0)


    def get_number_of_sequence_vars(self):
        """ Gets the number of sequence variables in the model.

        Returns:
            Number of sequence variables.
        """
        return self.get_info(INFO_NUMBER_OF_SEQUENCE_VARIABLES, 0)


    def get_number_of_constraints(self):
        """ Gets the number of constraints in the model.

        Returns:
            Number of constraints.
        """
        return self.get_info(INFO_NUMBER_OF_CONSTRAINTS, 0)


    def get_var_solution(self, name):
        """ Gets a variable solution from this model solution.

        Args:
            name: Variable name or variable expression.
        Returns:
            Variable solution (class CpoVarSolution), None if not found.
        """
        return self.solution.get_var_solution((name))


    def get_all_var_solutions(self):
        """ Gets the list of all variable solutions from this model solution.

        Returns:
            List of all variable solutions (class CpoVarSolution).
        """
        return self.solution.get_all_var_solutions()


    def get_value(self, name):
        """ Gets the value of a variable.

        For IntVar, value is an integer.
        For IntervalVar, value is a tuple (start, end, size), () if absent.
        For SequenceVar, value is list of interval variable solutions.
        For StateFunction, value is list of steps.

        Args:
            name: Variable name, or model variable descriptor.
        Returns:
            Variable value, None if variable is not found.
        """
        return self.solution.get_value(name)


    def _add_json_solution(self, jsol):
        """ Add a json solution to this solution descriptor

        Args:
            jsol:   JSON document representing solution, or string containing its JSON representation.
        """
        # Parse json string if needed
        if not isinstance(jsol, dict):
            jsol = json.loads(jsol, parse_constant=True)

        # Add solver status
        status = jsol.get('solutionStatus', ())
        self.solve_status = status.get('solveStatus', self.solve_status)
        self.fail_status = status.get('failStatus', self.fail_status)
        nsts = status.get('nextStatus')
        if nsts is not None:
            if nsts != 'NextTrue':
                self.fail_status = FAIL_STATUS_SEARCH_COMPLETED

        # Add parameters
        prms = jsol.get('parameters', None)
        if prms is not None:
            self.parameters = CpoParameters()
            self.parameters.update(prms)

        # Add information attributes
        cpinf = jsol.get('cpInfo', None)
        if cpinf is not None:
            self._add_infos(cpinf)

        # Add solution
        self.solution._add_json_solution(jsol)


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

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        # Select appropriate output
        if out is None:
            out = sys.stdout

        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
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
        #out.write("Solve status: " + str(self.get_solve_status()) + ", Fail status: " + str(self.get_fail_status()) + "\n")
        out.write("Solve status: " + str(self.get_solve_status()) + "\n")
        out.write("Solve time: " + str(round(self.get_solve_time(), 2)) + " sec\n")
        out.write("-------------------------------------------------------------------------------\n")

        if self.is_solution():
            self.solution._write_solution(out)


    def __str__(self):
        """ Build a string representation of this object.

        The string that is returned is the same than what is printed by calling :meth:`print_solution`.

        Returns:
            String representation of this object.
        """
        out = StringIO()
        self._write_solution(out)
        res = out.getvalue()
        out.close()
        return res


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)



class CpoRefineConflictResult(CpoRunResult):
    """ This class represents the result of a call to the conflict refiner.

    A conflict is a subset of the constraints and/or variables of the model which are
    mutually contradictory.

    The conflict refiner first examines the full infeasible model to identify portions of the conflict
    that it can remove. By this process of refinement, the conflict refiner arrives at a minimal conflict.
    A minimal conflict is usually smaller than the full infeasible model and thus makes infeasibility analysis easier.
    Since the conflict is minimal, removal of any one of these constraints will remove that particular cause
    for infeasibility.
    There may be other conflicts in the model; consequently, repair of a given conflict does not guarantee
    feasibility of the remaining model.
    If a model happens to include multiple independent causes of infeasibility,
    then it may be necessary for the user to repair one such cause and then repeat the diagnosis with further
    conflict analysis.
    """
    def __init__(self, model):
        # """ Creates a new empty conflict refiner result.
        #
        # Args:
        #    model: Related model
        # """
        super(CpoRefineConflictResult, self).__init__()
        self.model = model
        self.member_constraints = []    # List of member constraints
        self.possible_constraints = []  # List of possible member constraints
        self.member_variables = []      # List of member variables
        self.possible_variables = []    # List of possible member variables


    def get_all_member_constraints(self):
        """ Returns the list of all constraints that are certainly member of the conflict.

        Returns:
            List of model constraints (class CpoExpr) certainly member of the conflict.
        """
        return self.member_constraints


    def get_all_possible_constraints(self):
        """ Returns the list of all constraints that are possibly member of the conflict.

        Returns:
            List of model constraints (class CpoExpr) possibly member of the conflict.
        """
        return self.possible_constraints


    def get_all_member_variables(self):
        """ Returns the list of all variables that are certainly member of the conflict.

        Returns:
            List of model variables (class CpoVariable) certainly member of the conflict.
        """
        return self.member_variables


    def get_all_possible_variables(self):
        """ Returns the list of all variables that are possibly member of the conflict.

        Returns:
            List of model variables (class CpoVariable) possibly member of the conflict.
        """
        return self.possible_variables


    def is_conflict(self):
        """ Checks if this descriptor contains a valid conflict.

        Returns:
            True if there is a conflict, False otherwise.
        """
        return len(self.member_constraints) != 0 or len(self.possible_constraints) != 0 \
               or len(self.member_variables) != 0 or len(self.possible_variables) != 0


    def __nonzero__(self):
        """ Check if this descriptor contains a conflict.
        Equivalent to is_conflict()

        Returns:
            True if there is a conflict, False otherwise.
        """
        return self.is_conflict()


    def __bool__(self):
        """ Check if this descriptor contains a conflict.
        Equivalent to is_conflict()

        Equivalent to __nonzero__ for Python 3

        Returns:
            True if there is a conflict, False otherwise.
        """
        return self.is_conflict()


    def _add_json_solution(self, jsol):
        """ Add a json solution to this result descriptor

        Args:
            jsol:   JSON document representing result, or string containing its JSON representation.
        """
        # Parse json string if needed
        if not isinstance(jsol, dict):
            jsol = json.loads(jsol, parse_constant=True)

        # Get conflict data
        conflict = jsol.get('conflict')
        if conflict is None:
            return

        # Add constraints
        for name, status in conflict.get('constraints', {}).items():
            expr = self.model.get_expression(name)
            if expr is None:
                raise CpoException("INTERNAL ERROR: Conflict refiner returns a constraint '{}' that is not found in the model".format(name))
            if status == 'ConflictMember':
                self.member_constraints.append(expr)
            else:
                self.possible_constraints.append(expr)

        # Add variables
        vars = conflict.get('intVars', {}).copy()
        vars.update(conflict.get('intervalVars', {}))
        for name, status in vars.items():
            expr = self.model.get_expression(name)
            if expr is None:
                raise CpoException("INTERNAL ERROR: Conflict refiner returns a variable '{}' that is not found in the model".format(name))
            if status == 'ConflictMember':
                self.member_variables.append(expr)
            else:
                self.possible_variables.append(expr)


    def print_conflict(self, out=None):
        """ Prints this conflict on a given output.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        # Select appropriate output
        if out is None:
            out = sys.stdout

        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self._write_conflict(f)
        else:
            self._write_conflict(out)


    def _write_conflict(self, out):
        """ Write the conflict

        Args:
            out: Target output
        """
        out.write("Conflict refiner result:\n")
        if not self.is_conflict():
            out.write("   None\n")
            return
        # Print constraints in the conflict
        lc = self.get_all_member_constraints()
        if lc:
            out.write("Member constraints:\n")
            for c in lc:
                out.write("   {}\n".format(_build_conflict_constraint_string(c)))
        lc = self.get_all_possible_constraints()
        if lc:
            out.write("Possible member constraints:\n")
            for c in lc:
                out.write("   {}\n".format(_build_conflict_constraint_string(c)))
        # Print variables in the conflict
        lc = self.get_all_member_variables()
        if lc:
            out.write("Member variables:\n")
            for c in lc:
                out.write("   {}\n".format(c))
        lc = self.get_all_possible_variables()
        if lc:
            out.write("Possible member variables:\n")
            for c in lc:
                out.write("   {}\n".format(c))


    def __str__(self):
        """ Build a string representation of this object.

        The string that is returned is the same than what is printed by calling :meth:`print_conflict`.

        Returns:
            String representation of this object.
        """
        out = StringIO()
        self._write_conflict(out)
        res = out.getvalue()
        out.close()
        return res


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)




###############################################################################
##  Private functions
###############################################################################

# Constants conversion
_CONSTANTS_VALUES = {'intmin': INT_MIN, 'intmax': INT_MAX, 'intervalmin': INTERVAL_MIN, 'intervalmax': INTERVAL_MAX,
                     'NaN': float('nan'), 'Infinity': float('inf'), '-Infinity': -float('inf')}

# Marker of interval with holes
_HOLE_MARKER = "holes"

def _get_domain(val):
    """ Convert a solution value into domain.

    Args:
        val: Value to convert
    Returns:
        Variable domain
    """
    if is_array(val):
        res = []
        for v in val:
            if is_array(v):
                vl = len(v)
                if (vl == 2):
                    res.append((_get_num_value(v[0]), _get_num_value(v[1])))
                elif vl == 3:
                    res.append((_get_num_value(v[0]), _get_num_value(v[1]), _HOLE_MARKER))
                    assert v[2] == _HOLE_MARKER, "Domain interval with 3 elements must contains '{}' as last one".format(_HOLE_MARKER)
                else:
                    assert False, "Domain interval should contain only 2 elements"
            else:
                res.append(_get_num_value(v))
        return tuple(res)
    else:
        return _get_num_value(val)


def _get_num_value(val):
    """ Convert a solution value into number.
    Interpret intmin, intmax, intervalmin, intervalmax, NaN, Infinity if any.

    Args:
        val: Value to convert
    Returns:
        Converted value, itself if not found
    """
    return _CONSTANTS_VALUES.get(val, val)


def _check_arg_domain(val, name):
    """ Check that an argument is a correct domain and raise error if wrong

    Domain is:
       * a single integer for a fixed domain
       * a list of integers or intervals expressed as tuples.

    Args:
        val:  Argument value
        name: Argument name
    Returns:
        Domain to be set
    Raises:
        Exception if argument has the wrong format
    """
    # Check single integer
    if is_int(val):
        return val
    # Check list of integers or tuples
    assert is_array(val), "Argument '" + name + "' should be a list of integers and/or intervals"
    for v in val:
        if not is_int(v):
            assert _is_domain_interval(v), "Argument '" + name + "' should be a list of integers and/or intervals (tuples of 2 integers)"
    return val


def _is_domain_interval(val):
    """ Check if a value is representing a valid domain interval
    Args:
        val:  Value to check
    Returns:
        True if value is a tuple representing an interval
    """
    if not isinstance(val, tuple):
        return False
    if not (is_int(val[0]) and is_int(val[1]) and (val[1] >= val[0])):
        return False
    vl = len(val)
    if vl == 2:
        return True
    if vl == 3:
        return val[2] == _HOLE_MARKER
    return False


def _build_conflict_constraint_string(ctr):
    """ Build the string used to represent a constraint in conflict refiner
    Args:
        ctr:  Constraint to print
    Returns:
        Constraint string
    """
    return str(ctr)


