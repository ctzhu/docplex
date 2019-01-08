# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module implements appropriate software to solve a CPO model represented by a
:class:`docplex.cp.model.CpoModel` object.

It implements the following object classes:

 * :class:`CpoSolver` contains the public interface allowing to make solving requests with a model.
 * :class:`CpoSolverAgent` is an abstract class that is extended by the actual implementation(s) of
   the solving functions.

The :class:`CpoSolver` identifies and creates the required :class:`CpoSolverAgent` depending on the configuration
parameter *context.solver.agent' that contains the name of the agent to be used. This name is used to
access the configuration context *context.solver.<agent>* that contains the details about this agent.

For example, the default configuration refers to *docloud* as default solver agent, to solve model using *DOcplexcloud*
services. This means that at least following configuration elements must be set:
::
   context.solver.agent = 'docloud'
   context.solver.docloud.url = <URL of the service>
   context.solver.docloud.key = <Access key of the service>

The different methods that can be called on a CpoSolver object are:

 * :meth:`solve` simply solve the model and returns a solve result, if any.
   For convenience reason, this method is also directly available on the CpoModel object (:meth:`docplex.cp.model.CpoModel.solve`).
 * :meth:`search_next` and :meth:`end_search` allows to iterate on different solutions of the model.
 * :meth:`refine_conflict` calls the conflict refiner that identifies a minimal conflict for the infeasibility of
   the model.
 * :meth:`propagate` calls the propagation that communicates the domain reduction of a decision variable to
   all of the constraints that are stated over this variable.

Except :meth:`solve`, these functions are only available with a local solver with release strictly greater than 12.6.3.
When a method is not available, an exception *CpoNotSupportedException* is raised.

If the methods :meth:`search_next` and :meth:`end_search` are available in the underlying solver agent,
the :class:`CpoSolver` object can acts as an iterator. All solutions can be retrieved using a loop like:
::
   solver = CpoSolver(mdl)
   for sol in solver:
       sol.print_solution()

A such solution iteration can be interrupted at any time by calling end_search() that returns
a fail solution including the last solve status.
"""

import docplex.cp.config as config
from docplex.cp.utils import CpoException, CpoNotSupportedException, make_directories, Context
import docplex.cp.utils as utils
from docplex.cp.cpo_compiler import CpoCompiler
import docplex.cp.solver.environment_client as runenv

import time, importlib, inspect


###############################################################################
##  Public constants
###############################################################################

# Solver statuses
STATUS_IDLE              = "Idle"             # Solver created but inactive
STATUS_RELEASED          = "Released"         # Solver stopped with resources released.
STATUS_SOLVING           = "SolveRunning"     # Simple solve in progress
STATUS_SEARCH_WAITING    = "SearchWaiting"    # Search started or waiting to call next
STATUS_SEARCH_RUNNING    = "NextRunning"      # Search of next solution in progress
STATUS_REFINING_CONFLICT = "RefiningConflict" # Solver refine conflict in progress
STATUS_PROPAGATING       = "Propagating"      # Propagation in progress

###############################################################################
##  Public classes
###############################################################################

class CpoSolverAgent(object):
    """ This class is an abstract class that must be extended by every solver agent that intend
    to be called by :class:`CpoSolver` to solve a CPO model.
    """

    def __init__(self, model, params, context):
        """ Constructor

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  Solver agent context
        Raises:
            CpoException if jar file does not exists
        """
        super(CpoSolverAgent, self).__init__()
        self.model = model
        self.params = params
        self.context = context
        self.last_json_result = None

    def solve(self):
        """ Solve the model

        Returns:
            Model solve result (object of class CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().

        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def search_next(self):
        """ Search the next available solution.

        Returns:
            Next solve result (object of class CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) solve result with last solve information (type CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Returns:
            Conflict result (object of class CpoRefineConflictResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def propagate(self):
        """ This method invokes the propagation on the current model.

        Constraint propagation is the process of communicating the domain reduction of a decision variable to
        all of the constraints that are stated over this variable.
        This process can result in more domain reductions.
        These domain reductions, in turn, are communicated to the appropriate constraints.
        This process continues until no more variable domains can be reduced or when a domain becomes empty
        and a failure occurs.
        An empty domain during the initial constraint propagation means that the model has no solution.

        The result is a object of class CpoSolveResult, the same than the one returned by solve() method.
        However, in this case, variable domains may not be completely defined.

        Returns:
            Propagation result (object of class CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def end(self):
        """ End solver agent and release all resources.
        """
        self.model = None
        self.params = None
        self.context = None


    def _get_cpo_model_string(self):
        """ Get the CPO model as a string, according to configuration

        Return:
            String containing the CPO model in CPO file format
        """
        # Build string
        ctx = self.context
        cpostr = CpoCompiler(self.model, params=self.params).get_as_string()

        # Trace CPO model if required
        lout = ctx.get_log_output()
        if lout and ctx.trace_cpo:
            lout.write("Model '" + str(self.model.get_name()) + "' in CPO format:\n")
            lout.write(cpostr)
            lout.write("\n")
            self.model.print_information(lout)
            lout.write("\n")
            lout.flush()

        # Dump in dump directory if required
        if ctx.model.dump_directory:
            make_directories(ctx.model.dump_directory)
            file = ctx.model.dump_directory + "/" + utils.get_file_name_only(self.model.get_source_file()) + ".cpo"
            with utils.open_utf8(file, 'w') as f:
                f.write(cpostr)

        # Return
        return cpostr


    def _set_last_json_result_string(self, json):
        """ Set the string containing last received JSON result

        Args:
            json: JSON result string
        """
        self.context.log(3, "JSON result:\n", json)
        self.last_json_result = json


    def _get_last_json_result_string(self):
        """ Get the string containing last received JSON result

        Return:
            Last JSON result string, None if none
        """
        return self.last_json_result


    def _is_log_required(self):
        """ Check if solver log is required.

        Return:
            True if solver log is required, False otherwise
        """
        return self.context.add_log_to_solution or self.context.trace_log


    def _raise_not_supported(self):
        """ Raise an exception indicating that the calling method is not supported.
        """
        raise CpoNotSupportedException("Method '{}' is not available in solver agent '{}' ({})."
                                       .format(inspect.stack()[1][3], self.context.agent, type(self)))


class CpoSolver(object):
    """ This class represents the public API of the object allowing to solve a CPO model.

    It create the appropriate :class:`CpoSolverAgent` that actually implements solving functions, depending
    on the value of the configuration parameter *context.solver.agent*.
    """
    __slots__ = ('model',     # Model to solve
                 'context',   # Solving context
                 'solver',    # Solver agent
                 'status',    # Current solver status
                 'last_sol',  # Last returned solution
                )

    def __init__(self, model, **kwargs):
        """ Constructor:

        Args:
            model:     Model to solve
        Optional args:
            context:   Complete solving context. If not given, context is the default context that is set in config.py.
            params:    Solving parameters (CpoParameters) that overwrite those in the solving context
            url:       URL of the DOcplexcloud service that overwrites the one defined in the solving context.
            key:       Authentication key of the DOcplexcloud service that overwrites the one defined in the solving context.
            (others):  All other context parameters that can be changed.
        """
        super(CpoSolver, self).__init__()
        self.solver = None

        # Build effective context from args
        context = config._get_effective_context(**kwargs)

        # If defined, limit the number of threads
        mxt = context.solver.max_threads
        if isinstance(mxt, int):
            # Maximize number of workers
            nbw = context.params.Workers
            if (nbw is None) or (nbw > mxt):
                context.params.Workers = mxt
                print("WARNING: Number of workers has been reduced to " + str(mxt) + " to comply with platform limitations.")

        # Save attributes
        self.model = model
        self.context = context

        # Determine appropriate solver agent
        sctx = context.solver
        aname = sctx.agent
        if aname is None:
            aname = "docloud"
        else:
            aname = aname.lower()
        sctx.log(1, "Solve model '", self.model.get_name(), "' with agent '", aname, "'")

        # Retrieve solver agent class and create instance
        actx = sctx.get(aname)
        sclass = _get_solver_agent_class(aname, actx)
        self.solver = sclass(self.model, sctx.params, actx)

        # Initialize working variables
        self.last_sol = None
        self.status = STATUS_IDLE


    def __iter__(self):
        """  Define solver as an iterator """
        return self


    def __del__(self):
        # End solver
        self.end()


    def solve(self):
        """ Solve the model

        This function solves the model using CP Optimizer's built-in strategy.
        The built-in strategy is determined by setting the parameter SearchType (see docplex.cp.parameters).
        If the model contains an objective, then the optimal solution with respect to the objective will be calculated.
        Otherwise, a solution satisfying all problem constraints will be calculated.

        The function returns an object of the class CpoSolveResult (see docplex.cp.solution) that contains the solution
        if exists, plus different information on the solving process.

        Returns:
            Model solution (object of class CpoSolveResult)
        Raises:
            :class:`docplex.cp.utils.CpoException` (or derived) if error.
        """
        # Notify start solve to environment
        runenv.start_solve(self)

        # Solve model
        stime = time.time()
        self.status = STATUS_SOLVING
        msol = self.solver.solve()
        self.status = STATUS_IDLE
        stime = time.time() - stime
        self.context.solver.log(1, "Model '", self.model.get_name(), "' solved in ", round(stime, 2), " sec.")

        # Set solve time in solution if not done
        if msol.get_solve_time() == 0:
            msol._set_solve_time(stime)

        # Store last solution
        self.last_sol = msol

        # Notify end solve to environment
        runenv.end_solve(self)

        # Return solution
        return msol
        
     
    def search_next(self):
        """ Get the next available solution.

        This function is available only with local CPO solver with release number strictly greater than 12.6.3.

        Returns:
            Next model solution (object of class CpoModelSolution)
        Raises:
            CpoNotSupportedException: if method not available in the solver agent.
        """
        # Initiate search if needed
        if self.status == STATUS_IDLE:
            self.solver.start_search()
            self.status = STATUS_SEARCH_WAITING
        else:
            self._check_status(STATUS_SEARCH_WAITING)

        # Notify start solve to environment
        runenv.start_solve(self)

        # Solve model
        stime = time.time()
        self.status = STATUS_SEARCH_RUNNING
        msol = self.solver.search_next()
        self.status = STATUS_SEARCH_WAITING
        stime = time.time() - stime
        self.context.solver.log(1, "Model '", self.model.get_name(), "' next solution in ", round(stime, 2), " sec.")

        # Set solve time in solution if not done
        if msol.get_solve_time() == 0:
            msol._set_solve_time(stime)

        # End search if needed
        if not msol:
            self.solver.end_search()
            self.status = STATUS_IDLE

        # Store last solution
        self.last_sol = msol

        # Notify end solve to environment
        runenv.end_solve(self)

        # Return solution
        return msol


    def end_search(self):
        """ End current search.

        This function is available only with local CPO solver with release number strictly greater than 12.6.3.

        Returns:
            Last (fail) model solution with last solve information (type CpoModelSolution)
        Raises:
            CpoNotSupportedException: if method not available in the solver agent.
        """
        self._check_status(STATUS_SEARCH_WAITING)
        msol = self.solver.end_search()
        self.status = STATUS_IDLE
        return msol


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Conflict refiner is controled by the following parameters (that can be set at CpoSolver creation):

         * ConflictRefinerBranchLimit
         * ConflictRefinerFailLimit
         * ConflictRefinerIterationLimit
         * ConflictRefinerOnVariables
         * ConflictRefinerTimeLimit

        that are described in module :mod:`docplex.cp.parameters`.

        Note that the general *TimeLimit* parameter is used as a limiter for each conflict refiner iteration, but the
        global limitation in time must be set using *ConflictRefinerTimeLimit* that is infinite by default.



        This function is available only with local CPO solver with release number strictly greater than 12.6.3.

        Returns:
            List of constraints that cause the conflict (object of class CpoRefineConflictResult)
        Raises:
            CpoNotSupportedException: if method not available in the solver agent.
        """
        self._check_status(STATUS_IDLE)
        self.status = STATUS_REFINING_CONFLICT
        msol = self.solver.refine_conflict()
        self.status = STATUS_IDLE
        return msol


    def propagate(self):
        """ This method invokes the propagation on the current model.

        Constraint propagation is the process of communicating the domain reduction of a decision variable to
        all of the constraints that are stated over this variable.
        This process can result in more domain reductions.
        These domain reductions, in turn, are communicated to the appropriate constraints.
        This process continues until no more variable domains can be reduced or when a domain becomes empty
        and a failure occurs.
        An empty domain during the initial constraint propagation means that the model has no solution.

        The result is a object of class CpoSolveResult, the same than the one returned by solve() method.
        However, variable domains may not be completely defined.

        This function is available only with local CPO solver with release number strictly greater than 12.6.3.

        Returns:
            Propagation result (object of class CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._check_status(STATUS_IDLE)
        self.status = STATUS_PROPAGATING
        psol = self.solver.propagate()
        self.status = STATUS_IDLE
        return psol


    def get_last_solution(self):
        """ Get the last solution returned by this solver

        Returns:
            Last solution returned by this solver
        """
        return self.last_sol


    def end(self):
        # End this solver and release associated resources
        if (self.solver is not None) and (self.status != STATUS_RELEASED):
            self.solver.end()
            self.solver = None
            self.status = STATUS_RELEASED


    def next(self):
        """ For solution iteration, get the next available solution.

        This function is available only with local CPO solver with release number strictly greater than 12.6.3.

        Returns:
            Next model solution (object of class CpoModelSolution)
        """
        # Return solution
        msol = self.search_next()
        if msol:
            return msol
        else:
            raise StopIteration()


    def __next__(self):
        """ Get the next available solution (same as next() for compatibility with Python 3)

        This function is available only with local CPO solver with release number strictly greater than 12.6.3.

        Returns:
            Next model solution (object of class  CpoModelSolution)
        """
        return self.next()


    def _check_status(self, ests):
        """ Throws an exception if solver status is not the expected one

        Args:
            ests:  Expected status, or list of expected statuses
        Raise:
            CpoException if solver is not in the right status
        """
        if (self.status != ests):
           raise CpoException("Unexpected solver status. Should be '{}' instead of '{}'".format(ests, self.status))





###############################################################################
##  Private Functions
###############################################################################

def _get_solver_agent_class(aname, sctx):
    """ Get a solver agent class from its name

    Args:
        aname:  Solver agent name
        sctx:   Candidate solver context
    Returns:
        Solver agent class
    """
    # Check for solver agent context
    if not isinstance(sctx, Context):
        raise CpoException("Unknown solving agent '" + aname + "'. Check config.context.solver.agent parameter.")
    cpath = sctx.class_name
    if cpath is None:
        raise CpoException("Solving agent '" + aname + "' context does not contain attribute 'class_name'")

    # Split class name
    pnx = cpath.rfind('.')
    if pnx < 0:
        raise CpoException("Invalid class name '" + cpath + "' for solving agent '" + aname + "'. Should be <package>.<module>.<class>.")
    mname = cpath[:pnx]
    cname = cpath[pnx + 1:]

    # Load module
    try:
        module = importlib.import_module(mname)
    except Exception as e:
        raise CpoException("Module '" + mname + "' import error: " + str(e))

    # Create and check class
    sclass = getattr(module, cname, None)
    if sclass is None:
        raise CpoException("Module '" + mname + "' does not contain a class '" + cname + "'")
    if not inspect.isclass(sclass):
        raise CpoException("Agent class '" + cpath + "' is not a class.")
    if not issubclass(sclass, CpoSolverAgent):
        raise CpoException("Solver agent class '" + cpath + "' does not extend CpoSolverAgent.")

    # Return
    return sclass

