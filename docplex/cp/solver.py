# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model represented by a CpoModel object.

The solver is created passing the source model as first parameter.
The solving itself is performed by calling the solve() method.

Solving is executed as defined in the configuration (see config.py module to see how to customize it).
For an execution on DOcplexcloud, the configuration must contain the target URL and the authentication key.
"""

import docplex.cp.config as config
from docplex.cp.utils import CpoException, make_directories, DEFAULT, Context
import docplex.cp.utils as utils
import docplex.cp.cpo_compiler as cpo_compiler

import time, importlib, inspect


###############################################################################
##  Public classes
###############################################################################

class CpoSolutionIterator(object):
    """ Iterator over the different solver solutions provided by next() method.
    """

    def __init__(self, solver):
        """ Create a new solution iterator
        """
        self.solver = solver

    def __iter__(self):
        return self

    def next(self):
        """ Get the next available solution.
        Returns:
            Next model solution (object of class  CpoModelSolution)
        """
        msol = self.solver.next()
        if msol:
            return msol
        else:
            raise StopIteration()

    def __next__(self): # Compatibility for Python 3
        return self.next()


class CpoSolverAgent(object):
    """ CPO model abstract solver agent

    This class is extended by actual solver agents of CPO models that can be addressed from CpoSolver generic class.
    """

    def __init__(self, model, params, context):
        """ Create a new solver using DOcplexcloud web service

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  Solver context
        Raises:
            CpoException if jar file does not exists
        """
        super(CpoSolverAgent, self).__init__()
        self.model = model
        self.params = params
        self.context = context

    def solve(self):
        """ Solve the model

        Returns:
            Model solution (object of class CpoModelSolution)
        """
        raise CpoException("Method not implemented in this solver agent.")

    def next(self):
        """ Get the next available solution.

        (This method starts search automatically.)

        Returns:
            Next model solution (object of class  CpoModelSolution)
        """
        raise CpoException("Method not implemented in this solver agent.")

    def _get_cpo_model_string(self):
        """ Get the CPO model as a string, according to configuration

        Args:
            cpostr:  CPO model as a string in CPO format
        Return:
            String containing the CPO model in CPO file format
        """
        # Build string
        ctx = self.context
        cpostr = cpo_compiler.get_cpo_model(self.model, self.params, None if ctx is None else ctx.model)
        # Trace CPO model if required
        lout = ctx.log_output
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
            with open(file, "w") as f:
                f.write(cpostr)
        # Return
        return cpostr

    def _is_log_required(self):
        """ Check if solver log is required.

        Return:
            True if solver log is required, False otherwise
        """
        return self.context.add_log_to_solution or self.context.trace_log


class CpoSolver(object):
    """ Generic CPO model solver

    This class is the visible solver that creates appropriate actual solving class depending on configuration
    parameter context.solver.agent.
    """
    
    def __init__(self, model, context=DEFAULT, **kwargs):
        """ Create a new CPO model solver

        Args:
            model:          Model to solve
            context:        Global solving context. If not given, context is the default context that is set in config.py.
        Optional args:
            params:         Solving parameters (CpoParameters) that overwrites those in solving context
            etc             All other context parameters that can be changed
        """
        super(CpoSolver, self).__init__()

        # Clone default context and make changes
        if context is DEFAULT:
            context = config.context
        context = _update_context(context, kwargs)

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
        actx = sctx[aname]
        sclass = _get_solver_agent_class(aname, actx)
        self.solver = sclass(self.model, sctx.params, actx)


    def solve(self):
        """ Solve the model

        Returns:
            Model solution (object of class CpoModelSolution)
        """

        # Solve model
        stime = time.time()
        msol = self.solver.solve()
        stime = time.time() - stime
        self.context.solver.log(1, "Model '", self.model.get_name(), "' solved in ", round(stime, 2), " sec.")

        # Set solve time in solution if not done
        if msol.get_solve_time() == 0:
            msol._set_solve_time(stime)

        # Return solution
        return msol
        
     
    def next(self):
        """ Get the next available solution.

        (This method starts search automatically.)

        Returns:
            Next model solution (object of class CpoModelSolution)
        """
        # Solve model
        stime = time.time()
        msol = self.solver.next()
        stime = time.time() - stime
        self.context.solver.log(1, "Model '", self.model.get_name(), "' solved in ", round(stime, 2), " sec.")

        # Set solve time in solution if not done
        if msol.get_solve_time() == 0:
            msol._set_solve_time(stime)

        # Return solution
        return msol

    def solutions_iterator(self):
        """ Get an iterator on the sequence of solutions

        Returns:
            Solution iterator (each item is object of class CpoModelSolution)
        """
        return CpoSolutionIterator(self)


###############################################################################
##  Private Functions
###############################################################################

# Attribute values denoting a default value
DEFAULT_VALUES = ("ENTER YOUR KEY HERE", "ENTER YOUR URL HERE", "default")


def _update_context(ctx, kwargs):
    """ Build real context from source and list of replacements

    Args:
        ctx:     Source context
        kwargs:  Dictionary of replacements
    Returns:
        Updated context (source context is cloned)
    """
    ctx = ctx.clone()
    rplist = []  # List of replacements to be done in solving parameters
    for k, v in kwargs.items():
        if v not in DEFAULT_VALUES:
            rp = ctx.search_and_replace_attribute(k, v)
            # If not found, set in solving parameters
            if (rp is None):
                rplist.append((k, v))
    # Replace in parameters
    params = ctx.params
    for k, v in rplist:
        params.set_attribute(k, v)
    return ctx


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


