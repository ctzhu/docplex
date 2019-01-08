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
For an execution on DOcloud, the configuration must contain the target URL and the authentication key.
"""

import docplex.cp.config as config
from docplex.cp.utils import CpoException, make_directories, DEFAULT, Context
import docplex.cp.utils as utils
import docplex.cp.cpo_compiler as cpo_compiler

import time, importlib, inspect


###############################################################################
##  Public classes
###############################################################################

class CpoSolverAgent(object):
    """ CPO model abstract solver agent

    This class is extended by actual solver agents of CPO models that can be addressed from CpoSolver generic class.
    """

    def __init__(self, model, params, context):
        """ Create a new solver using DOcloud web service

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  DOcloud Solver context
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
            Model solution (type CpoModelSolution)
        """
        raise CpoException("The solve() method not implemented")

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

    def _set_solver_log(self, logstr, msol):
        """ Notify the content of the log, for storing in solution and tracing if required

        Args:
            logstr:  Solver log
            msol:    Model solution
        """
        '''
        lout = self.context.log_output
        if lout and (self.context.trace_log):
            lout.write("Model '" + str(self.model.get_name()) + "' solver log:\n")
            lout.write(logstr)
            lout.write("\n")
            lout.flush()
        '''
        if self.context.add_log_to_solution:
            msol._set_solver_log(logstr)


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
            url:            URL of the DOcloud service that overwrites the one defined in solving context.
            key:            Authentication key of the DOcloud service that overwrites the one defined in solving context.
            etc             All other context parameters that can be changed
        """
        super(CpoSolver, self).__init__()

        # Clone default context and make changes
        if context is DEFAULT:
            context = config.context
        context = _update_context(context, kwargs)

        # Save attributes
        self.model = model
        self.context = context

    def solve(self):
        """ Solve the model

        Returns:
            Model solution (type CpoModelSolution)
        """
        sctx = self.context.solver

        # Determine appropriate solver agent
        aname = sctx.agent
        if aname is None:
            aname = "docloud"
        else:
            aname = aname.lower()
        sctx.log(1, "Solve model '", self.model.get_name(), "' with agent '", aname, "'")

        # Retrieve solver agent class and create instance
        actx = sctx[aname]
        sclass = _get_solver_agent_class(aname, actx)
        solver = sclass(self.model, sctx.params, actx)

        # Solve model
        stime = time.time()
        msol = solver.solve()
        stime = time.time() - stime
        sctx.log(1, "Model '", self.model.get_name(), "' solved in ", round(stime, 2), " sec.")

        # Set solve time in solution if not done
        if msol.get_solve_time() == 0:
            msol._set_solve_time(stime)

        # Return solution
        return msol
        
     
###############################################################################
##  Private Functions
###############################################################################

def _update_context(ctx, kwargs):
    """ Build real context from source and list of replacements

    Args:
        ctx:     Source context
        kwargs:  Dixtionary of replacements
    Returns:
        Updated context (source context is cloned)
    """
    ctx = ctx.clone()
    rplist = []  # List of replacements to be done in solving parameters
    for k, v in kwargs.items():
        if v is not DEFAULT:
            rp = ctx.search_and_replace_attribute(k, v)
            # If not found, set in solving parameters
            if not rp:
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


