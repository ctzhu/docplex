# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the client that allows to notify solving environment
with relevant events.

The solving environment is typically local or a Python worker.

Real implementation of environment specifics is done in module
docplex.util.environment.py. The present module provides what is necessary
to call it with appropriate CPO solver data.

Note tha ta default null behavior is provided if the docplex.util.environment
can not be imported.
"""

from docplex.cp.solution import *
try:
    import docplex.util.environment as runenv
    ENVIRONMENT_PRESENT = True
except:
    ENVIRONMENT_PRESENT = False


###############################################################################
## Constants
###############################################################################

# Possible solve statuses
_STATUS_UNKNOWN                          = 0  # The algorithm has no information about the solution.
_STATUS_FEASIBLE_SOLUTION                = 1  # The algorithm found a feasible solution.
_STATUS_OPTIMAL_SOLUTION                 = 2  # The algorithm found an optimal solution.
_STATUS_INFEASIBLE_SOLUTION              = 3  # The algorithm proved that the model is infeasible.
_STATUS_UNBOUNDED_SOLUTION               = 4  # The algorithm proved the model unbounded.
_STATUS_INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5  # The model is infeasible or unbounded.

# Map of CPO solve status on environment status
_SOLVE_STATUS_MAP = {SOLVE_STATUS_FEASIBLE   : _STATUS_FEASIBLE_SOLUTION,
                     SOLVE_STATUS_INFEASIBLE : _STATUS_INFEASIBLE_SOLUTION,
                     SOLVE_STATUS_OPTIMAL    : _STATUS_OPTIMAL_SOLUTION}

###############################################################################
## Public functions
###############################################################################

def start_solve(solver):
    """ Process the start of a model solve

    Args:
       solver: Source CPO solver
    """
    _notify_start_solve(solver)


def end_solve(solver):
    """ Process the end of a model solve

    Args:
       solver: Source CPO solver
    """
    _update_solve_details(solver)
    _publish_solution(solver)
    _notify_end_solve(solver)


###############################################################################
## Private functions
###############################################################################

def _notify_start_solve(solver):
    """ Notify the start of a model solve

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment(solver, "solve_details")
    if env is None:
        return

    # Notify start solve, with no details
    env.notify_start_solve({})


def _update_solve_details(solver):
    """ Update solve details

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment(solver, "solve_details")
    if env is None:
        return

    # Get last solver solution
    msol = solver.get_last_solution()
    if msol is None:
        return

    # Build solve details
    infos = msol.get_infos()
    sdetails = {}
    nbintvars = infos.get("NumberOfIntegerVariables")
    if nbintvars is not None:
        sdetails["MODEL_DETAIL_INTEGER_VARS"] = nbintvars
    nbintervars = infos.get("NumberOfIntervalVariables")
    if nbintervars is not None:
        sdetails["MODEL_DETAIL_INTERVAL_VARS"] = nbintervars
    nbseqvars = infos.get("NumberOfSequenceVariables")
    if nbseqvars is not None:
        sdetails["MODEL_DETAIL_SEQUENCE_VARS"] = nbseqvars
    nbconstr = infos.get("NumberOfConstraints")
    if nbconstr is not None:
        sdetails["MODEL_DETAIL_CONSTRAINTS"] = nbconstr
    # Set detail type
    if (nbintervars in (0, None)) and (nbseqvars in (0, None)):
        sdetails["MODEL_DETAIL_TYPE"] = "CPO CP"
    else:
        sdetails["MODEL_DETAIL_TYPE"] = "CPO Scheduling"
    # Set objective if any
    objctv = msol.get_objective_values()
    if objctv is not None:
        sdetails["PROGRESS_CURRENT_OBJECTIVE"] = ';'.join([str(x) for x in objctv])
    # Submit details to environment
    env.update_solve_details(sdetails)


def _publish_solution(solver):
    """ Publish last solution in json format

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment(solver, "json_solution")
    if env is None:
        return

    # Write JSON solution as output
    json = solver.agent._get_last_json_result_string()
    if json is not None:
        with env.get_output_stream("solution.json") as fp:
            fp.write(json.encode('utf-8'))


def _notify_end_solve(solver):
    """ Notify the end of a solve

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment(solver, "solve_details")
    if env is None:
        return

    # Set solve status
    res = solver.get_last_solution()
    if res is None:
        status = _STATUS_UNKNOWN
    else:
        status = _SOLVE_STATUS_MAP.get(res.get_solve_status(), _STATUS_UNKNOWN)
    env.notify_end_solve(status)


def _get_environment(solver, prop):
    """ Get the environment to call
    Args:
        solver: Source CPO solver
        prop:   Auto_publish specific property that should be checked
    Returns:
        Environment to call, None if none
    """
    # Check if environment available
    if not ENVIRONMENT_PRESENT:
        return None

    # Skip if environment is local
    env = runenv.get_environment()
    if isinstance(env, runenv.LocalEnvironment):
        return None

    # Check auto_publish config
    pblsh = solver.context.solver.auto_publish
    if (pblsh is None) or not((pblsh is True) or pblsh.get_attribute(prop)):
        return None

    # Return
    return env


# Test environment class
class TestEnvironment(object):
    def get_input_stream(self, name):
        return open(name, "rb")
    def get_output_stream(self, name):
        return open(name, "wb")
    def notify_start_solve(self, details):
        print("Start solve, details: " + str(details))
    def update_solve_details(self, details):
        print("Update details, details: " + str(details))
    def notify_end_solve(self, status):
        print("End solve, status: " + str(status))
