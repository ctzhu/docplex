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
from docplex.cp.solver.solver_listener import CpoSolverListener
import os

try:
    import docplex.util.environment as runenv
    IS_IN_WORKER = isinstance(runenv.get_environment(), runenv.WorkerEnvironment)
except:
    IS_IN_WORKER = False


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
## Classes
###############################################################################

# Solver listener that interact with environment
class EnvSolverListener(CpoSolverListener):
    """ Cpo solver listener that interact with environment.
    This listener is added by the CpoSolver when it is created, if the environment exists.
    """
    def __init__(self):
        super(EnvSolverListener, self).__init__()


    def solver_created(self, solver):
        """ Notify the listener that the solver object has been created.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        # Check if calling environment is DODS (Decision Optimization for Data Science)
        env = _get_environment()
        if env is not None:
            # Check if solve must be transformed in start/next
            value = os.environ.get("IS_DODS")
            if str(value).lower() == "true":
                solver.context.solver.solve_with_start_next = True


    def start_solve(self, solver):
        """ Notify that the solve is started.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        _notify_start_solve(solver)


    def end_solve(self, solver):
        """ Notify that the solve is ended.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        _notify_end_solve(solver)


    def solution_found(self, solver, msol):
        """ Signal that a solution has been found.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            msol:   Model solution, object of class :class:`~docplex.cp.solution.CpoSolveResult`
        """
        _update_solve_details(solver)
        _publish_solution(solver)


###############################################################################
## Public functions
###############################################################################

def get_environment():
    """ Returns the Environment object that represents the actual execution environment.

    Returns:
        Environment descriptor, None if none.
    """
    # Check if environment available
    if not IS_IN_WORKER:
        return None
    return runenv.get_environment()


###############################################################################
## Private functions
###############################################################################

def _notify_start_solve(solver):
    """ Notify the start of a model solve

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment()
    if env is None:
        return

    if _get_property(solver, "solve_details", True):
        # Set ordered list of KPIs in solve details
        kpis = solver.get_model().get_kpis()
        if kpis:
            # Add ordered list of kpi names
            sdetails = {'MODEL_DETAIL_KPIS': json.dumps(list(kpis.keys()))}
            env.notify_start_solve(sdetails)


def _update_solve_details(solver):
    """ Update solve details

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment()
    if env is None:
        return

    if not _get_property(solver, "solve_details", True):
       return

    # Get last solver solution
    msol = solver.get_last_result()
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

    # Set KPIs if any
    kpis = msol.get_kpis()
    if kpis:
        # Add ordered list of kpi names
        sdetails["MODEL_DETAIL_KPIS"] = json.dumps(list(kpis.keys()))
        # Add KPIs
        for k, v in kpis.items():
            sdetails["KPI." + k] = v

    # Submit details to environment
    env.update_solve_details(sdetails)


def _publish_solution(solver):
    """ Publish last solution in json format

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment()
    if env is None:
        return

    # Write JSON solution as output
    resout = _get_property(solver, "result_output", 'solution.json')
    if resout:
        json = solver.agent._get_last_json_result_string()
        if json is not None:
            with env.get_output_stream(resout) as fp:
                fp.write(json.encode('utf-8'))

    # Publish kpis
    kpiout = _get_property(solver, "kpis_output", 'kpis.csv')
    if kpiout:
        sres = solver.get_last_result()
        if sres:
            kpis = sres.get_kpis()
            if kpis:
                with env.get_output_stream(kpiout) as fp:
                    fp.write('"NAME","VALUE"\n'.encode('utf-8'))
                    for k, v in kpis.items():
                        fp.write('{},{}\n'.format(encode_csv_string(k), v).encode('utf-8'))


def _notify_end_solve(solver):
    """ Notify the end of a solve

    Args:
       solver: Source CPO solver
    """
    # Get environment to be called
    env = _get_environment()
    if env is None:
        return

    # Set solve status
    if _get_property(solver, "solve_details", True):
        res = solver.get_last_result()
        if res is None:
            status = _STATUS_UNKNOWN
        else:
            status = _SOLVE_STATUS_MAP.get(res.get_solve_status(), _STATUS_UNKNOWN)
        env.notify_end_solve(status)


def _get_environment():
    """ Get the environment to call, checking if auto-publish is required.
    Returns:
        Environment to call, None if none
    """
    # Check if environment available
    if not IS_IN_WORKER:
        return None

    # Skip if environment is local
    env = runenv.get_environment()
    if isinstance(env, runenv.LocalEnvironment):
        return None

    return env


def _get_property(solver, prop, default):
    """ Get the value of an auto-publish property.
    Args:
        solver:  Source CPO solver
        prop:    Auto_publish specific property that should be checked
        default: Default property value
    Returns:
        Environment to call, None if none
    """
    # Check auto_publish config
    publish = solver.context.solver.auto_publish
    if (publish is None) or (publish is False):
        return None
    if publish is True:
        return default

    if isinstance(publish, Context):
        pval = publish.get_attribute(prop)
        return pval if pval else None

    # Not found
    return None
