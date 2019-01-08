# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using DOcplexcloud services.
"""

import docplex.cp.config as config
import docplex.cp.solution as solution
from docplex.cp.solver.docloud_client import JobClient
from docplex.cp.utils import CpoException
import docplex.cp.solver.solver as solver


###############################################################################
##  Public classes
###############################################################################

class CpoSolverDocloud(solver.CpoSolverAgent):
    """ Solver of CPO model using DOcplexcloud services. """
    
    def __init__(self, model, params, context):
        """ Create a new solver using DOcplexcloud web service

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  DOcplexcloud Solver context
        Raises:
            CpoException if jar file does not exists
        """
        if (context.key is None) or (' ' in context.key):
            raise CpoException("Your DOcplexcloud key has not been set")
        super(CpoSolverDocloud, self).__init__(model, params, context)
        self.log_data = []

    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: Calls to DOcplexcloud solving
         * 3: Detailed DOcplexcloud job information
         * 4: REST requests and response codes

        Returns:
            Model solve result (object of class CpoSolveResult)
        Raises:
            CpoException if error occurs
        """
        # Create DOcplexcloud client
        ctx = self.context
        client = JobClient(ctx)

        # Convert model into CPO format
        cpostr = self._get_cpo_model_string()

        # Solve model and retrieve solution
        name = self.model.get_name()
        maxwait = ctx.params.TimeLimit + ctx.request_timeout + ctx.result_wait_extra_time if ctx.params.TimeLimit else 0
        try:
            # Create job and start execution
            client.create_job(name, cpostr)
            client.execute_job()

            # Wait job termination
            lgntf = None
            if self._is_log_required():
                self.log_data = []
                lgntf = (lambda recs: self._log_records(recs))
            client.wait_job_termination(maxwait=maxwait, lognotif=lgntf)
            jinfo = client.get_info()

            # Trace response if required
            if ctx.is_log_enabled(3):
                ctx.log(3, "Job info:")
                for k in jinfo.keys():
                    ctx.log(3, k, " : ", jinfo[k])
                
            # Check failure
            fail = jinfo.get('failure', None)
            if (fail is not None):
                raise CpoException(fail.get('message', "Unknown failure"))
        
            # Create solution structure
            msol = solution.CpoSolveResult(self.model)
            _add_solve_status(msol, jinfo.get('executionStatus', None), jinfo.get('solveStatus', None))
            _add_details(msol, jinfo.get('details', None))

            # Add solve time
            msol._set_solve_time( (float(jinfo.get('endedAt', 0)) - float(jinfo.get('startedAt', 0))) / 1000)

            # Get response if any
            jsol = None
            if msol.is_solution():
                try:
                    jsol = client.get_attachment("solution.json")
                except Exception as e:
                    raise CpoException("Model solution access error: " + str(e))
            self._set_last_json_result_string(jsol)

            # Get log
            if self.context.add_log_to_solution:
                logstr = '\n'.join([rec['message'] for rec in self.log_data])
                msol._set_solver_log(logstr)

        finally:
            # Delete job
            if ctx.clean_job_after_solve:
                client.clean_job()

        # Append detailed solution values if any
        if jsol is not None:
            msol._add_json_solution(jsol)

        # Return
        return msol

    def _log_records(self, records):
        """ Method called when new log records are provided
        Args:
            records: List of new records
        """
        # Append to list of log records
        ctx = self.context
        if ctx.add_log_to_solution:
            self.log_data.extend(records)

        # Trace on output if required
        if ctx.trace_log:
            out = ctx.get_log_output()
            if out:
                for rec in records:
                    out.write(rec['message'])
                    out.write('\n')
                    out.flush()




###############################################################################
##  Model solution building functions
###############################################################################


def _add_solve_status(msol, dcest, dcsst):
    """ Add the solve status in the solution
    Args:
        msol:  Model solution to fill
        dcest: DOcplexcloud execution status
        dcsst: DOcplexcloud solve status
    """
    s = solution.SOLVE_STATUS_UNKNOWN
    if (dcest == "INTERRUPTED"):           s = solution.SOLVE_STATUS_JOB_ABORTED
    elif (dcest == "FAILED"):              s = solution.SOLVE_STATUS_JOB_FAILED
    elif (dcsst == "FEASIBLE_SOLUTION"):   s = solution.SOLVE_STATUS_FEASIBLE
    elif (dcsst == "INFEASIBLE_SOLUTION"): s = solution.SOLVE_STATUS_INFEASIBLE
    elif (dcsst == "OPTIMAL_SOLUTION"):    s = solution.SOLVE_STATUS_OPTIMAL
    msol._set_solve_status(s)
        
        
def _add_details(msol, detls):
    """ Add details to solution
    Args:
        msol: Model solution to fill
        detls: Details dictionary
    """
    if detls is None:
        return
    # Set objective value
    oval = detls.get('PROGRESS_CURRENT_OBJECTIVE', None)
    if oval is not None:
        msol.solution._set_objective_values([float(v) for v in oval.split(";")])
    # Set model characteristics
    nbctrs = int(detls.get('MODEL_DETAIL_CONSTRAINTS', 0))
    nbintvars = int(detls.get('MODEL_DETAIL_INTEGER_VARS', 0))
    nbitvvars = int(detls.get('MODEL_DETAIL_INTERVAL_VARS', 0))
    nbseqvars = int(detls.get('MODEL_DETAIL_SEQUENCE_VARS', 0))
    msol._set_model_attributes(nbintvars=nbintvars, nbitvvars=nbitvvars, nbseqvars=nbseqvars, nbctrs=nbctrs)
