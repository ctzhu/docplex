# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using DOcloud services.
"""

import docplex.cp.config as config
import docplex.cp.solution as solution
from docplex.cp.docloud_client import JobClient
from docplex.cp.utils import CpoException
import docplex.cp.cpo_compiler as cpo_compiler
import docplex.cp.solver as solver


###############################################################################
##  Public classes
###############################################################################

class CpoSolverDocloud(solver.CpoSolverAgent):
    """ Solver of CPO model using DOcloud services. """
    
    def __init__(self, model, params, context):
        """ Create a new solver using DOcloud web service

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  DOcloud Solver context
        Raises:
            CpoException if jar file does not exists
        """
        if (context.key is None) or (' ' in context.key):
            raise CpoException("Your DOcloud key has not been set")
        super(CpoSolverDocloud, self).__init__(model, params, context)

    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: Calls to DOcloud solving
         * 3: Detailed DOcloud job information
         * 4: REST requests and response codes

        Returns:
            Model solution (type CpoModelSolution)
        Raises:
            CpoException if error occurs
        """
        # Create DOcloud client
        ctx = self.context
        client = JobClient(ctx)

        # Convert model into CPO format
        cpostr = self._get_cpo_model_string()

        # Solve model and retrieve solution
        name = self.model.get_name()
        maxwait = ctx.params.TimeLimit + ctx.request_timeout + ctx.result_wait_extra_time if ctx.params.TimeLimit else 0
        try:
            client.create_job(name, cpostr)
            client.execute_job()
            client.wait_job_termination(maxwait=maxwait)
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
            msol = solution.CpoModelSolution()
            _add_solve_status(msol, jinfo.get('executionStatus', None), jinfo.get('solveStatus', None))
            _add_details(msol, jinfo.get('details', None))

            # Add solve time
            msol._set_solve_time( (float(jinfo.get('endedAt', 0)) - float(jinfo.get('startedAt', 0))) / 1000)

            # Get response if any
            jsol = None
            if msol.is_solution():
                try:
                    jsol = client.get_attachment("solution.json").decode('utf-8')
                except Exception as e:
                    raise CpoException("Model solution access error: " + str(e))

                ctx.log(3, "JSON solution:\n", jsol)

            # Get log
            if self._is_log_required():
                logstr = client.get_log()
                self._set_solver_log(logstr, msol)

        finally:
            # Delete job
            if ctx.clean_job_after_solve:
                client.clean_job()

        # Append detailed solution values if any
        if jsol is not None:
            msol._add_json_solution(self.model, jsol)

        # Return
        return msol


###############################################################################
##  Model solution building functions
###############################################################################


def _add_solve_status(msol, dcest, dcsst):
    """ Add the solve status in the solution
    Args:
        msol:  Model solution to fill
        dcest: DOcloud execution status
        dcsst: DOcloud solve status
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
        msol._set_objective_values([float(v) for v in oval.split(";")])
    # Set model characteristics
    nbctrs = detls.get('MODEL_DETAIL_CONSTRAINTS', 0)
    nbintvars = detls.get('MODEL_DETAIL_INTEGER_VARS', 0)
    nbitvvars = detls.get('MODEL_DETAIL_INTERVAL_VARS', 0)
    nbseqvars = detls.get('MODEL_DETAIL_SEQUENCE_VARS', 0)
    msol._set_model_attributes(nbintvars=nbintvars, nbitvvars=nbitvvars, nbseqvars=nbseqvars, nbctrs=nbctrs)
        
        
###############################################################################
##  Test program
###############################################################################

if __name__ == "__main__":
    # Solve sample model
    if True:
        import os
        from docplex.cp.cpo_parser import CpoParser
        mfile = os.path.dirname(__file__) + "/../../../UnitTests/cpomodels/Color.cpo"
        print("Solve model from file: " + mfile)
        prs = CpoParser()
        prs.parse(mfile)
        dcld = CpoSolverDocloud(prs.get_model(), params=prs.get_parameters(), context=config.context.solver.docloud)
        msol = dcld.solve()
        msol.print_solution()

    # List all jobs
    if False:
        jcli = JobClient(config.context.docloud)
        ljobs = jcli.get_all_jobs()
        print(str(len(ljobs)) + " jobs:")
        for job in ljobs:
            print(job['attachments'][0]['name'] + ": " + job['_id'] + ", " + job['solveStatus'])

    # Clear all jobs
    if False:
        jcli = JobClient(config.context.docloud)
        jcli.clean_all_jobs()
