# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint:skip-file
import json

from six import iteritems
from requests.exceptions import ConnectionError

from docplex.mp.utils import resolve_pattern, get_logger, normalize
from docloud.job import JobClient, DOcloudInterruptedException, DOcloudNotFoundError
from docloud.status import JobSolveStatus

# gendoc: ignore


class DOcloudConnectorException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class DOcloudConnector(object):
    # json keys
    JSON_LINEAR_CTS_KEY = 'linearConstraints'

    def __init__(self, docloud_context, verbose=False, log_output=None):
        """ Starts a connector which URL and authorization are stored in the specified context, 
        along with other connection parameters """
        if docloud_context is None or not docloud_context.has_credentials():
            raise DOcloudConnectorException("Please provide DOcloud credentials")

        url = docloud_context.url
        auth = docloud_context.key

        self._verbose = verbose or docloud_context.verbose

        self.client = JobClient(docloud_context.url, docloud_context.key)

        self.logger = get_logger('DOcloudConnector', self._verbose)

        if docloud_context.log_requests:
            self.client.rest_callback = \
                lambda m, u, *a, **kw: self._rest_callback(m, u, *a, **kw)

        self.logger.info("DOcloud connection using url = " + str(url) + " api_key = " + str(auth))

        self.client.verify = docloud_context.verify
        self.logger.info("DOcloud SSL verification = " + str(self.client.verify))

        self.waittime = docloud_context.waittime
        self.logger.info("   waittime = " + str(self.waittime))

        self.client.timeout = docloud_context.timeout if docloud_context.timeout is not None else None
        self.logger.info("   timeout = " + str(self.client.timeout))

        self._base_url = url
        # self._base_headers = {'X-IBM-Client-Id': api_key, 'Content-Type': 'application/json'}
        self.json = ""
        self.jobInfo = None
        self.run_deterministic = docloud_context.run_deterministic
        self.log_output = log_output
        self._hasSolution = False  # _submit_job change this
        self.__vars = None

    def _check_nonempty_json(self):
        if not self.json:
            raise DOcloudConnectorException("* empty JSON result!")

    @property
    def is_verbose(self):
        return self._verbose

    def log(self, msg, *args):
        if self.is_verbose:
            log_msg = "* {0}".format(resolve_pattern(msg, args))
            self.logger.info(log_msg)
            if self.log_output is not None:
                self.log_output.write(log_msg)

    def _as_string(self, content):
        resp_content_as_string = content
        if not isinstance(resp_content_as_string, str):
            resp_content_as_string = content.decode('utf-8')
        return resp_content_as_string

    def _submit_job(self, model_data, job_name=None, gzip=False, prm_data=None,
                    info_callback=None):
        self._hasSolution = False
        self.__vars = None

        try:
            # create job
            jobid = self.client.create_job(attachments=[{'name': job_name}, {'name': 'file.prm'}])
            self.log("job creation submitted, id is: {0!s}".format(jobid))
            if info_callback:
                info_callback({'jobid': jobid})
        except ConnectionError as c_e:
            raise DOcloudConnectorException("Cannot connect to {0}, error: {1}".format(self._base_url, str(c_e)))

        try:
            # upload prm 
            if prm_data is not None:
                encoded_prm_data = prm_data.encode('utf-8')
                self.client.upload_job_attachment(jobid, attid="file.prm",
                                                  data=encoded_prm_data,
                                                  gzip=gzip)
                self.log("CPLEX parameters file has been uploaded")

            # upload model
            self.client.upload_job_attachment(jobid,
                                              attid=job_name,
                                              data=model_data,
                                              gzip=gzip)
            self.log("model data '{attid}' has been uploaded".format(attid=job_name))

            # execute job
            self.client.execute_job(jobid)
            self.log("DOcloud execute submitted has been started")

            # get job execution status until it's processed or failed
            timedout = False
            try:
                if self.waittime is not None:
                    self.log("waiting for job completion with a wait time of {waittime} sec".format(waittime=self.waittime))
                else:
                    self.log("waiting for job completion with no wait time")
                wt = -1 if self.waittime is None else self.waittime
                self._executionStatus = self.client.wait_for_completion(jobid, waittime=wt)
            except DOcloudInterruptedException:
                timedout = True
            self.log("docloud execution has finished")

            # get job status. Do this before any time out handling
            self.jobInfo = self.client.get_job(jobid)

            if timedout:
                self._hasSolution = False
                self.log("Solve timed out after {waittime} sec".format(waittime=self.waittime))
                return None

            # get log as blog
            if self.log_output is not None:
                log_as_string = self.client.download_job_log(jobid).decode('utf-8')
                self.log_output.write(log_as_string)

            # get solution
            try:
                solution_as_string = self._as_string(self.client.download_job_attachment(jobid, attid="solution.json"))
                myjson = json.loads(solution_as_string, parse_constant='utf-8')['CPLEXSolution']
                self._hasSolution = bool(myjson)

            except DOcloudNotFoundError:
                myjson = None
                self._hasSolution = False
                self.log("no solution in attachment")
            self.log("docloud results have been received")

            return myjson

        finally:
            deleted = self.client.delete_job(jobid)
            self.log("delete status for job: {0!s} = {1!s}".format(jobid, deleted))

    def submit_model_data(self, mdl_name, mdl_data, extension, prm_data=None,
                          gzip=None, info_callback=None):
        job_name = normalize(mdl_name) + extension  # use extension or not ?
        self.json = self._submit_job(model_data=mdl_data, 
                                     prm_data=prm_data,
                                     job_name=job_name, 
                                     gzip=gzip,
                                     info_callback=info_callback)
        return self.json

    def has_solution(self):
        return self._hasSolution

    def get_cplex_details(self):
        if self.jobInfo:
            return self.jobInfo.get("details")

    def is_mip(self):
        return self.json and self.json['header']['solutionMethodString'] == 'mip'

    def variable_values(self):
        return self.get_variable_attr_map('value')

    def variable_reduced_costs(self):
        return self.get_variable_attr_map('reducedCost')

    def get_variable_attr_map(self, json_attr_name):
        assert json_attr_name
        if not self.json:
            return {}
        else:
            all_vars = self._getvars()
            attr_map = {int(v['index']): float(v[json_attr_name]) for v in all_vars}
            return attr_map

    def cplex_index_name_map(self):
        json_res = self.json
        if json_res:
            return {int(v['index']): v['name'] for v in self._getvars()}
        else:
            return {}

    def _getvars(self):
        if self.__vars is None:
            self.__vars = self.json['variables']
        return self.__vars

    def variable_results(self):
        if not self.json:
            return {}, {}
        else:
            all_vars = self._getvars()
            value_map = {int(v['index']): float(v['value']) for v in all_vars}
            rc_map = {} if self.is_mip() else {int(v['index']): float(v['reducedCost']) for v in all_vars}
            return value_map, rc_map

    def constraint_results(self):
        if not self.json or self.is_mip():
            return {}, {}
        else:
            lincst_key = self.JSON_LINEAR_CTS_KEY
            if lincst_key in self.json:
                all_linear_cts = self.json[lincst_key]
                dual_map = {int(v['index']): float(v['dual']) for v in all_linear_cts}
                slack_map = {int(v['index']): float(v['slack']) for v in all_linear_cts}
            else:
                dual_map = slack_map = {}
            return dual_map, slack_map

    def get_status_id(self):
        self._check_nonempty_json()
        return int(self.json['header']['solutionStatusValue'])

    def get_objective(self):
        self._check_nonempty_json()
        return float(self.json['header']['objectiveValue'])

    def get_solve_status(self):
        if 'solveStatus' in self.jobInfo:
            return JobSolveStatus[self.jobInfo['solveStatus']]
        else:
            return None

    def _rest_callback(self, method, url, *args, **kwargs):
        """The callback called by the DOcloud client to log REST operations
        """
        self.logger.info("{0} {1}".format(method, url))
        if len(args) > 0:
            self.logger.info("   Additionnal args : {0}".format(','.join(args)))
        for k, v in iteritems(kwargs):
            self.logger.info("   {0}: {1}".format(k, v))
