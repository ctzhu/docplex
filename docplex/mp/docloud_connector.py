# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint:skip-file
# gendoc: ignore

import json

from datetime import datetime
from six import iteritems, string_types
from requests.exceptions import ConnectionError

from docloud.job import JobClient, DOcloudInterruptedException, DOcloudNotFoundError
from docloud.status import JobSolveStatus, JobExecutionStatus

from docplex.mp.progress import ProgressData
from docplex.mp.utils import resolve_pattern, get_logger, normalize_basename
from docplex.mp.utils import CyclicLoop


def key_as_string(key):
    """For keys, we don't want the key to appear in INFO log outputs.
    Instead, we display the first 4 chars and the last 4 chars.
    """
    return (key[:4]+"*******"+ key[-4:]) if isinstance(key, string_types) else str(key)


class DOcloudConnectorException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class DOcloudEvent(object):
    """Used internally to manage events that must be processed in
    the main thread.

    The Progress and log monitor loop poll the service for progress and log
    information, then queue DOcloudEvents on an internal queue. That queue
    is processed by a loop in the main thread, making sure that logs are
    printed in the main thread and progress listeners are called in the
    main thread too.
    """
    def __init__(self, event_type, data):
        """The type if a string. Can be "log" or "progress"
        """
        self.type = event_type
        self.data = data


class DOcloudConnector(object):
    # json keys
    JSON_LINEAR_CTS_KEY = 'linearConstraints'

    def __init__(self, docloud_context, log_output=None):
        """ Starts a connector which URL and authorization are stored in the
        specified context, along with other connection parameters

        Args:
            log_output: The log output stream
        """
        if docloud_context is None or not docloud_context.has_credentials():
            raise DOcloudConnectorException("Please provide DOcplexcloud credentials")

        # store this for future usage
        self.docloud_context = docloud_context

        url = docloud_context.url
        auth = docloud_context.key

        self.logger = get_logger('DOcloudConnector', self.docloud_context.verbose)

        self.logger.info("DOcplexcloud connection using url = " + str(url) + " api_key = " + key_as_string(auth))
        self.logger.info("DOcplexcloud SSL verification = " + str(docloud_context.verify))

        self.logger.info("   waittime = " + str(docloud_context.waittime))
        self.logger.info("   timeout = " + str(docloud_context.timeout))

        self.json = ""
        self.jobInfo = None
        self.run_deterministic = docloud_context.run_deterministic
        self.log_output = log_output
        self._hasSolution = False  # _submit_job change this
        self.__vars = None

    def _check_nonempty_json(self):
        if not self.json:
            raise DOcloudConnectorException("* empty JSON result!")

    def log(self, msg, *args):
        if self.docloud_context.verbose:
            log_msg = "* {0}".format(resolve_pattern(msg, args))
            self.logger.info(log_msg)
            if self.log_output is not None:
                self.log_output.write(log_msg)

    def _as_string(self, content):
        resp_content_as_string = content
        if not isinstance(resp_content_as_string, str):
            resp_content_as_string = content.decode('utf-8')
        return resp_content_as_string

    def _submit_job(self, model_data, job_name=None, gzip=False,
                    prm_data=None,
                    warmstart_data=None,
                    warmstart_name=None,
                    info_callback=None,
                    info_to_monitor=None):
        """Submits a job to the cloud service.

        Args:
            model_data: The model data
            job_name: The name of the job
            gzip: If ``True``, data is gzipped before sent over the network
            prm_data: cplex prm to be used
            info_callback: A call back to be called when some info are available.
                That callback takes one parameter that is a dict containing
                the info as they are available.
        """
        self._hasSolution = False
        self.__vars = None

        if not info_to_monitor:
            info_to_monitor = {}

        client = JobClient(self.docloud_context.url,
                           self.docloud_context.key)

        # prepare client
        if self.docloud_context.log_requests:
            client.rest_callback = \
                lambda m, u, *a, **kw: self._rest_callback(m, u, *a, **kw)
        client.verify = self.docloud_context.verify
        client.timeout = self.docloud_context.timeout if self.docloud_context.timeout is not None else None

        try:
            try:
                # job attachment names
                # job name
                att_names = [{'name': job_name}]
                # prm
                if prm_data:
                    att_names.append({'name': 'file.prm'})
                # warmstart
                if warmstart_data:
                    att_names.append({'name': warmstart_name})
                # create job
                jobid = client.create_job(attachments=att_names)
                self.log("job creation submitted, id is: {0!s}".format(jobid))
                if info_callback and 'jobid' in info_to_monitor:
                    info_callback({'jobid': jobid})
            except ConnectionError as c_e:
                raise DOcloudConnectorException("Cannot connect to {0}, error: {1}".format(self.docloud_context.url, str(c_e)))

            try:
                # upload prm
                if prm_data is not None:
                    encoded_prm_data = prm_data.encode('utf-8')
                    client.upload_job_attachment(jobid, attid="file.prm",
                                                 data=encoded_prm_data, gzip=gzip)
                    self.log("CPLEX parameters file has been uploaded")

                # upload model
                client.upload_job_attachment(jobid, attid=job_name,
                                             data=model_data, gzip=gzip)
                self.log("model data '{attid}' has been uploaded".format(attid=job_name))

                # upload warmstart
                if warmstart_data:
                    client.upload_job_attachment(jobid,
                                                 attid=warmstart_name,
                                                 data=warmstart_data.encode('utf-8'),
                                                 gzip=gzip)
                    self.log("Warmstart data uploaded")

                # execute job
                client.execute_job(jobid)
                self.log("DOcplexcloud execute submitted has been started")
                # get job execution status until it's processed or failed
                timedout = False
                try:
                    self._executionStatus = self.wait_for_completion(client,
                                                                     jobid,
                                                                     info_callback=info_callback,
                                                                     info_to_monitor=info_to_monitor)
                except DOcloudInterruptedException:
                    timedout = True
                self.log("docloud execution has finished")
                # get job status. Do this before any time out handling
                self.jobInfo = client.get_job(jobid)
                if timedout:
                    self._hasSolution = False
                    self.log("Solve timed out after {waittime} sec".format(waittime=self.docloud_context.waittime))
                    return None
                # get solution
                try:
                    solution_as_string = self._as_string(client.download_job_attachment(jobid, attid="solution.json"))
                    myjson = json.loads(solution_as_string, parse_constant='utf-8')['CPLEXSolution']
                    self._hasSolution = bool(myjson)
                except DOcloudNotFoundError:
                    myjson = None
                    self._hasSolution = False
                    self.log("no solution in attachment")
                self.log("docloud results have been received")
                return myjson

            finally:
                deleted = client.delete_job(jobid)
                self.log("delete status for job: {0!s} = {1!s}".format(jobid, deleted))

        finally:
            client.close()

    def submit_model_data(self, mdl_name, mdl_data, extension,
                          prm_data=None,
                          warmstart_data=None,
                          gzip=None,
                          info_callback=None,
                          info_to_monitor=None):
        """Submits a model to the cloud service.
        
        Args:
            mdl_name: The name of the model
            mdl_data: The data for the model
            extension: The extension of the model (".lp", ".sav" etc...)
            gzip: If ``True``, data is gzipped before sent over the network
            prm_data: cplex prm to be used
            info_callback: A call back to be called when some info are available.
                That callback takes one parameter that is a dict containing
                the info as they are available.
            info_to_monitor: A set of information to monitor with info_callback.
                Currently, can be ``jobid`` and ``progress``.
        """
        job_name = normalize_basename(mdl_name) + extension  # use extension or not ?
        warmstart_name = normalize_basename(mdl_name) + ".mst"
        self.json = self._submit_job(model_data=mdl_data,
                                     prm_data=prm_data,
                                     warmstart_name=warmstart_name,
                                     warmstart_data=warmstart_data,
                                     job_name=job_name,
                                     gzip=gzip,
                                     info_callback=info_callback,
                                     info_to_monitor=info_to_monitor)
        return self.json

    def wait_for_completion(self, client, jobid,
                            info_callback=None, info_to_monitor=None):
        def status_poll(loop):
            """Callback to check execution status and stop the loop as soon
            as the job is finished. The loop is stopped as soon as the status
            is for a finished job,
            """
            status = loop.client.get_execution_status(loop.jobid)
            if JobExecutionStatus.isEnded(status):
                loop.status = status
                loop.stop()

        def waittime_timeout(loop):
            """Callback to stop completly the loop once the max waittime
            has been hit.
            """
            loop.stop()
            loop.timed_out = True

        def download_logs(loop, using_threads, log_output):
            """Function/Callback to download logs.

            We will use that function in threads (log_output will be None),
            were we want log events to be put in the loop's event_queue,
            and also after the loop has stopped(), to dump any log items
            remaining on the server (in this case, log_output will
            be the stream to write logs to)

            Args:
                loop: the loop
                using_threads: if true, we are using threads and should queue
                    events so that they are processed in the main thread
                log_output: the log_output
            """
            logs = loop.client.get_log_items(loop.jobid, loop.last_seqid, True)
            for log in logs:
                loop.last_seqid = log['seqid'] + 1
                for r in log['records']:
                    level = r['level'][:4]
                    date = r['date']
                    message = r['message'].rstrip()
                    d = datetime.utcfromtimestamp(int(float(date)*0.001))
                    m = "[{date}Z, {level}] {message}\n".format(date=d.isoformat(),
                                                                level=level,
                                                                message=message
                                                                )
                    if using_threads:
                        loop.event_queue.put(DOcloudEvent("log", m))
                    else:
                        log_output.write(m)

        def progress_poll(loop, using_threads, info_callback, info_to_monitor):
            """Function/Callback to poll and download progress.

            That function polls the service for progress, then queue
            progress events in the main loop.

            Args:
                loop: the loop
                using_threads: if true, we are using threads and should queue
                    events so that they are processed in the main thread
                info_callback: the info callback
                info_to_monitor: what info does the callback want to monitor
            """
            if 'progress' in info_to_monitor and info_callback:
                info = loop.client.get_job(loop.jobid)
                if 'details' in info:  # there are some info available
                    progress_data = self.map_job_info_to_progress_data(info)
                    if using_threads:
                        loop.event_queue.put(DOcloudEvent("progress", progress_data))
                    else:
                        info_callback({'progress': progress_data})

        class JobMonitor(CyclicLoop):
            """A cyclic loop with some encapsuled data"""
            def __init__(self, client, jobid, log_output):
                super(JobMonitor, self).__init__()
                self.client = client
                self.jobid = jobid
                self.log_output = log_output

                self.status = None
                self.timed_out = False
                self.last_seqid = 0

        if not info_to_monitor:
            info_to_monitor = {}

        # interval to check job status
        status_poll_interval = client.nice
        log_poll_interval = self.docloud_context.log_poll_interval
        if not log_poll_interval:
            log_poll_interval = client.nice * 3
        progress_poll_interval = self.docloud_context.progress_poll_interval
        if not progress_poll_interval:
            progress_poll_interval = client.nice * 3

        # The cyclic loop
        loop = JobMonitor(client, jobid, self.log_output)
        using_threads = False

        # configure status log event
        loop.enter(status_poll_interval, 1, status_poll, (loop, ))
        # configure log poll event
        if self.log_output:
            self.log("Polling logs every %s sec" % log_poll_interval)
            loop.enter(log_poll_interval, 1, download_logs,
                       (loop, using_threads, self.log_output))
        # configure progress poll event
        if 'progress' in info_to_monitor:
            self.log("Polling progress every %s sec" % progress_poll_interval)
            loop.enter(progress_poll_interval, 1,
                       progress_poll,
                       (loop, using_threads, info_callback, info_to_monitor))

        # If there's a waittime, configure an event to stop the loop after
        # ``waittime``
        if self.docloud_context.waittime:
            loop.enter(self.docloud_context.waittime, 1, waittime_timeout, (loop, ))
            self.log("waiting for job completion with a wait time of {waittime} sec".format(waittime=self.docloud_context.waittime))
        else:
            self.log("waiting for job completion with no wait time")

        # we want the dumps of the log to happen in the main thread
        # we also want the progress listener events to come in the main thread
        # this function is guaranteed to run in the main thread by the loop
        def main_thread_worker(event):
            if event.type == "log":
                loop.log_output.write(event.data)
            elif event.type == "progress":
                if info_callback:
                    info_callback({'progress': event.data})

        kwargs = {}
        if using_threads:
            kwargs['mt_worker'] = main_thread_worker
        loop.start(**kwargs)

        if self.log_output:
            # this will download the log items that were not downloaded in
            # the loop. Using using_threads=False to force a dump of the log
            # output
            download_logs(loop, False, self.log_output)

        if loop.timed_out:
            self.log("Job Timed out")
            raise DOcloudInterruptedException("Timeout after {0}".format(self.docloud_context.waittime), jobid=jobid)

        return loop.status

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
        """The callback called by the DOcplexcloud client to log REST operations
        """
        self.logger.info("{0} {1}".format(method, url))
        if len(args) > 0:
            self.logger.info("   Additionnal args : {0}".format(','.join(args)))
        for k, v in iteritems(kwargs):
            self.logger.info("   {0}: {1}".format(k, v))

    def map_job_info_to_progress_data(self, info):
        """ Map job info as downloaded from the cplex cloud worker to
        docplex.mp.progress.ProgressData

        Args:
            info: The info as a dict
        Returns:
            A ProgressData
        """

        pg = ProgressData()
        details = info.get('details')
        if details:
            pg.current_objective = float(details.get('PROGRESS_CURRENT_OBJECTIVE',
                                                     pg.current_objective))
            pg.best_bound = float(details.get('PROGRESS_BEST_OBJECTIVE',
                                              pg.best_bound))
            if 'PROGRESS_CURRENT_OBJECTIVE' in details and 'PROGRESS_BEST_OBJECTIVE' in details:
                if pg.current_objective > 0:
                    pg.mip_gap = abs(pg.current_objective - pg.best_bound) / pg.current_objective
            pg.current_nb_nodes = int(details.get('cplex.nodes.processed',
                                                  pg.current_nb_nodes))
            pg.remaining_nb_nodes = int(details.get('cplex.nodes.left',
                                                    pg.remaining_nb_nodes))
            # assume that there's an incubent if ther's a gap
            pg.has_incumbent = 'PROGRESS_GAP' in details
        pg.time = ((info.get('updatedAt')) - int(info.get('startedAt'))) / 1000
        return pg
