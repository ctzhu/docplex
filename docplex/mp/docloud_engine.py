# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from io import BytesIO

from six import iteritems

from docplex.mp.engine import IndexerEngine
from docplex.mp.docloud_connector import DOcloudConnector
from docplex.mp.printer_factory import ModelPrinterFactory
from docplex.mp.solution import SolveSolution, SolutionMSTPrinter
from docplex.mp.sdetails import SolveDetails
from docplex.mp.utils import DOcplexException, make_path
from docplex.mp.format import LP_format
from docplex.mp.compat23 import StringIO


# gendoc: ignore

# this is the default exchange format for docloud
_DEFAULT_EXCHANGE_FORMAT = LP_format


# noinspection PyProtectedMember
class DOcloudEngine(IndexerEngine):
    """ Engine facade stub to defer solve to drop-solve URL
    """

    def get_cplex(self):
        raise DOcplexException("{0} engine contains no instance of CPLEX".format(self.name()))

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        """
        No relaxation is available on the cloud (yet)
        :param relaxable_groups:
        :param optimize:
        :return:
        """
        raise DOcplexException("Engine {0} does not support relaxation (yet)".format(self.name()))

    def __init__(self, mdl, exchange_format=None, hide_user_names=False, **kwargs):
        IndexerEngine.__init__(self)

        if 'docloud_context' in kwargs:
            docloud_context = kwargs['docloud_context']
        else:
            docloud_context = None

        # --- log output can be overridden at solve time, so use te one from the context, not the model's
        actual_log_output = kwargs.get('log_output') or mdl.log_output

        self.__connector = DOcloudConnector(docloud_context, log_output=actual_log_output)
        self.__error_handler = mdl.error_handler
        self.__exchange_format = exchange_format

        # if no format was provided in constructor, look into the context
        if docloud_context.exchange_format and exchange_format is None:
            self.__exchange_format = docloud_context.exchange_format

        # fallback to default if no format was specified in constructor param
        # or in context
        if self.__exchange_format is None:
            self.__exchange_format = _DEFAULT_EXCHANGE_FORMAT

        self.__printer = None
        self.__hide_user_names = hide_user_names
        self._var_name_encoding = None
        self._solve_details = SolveDetails.make_dummy()

        self.debug_dump = docloud_context.debug_dump
        self.debug_dump_dir = docloud_context.debug_dump_dir

    def _lazy_get_printer(self, errh):
        if not self.__printer:
            self.__printer = ModelPrinterFactory.new_printer(self.__exchange_format, self.__hide_user_names)
        return self.__printer

    def name(self):
        return "docloud"

    def clear_objective(self, expr):
        # compatibility: nothing to do
        pass

    def can_solve(self):
        """
        :return: true, as this solver can solve!
        """
        return True

    def connect_progress_listeners(self, progress_listener_list):
        if progress_listener_list:
            self.__error_handler.warning("Progress listeners are not supported on DOcplexcloud.")


    def _docloud_cplex_version(self):
        # INTERNAL: returns the version of CPLEX used in DOcplexcloud
        # for now returns a string. maybe we could ping Docloud and get a dynamic answer.
        return "12.6.3.0"

    def _compute_prm_data_from_parameters(self, mdl_parameters, run_deterministic):
        # return a string in PRM format
        # overloaded params are:
        # - WRITE_LEVEL = 3, avoid zero values
        # - THREADS = 1, if deterministic, else not mentioned.
        # the resulting string will contain all non-default parameters,
        # AND those overloaded.
        # No side effect on actual model parameters
        overloaded_params = {mdl_parameters.output.writelevel: 3}
        if run_deterministic:
            # overloaded_params[mdl_parameters.threads] = 1 cf RTC28458
            overloaded_params[mdl_parameters.parallel] = 1  # 1 is deterministic

        # do we need to limit the version to the one use din docloud
        # i.e. if someone has a *newer* version than docloud??
        prm_data = mdl_parameters.export_prm_to_string(overloaded_params)
        return prm_data

    def _compute_prm_data(self, mdl, run_deterministic):
        # return a string in PRM format
        # overloaded params are:
        # - WRITE_LEVEL = 3, avoid zero values
        # - THREADS = 1, if deterministic, else not mentioned.
        # the resulting string will contain all non-default parameters,
        # AND those overloaded.
        # No side effect on actual model parameters
        mdl_parameters = mdl.parameters
        return self._compute_prm_data_from_parameters(mdl_parameters, run_deterministic)

    # noinspection PyProtectedMember
    def solve(self, mdl, parameters=None):
        self_connector = self.__connector
        mdl.notify_start_solve()

        # step 1 : prints the model in whatever exchange format
        printer = self._lazy_get_printer(mdl.error_handler)

        if self.__exchange_format.is_binary:
            filemode = "wb"
            oss = BytesIO()
        else:
            filemode = "w"
            oss = StringIO()

        #t = time.time()
        printer.printModel(mdl, oss)
        #elapsed_t = time.time() - t
        #self_connector.log("elapsed time in model printing = {0}".format(elapsed_t))

        self._var_name_encoding = printer.get_name_to_var_map(mdl)

        # DEBUG: dump request file
        if self.debug_dump:
            dump_path = make_path(error_handler=mdl.error_handler,
                                  basename=mdl.name,
                                  output_dir=self.debug_dump_dir,
                                  extension=printer.extension(),
                                  name_transformer="docloud_%s")
            print("DEBUG DUMP in " + dump_path)
            with open(dump_path, filemode) as out_file:
                out_file.write(oss.getvalue())

        # This block used to be try/catched for DOcloudConnector exceptions
        # and DOcloudException, but then infrastructure error were not
        # handled properly. Now we let the exception raise.
        job_name = "python_%s" % mdl.name
        if self.__exchange_format.is_binary:
            model_data = oss.getvalue()
        else:
            model_data = oss.getvalue().encode('utf-8')

        docloud_parameters = parameters if parameters is not None else mdl.parameters
        prm_data = self._compute_prm_data_from_parameters(docloud_parameters,
                                                          self_connector.run_deterministic)
        # export mipstart solution in CPLEX mst format, if any, else None
        mdl_mipstarts = mdl.mip_starts
        if mdl_mipstarts:
            warmstart_data = SolutionMSTPrinter.print_to_string(mdl_mipstarts)
        else:
            warmstart_data = None

        info_to_monitor = {'jobid'}
        if mdl.progress_listeners:
            info_to_monitor.add('progress')

        def notify_info(info):
            if "jobid" in info:
                mdl.fire_jobid(jobid=info["jobid"])
            if "progress" in info:
                mdl.fire_progress(progress_data=info["progress"])

        self_connector.submit_model_data(job_name, model_data,
                                         self.__exchange_format.extension,
                                         prm_data=prm_data,
                                         warmstart_data=warmstart_data,
                                         gzip=not self.__exchange_format.is_binary,
                                         info_callback=notify_info,
                                         info_to_monitor={'jobid', 'progress'})
        ok = self.__connector.has_solution()
        # cplex solve details
        json_details = self_connector.get_cplex_details()
        self._solve_details = SolveDetails.from_json(json_details)
        # --- end of block ---

        if not ok:
            mdl.notify_solve_failed()
            return None
        else:
            solution = self._make_solution(mdl)
            return solution

    def _get_var_by_cloud_name(self, mdl, cloud_name, local_encoding):
        if local_encoding:
            # LP format induces name changes
            return local_encoding.get(cloud_name)
        else:
            return mdl.get_var_by_name(cloud_name)

    def _make_solution(self, mdl):
        # Store the results of solve ina solution object.
        local_var_encoding = self._var_name_encoding
        raw_docloud_obj = self.__connector.get_objective()
        docloud_obj = mdl.round_objective_if_discrete(raw_docloud_obj)
        docloud_values_by_idx, docloud_var_rcs = self.__connector.variable_results()
        # CPLEX index to name map
        # for those variables returned by CPLEX.
        # all other are assumed to be zero
        index_name_map = self.__connector.cplex_index_name_map()
        # send an objective, a var-value dict and a string identifying the engine which solved.
        docloud_values_by_vars = {}
        keep_zeros = False
        count_nonmatching_cloud_vars = 0
        for cpx_idx, val in iteritems(docloud_values_by_idx):
            if val != 0 or keep_zeros:
                # first get the name from the cloud idx
                cloud_name = index_name_map[cpx_idx]
                if cloud_name:
                    dvar = self._get_var_by_cloud_name(mdl, cloud_name, local_var_encoding)
                    if dvar:
                        docloud_values_by_vars[dvar] = val
                    elif cloud_name.startswith("Rgc"):
                        # range variables
                        pass
                    else:
                        # one extra variable from docloud is OK
                        # it represents the constant term in objective
                        # more than one is an issue.
                        if count_nonmatching_cloud_vars:
                            mdl.info("Cannot find matching variable, cloud name is {0!s}", cloud_name)
                        count_nonmatching_cloud_vars += 1
                else:
                    mdl.warning("cannot find variable name from index: {0} - skipped".format(cpx_idx))

        sol = SolveSolution(mdl, obj=docloud_obj,
                            var_value_map=docloud_values_by_vars,
                            engine_name=self.name(),
                            keep_zeros=False,
                            rounding=True)
        sol._set_solve_status(self.get_solve_status())

        # attributes
        docloud_ct_duals, docloud_ct_slacks = self.__connector.constraint_results()
        sol._store_attribute_results(docloud_var_rcs, docloud_ct_duals, docloud_ct_slacks)
        return sol

    def get_solve_attribute(self, attr, index_seq):
        return {}

    def get_solve_status(self):
        return self.__connector.get_solve_status()

    def get_solve_details(self):
        return self._solve_details

