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
from docplex.mp.utils import normalize_basename

from docplex.mp.format import LP_format
from docplex.mp.compat23 import StringIO

import sys

# gendoc: ignore

# this is the default exchange format for docloud
_DEFAULT_EXCHANGE_FORMAT = LP_format

# default attachment name for PRMs
prm_name = "file.prm"


class FeasibilityPrinter(object):

    extension = ".feasibility"

    @classmethod
    def print_to_stream(cls, relaxables, out, extension=extension):
        if out is None:
            # prints on standard output
            cls.print_internal(sys.stdout, relaxables)

        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                cls.print_to_stream(relaxables, of)
        else:
            try:
                cls.print_internal(out, relaxables)

            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    @classmethod
    def print_internal(cls, out, relaxables):
        # INTERNAL: out is resolved here to a writeable stream...
        out.write("<CPLEXFeasopt>\n")
        out.write("  <rhs>\n")
        for relaxable_group in relaxables:
            pref, cts = relaxable_group
            for ct in cts:
                out.write("    <relax index=\"{0}\" preference=\"{1}\"/>".format(ct.index, pref))
                ctname = ct.name
                if ctname:
                    out.write("  <!-- {0} -->".format(ctname))
                out.write("\n")

        out.write("  </rhs>\n")
        out.write("</CPLEXFeasopt>\n")


# noinspection PyProtectedMember
class DOcloudEngine(IndexerEngine):
    """ Engine facade stub to defer solve to drop-solve URL
    """

    def _print_feasibility(self, out, relaxables):
        pass

    def get_cplex(self):
        raise DOcplexException("{0} engine contains no instance of CPLEX".format(self.name()))


    def __init__(self, mdl, exchange_format=None, hide_user_names=False, **kwargs):
        IndexerEngine.__init__(self)

        docloud_context = kwargs.get('docloud_context')
        # --- log output can be overridden at solve time, so use te one from the context, not the model's
        actual_log_output = kwargs.get('log_output') or mdl.log_output

        self._model = mdl
        self._connector = DOcloudConnector(docloud_context, log_output=actual_log_output)
        self._exchange_format = exchange_format or docloud_context.exchange_format or _DEFAULT_EXCHANGE_FORMAT

        self._printer = ModelPrinterFactory.new_printer(self._exchange_format, hide_user_names)

        # -- results.
        self._var_name_encoding = None
        self._solve_details = SolveDetails.make_dummy()

        # noinspection PyPep8
        self.debug_dump     = docloud_context.debug_dump
        self.debug_dump_dir = docloud_context.debug_dump_dir

    def name(self):
        return "docloud"

    def can_solve(self):
        """
        :return: true, as this solver can solve!
        """
        return True

    def connect_progress_listeners(self, progress_listener_list):
        if progress_listener_list:
            self._model.warning("Progress listeners are not supported on DOcplexcloud.")


    def _docloud_cplex_version(self):
        # INTERNAL: returns the version of CPLEX used in DOcplexcloud
        # for now returns a string. maybe we could ping Docloud and get a dynamic answer.
        return "12.6.3.0"

    def _serialize_parameters(self, parameters, user_overload_params_dict=None):
        # return a string in PRM format
        # overloaded params are:
        # - WRITE_LEVEL = 3, avoid zero values
        # - THREADS = 1, if deterministic, else not mentioned.
        # the resulting string will contain all non-default parameters,
        # AND those overloaded.
        # No side effect on actual model parameters
        if user_overload_params_dict is None:
            overloaded_params = dict()
        else:
            overloaded_params = user_overload_params_dict.copy()

        # WRITE_LEVEL = 3
        overloaded_params[parameters.output.writelevel] = 3

        if self._connector.run_deterministic:
            # overloaded_params[mdl_parameters.threads] = 1 cf RTC28458
            overloaded_params[parameters.parallel] = 1  # 1 is deterministic

        # do we need to limit the version to the one use din docloud
        # i.e. if someone has a *newer* version than docloud??
        prm_data = parameters.export_prm_to_string(overloaded_params)
        return prm_data

    def serialize_model(self, mdl):
        # step 1 : prints the model in whatever exchange format
        printer = self._printer

        if self._exchange_format.is_binary:
            filemode = "wb"
            oss = BytesIO()
        else:
            filemode = "w"
            oss = StringIO()

        printer.printModel(mdl, oss)

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

        if self._exchange_format.is_binary:
            model_data = oss.getvalue()
        else:
            model_data = oss.getvalue().encode('utf-8')
        return model_data

    def _serialize_relaxables(self, relaxables):
        oss = StringIO()
        FeasibilityPrinter.print_to_stream(out=oss, relaxables=relaxables)
        serialized_relaxables = oss.getvalue()
        return serialized_relaxables

    def _dump_if_required(self, data, mdl, basename, extension, is_binary=False, forced=False):
        # INTERNAL
        if self.debug_dump or forced:
            relax_path = make_path(error_handler=mdl.error_handler,
                                  basename=basename,
                                  output_dir=self.debug_dump_dir,
                                  extension=extension,
                                  name_transformer=None)
            fmode = "wb" if is_binary else "w"
            with open(relax_path, fmode) as out_file:
                out_file.write(data)

    def _make_attachment(self, attachment_name, attachment_data):
        # INTERNAL
        return {'name': attachment_name, 'data': attachment_data}

    def solve_relaxed(self, mdl, prio_name, relaxables, optimize, overwrite_params, parameters=None):
        # --- 1 serialize
        job_name = "docloud_%s" % mdl.name
        model_data = self.serialize_model(mdl)
        docloud_parameters = parameters if parameters is not None else mdl.parameters
        prm_data = self._serialize_parameters(docloud_parameters, user_overload_params_dict=overwrite_params)
        feasopt_data = self._serialize_relaxables(relaxables)

        # --- dump if need be
        relax_basename = normalize_basename("%s_%s_infeasible_feasopt" % (mdl.name, prio_name))
        prm_basename = normalize_basename("%s_infeasible_feasopt" % mdl.name)
        self._dump_if_required(model_data, mdl, basename=job_name, extension=".lp", forced=True)
        self._dump_if_required(feasopt_data, mdl, basename=relax_basename, extension=FeasibilityPrinter.extension, forced=True)
        self._dump_if_required(prm_data, mdl, basename=prm_basename, extension=".prm", forced=True)

        # --- submit job somehow...
        attachments = []
        model_name = normalize_basename(job_name) + self._exchange_format.extension
        attachments.append(self._make_attachment(model_name, model_data))

        attachments.append(self._make_attachment(prm_name, prm_data))
        attachments.append(self._make_attachment(normalize_basename(job_name) + FeasibilityPrinter.extension,
                                                 feasopt_data))



        # here we go...
        def notify_info(info):
            if "jobid" in info:
                mdl.fire_jobid(jobid=info["jobid"])
            if "progress" in info:
                mdl.fire_progress(progress_data=info["progress"])

        connector = self._connector
        mdl.notify_start_solve()
        connector.submit_model_data(attachments,
                                    gzip=not self._exchange_format.is_binary,
                                    info_callback=notify_info,
                                    info_to_monitor={'jobid', 'progress'})

        # --- cplex solve details
        json_details = connector.get_cplex_details()
        self._solve_details = SolveDetails.from_json(json_details)
        # ---

        # --- build a solution object, or None
        if not self._connector.has_solution():
            mdl.notify_solve_failed()
            return False, 0
        else:
            solution = self._make_solution(mdl)
            return True, solution.objective_value,


    # noinspection PyProtectedMember
    def solve(self, mdl, parameters=None):
        # Before submitting the job, we will build the list of attachments
        attachments = []

        # make sure model is the first attachment: that will be the name of the job on the console
        job_name = "python_%s" % mdl.name
        model_data = self.serialize_model(mdl)
        model_name = normalize_basename(job_name) + self._exchange_format.extension
        attachments.append({'name': model_name, 'data': model_data})

        # prm
        docloud_parameters = parameters if parameters is not None else mdl.parameters
        prm_data = self._serialize_parameters(docloud_parameters)
        attachments.append({'name': prm_name, 'data': prm_data})

        # warmstart_data
        # export mipstart solution in CPLEX mst format, if any, else None
        mdl_mipstarts = mdl.mip_starts
        if mdl_mipstarts:
            warmstart_data = SolutionMSTPrinter.print_to_string(mdl_mipstarts).encode('utf-8')
            warmstart_name = normalize_basename(job_name) + ".mst"
            attachments.append({'name': warmstart_name, 'data': warmstart_data})

        # info_to_monitor = {'jobid'}
        # if mdl.progress_listeners:
        #     info_to_monitor.add('progress')

        def notify_info(info):
            if "jobid" in info:
                mdl.fire_jobid(jobid=info["jobid"])
            if "progress" in info:
                mdl.fire_progress(progress_data=info["progress"])

        # This block used to be try/catched for DOcloudConnector exceptions
        # and DOcloudException, but then infrastructure error were not
        # handled properly. Now we let the exception raise.
        connector = self._connector
        mdl.notify_start_solve()
        connector.submit_model_data(attachments,
                                    gzip=not self._exchange_format.is_binary,
                                    info_callback=notify_info,
                                    info_to_monitor={'jobid', 'progress'})

        # --- cplex solve details
        json_details = connector.get_cplex_details()
        self._solve_details = SolveDetails.from_json(json_details)
        # ---

        # --- build a solution object, or None
        if not self._connector.has_solution():
            mdl.notify_solve_failed()
            solution = None
        else:
            solution = self._make_solution(mdl)
        # ---

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
        raw_docloud_obj = self._connector.get_objective()
        docloud_obj = mdl.round_objective_if_discrete(raw_docloud_obj)
        docloud_values_by_idx, docloud_var_rcs = self._connector.variable_results()
        # CPLEX index to name map
        # for those variables returned by CPLEX.
        # all other are assumed to be zero
        index_name_map = self._connector.cplex_index_name_map()
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
        docloud_ct_duals, docloud_ct_slacks = self._connector.constraint_results()
        sol._store_attribute_results(docloud_var_rcs, docloud_ct_duals, docloud_ct_slacks)
        return sol

    def get_solve_attribute(self, attr, index_seq):
        return {}

    def get_solve_status(self):
        return self._connector.get_solve_status()

    def get_solve_details(self):
        return self._solve_details

