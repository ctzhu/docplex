# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2021
# ----------------------------------

# gendoc: ignore

from itertools import zip_longest as  izip_longest

from docplex.mp.utils import is_iterable, OutputStreamAdapter
from docplex.mp.format import LPFormat
from docplex.mp.constants import EffortLevel, WriteLevel
from docplex.mp.solprinter import SolutionPrinter


class SolutionMSTPrinter(SolutionPrinter):

    # header contains the final newline
    mst_header = """<?xml version = "1.0" standalone="yes"?>
"""
    mst_extension = ".mst"

    one_solution_start_tag = "<CPLEXSolution version=\"1.0\">"
    one_solution_end_tag = "</CPLEXSolution>"

    # used when several solutions are present
    many_solution_start_tag = "<CPLEXSolutions version=\"1.0\">"
    many_solution_end_tag = "</CPLEXSolutions>"

    print_generated_vars = False

    def extension(self):
        return self.mst_extension

    @staticmethod
    def _print_signature(out, write_level):
        from docplex.version import docplex_version_string
        osa = OutputStreamAdapter(out)
        osa.write("<!-- This file has been generated by DOcplex version {}  -->\n".format(docplex_version_string))
        osa.write("<!-- Write level is WriteLevel.{0} -->\n".format(write_level.name))

    # @classmethod
    # def _print_to_stream2(cls, out, solutions, write_level, use_lp_names, effort_level=None):
    #     # no kwargs at this stage.
    #     # solutions can be either a plain solution or a sequence or an iterator
    #     if not is_iterable(solutions):
    #         cls.print_one_solution(solutions, out, use_lp_names=use_lp_names, write_level=write_level, effort_level=effort_level)
    #     else:
    #         sol_seq = list(solutions)
    #         nb_solutions = len(sol_seq)
    #         assert nb_solutions > 0
    #         if 1 == nb_solutions:
    #             cls.print_one_solution(sol_seq[0], out, use_lp_names, write_level=write_level, effort_level=effort_level[0])
    #         else:
    #             cls.print_many_solutions(sol_seq, out, use_lp_names, write_level, effort_level)

    def print_one_solution(self, sol, out, **kwargs):
        osa = OutputStreamAdapter(out)
        print_header = kwargs.get('print_header', True)
        use_lp_names = kwargs.get('use_lp_names', True)
        write_level  = kwargs.get('write_level', WriteLevel.Auto)
        effort_level = kwargs.get('effort_level', None)

        if print_header:
            osa.write(self.mst_header)
            self._print_signature(out, write_level)
        # <CPLEXSolution version="1.0">
        osa.write(self.one_solution_start_tag)
        osa.write("\n")

        # <header
        # problemName="foo"
        # objectiveValue="42"
        # />
        osa.write(" <header\n   problemName=\"{0}\"\n".format(sol.problem_name))
        if sol.has_objective():
            osa.write("   objectiveValue=\"{0:g}\"\n".format(sol.objective_value))
        if effort_level is not None:
            # assert this is an integer
            eff = EffortLevel.parse(effort_level)
            if eff != EffortLevel.Auto:
                osa.write("   MIPStartEffortLevel=\"{0:d}\"\n".format(eff.value))
        osa.write("  />\n")

        #  <variables>
        #    <variable name="x1" index ="1" value="3.14"/>
        #  </variables>
        osa.write(" <variables>\n")
        """
            osa.write("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:g}\" {3}/>\n"
              .format(var_name, var_index, var_value, rc_string))
        """
        print_generated = self.print_generated_vars
        filter_discrete = write_level.filter_nondiscrete()
        eps = 1e-16
        if write_level.filter_zeros():
            for dvar, _ in sol.iter_var_values():
                if print_generated or not dvar.is_generated():
                    if not filter_discrete or dvar.is_discrete():
                        var_name = LPFormat.lp_var_name(dvar) if use_lp_names else dvar.name
                        var_value = sol[dvar]
                        if abs(var_value) >= eps:
                            var_index = dvar.index
                            if var_name:
                                osa.write("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:g}\"/>\n"
                                          .format(var_name, var_index, var_value))
                            else:
                                osa.write("  <variable index=\"{1}\" value=\"{2:g}\"/>\n"
                                          .format(var_name, var_index, var_value))
        else:
            # iterate on all variebls
            for dvar in sol.model.generate_user_variables():
                if (not filter_discrete) or dvar.is_discrete():
                    var_name = LPFormat.lp_var_name(dvar) if use_lp_names else dvar.name
                    var_value = sol[dvar]
                    var_index = dvar.index
                    if var_name:
                        osa.write("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:g}\"/>\n"
                                  .format(var_name, var_index, var_value))
                    else:
                        osa.write("  <variable index=\"{1}\" value=\"{2:g}\"/>\n"
                                  .format(var_name, var_index, var_value))

        osa.write(" </variables>\n")

        #  </CPLEXSolution version="1.0">
        osa.write(self.one_solution_end_tag)
        osa.write("\n")

    def print_many_solutions(self, solutions, out, **kwargs):
        osa = OutputStreamAdapter(out)

        write_level  = kwargs.get('write_level', WriteLevel.Auto)
        effort_level = kwargs.get('effort_level', None)
        osa.write(self.mst_header)
        self._print_signature(out, write_level)
        # <CPLEXSolutions version="1.0">
        osa.write(self.many_solution_start_tag)
        osa.write("\n")

        efforts = [EffortLevel.Auto]
        if effort_level is None:
            pass
        elif is_iterable(effort_level):
            efforts = effort_level
        else:
            efforts = [effort_level]

        one_solution_kwargs = kwargs.copy()
        one_solution_kwargs['print_header'] = False
        for sol, effort in izip_longest(solutions, efforts, fillvalue=EffortLevel.Auto):
            one_solution_kwargs['effort_level'] = effort
            self.print_one_solution(sol, out, **one_solution_kwargs)

        # <CPLEXSolutions version="1.0">
        osa.write(self.many_solution_end_tag)
        osa.write("\n")
