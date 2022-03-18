# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2021
# ----------------------------------

# gendoc: ignore

from docplex.mp.utils import OutputStreamAdapter
from docplex.mp.format import LPFormat
from docplex.mp.solprinter import SolutionPrinter


class SolutionSolPrinter(SolutionPrinter):

    # header contains the final newline
    mst_header = """<?xml version = "1.0" standalone="yes"?>
"""
    sol_extension = ".sol"

    one_solution_start_tag = "<CPLEXSolution version=\"1.0\">"
    one_solution_end_tag = "</CPLEXSolution>"

    # used when several solutions are present
    many_solution_start_tag = "<CPLEXSolutions version=\"1.0\">"
    many_solution_end_tag = "</CPLEXSolutions>"


    def extension(self):
        return self.sol_extension

    @staticmethod
    def _print_signature(out):
        from docplex.version import docplex_version_string
        osa = OutputStreamAdapter(out)
        osa.write("<!-- This file has been generated by DOcplex version {}  -->\n".format(docplex_version_string))

    def print_one_solution(self, sol, out, **kwargs):
        osa = OutputStreamAdapter(out)
        print_header = kwargs.get('print_header', True)
        use_lp_names = kwargs.get('use_lp_names', True)
        filter_zeros = kwargs.get('filter_zeros', True)
        print_generated = kwargs.get('print_generated', False)
        precision = kwargs.get('precision', 1e-6)

        if print_header:
            osa.write(self.mst_header)
            self._print_signature(out)
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
        osa.write("  />\n")

        #  <variables>
        #    <variable name="x1" index ="1" value="3.14"/>
        #  </variables>
        osa.write(" <variables>\n")
        """
            osa.write("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:g}\" {3}/>\n"
              .format(var_name, var_index, var_value, rc_string))
        """
        if filter_zeros:
            for dvar, _ in sol.iter_var_values():
                if print_generated or not dvar.is_generated():
                    var_name = LPFormat.lp_var_name(dvar) if use_lp_names else dvar.name
                    var_value = sol[dvar]
                    if abs(var_value) >= precision:
                        var_index = dvar.index
                        if var_name:
                            osa.write("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:.5f}\"/>\n"
                                      .format(var_name, var_index, var_value))
                        else:
                            osa.write("  <variable index=\"{1}\" value=\"{2:g}\"/>\n"
                                      .format(var_name, var_index, var_value))
        else:
            # iterate on all variebls
            for dvar in sol.model.generate_user_variables():
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
        osa.write(self.mst_header)
        self._print_signature(out)
        # <CPLEXSolutions version="1.0">
        osa.write(self.many_solution_start_tag)
        osa.write("\n")
        one_solution_kwargs = kwargs.copy()
        one_solution_kwargs['print_header'] = False
        for sol in solutions:
            self.print_one_solution(sol, out, **one_solution_kwargs)

        # <CPLEXSolutions version="1.0">
        osa.write(self.many_solution_end_tag)
        osa.write("\n")
