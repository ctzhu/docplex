# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
from docplex.mp.constants import ComparisonType
from docplex.mp.linear import LinearExpr, Var
from docplex.mp.mprinter import TextModelPrinter, _ExportWrapper
from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType


class ModelPrettyPrinter(TextModelPrinter):
    vartype_map = {ContinuousVarType: "float", IntegerVarType: "int", BinaryVarType: "bool"}

    ct_symbol_map = {ComparisonType.EQ: "==",
                     ComparisonType.LE: "<=",
                     ComparisonType.GE: ">="}

    indented = ' ' * 2

    def __init__(self, nb_digits=6):
        # comment line is // as in OPL
        # do NOT forget about user names
        # no encoding is printed
        TextModelPrinter.__init__(self, indent=2, comment_start='//',
                                  nb_digits_for_floats=nb_digits,
                                  hide_user_names=False,
                                  encoding=None)

    def get_format(self):
        from docplex.mp.format import OPL_format

        return OPL_format

    def fix_name(self, mobj, prefix, local_index_map, hide_names):
        raw_name = mobj.name
        # ignore hide_names here
        if not raw_name or not mobj.has_user_name():
            return None # ignore automatic & generated objects
        else:
            if mobj.is_generated():
                mobj_origin = mobj.origin()
                if mobj is mobj_origin.functional_var:
                    # only the functional var is named after the functional expr
                    return str(mobj_origin)

            return self._translate_chars(raw_name)

    def _print_model_name(self, out, mdl):
        printed_name = mdl.name or "AnonymousModel"
        out.write("// model name is: {0:s}\n".format(printed_name))

    def _vartype_name(self, vartype):
        # INTERNA: returns a printable string for a vartype
        return self.vartype_map.get(type(vartype), "unknown")

    def _print_var_containers(self, out, mdl):
        gensym_count = 1
        printed_header = False
        for ctn in mdl.iter_var_containers():
            if not printed_header:
                self._print_line_comment(out, "var contrainer section")
                printed_header = True
            vartype_name = self._vartype_name(ctn.vartype)
            varctn_name = ctn.name
            if not varctn_name:
                varctn_name = 'x%d' % gensym_count
                gensym_count += 1
            out.write("dvar {0} {1}{2};\n".format(vartype_name, varctn_name, ctn.dimension_string))

        if printed_header:
            self._newline(out)

    def _print_single_vars(self, out, mdl):
        printed_header = False
        for v in mdl.iter_variables():
            if v.is_generated():
                continue
            var_ctn = v.get_container()
            if var_ctn is not None:
                continue

            if not printed_header:
                self._print_line_comment(out, "single vars section")
                printed_header = True
            vartype_name = self._vartype_name(v.vartype)
            var_printname = self._var_name_map.get(v._index, "???")
            out.write("dvar {0} {1};\n".format(vartype_name, var_printname))

        if printed_header:
            self._newline(out)

    def _print_objective(self, wrapper, model):
        wrapper.write(model.objective_sense.verb())
        wrapper.flush(print_newline=True)
        self._print_lexpr(wrapper, self._num_printer, self._var_name_map, model.objective_expr)
        wrapper.write(';', separator=False)
        wrapper.flush()

    def _pprint_expr(self, wrapper, expr):
        q = 0
        if expr.is_quad_expr() and expr.has_quadratic_term():
            q = self._print_qexpr_iter(wrapper, self._num_printer, self._var_name_map, expr.iter_sorted_quads(), use_double=False)
        self._print_expr_iter(wrapper, self._num_printer, self._var_name_map, expr.iter_terms(),
                              constant=expr.get_constant(),  # yes, print the constant
                              allow_empty=q > 0,
                              force_first_plus=q > 0  # force  a '+' if quadratic section is non-empty
                              )

    def _print_binary_constraint(self, wrapper, ct):
        left_expr = ct.left_expr
        right_expr = ct.right_expr
        self._pprint_expr(wrapper, left_expr)

        wrapper.write(self.ct_symbol_map[ct.type])

        self._pprint_expr(wrapper, right_expr)

    def _print_range_constraint(self, wrapper, rng):
        expr = rng.expr
        lb = rng.lb
        ub = rng.ub
        wrapper.write(self._num_to_string(lb))
        wrapper.write("<=")
        self._print_lexpr(wrapper, self._num_printer, self._var_name_map, expr, print_constant=True)
        wrapper.write("<=")
        wrapper.write(self._num_to_string(ub))


    def _print_indicator_constraint(self, wrapper, ind_ct):
        active = ind_ct.active_value
        linear_ct = ind_ct.linear_constraint
        indicator_varname = self._var_print_name(ind_ct.indicator_var)
        wrapper.write(indicator_varname)
        if 0 == active:
            wrapper.write("== 0")
        wrapper.write('<=')
        wrapper.write('(')
        self._print_binary_constraint(wrapper, linear_ct)
        wrapper.write(');', separator=False)

    def _print_linear_constraints(self, wrapper, model):
        wrapper.begin_line()
        for ct in model.iter_binary_constraints():
            if ct.is_generated():
                continue  # pragma: no cover

            wrapper.begin_line()
            ctname = self.ct_print_name(ct)
            if ctname:
                wrapper.set_indent('  ')  # two spaces
                wrapper.write(" %s:" % ctname)
                wrapper.flush()
            else:
                wrapper.begin_line(indented=True)

            self._print_binary_constraint(wrapper, ct)
            wrapper.write(';', separator=False)
            wrapper.set_indent(' ')
            wrapper.flush(print_newline=True, reset=False)

    def _print_ranges(self, wrapper, model):
        wrapper.begin_line()
        for ct in model.iter_range_constraints():

            wrapper.begin_line()
            ctname = self.ct_print_name(ct)
            if ctname:
                wrapper.set_indent(2 * ' ')
                wrapper.write(' %s:' % ctname)
                wrapper.flush()

            else:
                wrapper.begin_line(indented=True)

            self._print_range_constraint(wrapper, ct)
            wrapper.write(';', separator=False)
            wrapper.flush()
        wrapper.set_indent(' ')

    def _print_indicators(self, wrapper, model):
        for ct in model.iter_indicator_constraints():
            if not ct.is_generated():
                ctname = self.ic_print_name(ct)
                if ctname:
                    wrapper.set_indent('  ')
                    wrapper.write(" %s:" % ctname)
                    wrapper.flush()
                else:
                    wrapper.begin_line(indented=True)
                self._print_indicator_constraint(wrapper, ct)
                wrapper.set_indent(' ')
                wrapper.flush(reset=True)


    def _print_quadratic_cts(self, wrapper, model):
        for qct in model.iter_quadratic_constraints():

            ctname = self.qc_print_name(qct)
            if ctname:
                wrapper.set_indent('  ')
                wrapper.write(" %s:" % ctname)
                wrapper.flush()
            else:
                wrapper.begin_line(indented=True)

            self._print_binary_constraint(wrapper, qct)
            wrapper.write(';', separator=False)
            wrapper.set_indent(' ')
            wrapper.flush(reset=True)

    def _print_kpis(self, out, wrapper, model):
        printed_section_header = False
        for kpi in model.iter_kpis():
            if not kpi.requires_solution():
                continue

            if not printed_section_header:
                self._newline(out)
                self._print_line_comment(out, " KPI section")
                printed_section_header = True

            kpi_expr = kpi.as_expression()
            kpi_typename = 'int' if kpi_expr.is_discrete() else 'float'
            wrapper.write('dexpr {0} {1}'.format(kpi_typename, self._translate_chars(kpi.name)))
            wrapper.write('=')
            if isinstance(kpi_expr, LinearExpr):
                self._print_lexpr(wrapper, self._num_printer, self._var_name_map, kpi_expr, print_constant=True)
            elif isinstance(kpi_expr, Var):
                wrapper.write(kpi_expr.name)
            wrapper.write(';', separator=False)
            wrapper.flush(reset=True)
        if printed_section_header:
            wrapper.newline()

    def print_model_to_stream(self, out, model):
        wrapper = _ExportWrapper(oss=out, indent_str=' ', line_width=78)
        self.prepare(model)
        # header
        self._print_signature(out)
        self._print_encoding(out)
        self._print_model_name(out, model)

        # var containers
        self._print_var_containers(out, model)
        self._print_single_vars(out, model)
        # KPI section
        self._print_kpis(out, wrapper, model)

        self._print_objective(wrapper, model)
        wrapper.write("\nsubject to {")
        wrapper.flush()
        self._print_linear_constraints(wrapper, model)
        self._print_ranges(wrapper, model)
        self._print_indicators(wrapper, model)
        self._print_quadratic_cts(wrapper, model)
        out.write("}\n")
