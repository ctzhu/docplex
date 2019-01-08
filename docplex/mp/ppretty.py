# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
from docplex.mp.compat23 import StringIO
from docplex.mp.linear import LinearConstraintType, LinearExpr, Var
from docplex.mp.mprinter import TextModelPrinter
from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType


class ModelPrettyPrinter(TextModelPrinter):

    _vartype_map = {ContinuousVarType: "float", IntegerVarType: "int", BinaryVarType: "bool"}

    def __init__(self, nb_digits=6):
        # comment line is // as in OPL
        # do NOT forget about user names
        # no encoding is printed
        TextModelPrinter.__init__(self, indent=2, comment_start='//',
                                  nb_digits_for_floats=nb_digits,
                                  hide_user_names=False,
                                  encoding=None)
        # symbols for constraints, not including whitespace
        self.ct_symbol_map = {LinearConstraintType.EQ: "==",
                              LinearConstraintType.LE: "<=",
                              LinearConstraintType.GE: ">="}



    def get_format(self):
        from docplex.mp.format import OPL_format

        return OPL_format

    def fix_name(self, mobj, prefix, local_index_map, hide_names):
        raw_name = mobj.name
        # ignore hide_names here
        if not raw_name or mobj.has_automatic_name():
            return self._make_prefix_name(mobj, prefix, local_index_map, offset=1)
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
        return self._vartype_map.get(type(vartype), "unknown")

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

    def _print_objective(self, out, model):
        out.write(model.objective_sense.verb())
        self._newline(out)
        # ---
        oss = StringIO()
        oss.write(self._indent_space)
        self._expr_to_stringio(oss, model.objective_expr)
        oss.write(';')
        self.wrap_and_print(out, oss, subsequent_level=1)
        # ---
        self._newline(out)

    def _print_binary_constraint(self, oss, ct):
        left_expr = ct.left_expr
        right_expr = ct.right_expr
        self._expr_to_stringio(oss, left_expr)
        oss.write(' ')
        oss.write(self.ct_symbol_map[ct.type])
        oss.write(' ')
        self._expr_to_stringio(oss, right_expr)

    def _print_range_constraint(self, oss, rng):
        expr = rng.expr
        lb = rng.lb
        ub = rng.ub
        oss.write(self._num_to_string(lb))
        oss.write(" <= ")
        self._expr_to_stringio(oss, expr)
        oss.write(" <= ")
        oss.write(self._num_to_string(ub))
        oss.write(';')

    def _print_indicator_constraint(self, oss, ind_ct):
        active = ind_ct.active_value
        linear_ct = ind_ct.linear_constraint
        indicator_varname = self._var_print_name(ind_ct.indicator_var)
        oss.write(indicator_varname)
        if active is 0:
            oss.write(" == 0")
        oss.write(" <= ")
        oss.write('(')
        self._print_binary_constraint(oss, linear_ct)
        oss.write(')')
        oss.write(';')

    def _print_constraints(self, out, model):
        indent_one = self._indent_space
        for ct in model.iter_binary_constraints():
            if ct.is_generated():
                continue

            ctname = self.ct_print_name(ct)
            indent_level = 1
            if ctname:
                out.write("%s%s:\n" % (indent_one, ctname))
                indent_level += 1
            oss = StringIO()
            oss.write(indent_one * indent_level)
            self._print_binary_constraint(oss, ct)
            oss.write(';')
            self.wrap_and_print(out, oss, subsequent_level=indent_level)

    def _print_ranges(self, out, model):
        indent_one = self._indent_space
        for ct in model.iter_range_constraints():
            ctname = self.ct_print_name(ct)
            indent_level = 1
            if ctname:
                out.write("%s%s:\n" % (indent_one, ctname))
                indent_level += 1
            oss = StringIO()
            oss.write(indent_one * indent_level)
            self._print_range_constraint(oss, ct)
            self.wrap_and_print(out, oss, subsequent_level=indent_level)

    def _print_indicators(self, out, model):
        indent_one = self._indent_space
        for ct in model.iter_indicator_constraints():
            if ct.is_generated():
                continue

            ctname = self.ct_print_name(ct)
            indent_level = 1
            if ctname:
                out.write("%s%s:\n" % (indent_one, ctname))
                indent_level += 1
            oss = StringIO()
            oss.write(indent_one * indent_level)
            self._print_indicator_constraint(oss, ct)
            self.wrap_and_print(out, oss, subsequent_level=indent_level)

    def _print_kpis(self, out, model):
        printed_section_header = False
        for kpi in model.iter_kpis():
            if not kpi.requires_solution():
                continue

            if not printed_section_header:
                self._newline(out)
                self._print_line_comment(out, " KPI section")
                printed_section_header = True

            kpi_expr = kpi.as_expression()
            oss = StringIO()
            oss.write("dexpr ")
            if kpi_expr.is_discrete():
                oss.write("integer ")
            else:
                oss.write("float ")
            oss.write(self._translate_chars(kpi.name))
            oss.write(" = ")
            if isinstance(kpi_expr, LinearExpr):
                self._expr_to_stringio(oss, kpi_expr)
            elif isinstance(kpi_expr, Var):
                self._var_to_stringio(oss, kpi_expr)
            self.wrap_and_print(out, oss, subsequent_level=1)
        if printed_section_header:
            self._newline(out)

    def _var_to_stringio(self, oss, dvar):
        oss.write(dvar.name)

    def print_model_to_stream(self, out, model):
        self.prepare(model)
        self._print_signature(out)
        self._print_encoding(out)
        self._print_model_name(out, model)

        # var containers
        self._print_var_containers(out, model)
        self._print_single_vars(out, model)
        # KPI section
        self._print_kpis(out, model)

        self._print_objective(out, model)
        out.write("subject to {\n")
        self._print_constraints(out, model)
        self._print_ranges(out, model)
        self._print_indicators(out, model)
        out.write("}\n")
