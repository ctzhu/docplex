# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from __future__ import print_function
# from six import iteritems, itervalues

import re

from docplex.mp.linear import *
from docplex.mp.environment import env_is_64_bit
from docplex.mp.mprinter import TextModelPrinter, _ExportWrapper

from docplex.mp.format import LP_format

# gendoc: ignore


class LPModelPrinter(TextModelPrinter):
    _lp_re = re.compile(r"[a-df-zA-DF-Z!#$%&()/,;?@_`'{}|\"][a-zA-Z0-9!#$%&()/.,;?@_`'{}|\"]*")

    _lp_symbol_map = {LinearConstraintType.EQ: " = ",  # BEWARE NOT ==
                      LinearConstraintType.LE: " <= ",
                      LinearConstraintType.GE: " >= "}

    __new_line_sep = '\n'
    __expr_prefix = ' ' * 6

    float_precision_32 = 9
    float_precision_64 = 12  #

    def __init__(self, hide_user_names=False, indent_level=1):
        TextModelPrinter.__init__(self,
                                  indent=indent_level,
                                  comment_start='\\',
                                  hide_user_names=hide_user_names,
                                  nb_digits_for_floats=\
                                      self.float_precision_64 if env_is_64_bit() else
                                  self.float_precision_32)

        self.wrapper = self.create_wrapper(line_width=79, initial_indent=1)

        self._noncompliant_varname = None

    def get_format(self):
        return LP_format

    def encrypt_user_names(self):
        """
        Encrypt either if expliclity asked for or if some name is not LP-compliant
        :return:
        """
        return TextModelPrinter.encrypt_user_names(self) or self._noncompliant_varname

    def _print_ct_name(self, ct, name_map):
        lp_ctname = name_map.get(ct._index)
        indented = self._indent_level

        if lp_ctname is not None:
            ct_label = self._indent_space + lp_ctname + ':'
            indented += len(ct_label)
        else:
            ct_label = ''
        ct_indent_space = self._get_indent_from_level(indented)
        return ct_indent_space, ct_label

    def _print_binary_ct(self, wrapper, num_printer, var_name_map, binary_ct, _symbol_map=_lp_symbol_map):
        # ensure consistent ordering: left termes then right terms
        iter_diff_coeffs = binary_ct._iter_net_coeffs()
        self._print_expr_iter(wrapper, num_printer, var_name_map, iter_diff_coeffs)
        wrapper.write(_symbol_map.get(binary_ct.type, " ?? "), separator=False)
        wrapper.write(num_printer.to_string(binary_ct.rhs()), separator=False)
        wrapper.flush(print_newline=False)

    def _print_ranged_ct(self, wrapper, num_printer, var_name_map, ranged_ct):
        exp = ranged_ct.expr
        (varname, rhs, _) = self._rangeData[ranged_ct]
        self._print_expr(wrapper, num_printer, var_name_map, exp)
        wrapper.write('-', separator=False)
        wrapper.write(varname)
        wrapper.write('=')
        wrapper.write(self._num_to_string(rhs))
        wrapper.flush(print_newline=False)

    def _print_indicator_ct(self, wrapper, num_printer, var_name_map, indicator_ct):
        """
        Prints an indicator ct in LP
        :param wrapper:
        :param indicator_ct:
        :return:
        """
        INDICATOR_SYMBOL = " -> "
        binary_var = indicator_ct.indicator_var

        wrapper.write(self._var_print_name(binary_var))
        wrapper.write(" = ")
        wrapper.write("%d" % indicator_ct.logical_rhs)
        wrapper.write(INDICATOR_SYMBOL)
        self._print_binary_ct(wrapper, num_printer, var_name_map, indicator_ct.linear_constraint)

    def _print_constraint(self, wrapper, num_printer, var_name_map, ct):
        ct_label = None
        if isinstance(ct, LinearConstraint):
            if not ct.is_trivial_feasible():
                if self._hide_user_names:
                    indent_str = ''
                else:
                    indent_str, ct_label = self._print_ct_name(ct, name_map=self._ct_name_map)
                wrapper.reset_indent(indent_str)
                if ct_label is not None:
                    wrapper.write(ct_label)
                self._print_binary_ct(wrapper, num_printer, var_name_map, ct)
        elif isinstance(ct, RangeConstraint):
            if self._hide_user_names:
                indent_str = ''
            else:
                indent_str, ct_label = self._print_ct_name(ct, name_map=self._ct_name_map)
            wrapper.reset_indent(indent_str)
            if ct_label is not None:
                wrapper.write(ct_label)
            self._print_ranged_ct(wrapper, num_printer, var_name_map, ct)
        elif isinstance(ct, IndicatorConstraint):
            if self._hide_user_names:
                indent_str = ''
            else:
                indent_str, ct_label = self._print_ct_name(ct, name_map=self._ic_name_map)
            wrapper.reset_indent(indent_str)
            if ct_label is not None:
                wrapper.write(ct_label)
            self._print_indicator_ct(wrapper, num_printer, var_name_map, ct)
        else:
            ct.error("ERROR: unexpected constraint not printed: {0!s}".format(ct))  # pragma: no cover

        # EOL
        wrapper.newline()

    def _print_qexpr(self, wrapper, num_printer, var_name_map, quad_expr, force_initial_plus):
        # writes a quadratic expression
        # in the form [ 2a_ij a_i.a_j ] / 2
        # Note that all coefficients must be doubled due to the tQXQ formulation
        q = 0
        varname_getter = self._var_print_name
        if force_initial_plus:
            wrapper.write('+')
        if quad_expr.is_quadratic():
            wrapper.write('[')

            for qvp, qk in quad_expr.iter_quads():
                curr_token = ''
                if 0 == qk:
                    continue
                if qk < 0:
                    print_sign = '-'
                    abs_qk = - qk
                else:
                    print_sign = '+' if q > 0 else ''
                    abs_qk = qk
                curr_token += print_sign
                # all coefficients must be doubled because of the []/2 pattern.
                abs_qk2 = 2 * abs_qk
                if abs_qk2 != 1:
                    curr_token += num_printer.to_string(abs_qk2)
                    curr_token += ' '

                if qvp.is_square():
                    qv_name = varname_getter(qvp[0])
                    curr_token += "%s^2" % qv_name
                else:
                    qv1 = qvp[0]
                    qv2 = qvp[1]
                    curr_token += "%s*%s" % (varname_getter(qv1), varname_getter(qv2))

                wrapper.write(curr_token)

                q += 1

            # closing ]
            wrapper.write(']/2')


    def _print_expr(self, wrapper, num_printer, var_name_map, expr):
        # prints an expr to a stream
        term_iter = expr.iter_terms()
        self._print_expr_iter(wrapper, num_printer, var_name_map, term_iter)

    # @profile
    def _print_expr_iter(self, wrapper, num_printer, var_name_map, expr_iter):
        num2string_fn = num_printer.to_string
        c = 0
        for (v, coeff) in expr_iter:
            curr_token = ''
            if 0 == coeff:
                continue  # pragma: no cover

            if coeff < 0:
                curr_token += '-'
                wrote_sign = True
                coeff = - coeff
            elif c > 0:
                # here coeff is positive, we write the '+' only if term is non-first
                curr_token += '+'
                wrote_sign = True
            else:
                wrote_sign = False

            if 1 != coeff:
                if wrote_sign:
                    curr_token += ' '
                curr_token += num2string_fn(coeff)
            if wrote_sign or 1 != coeff:
                curr_token += ' '
            curr_token += var_name_map[v._index]

            wrapper.write(curr_token)
            c += 1

        if not c:
            # expr is empty, we must print something, so 0
            wrapper.write('0')

    def _print_var_block(self, wrapper, iter_vars, header):
        # Set wrapper with no indent
        wrapper.reset_indent('')
        wrapper.flush(print_newline=False)
        printed_header = False
        self_indent = self._indent_space
        for v in iter_vars:
            lp_name = self._var_print_name(v)
            if not printed_header:
                wrapper.newline()
                wrapper.write("%s" % header)
                printed_header = True
                # Configure indent for next lines
                wrapper.flush(print_newline=True)
                wrapper.reset_indent(self_indent)
                wrapper.flush(print_newline=False)
            wrapper.write(lp_name)
        if printed_header:
            wrapper.flush(print_newline=True)

    def _print_var_bounds(self, out, num_printer, varname, lb, ub, varname_indent=5 * ' '):
        LE = "<="
        FREE = "Free"

        if lb is None and ub is None:
            # try to indent with space of '0 <= ', that is 5 space
            out.write("%s %s %s\n" % (varname_indent, varname, FREE))
        elif lb is None:
            out.write("%s %s %s %s\n" % (varname_indent, varname, LE, num_printer.to_string(ub)))
        elif ub is None:
            out.write("%s %s %s\n" % (num_printer.to_string(lb), LE, varname))
        elif lb == ub:
            out.write("%s %s %s %s\n" % (varname_indent, varname, "=", num_printer.to_string(lb)))
        else:
            out.write("%s %s %s %s %s\n" % (num_printer.to_string(lb), LE, varname, LE, num_printer.to_string(ub)))

    TRUNCATE = 200

    @staticmethod
    def _non_compliant_lp_name_stop_here(name):
        pass

    def fix_name(self, mobj, prefix, local_index_map, hide_names):
        raw_name = mobj.name

        # anonymous constraints must be named in a LP (we follow CPLEX here)
        if hide_names or mobj.has_automatic_name() or mobj.is_generated() or not raw_name:
            return self._make_prefix_name(mobj, prefix, local_index_map, offset=1)
        elif not self._is_lp_compliant(raw_name):
            self._non_compliant_lp_name_stop_here(raw_name)
            return self._make_prefix_name(mobj, prefix, local_index_map, offset=1)
        else:
            # swap blanks with underscores
            fixed_name = self._translate_chars(raw_name)
            # truncate if necessary, again this does nothing if name is too short
            return fixed_name[:self.TRUNCATE]

    def _print_model_name(self, out, model):
        printed_name = model.name or 'CPLEX'
        out.write("\\Problem name: %s\n" % printed_name)

    @staticmethod
    def _is_lp_compliant(name, _lpname_regexp=_lp_re):
        if name is None:
            return True  # pragma: no cover
        # PUT THIS SOMEWHERE ELSE
        fixed_name = LPModelPrinter.fix_whitespace(name)
        lp_match = _lpname_regexp.match(fixed_name)
        return lp_match and lp_match.start() == 0 and lp_match.end() == len(fixed_name)

    @staticmethod
    def _is_injective(name_map):
        nb_keys = len(name_map)
        nb_different_names = len(set(name_map.values()))
        return nb_different_names == nb_keys

    #  @profile
    def print_model_to_stream(self, out, model):

        if not self._is_injective(self._var_name_map):
            # use indices to differentiate names
            sys.__stdout__.write("\DOcplex: refine variable names\n")
            k = 0
            for dv, lp_varname in iteritems(self._var_name_map):
                refined_name = "%s#%d" % (lp_varname, k)
                self._var_name_map[dv] = refined_name
                k += 1

        TextModelPrinter.prepare(self, model)
        self_num_printer = self._num_printer
        var_name_map = self._var_name_map

        self._print_signature(out)
        self._print_encoding(out)
        self._print_model_name(out, model)
        self._newline(out)

        # print objective
        out.write(model.objective_sense.name)
        self._newline(out)
        wrapper = _ExportWrapper(out, self.__expr_prefix)
        wrapper.write(' obj:')
        objexpr = model.objective_expr
        obj_offset = objexpr.constant
        obj_constant_term_varname = None
        if objexpr.is_constant():
            if obj_offset:
                wrapper.write(self._num_to_string(obj_offset))
        else:
            # if objexpr is constant just print nothing.

            # the new dummy var has name 'x' + (max_index+2
            # why +2 because name indices start from 1 and CPLEX start from 0
            if 0 != obj_offset:
                obj_constant_term_varname = self.get_extra_var_name(model, pattern='x%d')

            if objexpr.is_quad_expr():
                objlin = objexpr.linear_part
            else:
                objlin = objexpr

            if not objlin.is_constant():
                # write the linear part first
                self._print_expr(wrapper, self_num_printer, var_name_map, objlin)
                # for the constant part, one day remove this...
                if obj_constant_term_varname:
                    wrapper.write(' + %s' % obj_constant_term_varname, separator=False)
            elif obj_constant_term_varname:
                wrapper.write(obj_constant_term_varname)

            if objexpr.is_quad_expr() and objexpr.is_quadratic():
                # is there a linear part?
                self._print_qexpr(wrapper, self_num_printer, var_name_map, quad_expr=objexpr, force_initial_plus=not objlin.is_zero())

        wrapper.flush(print_newline=True)

        out.write("Subject To\n")

        for ct in model.iter_constraints():
            self._print_constraint(wrapper, self_num_printer, var_name_map, ct)

        out.write("\nBounds\n")
        for dvar in model.iter_variables():
            lp_varname = self._var_print_name(dvar)
            if dvar.is_binary():
                self._print_var_bounds(out, self_num_printer, lp_varname, 0, 1)
            else:
                var_lb = dvar.get_lb()
                var_ub = dvar.get_ub()
                free_lb = model.is_free_lb(var_lb)
                free_ub = model.is_free_ub(var_ub)
                if free_lb and free_ub:
                    self._print_var_bounds(out, self_num_printer, lp_varname, None, None)
                elif free_ub:
                    self._print_var_bounds(out, self_num_printer, lp_varname, var_lb, None)
                elif free_lb:
                    self._print_var_bounds(out, self_num_printer, lp_varname, None, var_ub)
                else:
                    self._print_var_bounds(out, self_num_printer, lp_varname, var_lb, var_ub)
        # add constant term
        if obj_constant_term_varname:
            self._print_var_bounds(out, self_num_printer, obj_constant_term_varname, obj_offset, obj_offset)
        # add ranged cts vars
        for rng in model.iter_range_constraints():
            (varname, _, ub) = self._rangeData[rng]
            self._print_var_bounds(out, self_num_printer, varname, 0, ub)

        self._print_var_block(wrapper, model.iter_binary_vars(), 'Binaries')
        self._print_var_block(wrapper, model.iter_integer_vars(), 'Generals')

        out.write("End\n")
