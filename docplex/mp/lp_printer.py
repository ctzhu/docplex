# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from __future__ import print_function
# from six import iteritems, itervalues

import re
import sys

from docplex.mp.linear import *
from docplex.mp.constants import ComparisonType
from docplex.mp.constr import LinearConstraint, RangeConstraint, IndicatorConstraint, QuadraticConstraint
from docplex.mp.environment import env_is_64_bit
from docplex.mp.mprinter import TextModelPrinter, _ExportWrapper, _NumPrinter

from docplex.mp.format import LP_format

# gendoc: ignore


class LPModelPrinter(TextModelPrinter):
    _lp_re = re.compile(r"[a-df-zA-DF-Z!#$%&()/,;?@_`'{}|\"][a-zA-Z0-9!#$%&()/.,;?@_`'{}|\"]*")

    _lp_symbol_map = {ComparisonType.EQ: " = ",  # BEWARE NOT ==
                      ComparisonType.LE: " <= ",
                      ComparisonType.GE: " >= "}

    __new_line_sep = '\n'
    __expr_indent = ' ' * 6

    float_precision_32 = 9
    float_precision_64 = 12  #

    def __init__(self, hide_user_names=False, indent_level=1):
        nb_digits = self.float_precision_64 if env_is_64_bit() else self.float_precision_32
        TextModelPrinter.__init__(self,
                                  indent=indent_level,
                                  comment_start='\\',
                                  hide_user_names=hide_user_names,
                                  nb_digits_for_floats=nb_digits)


        self._noncompliant_varname = None
        # specific printer for lp: do not print +inf/-inf inside constraints!
        self._lp_num_printer = _NumPrinter(nb_digits_for_floats=nb_digits,
                                           num_infinity=1e+20, pinf="1e+20", ninf="-1e+20")

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

    def _print_binary_ct(self, wrapper, num_printer, var_name_map, binary_ct, _symbol_map=_lp_symbol_map,
                         allow_empty=False, force_first_sign=False):
        # ensure consistent ordering: left termes then right terms
        iter_diff_coeffs = binary_ct.iter_net_linear_coefs()
        self._print_expr_iter(wrapper, num_printer, var_name_map, iter_diff_coeffs,
                              allow_empty=allow_empty,
                              force_first_plus=force_first_sign)
        wrapper.write(_symbol_map.get(binary_ct.type, " ?? "), separator=False)
        wrapper.write(num_printer.to_string(binary_ct.rhs()), separator=False)

    def _print_ranged_ct(self, wrapper, num_printer, var_name_map, ranged_ct):
        exp = ranged_ct.expr
        (varname, rhs, _) = self._rangeData[ranged_ct]
        self._print_expr(wrapper, num_printer, var_name_map, exp)
        wrapper.write('-', separator=False)
        wrapper.write(varname)
        wrapper.write('=')
        wrapper.write(self._num_to_string(rhs))

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


    def _print_quadratic_ct(self, wrapper, num_printer, var_name_map, qct):
        """
        Prints an indicator ct in LP
        :param wrapper:
        :param qct
        :return:
        """
        q = self._print_qexpr_iter(wrapper, num_printer, var_name_map, qct.iter_net_quads())
        # force a '+' ?
        has_quads = q > 0
        self._print_binary_ct(wrapper, num_printer, var_name_map, qct, allow_empty=has_quads, force_first_sign=has_quads)

    def _print_constraint_label(self, wrapper, ct, name_map):
        if self._hide_user_names:
            wrapper.set_indent('')
        else:
            indent_str, ct_label = self._print_ct_name(ct, name_map=name_map)
            wrapper.set_indent(indent_str)
            if ct_label is not None:
                wrapper.write(ct_label)

    def _print_constraint(self, wrapper, num_printer, var_name_map, ct):
        wrapper.begin_line()
        if isinstance(ct, LinearConstraint):
            if not ct.is_trivial_feasible():
                self._print_constraint_label(wrapper, ct, name_map=self._linct_name_map)
                self._print_binary_ct(wrapper, num_printer, var_name_map, ct)
        elif isinstance(ct, RangeConstraint):
            self._print_constraint_label(wrapper, ct, name_map=self._linct_name_map)
            self._print_ranged_ct(wrapper, num_printer, var_name_map, ct)
        elif isinstance(ct, IndicatorConstraint):
            self._print_constraint_label(wrapper, ct, name_map=self._ic_name_map)
            self._print_indicator_ct(wrapper, num_printer, var_name_map, ct)
        elif isinstance(ct, QuadraticConstraint):
            self._print_constraint_label(wrapper, ct, name_map=self._qc_name_map)
            self._print_quadratic_ct(wrapper, num_printer, var_name_map, ct)
        else:
            ct.error("ERROR: unexpected constraint not printed: {0!s}".format(ct))  # pragma: no cover

        wrapper.flush(print_newline=True, reset=True)



    def _print_var_block(self, wrapper, iter_vars, header):
        wrapper.begin_line()
        printed_header = False
        self_indent = self._indent_space
        for v in iter_vars:
            lp_name = self._var_print_name(v)
            if not printed_header:
                wrapper.newline()
                wrapper.write(header)
                printed_header = True
                wrapper.set_indent(self_indent)
                # Configure indent for next lines
                wrapper.flush(print_newline=True)
            wrapper.write(lp_name)
        if printed_header:
            wrapper.flush(print_newline=True)

    def _print_var_bounds(self, out, num_printer, varname, lb, ub, varname_indent=5 * ' ',
                          le_symbol='<=',
                          free_symbol='Free'):
        if lb is None and ub is None:
            # try to indent with space of '0 <= ', that is 5 space
            out.write("%s %s %s\n" % (varname_indent, varname, free_symbol))
        elif lb is None:
            out.write("%s %s %s %s\n" % (varname_indent, varname, le_symbol, num_printer.to_string(ub)))
        elif ub is None:
            out.write("%s %s %s\n" % (num_printer.to_string(lb), le_symbol, varname))
        elif lb == ub:
            out.write("%s %s %s %s\n" % (varname_indent, varname, "=", num_printer.to_string(lb)))
        else:
            out.write("%s %s %s %s %s\n" % (num_printer.to_string(lb), le_symbol, varname, le_symbol, num_printer.to_string(ub)))

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
        model_name = None
        if model.name:
            # make sure model name is ascii
            encoded = model.name.encode('ascii', 'backslashreplace')
            if sys.version_info[0] == 3:
                # in python 3, encoded is a bytes at this point. Make it a string again
                encoded = encoded.decode('ascii')
            model_name = encoded.replace('\\\\','_').replace('\\', '_')
        printed_name = model_name or 'CPLEX'
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
        self_num_printer = self._lp_num_printer
        var_name_map = self._var_name_map

        self._print_signature(out)
        self._print_encoding(out)
        self._print_model_name(out, model)
        self._newline(out)

        # print objective
        out.write(model.objective_sense.name)
        self._newline(out)
        wrapper = _ExportWrapper(out, indent_str=self.__expr_indent)
        wrapper.write(' obj:')
        objexpr = model.objective_expr
        obj_offset = objexpr.get_constant()
        #obj_constant_term_varname = None
        if objexpr.is_constant():
            if obj_offset:
                wrapper.write(self._num_to_string(obj_offset))
        else:
            # for now, introduce a new dummy var has name 'x' + (max_index+2
            # why +2 because name indices start from 1 and CPLEX start from 0
            # if 0 != obj_offset:
            #     obj_constant_term_varname = self.get_extra_var_name(model, pattern='x%d')

            if objexpr.is_quad_expr():
                objlin = objexpr.linear_part
            else:
                objlin = objexpr

            if objexpr.is_quad_expr() and objexpr.has_quadratic_term():
                q = self._print_qexpr_obj(wrapper, self_num_printer, var_name_map, quad_expr=objexpr, force_initial_plus=False)
            else:
                q = 0

            # write the linear part first
            self._print_expr(wrapper, self_num_printer, var_name_map, objlin,
                             print_constant=True,
                             allow_empty=True,
                             force_first_plus=q > 0)


        wrapper.flush(print_newline=True)

        out.write("Subject To\n")

        for ct in model.iter_constraints():
            self._print_constraint(wrapper, self_num_printer, var_name_map, ct)

        out.write("\nBounds\n")
        symbolic_num_printer = self._num_printer
        print_var_bounds_fn = self._print_var_bounds
        var_print_name_fn = self._var_print_name
        for dvar in model.iter_variables():
            lp_varname = var_print_name_fn(dvar)
            if dvar.is_binary():
                print_var_bounds_fn(out, self_num_printer, lp_varname, 0, 1)
            else:
                var_lb = dvar.get_lb()
                var_ub = dvar.get_ub()
                free_lb = model.is_free_lb(var_lb)
                free_ub = model.is_free_ub(var_ub)
                if free_lb and free_ub:
                    print_var_bounds_fn(out, self_num_printer, lp_varname, lb=None, ub=None)
                elif free_ub:
                    # avoid zero lb
                    if 0 != var_lb:
                        print_var_bounds_fn(out, symbolic_num_printer, lp_varname, var_lb, ub=None)
                    else:
                        # lb is zero, ub is infinity, we dont print anything
                        pass
                else:
                    # save the lb if is zero
                    printed_lb = None if 0 == var_lb else var_lb
                    print_var_bounds_fn(out, symbolic_num_printer, lp_varname, lb=printed_lb, ub=var_ub)

        # add ranged cts vars
        for rng in model.iter_range_constraints():
            (varname, _, ub) = self._rangeData[rng]
            self._print_var_bounds(out, self_num_printer, varname, 0, ub)

        self._print_var_block(wrapper, model.iter_binary_vars(), 'Binaries')
        self._print_var_block(wrapper, model.iter_integer_vars(), 'Generals')
        self._print_var_block(wrapper, model.iter_semicontinuous_vars(), 'Semi-continuous')
        self._print_sos_block(wrapper, model)
        out.write("End\n")


    def _print_sos_block(self, wrapper, mdl):
        if mdl.number_of_sos > 0:
            wrapper.write('SOS')
            wrapper.flush(print_newline=True)
            name_fn = self._var_print_name
            for sos in mdl.iter_sos():
                sos_name = sos.get_name()
                if sos_name:
                    wrapper.write('%s:' % sos_name)
                wrapper.write('S%d ::' % sos.sos_type.value)  # 1 or 2
                ranks = sos.get_ranks()
                for rank, sos_var in izip(ranks, sos._variables):
                    wrapper.write('%s : %d' % (name_fn(sos_var), rank))
                wrapper.flush(print_newline=True)




