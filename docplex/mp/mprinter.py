# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# coding=utf-8
# ------------------------------
from __future__ import print_function
import sys
import os

from docplex.mp.utils import normalize
from docplex.mp.linear import LinearConstraintType, LinearConstraint, IndicatorConstraint, RangeConstraint, Var, LinearExpr

from six import iteritems

from textwrap import TextWrapper

from docplex.mp.utils import StringIO


# gendoc: ignore

class _NumPrinter(object):
    """
    INTERNAL.
    """
    def __init__(self, nb_digits_for_floats, num_infinity=1e+20, pinf="+inf", ninf="-inf"):
        assert(nb_digits_for_floats >= 0)
        assert(isinstance(pinf, str))
        assert(isinstance(ninf, str))
        self.true_infinity = num_infinity
        self.__precision = nb_digits_for_floats
        self.__positive_infinity = pinf
        self.__negative_infinity = ninf
        # coin the format from the nb of digits
        # 2 -> %.2f
        self._double_format = "%." + ('%df' % nb_digits_for_floats)

    @property
    def precision(self):
        return self.__precision

    def to_string(self, num):
        if 0 == num:
            return '0'
        elif 1 == num:
            return '1'
        elif num >= self.true_infinity:
            return self.__positive_infinity
        elif num <= - self.true_infinity:
            return self.__negative_infinity
        elif num == int(num):
            return '%d' % int(num)
        else:
            return self._double_format % num

    def to_stringio(self, oss, num):
        int_num = int(num)
        if num == int_num:
            oss.write('%d' % int_num)
        else:
            oss.write(self._double_format % num)

    def __call__(self, num):
        return self.to_string(num)


class ModelPrinter(object):
    ''' Generic Printer code.
    '''

    def __init__(self):
        pass

    def get_format(self):
        """
        returns the Format object
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def extension(self):
        """
        :return: the extension of the format
        """
        return self.get_format().extension

    def _printModelToStream(self, mdl, ofs):
        from docplex.mp.utils import RedirectedOutputContext
        with RedirectedOutputContext(ofs, mdl.error_handler):
            self.printModelInternal(mdl)


    def printModel(self, mdl, out=None):
        """ Generic method.
            If passed with a string, uses it as a file name
            if None is passed, uses standard output.
            else assume a stream is passed and try it
        """
        if out is None:
            # prints on standard output
            self.printModelInternal(mdl)
        elif isinstance(out, str):
            # a string is interpreted as a path name
            ext = self.extension()
            path = out if out.endswith(ext) else out + ext
            # SAv format requires binary mode!
            write_mode = "wb" if self.get_format().is_binary else "w"
            with open(path, write_mode) as of:
                self._printModelToStream(mdl, of)
                # print("* file: %s overwritten" % path)
        else:
            try:
                self._printModelToStream(mdl, out)
            except AttributeError:   # pragma: no cover
                pass                 # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    def printModelInternal(self, mdl):
        raise NotImplementedError  # pragma: no cover
    
    def get_var_name_encoding(self):
        return None  # default is no encoding


# noinspection PyAbstractClass
class TextModelPrinter(ModelPrinter):

    DEFAULT_ENCODING = "ENCODING=ISO-8859-1"

    @staticmethod
    def create_wrapper(line_width=78, initial_indent=0):
        """
        :param initial_indent: basic indent
        :param line_width: maximum line width used by the wrapper
        :return: an instance of wrapper
        """
        wrapper = TextWrapper()
        wrapper.break_long_words = False
        wrapper.initial_indent = ' ' * initial_indent
        wrapper.width = line_width
        wrapper.break_on_hyphens = False
        wrapper.expand_tabs = False
        wrapper.drop_whitespace = False
        return wrapper

    def __init__(self, comment_start, indent=1,
                 hide_user_names=False,
                 nb_digits_for_floats=3,
                 encoding=DEFAULT_ENCODING,
                 wrap_lines=True):
        ModelPrinter.__init__(self)
        # should be elsewhere
        self.true_infinity = float('inf')

        self.wrap_lines = wrap_lines

        self._comment_start = comment_start
        self._hide_user_names = hide_user_names
        self._encoding = encoding  # None is a valid value, in which case no encoding is printed
        #
        self._var_name_map = {}
        self._ct_name_map = {}
        # created on demand if model is not fully indexed
        self._local_var_indices = None
        self._local_ct_indices = None
        self._rangeData = {}
        self._num_printer = _NumPrinter(nb_digits_for_floats)
        self._indent_level = indent
        self._indent_space = ' ' * indent

    @property
    def nb_digits_for_floats(self):
        return self._num_printer.precision

    def _get_hide_user_names(self):
        """
        returns true if user names for variables and constraints should be forgotten.
        If yes, generic names (e.g. x1,x3, c45.. are generated and used everywhere).
        This is done on purpose to obfuscate the file.
        :return:
        """
        return self._hide_user_names

    def _set_hide_user_names(self, hide):
        self._hide_user_names = hide

    forget_user_names = property(_get_hide_user_names, _set_hide_user_names)

    def encrypt_user_names(self):
        """
        Actually used to decide whether to encryupt or noyt
        :return:
        """
        return self._hide_user_names

    def _print_line_comment(self, comment_text):
        print(self._comment_start, comment_text)

    def _print_encoding(self):
        """
        prints the file encoding
        :return:
        """
        if self._encoding:
            self._print_line_comment(self._encoding)

    def _print_model_name(self, mdl):
        """ Redefine this method to print the model name, if necessary
        :param mdl: the model to be printed
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def _print_signature(self):
        """
        Prints a signature message denoting this file comes from Python Modeling Layer
        :return:
        """
        self._print_line_comment("This file has been generated by DOcplex")

    def _precompute_name_dict(self, mobj_seq, prefixer, local_index_map):
        ''' Returns a name dictionary from a sequence of modeling objects.'''
        fixed_name_dir = {}
        all_names = set({})
        for mobj in mobj_seq:
            prefix = prefixer(mobj)
            fixed_name = self.fix_name(mobj, prefix, local_index_map)
            if fixed_name:
                if fixed_name in all_names:
                    mobj.trace("duplicated name {0} obj is {0!s}".format(fixed_name, mobj))
                fixed_name_dir[mobj] = fixed_name
                all_names.add(fixed_name)
            else:
                pass
                # sys.__stdout__.write("\n-- object has no name: {0!s}".format(mobj))

        # prefix if not unique
        global_name_set = set(fixed_name_dir.values())
        k = 1
        if len(global_name_set) < len(fixed_name_dir):
            sys.__stdout__.write("\n--suffixing names")
            for (mobj, lp_name_pass1) in iteritems(fixed_name_dir):
                fixed_name_dir[mobj] = "%s#%d" % (lp_name_pass1, k)
                k += 1
        return fixed_name_dir

    def _num_to_string(self, num):
        # INTERNAL
        return self._num_printer.to_string(num)

    def _num_to_stringio(self, oss, num):
        return self._num_printer.to_stringio(oss, num)

    def prepare(self, model):
        """
        :param model: the model being printed
        """
        #if not model._is_fully_indexed():
            # use printer local indexing for name generation.
        self._local_var_indices = {dv: k for k, dv in enumerate(model.iter_variables())}
        self._local_ct_indices  = {ct: k for k, ct in enumerate(model.iter_constraints())}

        var_prefixer = lambda _: 'x'
        self._var_name_map = self._precompute_name_dict(model.iter_variables(), var_prefixer, self._local_var_indices)
        ct_type_to_prefix_map = {LinearConstraint: 'c', IndicatorConstraint: 'ic', RangeConstraint: 'c'}
        ct_prefixer = lambda ct: ct_type_to_prefix_map.get(type(ct), 'c')
        self._ct_name_map = self._precompute_name_dict(model.iter_constraints(), ct_prefixer, self._local_ct_indices)

        self._rangeData = {}
        for rng in model.iter_range_constraints():
            # precompute data for ranges
            # 1 name ?
            # 2 rhs is lb - constant
            # 3 bounds are (0, ub-lb)
            varname = 'Rg%s' % self.ct_print_name(rng)
            rhs = rng.rhs()
            ub = rng.range()
            self._rangeData[rng] = (varname, rhs, ub)

    @staticmethod
    def fix_whitespace(name):
        """
        Swaps white spaces by underscores. Names with no blanks are not copied.
        :param name:
        :return:
        """
        return name.replace(" ", "_")

    def _var_print_name(self, dvar):
        # INTERNAL
        return self._var_name_map[dvar]

    def get_var_name_encoding(self):
        return self._var_name_map

    def get_ct_name_encoding(self):
        return self._ct_name_map

    def ct_print_name(self, ct):
        return self._ct_name_map.get(ct)

    def max_var_name_len(self):
        """
        :return: the maximum length of variable names
        """
        return max([len(vn) for vn in self._var_name_map.values()]) if self._var_name_map else 0

    def max_ct_name_len(self):
        """
        :return: the maximum length of constraint names
        """
        return max([len(cn) for cn in self._ct_name_map.values()]) if self._ct_name_map else 0

    def get_extra_var_name(self, model, pattern='x%d'):
        """
        :param pattern: a format string with one %d
        :return: a variable name f the form pattern %k
        where k is an integer, starting at max variable index+2
        we loop until a free name is found.
        """
        if model.number_of_variables:
            safe_index = max([dv.index for dv in model.iter_variables()]) + 2  # add1 for next, add 1 for start at 1
        else:
            safe_index = 1
        model_var_names = {dv.name for dv in model.iter_variables() if dv.name is not None}

        safe_name = pattern % safe_index
        nb_tries = 0
        while safe_name in model_var_names and nb_tries <= 1000:
            safe_index += 1
            safe_name = pattern % safe_index
            nb_tries += 1
        if nb_tries == 1000:
            return "_zorglub"
        else:
            return safe_name

    @staticmethod
    def _is_automatic_name(name):
        return name[0] == '_'

    @staticmethod
    def _make_prefix_name(mobj, prefix, local_index_map, offset=1):
        index = local_index_map[mobj] if local_index_map is not None else mobj.unchecked_index
        prefixed_name = "{0:s}{1:d}".format(prefix, index + offset)
        return prefixed_name

    def translate_chars(self, raw_name):
        from docplex.mp.utils import mktrans
        table = mktrans(" -+/\\<>", "_mpd___")
        return raw_name.translate(table)

    def fix_name(self, mobj, prefix, local_index_map):
        """
        default implementation does nothing but return the raw name
        :param mobj: a modeling object
        :param prefix: a naming pattern with a slot for a counter
        :return: the new modified name if necessary, here does nothing/
        """
        raw_name = mobj.name
        if self._hide_user_names or not raw_name or self._is_automatic_name(raw_name):
            return self._make_prefix_name(mobj, prefix, local_index_map, offset=1)
        else:
            return self.translate_chars(raw_name)

    def printModelInternal(self, mdl):
        """ This is yet another abstract class
        :param mdl:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def _expr_to_stringio(self, oss, expr):
        # nb digits
        # product symbol is '*'
        # no spaces
        expr.to_stringio(oss, self.nb_digits_for_floats, prod_symbol='*', use_space=True, var_namer=lambda v: self._var_name_map[v])

    def _expr_to_string(self, expr):
        oss = StringIO()
        self._expr_to_stringio(oss, expr)
        return oss.getvalue()

    def wrap_and_print(self, oss, wrapper, subsequent_level=1):
        """
        Takes input from a stringIO object, wraps it using the wrapper object, prints the
        wrapped text, and sets the subsequent indent.
        :param oss:
        :param wrapper:
        :param subsequent_level:
        """
        raw = oss.getvalue()
        if self.wrap_lines:
            wrapper.subsequent_indent = self._indent_space * subsequent_level
            print(wrapper.fill(raw))
        else:
            print(raw)


class ModelPrettyPrinter(TextModelPrinter):

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

        self.wrapper = self.create_wrapper(line_width=78, initial_indent=0)

    def get_format(self):
        from docplex.mp.format import OPL_format
        return OPL_format

    def fix_name(self, mobj, prefix, local_index_map):
        raw_name = mobj.name
        if not raw_name or mobj.has_automatic_name():
            return self._make_prefix_name(mobj, prefix, local_index_map, offset=1)
        else:
            if mobj.is_generated():
                mobj_origin = mobj.origin()
                if mobj is mobj_origin.functional_var:
                    # only the functional var is named after the functional expr
                    return str(mobj_origin)

            return normalize(raw_name, force_lowercase=False)

    def _print_model_name(self, mdl):
        printed_name = mdl.name or "AnonymousModel"
        print("// model name is: {0:s}".format(printed_name))
        print()

    def _print_var_containers(self, mdl):
        gensym_count = 1
        printed_header = False
        for ctn in mdl.iter_var_containers():
            if not printed_header:
                self._print_line_comment("var contrainer section")
                printed_header = True
            vartype_name = ctn.vartype.short_name
            varctn_name = ctn.name
            if not varctn_name:
                varctn_name = 'x%d' % gensym_count
                gensym_count += 1
            print("dvar {0} {1}{2};".format(vartype_name, varctn_name, ctn.dimension_string))

        if printed_header:
            print()

    def _print_single_vars(self, mdl):
        printed_header = False
        for v in mdl._iter_single_vars():
            if v.is_generated():
                continue

            if not printed_header:
                self._print_line_comment("single vars section")
                printed_header = True
            vartype_name = v.vartype.short_name
            var_printname = self._var_name_map.get(v, "???")
            print("dvar {0} {1};".format(vartype_name, var_printname))

        if printed_header:
            print()

    def _print_objective(self, model):
        print(model.objective_sense.verb())
        oss = StringIO()
        oss.write(self._indent_space)
        self._expr_to_stringio(oss, model.objective_expr)
        oss.write(';')
        self.wrap_and_print(oss, self.wrapper, subsequent_level=1)
        print()

    def _print_binary_constraint(self, oss, ct):
        left_expr = ct.left_expr
        right_expr = ct.right_expr
        self._expr_to_stringio(oss, left_expr)
        oss.write(' ')
        oss.write(self.ct_symbol_map[ct.type])
        oss.write(' ')
        self._expr_to_stringio(oss, right_expr)
        oss.write(';')

    def _print_range_constraint(self, oss, rng):
        expr = rng.expr
        lb = rng.lb
        ub = rng.ub
        self._num_to_stringio(oss, lb)
        oss.write(" <= ")
        self._expr_to_stringio(oss, expr)
        oss.write(" <= ")
        self._num_to_stringio(oss, ub)
        oss.write(';')

    def _print_indicator_constraint(self, oss, ind_ct):
        active = ind_ct.active_value
        linear_ct = ind_ct.linear_constraint
        indicator_varname = self._var_print_name(ind_ct.indicator_var)
        oss.write(indicator_varname)
        if active is 0:
            oss.write(" == 0")
        oss.write(" <= ")
        self._print_binary_constraint(oss, linear_ct)
        oss.write(';')

    def _print_constraints(self, model):
        indent_one = self._indent_space
        for ct in model.iter_binary_constraints():
            if ct.is_generated():
                continue

            ctname = self.ct_print_name(ct)
            indent_level = 1
            if ctname:
                print("%s%s:" % (indent_one, ctname))
                indent_level += 1
            oss = StringIO()
            oss.write(indent_one * indent_level)
            self._print_binary_constraint(oss, ct)
            self.wrap_and_print(oss, self.wrapper, subsequent_level=indent_level)

    def _print_ranges(self, model):
        indent_one = self._indent_space
        for ct in model.iter_range_constraints():
            ctname = self.ct_print_name(ct)
            indent_level = 1
            if ctname:
                print("%s%s:" % (indent_one, ctname))
                indent_level += 1
            oss = StringIO()
            oss.write(indent_one * indent_level)
            self._print_range_constraint(oss, ct)
            self.wrap_and_print(oss, self.wrapper, subsequent_level=indent_level)

    def _print_indicators(self, model):
        indent_one = self._indent_space
        for ct in model.iter_indicator_constraints():
            if ct.is_generated():
                continue

            ctname = self.ct_print_name(ct)
            indent_level = 1
            if ctname:
                print("%s%s:" % (indent_one, ctname))
                indent_level += 1
            oss = StringIO()
            oss.write(indent_one * indent_level)
            self._print_indicator_constraint(oss, ct)
            self.wrap_and_print(oss, self.wrapper, subsequent_level=indent_level)

    def _print_kpis(self, model):
        printed_section_header = False
        for kpi in model.iter_kpis():
            if not kpi.requires_solution():
                continue

            if not printed_section_header:
                print()
                self._print_line_comment(" KPI section")
                printed_section_header = True

            kpi_expr = kpi.as_expression()
            oss = StringIO()
            oss.write("dexpr ")
            if kpi_expr.is_discrete():
                oss.write("integer ")
            else:
                oss.write("float ")
            oss.write(normalize(kpi.name, force_lowercase=False))
            oss.write(" = ")
            if isinstance(kpi_expr, LinearExpr):
                self._expr_to_stringio(oss, kpi_expr)
            elif isinstance(kpi_expr, Var):
                self._var_to_stringio(oss, kpi_expr)
            self.wrap_and_print(oss, self.wrapper, subsequent_level=1)
        if printed_section_header:
            print()

    def _var_to_stringio(self, oss, dvar):
        oss.write(dvar.name)


    def printModelInternal(self, model):
        self.prepare(model)
        self._print_signature()
        self._print_encoding()
        self._print_model_name(model)

        # var containers
        self._print_var_containers(model)
        self._print_single_vars(model)
        # KPI section
        self._print_kpis(model)

        self._print_objective(model)
        print("subject to {")
        self._print_constraints(model)
        self._print_ranges(model)
        self._print_indicators(model)
        print("}")

