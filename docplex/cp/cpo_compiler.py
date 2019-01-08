# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Compiler converting internal model representation to CPO file format
"""

from docplex.cp.expression import *
from docplex.cp.solution import *
from docplex.cp.utils import *
import docplex.cp.config as config

import sys

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


###############################################################################
## Public classes
###############################################################################

class CpoCompiler(object):
    """ Compiler to CPO file format """
    __slots__ = ('model',                  # Source model
                 'params',                 # Solving parameters
                 'sourceloc',              # Indicator to add location traces in generated output
                 'alias_min_name_length',  # Minimum variable name length to replace it by an alias
                 'id_strings',             # Dictionary of printable string for each identifier
                 'last_loc',               # Last source location (file, line)
                 'exprset',                # Set of ids of named expressions already compiled
                 )

    def __init__(self, model, **kwargs):
        """ Create a new compiler

        Args:
            model:  Source model
        Optional args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        """
        super(CpoCompiler, self).__init__()

        # Build effective context
        context = config._get_effective_context(**kwargs)

        # Initialize processing
        self.model = model
        self.params = context.params
        self.sourceloc = True
        self.alias_min_name_length = None
        self.id_strings = {}

        # Set model parameters
        mctx = context.model
        if mctx is not None:
            self.sourceloc = mctx.add_source_location
            self.alias_min_name_length = mctx.length_for_alias

    def print_model(self, out=None):
        """ Compile the model and print the CPO file format in a given output.

        Args:
            out: Target output, stream or file name. Default is sys.stdout.
        """
        # Check file name
        if out is None:
            out = sys.stdout
        if isinstance(out, str):
            with open(os.path.abspath(out), 'w') as f:
                self._write_model(f)
        else:
            self._write_model(out)

    def get_as_string(self):
        """ Compile the model in CPO file format into a string

        Returns:
            String containing the model
        """
        # Print the model into a string
        out = StringIO()
        self._write_model(out)
        res = out.getvalue()
        out.close()
        return res

    def _write_model(self, out):
        """ Compile the model

        Args:
            out: target output
        """
        # Expand model expressions if not done
        model = self.model
        self.exprset = set()
        self.last_loc = None

        # Write header
        banner = "/" * 79 + "\n"
        sfile = model.get_source_file()
        out.write(banner)
        out.write("// CPO file generated from:\n")
        out.write("// " + sfile + "\n")
        out.write(banner)

        # Write variables
        out.write("\n//--- Variables ---\n")
        vlist = model.get_all_variables()
        for v in self._expand_expressions(vlist):
            self._write_expression(out, v)

        # If aliases are requested, print as comment list of aliases
        mnl = self.alias_min_name_length
        if mnl is not None:
            # Preload string map with aliases when relevant
            aliasfound = False
            alias_gen = IdAllocator('_A_', "0123456789abcdefghijklmnopqrstuvwxyz")
            for v in model.get_all_variables():
                if isinstance(v, CpoStateFunction):
                    continue
                # Compute CPO printable variable name
                vname = v.get_name()
                vpname = strname = to_printable_string(vname)
                if len(vpname) > mnl:
                    # Replace by an alias
                    vpname = alias_gen.allocate()
                    # Trace alias
                    if not aliasfound:
                        aliasfound = True
                        out.write("\n//--- Aliases ---\n")
                        out.write("// To reduce CPO file size, the following aliases have been used to replace variable names longer than " + str(mnl) + "\n")
                    out.write(vpname + " = " + strname + ";\n")
                self.id_strings[vname] = vpname

        # Write expressions
        out.write("\n//--- Expressions ---\n")
        self.last_loc = None
        lexpr = model.get_expressions()
        for x in self._expand_expressions(lexpr):
            self._write_expression(out, x)

        # Write search phases 
        phases = model.get_search_phases()
        if phases:
            out.write("\n//--- Search phases ---\n")
            out.write("search {\n")
            for x in self._expand_expressions(phases):
                self._write_expression(out, x)
            out.write("}\n")

        # Write starting point
        spoint = model.get_starting_point()
        if spoint is not None:
            out.write("\n//--- Starting point ---\n")
            if self.last_loc is not None:
                out.write("#line off\n")
            out.write("startingPoint {\n")
            for var in spoint.get_all_var_solutions():
                self._write_starting_point(out, var)
            out.write("}\n")

        # Write parameters
        out.write("\n//--- Parameters ---\n")
        if self.params and (len(self.params) > 0):
            if self.last_loc is not None:
                out.write("#line off\n")
            out.write("parameters {\n")
            for k in sorted(self.params.keys()):
                v = self.params[k]
                if v is not None:
                    out.write("   " + k + " = " + str(v) + ";\n")
            out.write("}\n")
        else:
            out.write("// None\n")

    def _write_expression(self, out, xnode):
        """ Write model expression

        Args:
            out:    Target output
            xnode:  Expression node (expr, loc, root)
        """
        # Trace location if required
        expr, loc, root = xnode
        lloc = self.last_loc
        if self.sourceloc and (loc is not None) and (loc != lloc):
            (file, line) = loc
            out.write("#line ")
            out.write(str(line))
            if (lloc is None) or (file != lloc[0]):
                out.write(' "')
                out.write(file)
                out.write('"')
            out.write("\n")
            self.last_loc = loc

        # Write expression
        id = expr.get_name()
        if id is not None:
            wid = self._get_id_string(id)
            out.write(wid)
            out.write(" = ")
        out.write(self._compile_expression(expr))
        out.write(";\n")
        if root and id is not None:
            out.write(wid)
            out.write(";\n")

    def _write_starting_point(self, out, var):
        """ Write a starting point variable

        Args:
            out:  Target output
            var:  Variable solution
        """
        # Build starting point declaration
        cout = []
        if isinstance(var, CpoIntVarSolution):
            self._compile_int_var_starting_point(var, cout)
        elif isinstance(var, CpoIntervalVarSolution):
            self._compile_interval_var_starting_point(var, cout)
        else:
            raise CpoException("Internal error: unsupported starting point variable: " + str(var))
        # Write variable starting point
        out.write(self._get_id_string(var.get_name()))
        out.write(" = ")
        out.write(''.join(cout))
        out.write(";\n")

    def _get_id_string(self, id):
        """ Get the string representing an identifier

        Args:
            id: Identifier name
        Returns:
            CPO identifier string, including double quotes and escape sequences if needed if not only chars and integers
        """
        # Check if already converted
        res = self.id_strings.get(id, None)
        if res is None:
            # Convert id into string and store result for next call
            res = to_printable_string(id)
            self.id_strings[id] = res
        return res

    def _compile_expression(self, expr, root=True):
        """ Compile an expression in a string in CPO format

        Args:
            expr: Expression to compile
            root: Root expression indicator
        Returns:
            String representing this expression in CPO format
        """
        # Initialize working variables
        cout = []  # Result list of strings
        estack = [[expr, -1]]  # Expression stack [Expression, child index]

        # Loop while expression stack is not empty
        while estack:
            # Get expression to compile
            edscr = estack[-1]
            e = edscr[0]

            # Check if expression is named and not root (named expression and variable)
            if (not root or (e is not expr)) and e.has_name():
                cout.append(self._get_id_string(e.get_name()))
                estack.pop()
                continue

            # Check constant expressions
            t = e.get_type()
            if t.is_constant():
                estack.pop()
                if t.is_array():
                    vals = e.get_value()
                    if len(vals) == 0:
                        cout.append("intArray[]")
                    else:
                        cout.append('[')
                        cout.append(', '.join(str(v) for v in vals))
                        cout.append(']')
                elif (t == Type_Bool):
                    cout.append("true()" if e.get_value() else "false()")
                elif (t == Type_TransitionMatrix):
                    self._compile_transition_matrix(e, cout)
                elif (t == Type_TupleSet):
                    self._compile_tuple_set(e, cout)
                elif (t == Type_StepFunction):
                    self._compile_step_function(e, cout)
                elif (t == Type_SegmentedFunction):
                    self._compile_segmented_function(e, cout)
                else:
                    cout.append(str(e.get_value()))

            # Check variables
            elif t.is_variable():
                estack.pop()
                if (t == Type_IntVar):
                    self._compile_int_var(e, cout)
                elif (t == Type_IntervalVar):
                    self._compile_interval_var(e, cout)
                elif (t == Type_SequenceVar):
                    self._compile_sequence_var(e, cout)
                elif (t == Type_StateFunction):
                    self._compile_state_function(e, cout)

            # Check expression array
            elif (t.is_array()):
                oprnds = e._get_children()
                cnx = edscr[1]
                if (cnx < 0):
                    cout.append("[")
                cnx += 1
                if (cnx >= len(oprnds)):
                    cout.append("]")
                    estack.pop()
                else:
                    edscr[1] = cnx
                    if (cnx > 0):
                        cout.append(", ")
                    estack.append([oprnds[cnx], -1])

            else:
                # Get signature
                sign = e.get_signature()
                if (sign is None):
                    cout.append(e.name)
                    estack.pop()
                    continue

                # Get operation elements
                oper = sign.operation
                prio = oper.priority
                oprnds = e.get_operands()
                cnx = edscr[1]

                # Check if function call
                if (prio < 0):
                    # Check first call
                    if (cnx < 0):
                        cout.append(oper.keyword)
                        cout.append("(")
                    cnx += 1
                    if (oprnds is None) or (cnx >= len(oprnds)):
                        cout.append(")")
                        estack.pop()
                    else:
                        edscr[1] = cnx
                        if (cnx > 0):
                            cout.append(", ")
                        estack.append([oprnds[cnx], -1])

                else:
                    # Check parenthesis required
                    parents = False
                    if (len(estack) > 1):
                        oprio = estack[-2][0].get_priority()
                        if (oprio >= 0):
                            parents = (prio > oprio) or (prio > 5)

                    # Write operation
                    if (cnx < 0):
                        if parents:
                            cout.append("(")
                        if (oprnds is None) or (len(oprnds) == 1):
                            cout.append(oper.keyword)
                    cnx += 1
                    if (oprnds is None) or (cnx >= len(oprnds)):
                        if parents:
                            cout.append(")")
                        estack.pop()
                    else:
                        edscr[1] = cnx
                        if (cnx > 0):
                            cout.append(" ")
                            cout.append(oper.keyword)
                            cout.append(" ")
                        estack.append([oprnds[cnx], -1])

        # Check output exists
        if not cout:
            raise CpoException("Internal error: unable to compile expression: " + str(expr))
        return ''.join(cout)

    def _compile_int_var(self, v, cout):
        """ Compile a IntVar in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        cout.append("intVar(")
        self._compile_var_domain(v.get_domain(), cout)
        cout.append(")")

    def _compile_int_var_starting_point(self, v, cout):
        """ Compile a starting point IntVar in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        cout.append("(")
        self._compile_var_domain(v.get_value(), cout)
        cout.append(")")

    def _compile_interval_var(self, v, cout):
        """ Compile a IntervalVar in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        cout.append("intervalVar(")
        if v.is_absent():
            cout.append("absent")
        elif v.is_present():
            cout.append("present")
        else:
            cout.append("optional")
        if (v.start != DEFAULT_INTERVAL):
            cout.append(", start=")
            cout.append(_build_interval_var_domain_string(v.start))
        if (v.end != DEFAULT_INTERVAL):
            cout.append(", end=")
            cout.append(_build_interval_var_domain_string(v.end))
        if (v.length != DEFAULT_INTERVAL):
            cout.append(", length=")
            cout.append(_build_interval_var_domain_string(v.length))
        if (v.size != DEFAULT_INTERVAL):
            cout.append(", size=")
            cout.append(_build_interval_var_domain_string(v.size))
        if (v.intensity is not None):
            cout.append(", intensity=")
            cout.append(self._compile_expression(v.intensity, root=False))
        if (v.granularity is not None):
            cout.append(", granularity=")
            cout.append(str(v.granularity))
        cout.append(")")

    def _compile_interval_var_starting_point(self, v, cout):
        """ Compile a starting IntervalVar in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        if v.is_absent():
            cout.append("absent")
            return
        cout.append("(")
        cout.append("present" if v.is_present() else "optional")
        rng = v.get_start()
        if rng is not None:
            cout.append(", start=")
            self._compile_var_domain([rng], cout)
        rng = v.get_end()
        if rng is not None:
            cout.append(", end=")
            self._compile_var_domain([rng], cout)
        rng = v.get_size()
        if rng is not None:
            cout.append(", size=")
            self._compile_var_domain([rng], cout)
        cout.append(")")

    def _compile_sequence_var(self, sv, cout):
        """ Compile a SequenceVar in a string in CPO format
        Args:
            sv:   Sequence variable
            cout: Output string list
        """
        cout.append("sequenceVar(")
        cout.append("[" + ", ".join(self._get_id_string(v.get_name()) for v in sv.get_interval_variables()) + "]")
        types = sv.get_types()
        if (types is not None):
            cout.append(", [" + ", ".join(str(t) for t in types) + "]")
        cout.append(")")

    def _compile_state_function(self, stfct, cout):
        """ Compile a State in a string in CPO format

        Args:
           stfct: Segmented function
           cout:  Output string list
        """
        cout.append("stateFunction(")
        cout.append(self._compile_expression(stfct.get_transition_matrix(), root=False))
        cout.append(")")

    def _compile_transition_matrix(self, tm, cout):
        """ Compile a TransitionMatrix in a string in CPO format

        Args:
            tm:   Transition matrix
            cout: Output string list
        """
        cout.append("transitionMatrix(")
        cout.append(", ".join(str(v) for v in tm.get_matrix()))
        cout.append(")")

    def _compile_tuple_set(self, tplset, cout):
        """ Compile a TupleSet in a string in CPO format

        Args:
           tplset: Tuple set
           cout:   Output string list
        """
        cout.append("[")
        for i, tpl in enumerate(tplset.get_tuple_set()):
            if i > 0:
                cout.append(", ")
            cout.append("[")
            self._compile_list_of_integers(tpl, cout)
            cout.append("]")
        cout.append("]")

    def _compile_var_domain(self, dom, cout):
        """ Compile a variable domain in CPO format

        Args:
            dom:   Variable domain
            cout:  Output string list
        """
        if is_array(dom):
            for i, d in enumerate(dom):
                if i > 0:
                    cout.append(", ")
                if (isinstance(d, (list, tuple))):
                    cout.append(_build_int_var_domain_string(d))
                else:
                    cout.append(str(d))
        else:
            cout.append(str(dom))

    def _compile_list_of_integers(self, lint, cout):
        """ Compile a list of integers in CPO format

        Args:
            lint:  List of integers
            cout:  Output string list
        """
        llen = len(lint)
        i = 0
        while i < llen:
            if i > 0:
                cout.append(", ")
            j = i + 1
            while (j < llen) and (lint[j] == lint[j - 1] + 1):
                j += 1
            if (j > i + 1):
                cout.append(str(lint[i]) + ".." + str(lint[j - 1]))
            else:
                cout.append(str(lint[i]))
            i = j

    def _compile_step_function(self, stfct, cout):
        """ Compile a StepFunction in a string in CPO format

        Args:
           stfct: Step function
           cout:  Output string list
        """
        cout.append("stepFunction(")
        cout.append(", ".join(map(to_string, stfct.get_step_list())))
        cout.append(")")

    def _compile_segmented_function(self, sgfct, cout):
        """ Compile a SegmentedFunction in a string in CPO format

        Args:
           sgfct: Segmented function
           cout:  Output string list
        """
        cout.append("segmentedFunction(")
        cout.append(", ".join(map(to_string, sgfct.get_segment_list())))
        cout.append(")")

    def _expand_expressions(self, lexpr):
        """ Scan a list of expressions and extract named expression before usage.

        Expressions may be named if:
         * used multiple times,
         * explicitly named by end-user

        Args:
            lexpr:  List of expressions
        Returns:
            New list of expressions with named sub-expressions placed before expressions
        """
        # Initialize processing
        nlexpr = []  # New list of expressions
        exprset = self.exprset  # Set of named expressions already compiled

        # Scan all expressions
        for v in lexpr:
            if isinstance(v, CpoExpr):
                expr = v
                loc = None
                root = False
            else:
                (expr, loc, root) = v
            # Get all identified sub-expressions in the expression
            lsexpr = reversed(_get_id_sub_expressions(expr))
            # Add them to result if not already in
            for se in lsexpr:
                eid = id(se)
                if (eid not in exprset):
                    nlexpr.append((se, loc, root if se is expr else False))
                    exprset.add(eid)
            # Add initial expression if not named (processed above)
            if expr.is_variable() or not (expr.has_name()):
                nlexpr.append((expr, loc, root))

        # Return new list of expressions
        return nlexpr


###############################################################################
## Public functions
###############################################################################

def get_cpo_model(model, **kwargs):
    """ Convert a model into a string with CPO file format.

        Args:
            model:  Source model
        Optional args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
    Returns:
        String of the model in CPO file format
    """
    cplr = CpoCompiler(model, **kwargs)
    return cplr.get_as_string()


###############################################################################
## Private functions
###############################################################################

def _build_int_var_domain_bound_string(ibv):
    """ Build the string representing an integer variable domain bound

    This methods checks for special values INT_MIN and INT_MAX.

    Args:
        ibv: Interval bound value
    Returns:
        String representation of the interval
    """
    if (ibv == INT_MIN):
        return ("intmin")
    elif (ibv == INT_MAX):
        return ("intmax")
    else:
        return str(ibv)


def _build_int_var_domain_string(intv):
    """ Build the string representing an interval domain

    Args:
       intv: Domain interval (list or tuple of 2 integers)
    Returns:
        String representation of the interval
    """
    return _build_int_var_domain_bound_string(intv[0]) + ".." + _build_int_var_domain_bound_string(intv[1])


def _build_interval_var_domain_bound_string(ibv):
    """ Build the string representing an interval variable domain bound

    This methods checks for special values INTERVAL_MIN and INTERVAL_MAX.

    Args:
        ibv: Interval bound value
    Returns:
        String representation of the interval
    """
    if (ibv == INTERVAL_MIN):
        return ("intervalmin")
    elif (ibv == INTERVAL_MAX):
        return ("intervalmax")
    else:
        return str(ibv)


def _build_interval_var_domain_string(intv):
    """ Build the string representing an interval_var domain

    Args:
        intv: Domain interval
    Returns:
        String representation of the domain
    """
    smn = intv[0]
    smx = intv[1]
    if (smn == smx):
        return _build_interval_var_domain_bound_string(smn)
    return _build_interval_var_domain_bound_string(smn) + ".." + _build_interval_var_domain_bound_string(smx)


def _get_id_sub_expressions(expr):
    """ Build a list of all identifiable sub-expressions:
     * referenced more than once in an expression
     * or associated to a name

    Args:
        expr: Expression to scan
    Returns:
        List of identifiable sub-expressions
    """
    lexpr = []  # Result list of expressions
    estack = [expr]
    # Loop while expression stack is not empty
    while estack:
        # Get expression to compile
        e = estack.pop()
        # Check if expression is CPO
        if isinstance(e, CpoExpr):
            # Check if expression is named
            if e.has_name() and not (e.is_variable()):
                lexpr.append(e)
            # Stack children expressions
            chldrn = e._get_children()
            if chldrn is not None:
                estack.extend(chldrn)

    # Return list of sub-expressions
    return lexpr
