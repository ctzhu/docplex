# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Parser converting a FZN file to internal model representation.

This parser does not support the complete set of predicates described in the specifications of FlatZinc
that can be found here: http://www.minizinc.org/downloads/doc-1.6/flatzinc-spec.pdf

Basically, it supports essentially integer expressions, some floating point expressions and custom
predicates related to scheduling.

The predicates that are supported are:

 * *array predicates*

   array_bool_and, array_bool_element, array_bool_or, array_bool_xor,
   array_float_element, array_int_element, array_set_element,
   array_var_bool_element, array_var_float_element, array_var_int_element, array_var_set_element.

 * *boolean predicates*

   bool2int, bool_and, bool_clause, bool_eq, bool_eq_reif, bool_le, bool_le_reif,
   bool_lin_eq, bool_lin_le, bool_lt, bool_lt_reif, bool_not, bool_or, bool_xor.

 * *integer predicates*

   int_abs, int_div, int_eq, int_eq_reif, int_le, int_le_reif, int_lin_eq, int_lin_eq_reif,
   int_lin_le, int_lin_le_reif, int_lin_ne, int_lin_ne_reif, int_lt, int_lt_reif, int_max, int_min,
   int_mod, int_ne, int_ne_reif, int_plus, int_times, int2float.

 * *float predicates*

   float_abs, float_exp, float_ln, float_log10, float_log2, float_sqrt, float_eq, float_eq_reif,
   float_le, float_le_reif, float_lin_eq, float_lin_eq_reif, float_lin_le, float_lin_le_reif, float_lin_lt,
   float_lin_lt_reif, float_lin_ne, float_lin_ne_reif, float_lt, float_lt_reif, float_max, float_min,
   float_ne, float_ne_reif, float_plus.

 * *set predicates*

   set_in, set_in_reif.

 * *custom predicates*

   all_different_int, subcircuit, count_eq_const, table_int, inverse,
   lex_lesseq_bool, lex_less_bool, lex_lesseq_int, lex_less_int, int_pow, cumulative


Detailed description
--------------------
"""

from docplex.cp.fzn_tokenizer import *
from docplex.cp.expression import *
from docplex.cp.solution import *
from docplex.cp.model import CpoModel
import docplex.cp.modeler as modeler
import docplex.cp.config as config
import docplex.cp.expression as expression
import collections


###############################################################################
## Constants
###############################################################################

# Mapping of FlatZinc predicates to CPO expressions expressed as a list of couples
# Key is the name or list of names list of names of FZN predicates,
# Value is a lambda expression taking predicate arguments and returning CPO expression
PREDICATES_MAP = \
{
    # Basic predicates
    'array_bool_and': lambda t, r: modeler.equal(r, modeler.min_of(t)),
    'array_bool_element': lambda x, t, r: modeler.equal(r, modeler.element(t, x - 1)),
    'array_bool_or': lambda t, r: modeler.equal(r, modeler.max_of(t)),
    'array_bool_xor': None,
    'array_float_element': lambda x, t, r: modeler.equal(r, modeler.element(t, x - 1)),
    'array_int_element': lambda x, t, r: modeler.equal(r, modeler.element(t, x - 1)),
    'array_set_element': None,
    'array_var_bool_element': lambda x, t, r: modeler.equal(r, modeler.element(t, x - 1)),
    'array_var_float_element': lambda x, t, r: modeler.equal(r, modeler.element(t, x - 1)),
    'array_var_int_element': lambda x, t, r: modeler.equal(r, modeler.element(t, x - 1)),
    'array_var_set_element': None,
    #'bool2int': lambda a, b: a == b,
    'bool_and': lambda a, b: modeler.logical_and(a, b),
    #'bool_clause': lambda a, b: (modeler.max_of(a) > 0) | (modeler.min_of(b) == 0),
    'bool_clause': lambda a, b: _bool_clause(a, b),
    'bool_eq': lambda a, b: a == b,
    'bool_eq_reif': lambda a, b, r: modeler.equal(r, (a == b)),
    'bool_le': lambda a, b: a <= b,
    'bool_le_reif': lambda a, b, r: modeler.equal(r, (a <= b)),
    'bool_lin_eq': lambda a, b, r: _scal_prod(a, b, r, modeler.equal),
    'bool_lin_le': lambda a, b, r: _scal_prod(a, b, r, modeler.less_or_equal),
    'bool_lt': lambda a, b: a < b,
    'bool_lt_reif': lambda a, b, r: r == (a < b),
    'bool_not': lambda a, b: b == modeler.logical_not(a),
    'bool_or': lambda a, b, r: r == modeler.logical_or(a, b),
    'bool_xor': lambda a, b, r: modeler.equal(r, (a != b)),
    'float_abs': lambda a, r: modeler.equal(r, modeler.abs_of(a)),
    'float_acos': None,
    'float_asin': None,
    'float_atan': None,
    'float_cos': None,
    'float_cosh': None,
    'float_exp': lambda a, r: modeler.equal(r, modeler.exponent(a)),
    'float_ln': lambda a, r: modeler.equal(r, modeler.log(a)),
    'float_log10': lambda a, r: modeler.equal(r, (modeler.log(a) / modeler.log(10))),
    'float_log2': lambda a, r: modeler.equal(r, (modeler.log(a) / modeler.log(2))),
    'float_sqrt': lambda a, r: modeler.equal(r, modeler.power(a, 0.5)),
    'float_sin': None,
    'float_sinh': None,
    'float_tan': None,
    'float_tanh': None,
    'float_eq': lambda a, b: a == b,
    'float_eq_reif': lambda a, b, r: modeler.equal(r, (a == b)),
    'float_le': lambda a, b: a <= b,
    'float_le_reif': lambda a, b, r: modeler.equal(r, (a <= b)),
    'float_lin_eq': lambda a, b, r: _scal_prod(a, b, r, modeler.equal),
    'float_lin_eq_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.equal),
    'float_lin_le': lambda a, b, r: _scal_prod(a, b, r, modeler.less_or_equal),
    'float_lin_le_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.less_or_equal),
    'float_lin_lt': lambda a, b, r: _scal_prod(a, b, r, modeler.less),
    'float_lin_lt_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.less),
    'float_lin_ne': lambda a, b, r: _scal_prod(a, b, r, modeler.diff),
    'float_lin_ne_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.diff),
    'float_lt': lambda a, b: a < b,
    'float_lt_reif': lambda a, b, r: r == (a < b),
    'float_max': lambda a, b, r: modeler.equal(r, modeler.max_of(a, b)),
    'float_min': lambda a, b, r: modeler.equal(r, modeler.min_of(a, b)),
    'float_ne': lambda a, b: a != b,
    'float_ne_reif': lambda a, b, r: modeler.equal(r, (a != b)),
    'float_plus': lambda a, b, r: modeler.equal(r, (a + b)),
    'int_abs': lambda a, r: r == modeler.abs_of(a),
    'int_div': lambda a, b, r: modeler.equal(r, (a // b)),
    'int_eq': lambda a, b: a == b,
    'int_eq_reif': lambda a, b, r: modeler.equal(r, (a == b)),
    'int_le': lambda a, b: a <= b,
    'int_le_reif': lambda a, b, r: modeler.equal(r, (a <= b)),
    #'int_lin_eq': lambda a, b, r: _scal_prod(a, b, r, modeler.equal),
    'int_lin_eq_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.equal),
    'int_lin_le': lambda a, b, r: _scal_prod(a, b, r, modeler.less_or_equal),
    'int_lin_le_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.less_or_equal),
    'int_lin_ne': lambda a, b, r: _scal_prod(a, b, r, modeler.diff),
    'int_lin_ne_reif': lambda a, b, r, d: d == _scal_prod(a, b, r, modeler.diff),
    'int_lt': lambda a, b: a < b,
    'int_lt_reif': lambda a, b, r: modeler.equal(r, (a < b)),
    'int_max': lambda a, b, r: modeler.equal(r, modeler.max_of(a, b)),
    'int_min': lambda a, b, r: modeler.equal(r, modeler.min_of(a, b)),
    'int_mod': lambda a, b, r: modeler.equal(r, (a % b)),
    'int_ne': lambda a, b: a != b,
    'int_ne_reif': lambda a, b, r: modeler.equal(r, (a != b)),
    'int_plus': lambda a, b, r: modeler.equal(r, (a + b)),
    'int_times': lambda a, b, r: modeler.equal(r, (a * b)),
    'int2float': lambda a, b: modeler.equal(a, b),
    'set_card': None,
    'set_diff': None,
    'set_eq': None,
    'set_eq_reif': None,
    'set_in': lambda a, b: modeler.allowed_assignments(a, b),
    'set_in_reif': lambda a, b, r: modeler.equal(r, modeler.allowed_assignments(a, b)),
    'set_intersect': None,
    'set_le': None,
    'set_lt': None,
    'set_ne': None,
    'set_ne_reif': None,
    'set_subset': None,
    'set_subset_reif': None,
    'set_symdiff': None,
    'set_union': None,

    # Custom predicates
    'all_different_int': lambda a: modeler.all_diff(a),
    'subcircuit': lambda t: _subcircuit(t),
    'count_eq_const': lambda t, v, r: r == modeler.count(t, v),
    'table_int': lambda t, a: _table_int(t, a),
    'inverse': lambda a, b: _inverse(a, b),
    'lex_lesseq_int': lambda a, b: modeler.lexicographic(a, b),
    'lex_lesseq_bool': lambda a, b: modeler.lexicographic(a, b),
    'lex_less_int': lambda a, b: _lex_less_int(a, b),
    'lex_less_bool': lambda a, b: _lex_less_int(a, b),
    'int_pow': lambda a, b, r: r == (a ** b),
}

# Dictionary of predicates implemented as parser methods
# Key is name of the predicate, Value is name of the CpoParser function that implements the predicate.
SELF_PREDICATES_MAP = \
{
    'cumulative': '_pred_cumulative',
    'bool2int':   '_pred_bool2int',
    'int_lin_eq': '_pred_int_lin_eq'
}


###############################################################################
## Public classes
###############################################################################

class FznParserException(CpoException):
    """ The base class for exceptions raised by the CPO parser
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(FznParserException, self).__init__(msg)

# Parameter descriptor
FznParameter = collections.namedtuple('FznParameter', ('name',   # Variable name
                                                       'type',   # Variable type (string)
                                                       'size',   # Array size (if array), None for variable
                                                       'value',  # Value
                                                       ))

# Variable descriptor
FznVariable = collections.namedtuple('FznVariable', ('name',       # Variable name
                                                     'domain',     # Domain
                                                     'introduced', # Introduced indicator
                                                     'size',       # Array size (if array), None for variable
                                                     'value',      # Value
                                                     ))

# Constraint descriptor
FznConstraint = collections.namedtuple('FznConstraint', ('predicate',   # Name of the predicate
                                                         'args',        # Arguments
                                                         'defvar',      # Name of the variable defined by this constraint
                                                         'annotations', # Annotations
                                                         ))

# Constraint descriptor
FznObjective = collections.namedtuple('FznObjective', ('operation',   # Objective operation in 'satisfy', 'minimize', 'maximize'
                                                       'expr',        # Target expression
                                                       'annotations', # Annotations
                                                       ))


class FznReader(object):
    """ Reader of FZN file format """
    __slots__ = ('source_file',  # Source file
                 'tokenizer',    # Reading tokenizer
                 'token',        # Last read token

                 'parameters',   # List of parameters
                 'variables',    # List of variables
                 'constraints',  # List of model constraints
                 'objective',    # Model objective
                 )

    def __init__(self):
        """ Create a new FZN reader
        """
        super(FznReader, self).__init__()
        self.source_file = None
        self.tokenizer = None
        self.token = None
        self.parameters = []
        self.variables = []
        self.constraints = []
        self.objective = None


    def parse(self, cfile):
        """ Parse a FZN file

        Args:
            cfile: FZN file to read
        Raises:
            FznParserException: Parsing exception
        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        """
        # Store file name if first file
        self.source_file = cfile
        with open_utf8(cfile, mode='r') as f:
            self.tokenizer = FznTokenizer(cfile, f)
            self._read_document()
            self.tokenizer = None


    def parse_string(self, str):
        """ Parse a string

        Result of the parsing is added to the current result model.

        Args:
            str: String to parse
        """
        self.tokenizer = FznTokenizer("String", str)
        self._read_document()
        self.tokenizer = None


    def write(self, out=None):
        """ Write the model.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional): Target output stream or file name. If not given, default value is sys.stdout.
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        # Write model content
        for x in self.parameters:
            print(str(x))
        for x in self.variables:
            print(str(x))
        for x in self.constraints:
            print(str(x))
        out.flush()


    def _read_document(self):
        """ Read all FZN document
        """
        try:
            self._next_token()
            while self._read_predicate():
                pass
            while self._read_parameter_or_variable():
                pass
            while self._read_constraint():
                pass
            self._read_objective()


        except Exception as e:
            import traceback
            traceback.print_exc()
            self._raise_exception(str(e))

        if self.token != TOKEN_EOF:
            self._raise_exception("Unexpected token '{}'".format(self.token))


    def _read_predicate(self):
        """ Read a predicate declaration

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a predicate has been read, False if nothing to process
        """
        if self.token.value != "predicate":
            return False

        # Read predicate declaration
        while self.token not in (TOKEN_SEMICOLON, TOKEN_EOF):
            self._next_token()
        if self.token != TOKEN_SEMICOLON:
            self._raise_exception("Semicolon ';' expected at the end of a predicate declaration.")
        self._next_token()
        return True


    def _read_parameter_or_variable(self):
        """ Read a parameter or variable declaration

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a parameter has been read, False if nothing to process
        """
        tok = self.token
        if tok.type != TOKEN_SYMBOL:
            return False

        # Read array size if any
        arsize = self._read_array_size()

        # Check if variable declaration
        tok = self.token
        if tok.value == 'var':
            self._next_token()
            return self._read_variable(arsize)

        # Check type name
        if tok.value not in ('bool', 'float', 'int', 'set'):
            return False
        typ = tok.value
        if typ == 'set':
            self._check_token(self._next_token(), TOKEN_SYMBOL_OF)
            self._check_token(self._next_token(), TOKEN_SYMBOL_INT)

        # Check separating colon
        self._check_token(self._next_token(), TOKEN_COLON)

        # Check parameter name
        tok = self._next_token()
        if tok.type != TOKEN_SYMBOL:
            self._raise_exception("Symbol expected as parameter name.")
        pid = tok.value
        self._check_token(self._next_token(), TOKEN_EQUAL)

        # Read expression
        self._next_token()
        expr = self._read_expression()
        if typ == 'set':
            arsize = len(expr)

        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Build result
        self.parameters.append(FznParameter(pid, typ, arsize, expr))

        return True


    def _read_variable(self, arsize):
        """ Read a variable declaration

        This function is called with first token already read and terminates with next token already read.

        Args:
            arsize:  Array size if any
        Returns:
            True if a variable has been read, False if nothing to process
        """
        # Read domain
        dom = self._read_var_domain()
        self._check_token(self.token, TOKEN_COLON)
        tok = self._next_token()
        if tok.type != TOKEN_SYMBOL:
            self._raise_exception("Symbol expected as variable name.")
        vid = tok.value
        self._next_token()

        # Check annotations
        annotations = self._read_annotations()
        # print("Annotations: {}".format(annotations))
        is_introduced = 'var_is_introduced' in annotations

        # Check expression
        expr = None
        if self.token == TOKEN_EQUAL:
            self._next_token()
            expr = self._read_expression()
            if expr is True:
                dom = (1,)
            elif expr is False:
                dom = (0,)
            elif is_int(expr):
                dom = (expr,)

        # Read ending semicolon
        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Create variable
        self.variables.append(FznVariable(vid, dom, is_introduced, arsize, expr))

        return True


    def _read_constraint(self):
        """ Read a constraint

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a variable has been read, False if nothing to process
        """
        # Check constraint token
        if self.token != TOKEN_SYMBOL_CONSTRAINT:
            return False

        # Read constraint name
        tok = self._next_token()
        if tok.type != TOKEN_SYMBOL:
            self._raise_exception("Constraint name '{}' should be a symbol.".format(tok))
        cname = tok.value

        # Read parameters
        args = []
        self._check_token(self._next_token(), TOKEN_PARENT_OPEN)
        self._next_token()
        while self.token != TOKEN_PARENT_CLOSE:
            args.append(self._read_expression())
            if self.token is TOKEN_COMMA:
                self._next_token()
        self._next_token()

        # Check annotations
        annotations = self._read_annotations()
        defvar = annotations.get('defines_var', (None,))[0]

        # Read ending semicolon
        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Store constraint
        self.constraints.append(FznConstraint(cname, args, defvar, annotations))

        return True


    def _read_objective(self):
        """ Read solve objective

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a variable has been read, False if nothing to process
        """
        # Check constraint token
        if self.token != TOKEN_SYMBOL_SOLVE:
            return False
        self._next_token()

        # Check annotations
        annotations = self._read_annotations()

        # Read solve objective
        tok = self.token
        if (tok.type != TOKEN_SYMBOL) or (tok.value not in ('satisfy', 'minimize', 'maximize')):
            self._raise_exception(
                "Solve objective '{}' should be a symbol in 'satisfy', 'minimize', 'maximize'.".format(tok))
        obj = tok.value
        self._next_token()

        # Read expression if any
        expr = None if obj == 'satisfy' else self._read_expression()

        # Read ending semicolon
        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Store objective
        self.objective = FznObjective(obj, expr, annotations)

        return True


    def _read_expression(self):
        """ Read an expression

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Expression that has been read
        """
        tok = self.token
        self._next_token()

        # Check int constant
        if tok.type == TOKEN_INTEGER:
            v1 = int(tok.value)
            # Check set const
            if self.token == TOKEN_INTERVAL:
                tok2 = self._next_token()
                if tok2.type != TOKEN_INTEGER:
                    self._raise_exception("Set upper bound {} should be an integer constant.".format(tok2))
                self._next_token()
                v2 = int(tok2.value)
                return (v1,) if v1 == v2 else (v1, v2)
            else:
                return v1

        # Check float constant
        if tok.type == TOKEN_FLOAT:
            return float(tok.value)

        # Set of integer constant
        if tok == TOKEN_BRACE_OPEN:
            lints = []
            tok = self.token
            while tok != TOKEN_BRACE_CLOSE:
                if tok.type != TOKEN_INTEGER:
                    self._raise_exception("Set element {} should be an integer constant.".format(tok))
                lints.append(int(tok.value))
                tok = self._next_token()
                if tok == TOKEN_COMMA:
                    tok = self._next_token()
            self._next_token()
            return lints

        # Check symbols
        if tok.type == TOKEN_SYMBOL:
            sid = tok.value
            # Check boolean constant
            if sid == "true":
                return True
            if sid == "false":
                return False
            # Check array access
            if self.token == TOKEN_HOOK_OPEN:
                tok2 = self._next_token()
                if tok2.type != TOKEN_INTEGER:
                    self._raise_exception("Array index '{}' should be an integer constant.".format(tok2))
                self._check_token(self._next_token(), TOKEN_HOOK_CLOSE)
                self._next_token()
                # Build array access as a tuple (arr_name, index)
                return sid, int(tok2.value)
            # Check annotation function call
            elif self.token == TOKEN_PARENT_OPEN:
                lexprs = [sid]
                self._next_token()
                while self.token != TOKEN_PARENT_CLOSE:
                    lexprs.append(self._read_expression())
                    if self.token == TOKEN_COMMA:
                        self._next_token()
                self._next_token()
                return tuple(lexprs)
            else:
                return sid

        # Array of expressions
        if tok == TOKEN_HOOK_OPEN:
            lexprs = []
            while self.token != TOKEN_HOOK_CLOSE:
                lexprs.append(self._read_expression())
                if self.token == TOKEN_COMMA:
                    self._next_token()
            self._next_token()
            return lexprs

        # Unknown
        self._raise_exception("Invalid expression start: '{}'.".format(tok))


    def _read_array_size(self):
        """ Read an array size declaration

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Array size as int if given,
            -1 if size is not precised,
            None if no array specified
        """
        # Check array token
        if self.token != TOKEN_SYMBOL_ARRAY:
            return None

        # Read array specs
        self._check_token(self._next_token(), TOKEN_HOOK_OPEN)
        tok = self._next_token()
        if tok == TOKEN_SYMBOL_INT:
            arsize = -1
        else:
            self._check_token(tok, TOKEN_INTEGER_ONE)
            self._check_token(self._next_token(), TOKEN_INTERVAL)
            tok = self._next_token()
            if tok.type != TOKEN_INTEGER:
                self._raise_exception("Array size '{}' should be integer.".format(tok))
            arsize = int(tok.value)
        self._check_token(self._next_token(), TOKEN_HOOK_CLOSE)
        self._check_token(self._next_token(), TOKEN_SYMBOL_OF)
        self._next_token()

        return arsize


    def _read_var_domain(self):
        """ Read the domain of a variable.

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Variable domain
        """
        # Get token
        tok = self.token

        # Check boolean domain
        if tok.value == 'bool':
            self._next_token()
            return BINARY_DOMAIN

        # Check undefined domain
        if tok.value == 'int':
            self._next_token()
            return DEFAULT_INTEGER_VARIABLE_DOMAIN

        # Read set of integers or interval
        if tok == TOKEN_BRACE_OPEN:
            lint = sorted(self._read_expression())
            dom = []
            llen = len(lint)
            i = 0
            while i < llen:
                j = i + 1
                while (j < llen) and (lint[j] == lint[j - 1] + 1):
                    j += 1
                if j > i + 1:
                    dom.append((lint[i], lint[j - 1]))
                else:
                    dom.append(lint[i])
                i = j
            return tuple(dom)

        # Check integer domain
        if tok.type != TOKEN_INTEGER:
            self._raise_exception("Variable domain should start by an integer constant.")
        self._next_token()
        if self.token == TOKEN_INTERVAL:
            tok2 = self._next_token()
            if tok2.type != TOKEN_INTEGER:
                self._raise_exception("Domain upper bound {} should be an integer constant.".format(tok2))
            self._next_token()
            v1 = int(tok.value)
            v2 = int(tok2.value)
            if v1 == v2:
                return v1,
            return ((v1, v2),)

        if tok.type != TOKEN_INTEGER:
            self._raise_exception("Variable domain should end with an integer constant.")
        return (int(tok.value),)


    def _read_annotations(self):
        """ Read a list of annotations

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Dictionary of annotations. Key is name, value is list or parameters.
        """
        result = {}

        # Check annotation start token
        while self.token == TOKEN_DOUBLECOLON:
            # Read annotation name
            anm = self._next_token()
            if anm.type != TOKEN_SYMBOL:
                self._raise_exception("Annotation name '{}' should be a symbol.".format(anm))
            args = []
            tok = self._next_token()
            if tok == TOKEN_PARENT_OPEN:
                self._next_token()
                while self.token != TOKEN_PARENT_CLOSE:
                    args.append(self._read_expression())
                    if self.token == TOKEN_COMMA:
                        self._next_token()
                self._next_token()
            result[anm.value] = tuple(args)

        return result


    def _next_token(self):
        """ Read next token
        Returns:
            Next read token, None if end of input
        """
        self.token = self.tokenizer.next_token()
        # print("Line {}, col {}, tok '{}'".format(self.tokenizer.line_number, self.tokenizer.read_index, self.token))
        return self.token


    def _check_token(self, tok, etok):
        """ Check that a read token is a given one an raise an exception if not
        Args:
            tok: Read token
            etok: Expected token
        """
        if tok != etok:
            self._raise_unexpected_token(etok, tok)


    def _raise_unexpected_token(self, expect=None, tok=None):
        """ Raise a "Unexpected token" exception
        Args:
            tok:  Unexpected token
        """
        if tok is None:
            tok = self.token
        if expect is None:
            self._raise_exception("Unexpected token '" + str(tok) + "'")
        self._raise_exception("Read '" + str(tok) + "' instead of expected '" + str(expect) + "'")


    def _raise_exception(self, msg):
        """ Raise a Parsing exception
        Args:
            msg:  Exception message
        """
        raise FznParserException(self.tokenizer.build_error_string(msg))



class FznParser(object):
    """ Reader of FZN file format """
    __slots__ = ('model',        # Read model
                 'compiled',     # Model compiled indicator
                 'reader',       # FZN reader
                 'cpo_exprs',    # Dictionary of CPO expressions. Key=name, value = CPO expr
                 'predicates',   # Dictionnary of predicates
                 'reduce',       # Reduce model indicator
                 'defined_var',  # Variable defined by the currently compiled constraint
                 'interval_gen', # Name generator for interval var expressions
                 'cumul_gen',    # Name generator for cumul atom expressions
                 )

    def __init__(self, mdl=None):
        """ Create a new CPO format parser

        Args:
            mdl:  Model to fill, None (default) to create a new one.
        """
        super(FznParser, self).__init__()
        self.model = mdl if mdl is not None else CpoModel()
        self.compiled = False
        self.reader = FznReader()
        self.interval_gen = IdAllocator("IntervalVar_")
        self.cumul_gen = IdAllocator("VarCumulAtom_")

        # Do not store location information (would store parser instead of real lines)
        self.model.source_loc = False

        # Initialize predicates map with those implemented in parser
        self.predicates = dict(PREDICATES_MAP)
        for k, v in SELF_PREDICATES_MAP.items():
            self.predicates[k] = self.__getattribute__(v)

        # Set model reduction indicator
        self.reduce = config.context.parser.fzn_reduce


    def get_model(self):
        """ Get the model that have been parsed

        Return:
            CpoModel result of the parsing
        """
        if not self.compiled:
            self.compiled = True
            self._compile_to_model()
        return self.model


    def parse(self, cfile):
        """ Parse a FZN file

        Args:
            cfile: FZN file to read
        Raises:
            FznParserException: Parsing exception
        """
        if self.model.source_file is None:
            self.model.source_file = cfile
        self.reader.parse(cfile)


    def parse_string(self, str):
        """ Parse a string

        Result of the parsing is added to the current result model.

        Args:
            str: String to parse
        """
        self.reader.parse_string(str)


    def _get_cpo_expr_map(self):
        """ For testing, get the map of CPO expressions
        """
        self.get_model()
        return self.cpo_exprs


    def _compile_to_model(self):
        """ Compile FZN model into CPO model
        """
        self.cpo_exprs = {}

        # Compile parameters
        for x in self.reader.parameters:
            self._compile_parameter(x)
        # Compile variables
        for x in self.reader.variables:
            self._compile_variable(x)
        # Compile constraints
        # for k, v in self.cpo_exprs.items():
        #     print("{} -> {}".format(k, id(v)))
        for x in self.reader.constraints:
            self._compile_constraint(x)
        # Compile objective
        self._compile_objective(self.reader.objective)


    def _compile_parameter(self, fp):
        """ Compile a FZN parameter into CPO model
        Args:
            fp: Flatzinc parameter
        """
        if fp.size:
            # Build array
            expr = build_cpo_expr([self._get_cpo_expr(e) for e in fp.value])
        else:
            expr = build_cpo_expr(fp.value)
        expr.set_name(fp.name)
        self.cpo_exprs[fp.name] = expr


    def _compile_variable(self, fv):
        """ Compile a FZN variable into CPO model
        Args:
            fv: Flatzinc variable
        """
        #print("Compile var: {}".format(fv))
        if fv.size:
            # Build array of variables
            if fv.value:
                arr = [self._get_cpo_expr(e) for e in fv.value]
            else:
                arr = []
                for i in range(fv.size):
                    arr.append(integer_var(name=fv.name + '[' + str(i + 1) + ']', domain=fv.domain))
            expr = build_cpo_expr(arr)
        else:
            expr = integer_var(domain=fv.domain)
        expr.set_name(fv.name)
        self.cpo_exprs[fv.name] = expr


    def _compile_constraint(self, fc):
        """ Compile a FZN constraint into CPO model
        Args:
            fv: Flatzinc constraint
        """
        #print("Compile constraint {}".format(fc))
        # Search for corresponding predicate
        pred = self.predicates.get(fc.predicate)
        if pred is None:
            self._raise_exception("Unable to convert predicate '{}'.".format(fc.predicate))
        # Convert arguments
        args = [self._get_cpo_expr(e) for e in fc.args]
        # Call predicate implementation to build corresponding CPO expression
        self.defined_var = None if fc.defvar is None else self._get_cpo_expr(fc.defvar)
        cexpr = pred(*args)
        # Add to model
        self.model.add(cexpr)


    def _compile_objective(self, fo):
        """ Compile a FZN objective into CPO model
        Args:
            fo: Flatzinc objective
        """
        #print("Compile objective {}".format(fo))
        if fo is None:
            return
        if fo.operation != 'satisfy':
            expr = self._get_cpo_expr(fo.expr)
            oxpr = modeler.maximize(expr) if fo.operation == 'maximize' else modeler.minimize(expr)
            self.model.add(oxpr)


    def _get_cpo_expr(self, expr):
        """ retrieve a CPO expression from its FZN representation
        Args:
            expr:  FZN expression
        """
        # Check basic types
        etyp = type(expr)
        if etyp in INTEGER_TYPES:
            return expr
        if etyp in STRING_TYPES:
            v = self.cpo_exprs.get(expr)
            if v is None:
                raise FznParserException("Can not find element {}".format(expr))
            return v

        # Check array access
        if etyp is tuple:
            # Check tuple of integers
            if is_int(expr[0]):
                if len(expr) > 1:
                    return [i for i in range(expr[0], expr[1] + 1)]
                return [expr[0]]
            # Access to array element
            arr = self.cpo_exprs.get(expr[0])
            if arr is None:
                raise FznParserException("Can not find array {}".format(expr[0]))
            return _get_value(arr)[expr[1] - 1]

        # List
        if etyp is list:
            # res = build_cpo_expr([self._get_cpo_expr(x) for x in expr])
            # # For testing purpose, check if each array element is correct
            # for x, n in zip(res.value, expr):
            #     if isinstance(x, CpoExpr) and x.name != n:
            #         print("Cache size: {}".format(expression._CPO_VALUES_FROM_PYTHON.size()))
            #         raise Exception("Array element found with wrong name {} instead of {} in {}".format(x.name, n, expr))
            # return res
            return build_cpo_expr([self._get_cpo_expr(x) for x in expr])

        # Boolean
        if etyp is bool:
            return modeler.true() if expr else modeler.false()

        # Unknown
        raise FznParserException("Can not find element {}".format(expr))


    def _pred_cumulative(self, stime, tdur, rreq, bnd):
        """ Requires that a set of tasks given by start times s, durations d, and resource requirements r,
        never require more than a global resource bound b at any one time.
        Args:
            stime:  Tasks start time
            tdur:   Tasks tasks durations
            rreq:   Task resource requirements
            bnd:    Global resource bound
        """
        # Create interval vars and cumul atoms
        cumul_atoms = []
        for s, d, r in zip(_get_value(stime), _get_value(tdur), _get_value(rreq)):
            vname = None
            # Get start time
            ds = _get_domain_bounds(s)
            if isinstance(s, CpoIntVar):
                vname = s.name
            # Get duration
            dd = _get_domain_bounds(d)
            if vname is None and isinstance(d, CpoIntVar):
                vname = d.name
            # Get requirement
            dr = _get_domain_bounds(r)
            if vname is None and isinstance(r, CpoIntVar):
                vname = r.name

            # Create interval variable
            if vname is None:
                vname = self.interval_gen.allocate()
            else:
                vname = "Itv_" + vname
                if vname in self.cpo_exprs:
                    cnt = 1
                    nname = vname + "@1"
                    while nname in self.cpo_exprs:
                        cnt += 1
                        nname = vname + "@" + str(cnt)
                    vname = nname
            # Create interval variable
            ivar = interval_var(start=ds, end=(INTERVAL_MIN, INTERVAL_MAX), size=dd, name=vname)
            self.cpo_exprs[vname] = ivar

            # Create pulse
            pulse = modeler.pulse(ivar, dr)
            cumul_atoms.append(pulse)
            # Replace previous variable by a start of interval variable
            if isinstance(s, CpoIntVar):
                self.model.add(s == modeler.start_of(ivar))
            if isinstance(d, CpoIntVar):
                self.model.add(d == modeler.size_of(ivar))
            if isinstance(r, CpoIntVar):
                self.model.add(r == modeler.height_at_start(ivar, pulse))

        # Create final constraint
        # cumf = CpoFunctionCall(Oper__cumul_function, Type_CumulFunction, (CpoValue(cums, Type_CumulAtomArray), CpoValue([], Type_CumulAtomArray)))
        cumf = CpoFunctionCall(Oper_sum, Type_CumulFunction, (CpoValue(cumul_atoms, Type_CumulAtomArray),))
        return modeler.greater_or_equal(bnd, cumf)


    @staticmethod
    def _pred_bool2int(bx, ix):
        """ Convert a boolean expression into integer
        Args:
            bx:  Boolean expression
            ix:  Integer expression
        """
        return bx == ix


    def _pred_int_lin_eq(self, coefs, vars, res):
        """ Scalar product
        Args:
            coefs:  Array of integer coefficients
            vars:   Array of variables
            res:    Result
        """
        defvar = self.defined_var
        if defvar is None:
            # Call default implementation
            return _scal_prod(coefs, vars, res, modeler.equal)

        # Arrange expression to have defined variable on the left
        coefs = _get_value(coefs)
        vars = _get_value(vars)
        # print("Def_var: {}, id:{}".format(defvar, id(defvar)))
        # print("Vars:")
        # for v in vars:
        #     print("   {}, id:{}".format(v, id(v)))
        vx = 0
        try:
            while vars[vx] is not defvar:
                vx += 1
        except IndexError:
            raise FznParserException("Defined variable {} not found in array of variables".format(defvar.name))
        vcoef = coefs[vx]
        vars = vars[:vx] + vars[vx + 1:]
        coefs = coefs[:vx] + coefs[vx + 1:]
        if vcoef < 0:
            vcoef = -vcoef
            coefs = list([-c for c in coefs])
            res = -res

        # Check number of coefs not to 1
        # if [c != 1 and c != -1 for c in coefs].count(True) > 1:
        if any(c != 1 and c != -1 for c in coefs):
            coefs = list([-c for c in coefs])
            if res is 0:
                res = modeler.scal_prod(coefs, vars)
            else:
                res = res + modeler.scal_prod(coefs, vars)
        else:
            # Build expression as a sum
            for c, v in zip(coefs, vars):
                if res is 0:
                    res = _mutl_by_int(v, -c)
                elif c < 0:
                    res = res + _mutl_by_int(v, -c)
                else:
                    res = res - _mutl_by_int(v, c)

        # Add result coef if any
        if vcoef != 1:
            res = res / vcoef

        # Build final equality
        return modeler.equal(defvar, res)


###############################################################################
## Custom predicates implementation
###############################################################################

def _scal_prod(vals, vars, res, op):
    """ Scalar product
    Args:
        vals:  Array of values
        vars:  Array of variables
        res:   Expected result
        op:    Comparison operation
    """
    # Check if vals are only 1 or -1
    vvals = _get_value(vals)
    if all(x == 1 or x == -1 for x in vvals):
        # Replace with a sum
        left = []
        right = []
        for f, v in zip(vvals, _get_value(vars)):
            if f > 0:
                left.append(v)
            else:
                right.append(v)
        if res != 0 or not right:
            right.append(res)
        if not left:
            left = [-right.pop()]
        return op(_create_sum_of_expr(left), _create_sum_of_expr(right))

    # General case
    return op(modeler.scal_prod(vals, vars), res)


def _subcircuit(x):
    """ Constrains the elements of x to define a subcircuit where x[i] = j means that j is the successor of i and x[i] = i means that i is not in the circuit.
    Args:
        x: Array of variables
    """
    return CpoFunctionCall(Oper__sub_circuit, Type_Constraint, (build_cpo_expr(_insert_zero(_get_value(x))),))


def _inverse(f, invf):
    """ Constrains two arrays of int variables, f and invf, to represent inverse functions. All the values in each array must be within the index set of the other array.
    Args:
        f:     First function as array of int
        invf:  Inverse function
    """
    f = _insert_zero(_get_value(f))
    invf = _insert_zero(_get_value(invf))
    return CpoFunctionCall(Oper_inverse, Type_Constraint, (build_cpo_expr(f), build_cpo_expr(invf)))


def _table_int(vars, values):
    """ Implement custom predicate table_int
    Args:
        vars:    Array of variables
        values:  List of values
    """
    # Split value array in tuples
    vars = _get_value(vars)
    tsize = len(vars)
    if tsize == 0:
        return modeler.true()
    values = _get_value(values)
    tuples = [values[i : i+tsize] for i in range(0, len(values), tsize)]

    # Build allowed assignment expression
    return modeler.allowed_assignments(vars, tuples)


def _lex_less_int(vars1, vars2):
    """ Requires that the array vars1 is strictly lexicographically less than array vars2
    Args:
        vars1:  First array of variables
        vars2:  Second array of variables
    """
    # Add 0 and 1 at the end of arrays to force inequality
    vars1 = list(_get_value(vars1))
    vars1.append(1)
    vars2 = list(_get_value(vars2))
    vars2.append(0)

    # Build lexicographic expression
    return modeler.lexicographic(vars1, vars2)


def _bool_clause(a, b):
    """ Implementation of bool_clause predicate
    Args:
        a:  First array of booleans
        b:  Second array of booleans
    Returns:
        Model expression
    """
    # Default implementation
    exprs = list(_get_value(a))
    for x in _get_value(b):
        exprs.append(1 - x)
    return modeler.sum_of(exprs) >= 1

    # Alternative implementation
    # return (modeler.max_of(a) > 0) | (modeler.min_of(b) == 0)

    # Other alternative implementation
    # expr = None
    # for x in _get_value(a):
    #     x = x > 0
    #     expr = x if expr is None else modeler.logical_or(expr, x)
    # for x in _get_value(b):
    #     x = x == 0
    #     expr = x if expr is None else modeler.logical_or(expr, x)
    # return expr



###############################################################################
## Utility functions
###############################################################################

def _get_value(expr):
    """ Get the python value of an expression
    Args:
        expr: Expression (python or CPO)
    Returns:
        Python value
    """
    return expr.value if isinstance(expr, CpoValue) else expr


def _insert_zero(arr):
    """ Insert a zero at the beginning of a python array
    Args:
        arr: Source array (list)
    Returns:
        Same list with zero inserted at the beginning
    """
    res = [0]
    res.extend(arr)
    return res


def _get_domain_bounds(x):
    """ Get min and max bounds of an expression, integer variable or integer
    Args:
        expr: Integer variable or domain
    Returns:
        Tuple (min, max)
    """
    if isinstance(x, CpoIntVar):
        return x.get_domain_min(), x.get_domain_max()
    if isinstance(x, CpoValue):
        x = x.value
    if is_int(x):
        return x, x
    return x


def _create_sum_of_expr(lexprs):
    """ Create an expression building the sum of expressions
    Args:
        lexprs: List of expressions
    Returns:
        Expression that sums all expressions
    """
    nbexpr = len(lexprs)
    if nbexpr == 0:
        return 0
    if nbexpr == 1:
        return lexprs[0]
    if nbexpr > 2:
        return modeler.sum(lexprs)
    return modeler.plus(lexprs[0], lexprs[1])


def _constraint_expr_domain(expr, var):
    """ Create an expression that constrain the domain of an expression to the domain of a variable
    Args:
        expr: Expression to constrain
        var:  Integer var to take domain from
    Returns:
        Constraint expression
    """
    dom = var.get_domain()
    if len(dom) == 1:
        # Use range
        dom = dom[0]
        return modeler.range(expr, dom[0], dom[1])
    # Use allowed assignment
    return modeler.allowed_assignments(expr, var.domain_iterator())


def _mutl_by_int(expr, val):
    """ Create an expression that multiply an expression by an integer
    Args:
        expr: Expression to constrain
        val:  Integer value to multiply expression with
    Returns:
        New expression
    """
    if val == 1:
        return expr
    if val == -1:
        return -expr
    if val == 0:
        return 0
    return val * expr

