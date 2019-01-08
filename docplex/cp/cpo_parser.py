# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Parser converting a CPO file to internal model representation.
"""

from docplex.cp.cpo_tokenizer import *
from docplex.cp.expression import *
from docplex.cp.expression import _create_operation, _get_cpo_type
from docplex.cp.function import *
from docplex.cp.model import CpoModel
from docplex.cp.catalog import *
from docplex.cp.parameters import *


###############################################################################
## Constants
###############################################################################

# Minimum CPO format version number
MIN_CPO_VERSION_NUMBER = "12.6.3.0"

# Map of all operators. Key is operator, value is list of corresponding operation descriptors
_ALL_OPERATORS = {}

# Map of all operations. Key is CPO operation name, value is list corresponding operation descriptor
_ALL_OPERATIONS = {}

# Initialization code
for op in ALL_OPERATIONS:
    if (op.get_priority() >= 0):
        kwd = op.get_keyword()
        oplist = _ALL_OPERATORS.get(kwd, None)
        if (oplist is None):
            _ALL_OPERATORS[kwd] = [op]
        else:
            oplist.append(op)
    _ALL_OPERATIONS[op.get_cpo_name()] = op
_ALL_OPERATIONS['alldiff'] = Oper_all_diff
_ALL_OPERATIONS['allDiff'] = Oper_all_diff


# Known identifiers
_KNOWN_IDENTIFIERS = {"intmax": INT_MAX, "intmin": INT_MIN,
                      "inf": INFINITY,
                      "intervalmax": INTERVAL_MAX, "intervalmin": INTERVAL_MIN,
                      "no": False}

# Map of array types for each CPO name
_ARRAY_TYPES = {'intArray': Type_IntArray, 'floatArray': Type_FloatArray,
                'intExprArray': Type_IntExprArray, 'floatExprArray': Type_FloatExprArray,
                'intervalVarArray': Type_IntervalVarArray, 'sequenceVarArray': Type_SequenceVarArray,
                '_cumulAtomArray' : Type_CumulAtomArray}


###############################################################################
## Public classes
###############################################################################

class CpoParserException(CpoException):
    """ The base class for exceptions raised by the CPO parser
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(CpoParserException, self).__init__(msg)


class CpoUnsupportedFormatVersionException(CpoParserException):
    """ Exception raised when a version mismatch is detected
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(CpoUnsupportedFormatVersionException, self).__init__(msg)



class CpoParser(object):
    """ Reader of CPO file format """
    __slots__ = ('model',          # Read model
                 'params',         # Read solving parameters
                 'tokenizer',      # Reading tokenizer
                 'token',          # Last read token
                 'pushtoken',      # Pushed token
                 )
    
    def __init__(self):
        """ Create a new CPO format parser 
        """
        super(CpoParser, self).__init__()
        self.model = CpoModel()
        self.params = CpoParameters()
        self.token = None
        self.pushtoken = None

        # Do not store location information (would store parser instead of real lines)
        self.model.source_loc = False

        # TODO: parse and include source information

    def get_model(self):
        """ Get the model that have been parsed

        Return:
            CpoModel result of the parsing
        """
        return self.model
            
    def get_parameters(self):
        """ Get the solving parameters that have been parsed

        Return:
            CpoParameters result of the parsing
        """
        return self.params

    def get_format_version(self):
        """ Get the version of the CPO format

        Return:
            Cpo format version number, as a string. None if not present in the file.
        """
        return self.format_version

    def parse(self, cfile):
        """ Parse a CPO file

        Args:
            cfile: CPO file to read
        Raises:
            CpoParserException: Parsing exception
            CpoVersionMismatchException: Read CPO format is under MIN_CPO_VERSION_NUMBER
        """
        # Store file name if first file
        if self.model.source_file is None:
            self.model.source_file = cfile
        with open_utf8(cfile, mode='r') as f:
            self.tokenizer = CpoTokenizer(cfile, f)
            while self._read_statement():
                pass
            self.tokenizer = None

    def parse_string(self, str):
        """ Parse a string

        Args:
            str: String to parse
        """
        self.tokenizer = CpoTokenizer("String", str)
        while self._read_statement():
            pass
        self.tokenizer = None

    def _read_statement(self):
        """ Read a statement or a section

        This functions reads the first token and exits with current token that is
        the last of the statement.
        Returns:
            True if something has been read, False if end of input
        """
        try:
            tok1 = self._next_token()
            if tok1 is TOKEN_NONE:
                return False
            tok2 = self._next_token()
            if (tok1.value == '#'):
                self._read_directive(tok2.value)
            elif (tok2.value == '='):
                self._read_assignment(tok1.get_string())
            elif (tok2.value == '{'):
                self._read_section(tok1.value)
            else:
                # Read expression
                self._push_token(tok1)
                expr = self._read_expression()
                #print("Expression read in statement: Type: " + str(expr.get_type()) + ", val=" + str(expr))
                self.model.add(expr)
                self._check_token_value(self.token, ';')
        except Exception as e:
            if isinstance(e, CpoParserException):
                raise e
            import traceback
            traceback.print_exc()
            self._raise_exception(str(e))
        return True

    def _read_directive(self, name):
        """ Read a directive

        Args:
            name:  Directive name
        """
        if name == "include":
            self._raise_exception("#include currently not supported")
        elif name == "line":
            # Skip line
            self.tokenizer.get_line_reminder()
        else:
            self._raise_exception("Unknown directive '" + name + "'")
        
    def _read_assignment(self, name):
        """ Read an assignment

        Args:
            name:  Assignment name
        """
        tok = self._next_token()
        if tok.value in ("intVar", "_intVar"):
            v = self._read_int_var(name)
            self.model._add_variable(v)

        elif tok.value == "intervalVar":
            v = self._read_interval_var(name)
            self.model._add_variable(v)

        elif tok.value == "sequenceVar":
            v = self._read_sequence_var(name)
            self.model._add_variable(v)

        elif tok.value == "stateFunction":
            v = self._read_state_function(name)
            self.model._add_variable(v)

        else:
            # Read expression
            expr = self._read_expression()
            self._push_token('')
            # Build CPO expression if needed
            if not(isinstance(expr, CpoExpr)):
                expr = create_cpo_expr(expr)
            # Add expression to model
            if not expr.has_name():
                expr.set_name(name)
                self.model._add_named_expr(expr, name)
            elif expr.is_constraint_or_bool_expr() or name != expr.get_name():
                self.model._add_named_expr(expr, name)
            else:
                self.model.add(expr)
        self._check_token_value(self._next_token(), ';')
            
    def _read_int_var(self, name):
        """ Read a int_var declaration

        Args:
            name:  Variable name
        Returns:
            CpoIntVar variable expression
        """
        # Read arguments
        self._check_token_value(self._next_token(), '(')
        args = self._read_expression_list(')')
        return CpoIntVar(args, name)
            
    def _read_interval_var(self, name):
        """ Read a interval_var declaration
        Args:
            name:  Variable name
        Returns:
            CpoIntervalVar variable expression
        """
        res = interval_var(name=name)
        self._check_token_value(self._next_token(), '(')
        tok = self._next_token()
        while (not tok.is_value(')')):
            # Read argument name
            self._check_token_string(tok)
            aname = tok.value
            if (aname == "present"):
                res.set_present()
                self._next_token()
            elif (aname == "absent"):
                res.set_absent()
                self._next_token()
            elif (aname == "optional"):
                res.set_optional()
                self._next_token()
            else:
                self._check_token_value(self._next_token(), '=')
                if (aname in ("start", "end", "length", "size")):
                    # Read interval
                    self._next_token()
                    intv = self._read_expression()
                    if isinstance(intv, int):
                        intv = (intv, intv)
                    elif not isinstance(intv, (list, tuple)):
                        self._raise_exception("'start', 'end', 'length' or 'size' should be an integer or an interval")
                    setattr(res, aname, intv)
                elif (aname == "intensity"):
                    self._next_token()
                    res.set_intensity(self._read_expression())
                elif (aname == "granularity"):
                    tok = self._next_token()
                    self._check_token_integer(tok)
                    res.set_granularity(int(tok.value))
                    self._next_token()
                else:
                    self._raise_exception("Unknown IntervalVar attribute argument '" + aname + "'")
            # Read comma
            tok = self.token
            if (tok.is_value(',')):
                tok = self._next_token()
        return res

    def _read_sequence_var(self, name):
        """ Read a sequence_var declaration
        Args:
            name:  Variable name
        Returns:
            CpoSequenceVar variable expression
        """
        self._check_token_value(self._next_token(), '(')
        args = self._read_expression_list(')')
        if (len(args) == 1):
            lvars = args[0]
            ltypes = None
        else:
            if (len(args) != 2):
                self._raise_exception("'sequenceVar' should have 1 or 2 arguments")
            lvars = args[0]
            ltypes = args[1]
        return CpoSequenceVar(lvars, ltypes, name)

    def _read_state_function(self, name):
        """ Read a state function declaration
        Args:
            name:  Variable name
        Returns:
            CpoStateFunction variable expression
        """
        self._check_token_value(self._next_token(), '(')
        args = self._read_expression_list(')')
        nbargs = len(args)
        if (nbargs == 0):
            trmx = None
        elif (nbargs == 1):
            trmx = args[0]
        else:
            self._raise_exception("'stateFunction' should have 0 or 1 argument")
        return CpoStateFunction(trmx, name)

    def _read_expression(self):
        """ Read an expression

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Expression that has been read
        """

        # Read sub-expression
        expr = self._read_sub_expression()
        tok = self.token
        if not (tok.is_type(TOKEN_OPERATOR)):
            return expr

        # Initialize elements stack
        stack = [expr]
        while tok.is_type(TOKEN_OPERATOR):
            op = self._get_and_check_operator(tok)
            self._next_token()
            expr = self._read_sub_expression()
            tok = self.token

            # Reduce stack if possible
            while (len(stack) > 1) and op[0].get_priority() >= stack[-2][0].get_priority():
                oexpr = stack.pop()
                oop = stack.pop()
                stack[-1] = self._create_operation(oop, (stack[-1], oexpr))

            stack.append(op)
            stack.append(expr)

        # Build final expression
        expr = stack.pop()
        while stack:
            op = stack.pop()
            expr = self._create_operation(op, (stack.pop(), expr))
        return expr

    def _read_sub_expression(self):
        """ Read a sub-expression

        First expression token is already read.
        Function exits with current token following the last expression token
        Return:
            Expression that has been read
        """
        # Check integer constant or interval
        tok = self.token
        #print("--> readSubExpression, tok=" + str(tok))
        if (tok.type == TOKEN_INTEGER) or (tok.value in _KNOWN_IDENTIFIERS):
            if (tok.type == TOKEN_INTEGER):
                sval = int(tok.value)
            else:
                sval = _KNOWN_IDENTIFIERS[tok.value]
            tok = self._next_token()
            if tok.is_value('..'):
                tok = self._next_token()
                if (tok.type == TOKEN_INTEGER):
                    eval = int(tok.value)
                elif (tok.value in _KNOWN_IDENTIFIERS):
                    eval = _KNOWN_IDENTIFIERS[tok.value]
                else:
                    self._raise_exception("Expression after '..' should be an integer")
                self._next_token()
                return (sval, eval)
            else:
                return sval

        # Check float constant
        if (tok.type == TOKEN_FLOAT):
            self._next_token()
            return float(tok.value)
        
        # Check unary operator
        if tok.is_type(TOKEN_OPERATOR):
            # Retrieve operation descriptor
            op = self._get_and_check_operator(tok)
            # Read next expression
            self._next_token()
            expr = self._read_sub_expression()
            if isinstance(expr, (int, float)):
                if tok.is_value('-'):
                    return -expr
                if tok.is_value('+'):
                    return expr
            return self._create_operation(op, (expr,))
        
        # Check symbol
        if (tok.type == TOKEN_SYMBOL):
            ntok = self._next_token()
            if (ntok.is_value('(')):
                # Read function arguments
                if tok.is_value("transitionMatrix"):
                    # Check arguments list to support presolved version
                    tok = self._next_token()
                    self._push_token(tok)
                    if tok.is_value("matrixSize"):
                        args = self._read_arguments_list(')')
                        for name, mtrx in args:
                            if name == 'matrix':
                                break
                    else:
                        mtrx = self._read_expression_list(')')
                    self._next_token()
                    return CpoTransitionMatrix(values=mtrx)
                else:
                    args = self._read_expression_list(')')
                    self._next_token()
                    # Check predefined functions
                    if tok.is_value("stepFunction"):
                        return CpoStepFunction(args)
                    if tok.is_value("segmentedFunction"):
                        return CpoSegmentedFunction(args[0], args[1:])
                    if tok.is_value("sequenceVar"):
                        return CpoSequenceVar(*args)
                    # General function call, retrieve operation descriptor
                    opname = tok.value
                    op = _ALL_OPERATIONS.get(opname, None)
                    if op is None:
                        # # Check private operation
                        # if opname.startswith('_'):
                        #     # Create special operation descriptor
                        #     sgn = CpoSignature(Type_IntExpr, (_get_cpo_type(x) for x in args))
                        #     op = CpoOperation(opname, opname, None, -1, (sgn, ))
                        # else:
                        #     self._raise_exception("Unknown operation '" + str(tok.value) + "'")
                        self._raise_exception("Unknown operation '" + str(tok.value) + "'")
                    return self._create_operation((op,), args)

            elif (ntok.is_value('[')):
                # Read typed array
                expr = self._read_expression_list(']')
                self._next_token()
                # Search type
                typ = _ARRAY_TYPES.get(tok.value)
                if typ is None:
                    self._raise_exception("Unknown array type '" + str(tok.value) + "'")
                # Compute best type if array not empty
                if len(expr) > 0:
                    ct = _get_cpo_type(expr)
                    if ct.is_kind_of(typ):
                        typ = ct
                return CpoValue(expr, typ)

            else:
                # Token is an expression id
                return self._get_identifier_value(tok.value)
        
        # Check expression in parenthesis
        if tok.is_value('('):
            expr = self._read_expression_list(')')
            self._next_token()
            if len(expr) == 1:
                return expr[0]
            return expr

        # Check array with no type
        if tok.is_value('['):
            expr = self._read_expression_list(']')
            self._next_token()
            return expr
            
        # Check reference to a model expression or variable
        if tok.is_type(TOKEN_STRING):
            self._next_token()
            return self._get_identifier_value(tok.get_string())
                
        # Unknown expression
        self._raise_exception("Invalid start of expression: '" + str(tok) + "'")
            
    def _read_expression_list(self, etok):
        """ Read a list of expressions

        This method supposes that the list start token is read (for example '(' or '[').
        When returning, current token is list ending token
        Args:
           etok: Expression list ending token string (for example ')' or ']')
        Returns:
            Array of expressions
        """
        lxpr = []
        self._next_token()
        while not(self.token.is_value(etok)):
            lxpr.append(self._read_expression())
            if (self.token.is_value(',')):
                self._next_token()
        return lxpr
        
    def _read_arguments_list(self, etok):
        """ Read a list of arguments that are possibly named

        This method supposes that the list start token is read (for example '(' or '[').
        When returning, current token is list ending token
        Args:
           etok: Expression list ending token string (for example ')' or ']')
        Returns:
            Array of couples (name, expression)
        """
        lxpr = []
        self._next_token()
        while not(self.token.is_value(etok)):
            if self.token.is_type(TOKEN_SYMBOL):
                name = self.token
                if self._next_token().is_value('='):
                    self._next_token()
                else:
                    self._push_token(name)
                    name = None
            else:
                name = None
            lxpr.append((name, self._read_expression()))
            if (self.token.is_value(',')):
                self._next_token()
        return lxpr

    def _read_section(self, name):
        """ Read a section.
        Current token is the opening brace
        Args:
            name:  Section name
        """
        if (name == "parameters"):
            self._read_section_parameters()
        elif (name == "internals"):
            self._read_section_internals()
        elif (name == "search"):
            self._read_section_search()
        else:
            self._raise_exception("Unknown section '" + name + "'")
            
    def _read_section_parameters(self):
        """ Read a parameters section
        """
        tok = self._next_token()
        while not tok.is_value('}'):
            vname = self._check_token_string(tok)
            self._check_token_value(self._next_token(), '=')
            value = self._next_token()
            self._check_token_value(self._next_token(), ';')
            self.params.set_attribute(vname, value.get_string())
            tok = self._next_token()

    def _read_section_internals(self):
        """ Read a internals section
        """
        # Skip all until section end
        tok = self._next_token()
        while (tok is not TOKEN_NONE) and not(tok.is_value('}')):
            if self.token.is_value("version"):
                self._check_token_value(self._next_token(), '(')
                ver = self.tokenizer.get_up_to(')')
                self.model.format_version = ver
                # if ver < MIN_CPO_VERSION_NUMBER:
                #     raise CpoUnsupportedFormatVersionException("Can not parse a CPO file with version {}, lower than {}"
                #                                                .format(ver, MIN_CPO_VERSION_NUMBER))
                self._check_token_value(self._next_token(), ')')
            tok = self._next_token()
        
    def _read_section_search(self):
        """ Read a search section
        """
        # Read statements up to end of section
        tok = self._next_token()
        while (tok is not TOKEN_NONE) and not(tok.is_value('}')):
            self._push_token(tok)
            self._read_statement()
            tok = self._next_token()
        
    def _read_domain_list(self):
        """ Read a domain definition list.

        This method starts by reading first '(' and ends by reading last ')'
        Returns:
            List of integers or couples of integers
        """
        res = []
        self._check_token_value(self._next_token(), '(')
        tok = self._next_token()
        while not tok.is_value(')'):
            v1 = self._check_token_integer(tok)
            tok = self._next_token()
            if tok.is_value('..'):
                tok = self._next_token()
                v2 = self._check_token_integer(tok)
                res.append((v1, v2))
                tok = self._next_token()
            else:
                res.append(v1)
            if tok.is_value(','):
                tok = self._next_token()
        return(res)       
        
    def _check_token_value(self, tok, val):
        """ Check that a read token has a given value an raise an exception if not
        Args:
            tok: Read token
            val: Expected value
        """
        if not(tok.is_value(val)):
            self._raise_unexpected_token(val, tok)
            
    def _get_and_check_operator(self, tok):
        """ Get an operator descriptor and raise an exception if not found
        Args:
            tok:  Operator token
        Returns:
            List of Operation descriptor for this keyword
        Raises:
            CpoException if operator does not exists
        """
        op = _ALL_OPERATORS.get(tok.value, None)
        # print("Operator for '" + tok.value + "': " + ", ".join(str(x) for x in op) + ", priority=" + str(op[0].get_priority()))
        if op is None:
            self._raise_exception("Unknown operator '" + str(tok.value) + "'")
        return op
            
    def _check_token_string(self, tok):
        """ Check that a token is a string and raise an exception if not
        Args:
            tok: Token
        Returns:
            String value of the token            
        """
        if tok.is_type(TOKEN_SYMBOL):
            return tok.value
        if tok.is_type(TOKEN_STRING):
            return tok.get_string()
        self._raise_exception("String expected")
    
    def _check_token_integer(self, tok):
        """ Check that a token is an integer and raise an exception if not
        Args:
            tok: Token
        Returns:
            integer value of the token
        """
        if tok.is_type(TOKEN_INTEGER):
            return(int(tok.value))
        if tok.value in _KNOWN_IDENTIFIERS:
            return _KNOWN_IDENTIFIERS[tok.value]
        self._raise_exception("Integer expected instead of '" + tok.value + "'")
    
    def _get_identifier_value(self, eid):
        """ Get an expression associated to an identifier
        Args:
            eid:  Expression identifier
        """
        expr = self.model.get_expression(eid)
        if expr is None:
            self._raise_exception("Unknown identifier '" + str(eid) + "'")
        return(expr)

    def _create_operation(self, lops, args):
        """ Create a model operation

        Args:
            lops:  List of candidate operation descriptor
            args:  Operation arguments
        Returns:
            Model expression
        Raises:
            Cpo exception if error
        """
        lastex = None # Last error found, thrown only if no viable solution has been found
        for op in lops:
            try:
                return _create_operation(op, args)
            except Exception as e:
                lastex = e
        if lastex is None:
            lastex = Exception("No valid operation found for " + str(lops[0].get_cpo_name()))
        self._raise_exception(str(lastex))
        
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
        raise CpoParserException(self.tokenizer.build_error_string(msg))
    
    def _next_token(self):
        """ Read next token
        Returns:
            Next read token, None if end of input
        """
        # Check if a token has been pushed
        if self.pushtoken is not None:
            tok = self.pushtoken
            self.pushtoken = None
        else:
            tok = self.tokenizer.next_token()
        self.token = tok
        #print("Tok='" + str(tok) + "'")
        return tok
        
    def _push_token(self, tok):
        """ Push current token
        Args:
            tok: New current token 
        """
        self.pushtoken = self.token
        self.token = tok
        
