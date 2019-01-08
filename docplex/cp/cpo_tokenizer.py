# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Tokenizer for reading CPO file format
"""

from docplex.cp.utils import to_internal_string, is_string


###############################################################################
## Utility classes
###############################################################################

class CpoToken(object):
    """ Token returned by tokenizer  """
    __slots__ = ('type',  # Token type
                 'value',  # Token string value (with quotes for strings)
                 )

    def __init__(self, type, value):
        """ Create a new token
        Args:
            type:  Token type
            value: Token value
        """
        super(CpoToken, self).__init__()
        self.type = type
        self.value = value

    def get_string(self):
        """ Get the string corresponding to the value, interpreting escape sequences if necessary
        Returns:
            Expanded string value
        """
        return self.value if self.type != TOKEN_STRING else to_internal_string(self.value)

    def __str__(self):
        """ Build a string representing this token
        Returns:
            String representing this token
        """
        return self.value

    def __eq__(self, other):
        """ Check if this token is equal to another object
        Args:
            other:  Object to compare with
        Returns:
            True if 'other' is a token with the same value then this one, False otherwise
        """
        return isinstance(other, CpoToken) and (other.type == self.type) and (other.value == self.value)

    def __ne__(self, other):
        """ Overwrite inequality comparison """
        return not self.__eq__(other)


###############################################################################
## Constants
###############################################################################

# Token types
TOKEN_NONE        = 0
TOKEN_INTEGER     = 1
TOKEN_FLOAT       = 2
TOKEN_PUNCTUATION = 3
TOKEN_OPERATOR    = 4
TOKEN_SYMBOL      = 5
TOKEN_STRING      = 6
TOKEN_VERSION     = 7

# Miscellaneous directly used tokens
TOKEN_EOF = CpoToken(TOKEN_NONE, "EOF")

# Punctuation tokens
TOKEN_COMMA        = CpoToken(TOKEN_PUNCTUATION, ',')
TOKEN_COLON        = CpoToken(TOKEN_PUNCTUATION, ':')
TOKEN_SEMICOLON    = CpoToken(TOKEN_PUNCTUATION, ';')
TOKEN_BRACE_OPEN   = CpoToken(TOKEN_PUNCTUATION, '{')
TOKEN_BRACE_CLOSE  = CpoToken(TOKEN_PUNCTUATION, '}')
TOKEN_HOOK_OPEN    = CpoToken(TOKEN_PUNCTUATION, '[')
TOKEN_HOOK_CLOSE   = CpoToken(TOKEN_PUNCTUATION, ']')
TOKEN_PARENT_OPEN  = CpoToken(TOKEN_PUNCTUATION, '(')
TOKEN_PARENT_CLOSE = CpoToken(TOKEN_PUNCTUATION, ')')
TOKEN_HASH         = CpoToken(TOKEN_PUNCTUATION, '#')

# Operator tokens
TOKEN_ASSIGN        = CpoToken(TOKEN_OPERATOR, '=')
TOKEN_SLASH         = CpoToken(TOKEN_OPERATOR, '/')
TOKEN_GREATER       = CpoToken(TOKEN_OPERATOR, '>')
TOKEN_GREATER_EQUAL = CpoToken(TOKEN_OPERATOR, '>=')
TOKEN_LOWER         = CpoToken(TOKEN_OPERATOR, '<')
TOKEN_LOWER_EQUAL   = CpoToken(TOKEN_OPERATOR, '<=')
TOKEN_EQUAL         = CpoToken(TOKEN_OPERATOR, '==')
TOKEN_DIFFERENT     = CpoToken(TOKEN_OPERATOR, '!=')
TOKEN_IMPLY         = CpoToken(TOKEN_OPERATOR, '=>')
TOKEN_AND           = CpoToken(TOKEN_OPERATOR, '&&')
TOKEN_OR            = CpoToken(TOKEN_OPERATOR, '||')
TOKEN_NOT           = CpoToken(TOKEN_OPERATOR, '!')
TOKEN_INTERVAL      = CpoToken(TOKEN_OPERATOR, '..')


# Array of single character tokens
_CHAR_TOKENS = [None] * 127
_CHAR_TOKENS[ord(',')] = TOKEN_COMMA
_CHAR_TOKENS[ord(':')] = TOKEN_COLON
_CHAR_TOKENS[ord(';')] = TOKEN_SEMICOLON
_CHAR_TOKENS[ord('#')] = TOKEN_HASH
_CHAR_TOKENS[ord('(')] = TOKEN_PARENT_OPEN
_CHAR_TOKENS[ord(')')] = TOKEN_PARENT_CLOSE
_CHAR_TOKENS[ord('[')] = TOKEN_HOOK_OPEN
_CHAR_TOKENS[ord(']')] = TOKEN_HOOK_CLOSE
_CHAR_TOKENS[ord('{')] = TOKEN_BRACE_OPEN
_CHAR_TOKENS[ord('}')] = TOKEN_BRACE_CLOSE

for c in "+-*%^":
    _CHAR_TOKENS[ord(c)] = CpoToken(TOKEN_OPERATOR, c)
_CHAR_TOKENS[ord('/')] = TOKEN_SLASH

_CHAR_TOKENS = tuple(_CHAR_TOKENS)


###############################################################################
## Public classes
###############################################################################

class CpoTokenizer(object):
    """ Tokenizer for CPO file format """
    __slots__ = ('name',         # Input name (for error string build)
                 'input',        # Input stream
                 'line',         # Current input line
                 'token_start',  # Index of token start in current line
                 'line_length',  # Current input line length
                 'read_index',   # Current read index in the line
                 'line_number',  # Current line number
                 )

    def __init__(self, name, input):
        """ Create a new tokenizer
        Args:
            input: Input stream or string
        """
        super(CpoTokenizer, self).__init__()
        self.name = name
        if is_string(input):
            self.input = None
            self.line = input
        else:
            self.input = input
            self.line = ""
        self.line_length = len(self.line)
        self.read_index = 0
        self.line_number = 1
        self.token_start = 0


    def next_token(self):
        """ Get the next token

        Returns:
            Next available token (type CpoToken), TOKEN_NONE if end of input
        """
        # Skip separators and comments
        while True:
            c = self._next_char()
            while c and (c <= ' '):
                c = self._next_char()
            if c is None:
                return TOKEN_EOF

            # Check start comment
            if c == '/':
                c = self._peek_char()
                if c == '/':
                    # Skip characters until end of line
                    self.get_line_reminder()
                elif c == '*':
                    # Search end of comment
                    self._skip_char()
                    pc = ""
                    c = self._next_char()
                    while c and ((c != '/') or (pc != '*')):
                        pc = c
                        c = self._next_char()
                else:
                    return TOKEN_SLASH
            else:
                break

        # Reset current token
        self.token_start = self.read_index - 1

        # Check single character tokens
        t = _CHAR_TOKENS[ord(c)]
        if t:
            return t

        # Double character operators
        elif c == '>':
            if self._peek_char() == '=':
                self._next_char()
                return TOKEN_GREATER_EQUAL
            return TOKEN_GREATER

        elif c == '<':
            if self._peek_char() == '=':
                self._next_char()
                return TOKEN_LOWER_EQUAL
            return TOKEN_LOWER

        elif c == '=':
            c = self._peek_char()
            if c == '=':
                self._next_char()
                return TOKEN_EQUAL
            if c == '>':
                self._next_char()
                return TOKEN_IMPLY
            return TOKEN_ASSIGN

        elif c == '!':
            if self._peek_char() == '=':
                self._next_char()
                return TOKEN_DIFFERENT
            return TOKEN_NOT

        elif c == '&':
            if self._next_char() != '&':
                raise SyntaxError(self.build_error_string("Unknown operator '&'"))
            return TOKEN_AND

        elif c == '|':
            if self._next_char() != '|':
                raise SyntaxError(self.build_error_string("Unknown operator '|'"))
            return TOKEN_OR

        elif c == '.':
            if self._next_char() != '.':
                raise SyntaxError(self.build_error_string("Unknown operator '.'"))
            return TOKEN_INTERVAL

        # Check string
        elif c == '"':
            c = ''
            # Read character sequence
            while (c is not None) and (c != '"'):
                c = self._next_char()
                if c == '\\':
                    self._next_char()
                    c = ''
            if c is None:
                raise SyntaxError(self.build_error_string("String not ended before end of stream"))
            return CpoToken(TOKEN_STRING, self._get_token())

        # Check symbol
        elif ((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')) or (c == '_'):
            # Read symbol
            c = self._peek_char()
            while c and (((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')) or ((c >= '0') and (c <= '9')) or (c == '_')):
                c = self._skip_char()
            s = self._get_token()
            return CpoToken(TOKEN_OPERATOR if s == 'div' else TOKEN_SYMBOL, s)

        # Check number
        elif (c >= '0') and (c <= '9'):
            # Read number
            typ = TOKEN_INTEGER
            c = self._peek_char()
            while c and (c >= '0') and (c <= '9'):
                c = self._skip_char()
            if c == '.':
                c = self._skip_char()
                if c == '.':
                    # Case of '..' used to specify intervals
                    self.read_index -= 1
                    return CpoToken(typ, self._get_token())
                typ = TOKEN_FLOAT
                while c and (c >= '0') and (c <= '9'):
                    c = self._skip_char()
                if (c == 'e') or (c == 'E'):
                    c = self._skip_char()
                    if (c == '-') or (c == '+'):
                        c = self._skip_char()
                    while c and (c >= '0') and (c <= '9'):
                        c = self._skip_char()
                elif c == '.':
                    typ = TOKEN_VERSION
                    while (c == '.') or ((c >= '0') and (c <= '9')):
                        c = self._skip_char()
            elif (c == 'e') or (c == 'E'):
                typ = TOKEN_FLOAT
                c = self._skip_char()
                if (c == '-') or (c == '+'):
                    c = self._skip_char()
                while c and (c >= '0') and (c <= '9'):
                    c = self._skip_char()
            return CpoToken(typ, self._get_token())

        # Unknown token
        raise SyntaxError(self.build_error_string("Unknown token starting by '{}'".format(c)))


    def get_line_reminder(self):
        """ Get reminder of the line
        Returns:
            Line remainder content, without ending \n
        """
        start = self.read_index
        c = self._next_char()
        while c and (c != '\n'):
            c = self._next_char()
        return self.line[start:self.read_index]


    def _get_token(self):
        """ Get the last read token
        """
        return self.line[self.token_start:self.read_index]


    def _peek_char(self):
        """ Peek (but not get) next input character
        Returns:
            Next available character, None if end of input
        """
        # Check end of stream
        if self.line is None:
            return None

        # Check end of line
        if self.read_index >= self.line_length:
            return '\n'
        return self.line[self.read_index]


    def _skip_char(self):
        """ Skip next input character and peek next one
        Returns:
            Next available character, None if end of input
        """
        # Check end of line
        self.read_index += 1
        if self.read_index >= self.line_length:
            return '\n'
        c = self.line[self.read_index]
        if c == '\n':
            self.line_number += 1
        return c


    def _next_char(self):
        """ Get next input character
        Returns:
            Next available character, None if end of input
        """
        # Check end of stream
        line = self.line
        if line is None:
            return None

        # Check end of line
        if self.read_index >= self.line_length:
            # Read next line and check end of file
            line = "" if self.input is None else self.input.readline()
            if line == "":
                self.line = None
                self.line_length = 0
                return None
            self.line = line
            self.line_length = len(line)
            self.read_index = 0
        c = line[self.read_index]
        self.read_index += 1
        if c == '\n':
            self.line_number += 1
        return c


    def build_error_string(self, msg):
        """ Build error string for exception
        """
        return "Error in '" + self.name + "' at line " + str(self.line_number) + " index " + str(
            self.read_index) + ": " + msg

