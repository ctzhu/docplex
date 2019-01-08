# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Tokenizer for reading CPO file format
"""

from docplex.cp.utils import CpoException, to_internal_string, is_string


###############################################################################
## Constants
###############################################################################

# Operator characters
_OPERATOR_CHARS = "+-*/<>=^%!&.|"

# Punctuation characters
_PUNCTUATION_CHARS = ",;[](){}#"

# Token types
TOKEN_EMPTY       = 0
TOKEN_INTEGER     = 1
TOKEN_FLOAT       = 2
TOKEN_PUNCTUATION = 3
TOKEN_OPERATOR    = 4
TOKEN_SYMBOL      = 5
TOKEN_STRING      = 6


###############################################################################
## Public classes
###############################################################################

class CpoToken(object):
    """ Token returned by tokenizer  """
    __slots__ = ('type',  # Token type
                 'value', # Token string value (with quotes for strings)
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
        
    def is_type(self, type):
        """ Check if a token has a given type
        Args:
            type: Expected type
        Returns:
            True if type is as expected, False otherwise
        """
        return (self.type == type)
        
    def is_value(self, value):
        """ Check if a token has a given value
        Args:
            value: Expected value
        Returns:
            True if value is as expected, false otherwise
        """
        return (self.value == value)
        
    def get_string(self):
        """ Get the string corresponding to the value, interpreting escape sequences if necessary
        Returns:
            Expanded string value
        """
        if (self.type != TOKEN_STRING):
            return(self.value)
        return(to_internal_string(self.value))
        
    def __str__(self):
        """ Build a string representing this token
        Returns:
            String representing this token
        """
        return str(self.value)

    def __eq__(self, other):
        """ Build a string representing this token
        Returns:
            String representing this token
        """
        return isinstance(other, CpoToken) and (other.type == self.type) and (other.value == self.value)


# Null token
TOKEN_NONE = CpoToken(TOKEN_EMPTY, "")

# Token for '/'
_TOKEN_SLASK = CpoToken(TOKEN_OPERATOR, '/')


class CpoTokenizer(object):
    """ Tokenizer for CPO file format """
    __slots__ = ('name',         # Input name (for error string build)
                 'input',        # Input stream
                 'token',        # Current token (list of characters)
                 'line',         # Current input line
                 'rindex',       # Current read index in the line
                 'lineNumber',   # Current line number
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
        self.rindex = 0
        self.lineNumber = 1
        self.token = []

    def next_token(self):
        """ Get the next token

        Returns:
            Next available token (type CpoToken), TOKEN_NONE if end of input
        """
        # Skip separators and comments
        c = ' '
        while (True):
            c = self._next_char()
            while (c is not None) and (c <= ' '):
                c = self._next_char()
            if c is None:
                return TOKEN_NONE    
            
            # Check start comment
            if c == '/':
                c = self._next_char()
                if c == '/':
                    # Skip characters until end of line
                    self.get_line_reminder()
                elif c == '*':
                    # Search end of comment
                    pc = ""
                    c = self._next_char()
                    while (c is not None) and ((c != '/') or (pc != '*')):
                        pc = c
                        c = self._next_char()
                else:
                    self._back_char()
                    return _TOKEN_SLASK
            else:
                break
            
        # Reset current token
        self._reset_token()
        if c == '"':
            c = ''
            # Read character sequence
            while (c is not None) and (c != '"'):
                c = self._next_char()
                if c == '\\':
                    self._next_char()
                    c = ''
            if c is None:
                raise CpoException(self.build_error_string("String not ended before end of stream"))
            return CpoToken(TOKEN_STRING, self._get_token())
        
        elif (c >= '0') and (c <= '9'):
            # Read number
            typ = TOKEN_INTEGER
            while (c >= '0') and (c <= '9'):
                c = self._next_char()
            if (c == '.') or (c == 'e') or (c == 'E'):
                c = self._next_char()
                if (c == '.'):
                    self._back_char()
                    self._back_char()
                    return CpoToken(typ, self._get_token())
                typ = TOKEN_FLOAT
                while c and (c >= '0') and (c <= '9'):
                    c = self._next_char()
                if (c == 'e') or (c == 'E'):
                    c = self._next_char()
                    if (c == '-') or (c == '+'): 
                        c = self._next_char()
                    while (c >= '0') and (c <= '9'):
                        c = self._next_char()
            self._back_char()
            return CpoToken(typ, self._get_token())
        
        elif ((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')) or (c == '_'):
            # Read symbol
            c = self._next_char()
            while c and (((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')) or ((c >= '0') and (c <= '9')) or (c == '_')):
                c = self._next_char()
            self._back_char()
            s = self._get_token()
            return CpoToken(TOKEN_OPERATOR if s == 'div' else TOKEN_SYMBOL, s)
        
        elif c in _PUNCTUATION_CHARS:
            return CpoToken(TOKEN_PUNCTUATION, c)

        elif c in _OPERATOR_CHARS:
            c2 = self._next_char()
            # Accept next char as operator if possible
            if not(c2 in _OPERATOR_CHARS) or (c2 == '-'):
                self._back_char()
            return CpoToken(TOKEN_OPERATOR, self._get_token())
        
        else:
            raise CpoException(self.build_error_string("Unknown token starting by '" + c + "'"))
            
    def get_line_reminder(self):
        """ Get reminder of the line
        Returns:
            Line remainder content, without ending \n
        """
        start = self.rindex
        c = self._next_char()
        while (c is not None) and (c != '\n'):
            c = self._next_char()
        return(self.line[start:self.rindex])
        
    def get_up_to(self, echar):
        """ Get string from current character to a given one (excluded)
        Args:
            echar:  Ending character
        Returns:
            Line content, without ending \n
        """
        res = []
        c = self._next_char()
        while (c is not None) and (c != echar):
            res.append(c)
            c = self._next_char()
        self._back_char()
        return(''.join(res))

    def _reset_token(self):
        """ Reset the current token
        """
        self.token = [self.token.pop()]
        
    def _get_token(self):
        """ Get the last read token
        """
        return ''.join(self.token)
                        
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
        slen = len(line)
        if self.rindex >= slen:
            # Read next line and check end of file
            line = "" if self.input is None else self.input.readline()
            if line == "":
                self.line = None
                return None
            self.line = line
            self.rindex = 0
        c = line[self.rindex]
        self.rindex += 1
        if c == '\n':
            self.lineNumber += 1
        # Push last character in token
        self.token.append(c)
        return c
            
    def _back_char(self):
        """ Push back one character
        """
        if self.line is not None:
            self.rindex -= 1
            self.token.pop()
        
    def build_error_string(self, msg):
        """ Build error string for exception
        """
        return "Error in '" + self.name + "' at line " + str(self.lineNumber) + " index " + str(self.rindex) + ": " + msg
