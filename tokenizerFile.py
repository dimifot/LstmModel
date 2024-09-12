import re

def tokenizer(expression):

    token_patterns = [
        (r'def', 'DEF'),  # Matches 'def' keyword
        (r'float', 'FLOAT'),  # Matches 'float' keyword
        (r'while', 'WHILE'),  # Matches 'while' keyword
        (r'if', 'IF'),  # Matches 'if' keyword
        (r'[+\-*/]', 'OPERATOR'),  # Matches arithmetic operators
        (r'\d+\.\d+', 'FLOAT_LITERAL'),  # Matches floating point numbers
        (r'\d+', 'INT_LITERAL'),  # Matches integers
        (r'[;,:]', 'SEPARATOR'),  # Matches separators like ';' and ','
        (r'\(', 'LEFT_PAREN'),  # Matches left parenthesis
        (r'\)', 'RIGHT_PAREN'),  # Matches right parenthesis
        (r'=', 'ASSIGNMENT'),  # Matches assignment operator
        (r'!', 'EXCLAMATION'),  # Matches exclamation mark
        (r'object', 'OBJECT'),  # Matches 'object' keyword
        (r'\{|\}', 'BRACE'),  # Matches braces '{' and '}'
        (r'[a-zA-Z]\w*', 'IDENTIFIER'),  # Matches identifiers (variable names)
        (r'\s+', None)  # Matches and skips whitespace
    ]

    # Combine token patterns into a single regex
    token_regex = '|'.join(f'(?P<{name}>{pattern})' for pattern, name in token_patterns if name)

    # Tokenize the input expression
    tokens = []
    for match in re.finditer(token_regex, expression):
        kind = match.lastgroup
        value = match.group()
        if kind:
            tokens.append(value if kind not in ["FLOAT_LITERAL", "INT_LITERAL"] else kind)

    return tokens

def py_tokenizer(expression):
    token_patterns = [
        (r'\bdef\b', 'DEF'),  # Matches 'def' keyword
        (r'\bfloat\b', 'FLOAT'),  # Matches 'float' keyword
        (r'\bwhile\b', 'WHILE'),  # Matches 'while' keyword
        (r'\bif\b', 'IF'),  # Matches 'if' keyword
        (r'\belse\b', 'ELSE'),  # Matches 'else' keyword
        (r'\bfor\b', 'FOR'),  # Matches 'for' keyword
        (r'\bin\b', 'IN'),  # Matches 'in' keyword
        (r'\band\b', 'AND'),  # Matches 'and' keyword
        (r'\bor\b', 'OR'),  # Matches 'or' keyword
        (r'\bnot\b', 'NOT'),  # Matches 'not' keyword
        (r'\breturn\b', 'RETURN'),  # Matches 'return' keyword
        (r'\bobject\b', 'OBJECT'),  # Matches 'object' keyword
        (r'[+\-*/%]', 'OPERATOR'),  # Matches arithmetic operators
        (r'==|!=|<=|>=|<|>', 'COMPARISON'),  # Matches comparison operators
        (r'\d+\.\d+', 'FLOAT_LITERAL'),  # Matches floating point numbers
        (r'\d+', 'INT_LITERAL'),  # Matches integers
        (r'".*?"', 'DOUBLE_STRING_LITERAL'),  # Matches double-quoted string literals
        (r"'.*?'", 'SINGLE_STRING_LITERAL'),  # Matches single-quoted string literals
        (r'[;,:]', 'SEPARATOR'),  # Matches separators like ';' and ','
        (r'\(', 'LEFT_PAREN'),  # Matches left parenthesis
        (r'\)', 'RIGHT_PAREN'),  # Matches right parenthesis
        (r'\{', 'LEFT_BRACE'),  # Matches left brace
        (r'\}', 'RIGHT_BRACE'),  # Matches right brace
        (r'\[', 'LEFT_BRACKET'),  # Matches left bracket
        (r'\]', 'RIGHT_BRACKET'),  # Matches right bracket
        (r'=', 'ASSIGNMENT'),  # Matches assignment operator
        (r'!', 'EXCLAMATION'),  # Matches exclamation mark
        (r'\s+', None),  # Matches and skips whitespace
        (r'#[^\n]*', None),  # Matches and skips comments
        (r'[a-zA-Z_]\w*', 'IDENTIFIER'),  # Matches identifiers (variable names)
    ]

    # Combine token patterns into a single regex
    token_regex = '|'.join(f'(?P<{name}>{pattern})' for pattern, name in token_patterns if name)

    # Tokenize the input expression
    tokens = []
    for match in re.finditer(token_regex, expression):
        kind = match.lastgroup
        value = match.group()
        if kind:
            tokens.append(value if kind not in ["FLOAT_LITERAL", "INT_LITERAL", "DOUBLE_STRING_LITERAL", "SINGLE_STRING_LITERAL"] else kind)

    return tokens

expression = """
float d = 32.2;
float h = 100.0;
while d<=h: { float e = 1 + d/5; d = d +1; h = h - e; }
float a = 0.6 + 18.7 + 8.8;
while a<=d: { float h = a /3; a = a +h; }
def fun2(float g, float c){ object g = object(); float h = c / 2.9; float g.f = 12.6 + c + h; return g; }
fun2 f = fun2(6.465, 19.68);
"""

tokens = tokenizer(expression)
for token in tokens:
    print(token)

code = """
def func2(b):
    y = 18.257
    y = y / a
c = 9.933
while c < 8.513:
    z = z * x
def func1(x):
    x = 6.65
    y = y * c
"""

tokens = py_tokenizer(code)
print(tokens)