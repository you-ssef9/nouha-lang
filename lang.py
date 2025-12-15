#!/usr/bin/env python3
"""
Nouha Language - Advanced Interpreter
A dynamically typed scripting language
"""

import sys
import os
import math
import re
import json
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import readline  # For better REPL experience

# ==================== AST NODES ====================
class NodeType(Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    CALL = "call"
    INDEX = "index"
    ASSIGNMENT = "assignment"
    BLOCK = "block"
    IF = "if"
    WHILE = "while"
    FOR = "for"
    FUNCTION = "function"
    RETURN = "return"
    LIST = "list"
    DICT = "dict"
    IMPORT = "import"
    TRY_CATCH = "try_catch"

@dataclass
class ASTNode:
    type: NodeType
    value: Any = None
    children: List['ASTNode'] = None
    line: int = 0
    col: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

# ==================== TOKENIZER ====================
class TokenType(Enum):
    # Keywords
    IF = "if"
    ELSE = "else"
    ELIF = "elif"
    WHILE = "while"
    FOR = "for"
    IN = "in"
    FUNCTION = "func"
    RETURN = "return"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPORT = "import"
    TRY = "try"
    CATCH = "catch"
    THROW = "throw"
    CLASS = "class"
    
    # Operators
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"
    
    # Comparison
    EQ = "=="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    
    # Assignment
    ASSIGN = "="
    PLUS_EQ = "+="
    MINUS_EQ = "-="
    MULT_EQ = "*="
    DIV_EQ = "/="
    
    # Delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    DOT = "."
    COLON = ":"
    SEMICOLON = ";"
    
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

class Tokenizer:
    KEYWORDS = {
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'elif': TokenType.ELIF,
        'while': TokenType.WHILE,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'func': TokenType.FUNCTION,
        'return': TokenType.RETURN,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'null': TokenType.NULL,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'import': TokenType.IMPORT,
        'try': TokenType.TRY,
        'catch': TokenType.CATCH,
        'throw': TokenType.THROW,
        'class': TokenType.CLASS,
    }
    
    OPERATORS = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.MULTIPLY,
        '/': TokenType.DIVIDE,
        '%': TokenType.MODULO,
        '**': TokenType.POWER,
        '=': TokenType.ASSIGN,
        '==': TokenType.EQ,
        '!=': TokenType.NEQ,
        '<': TokenType.LT,
        '>': TokenType.GT,
        '<=': TokenType.LTE,
        '>=': TokenType.GTE,
        '+=': TokenType.PLUS_EQ,
        '-=': TokenType.MINUS_EQ,
        '*=': TokenType.MULT_EQ,
        '/=': TokenType.DIV_EQ,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.col = 1
        
    def tokenize(self) -> List[Token]:
        tokens = []
        while self.position < len(self.source):
            char = self.source[self.position]
            
            # Whitespace
            if char in ' \t':
                self.advance()
                continue
            elif char == '\n':
                self.line += 1
                self.col = 1
                self.advance()
                continue
            
            # Comments
            elif char == '#':
                while self.position < len(self.source) and self.source[self.position] != '\n':
                    self.advance()
                continue
            
            # Strings
            elif char in '"\'`':
                tokens.append(self.read_string(char))
                continue
            
            # Numbers
            elif char.isdigit():
                tokens.append(self.read_number())
                continue
            
            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                tokens.append(self.read_identifier())
                continue
            
            # Operators
            elif char in self.OPERATORS:
                # Check for two-character operators
                if self.position + 1 < len(self.source):
                    two_char = char + self.source[self.position + 1]
                    if two_char in self.OPERATORS:
                        tokens.append(Token(
                            self.OPERATORS[two_char],
                            two_char,
                            self.line,
                            self.col
                        ))
                        self.advance(2)
                        continue
                
                tokens.append(Token(
                    self.OPERATORS[char],
                    char,
                    self.line,
                    self.col
                ))
                self.advance()
                continue
            
            # Delimiters
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, char, self.line, self.col))
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, char, self.line, self.col))
            elif char == '{':
                tokens.append(Token(TokenType.LBRACE, char, self.line, self.col))
            elif char == '}':
                tokens.append(Token(TokenType.RBRACE, char, self.line, self.col))
            elif char == '[':
                tokens.append(Token(TokenType.LBRACKET, char, self.line, self.col))
            elif char == ']':
                tokens.append(Token(TokenType.RBRACKET, char, self.line, self.col))
            elif char == ',':
                tokens.append(Token(TokenType.COMMA, char, self.line, self.col))
            elif char == '.':
                tokens.append(Token(TokenType.DOT, char, self.line, self.col))
            elif char == ':':
                tokens.append(Token(TokenType.COLON, char, self.line, self.col))
            elif char == ';':
                tokens.append(Token(TokenType.SEMICOLON, char, self.line, self.col))
            else:
                raise SyntaxError(f"Unexpected character '{char}' at line {self.line}, col {self.col}")
            
            self.advance()
        
        tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return tokens
    
    def advance(self, n=1):
        self.position += n
        self.col += n
    
    def read_string(self, quote_char: str) -> Token:
        start_line = self.line
        start_col = self.col
        self.advance()  # Skip opening quote
        
        value = ""
        while self.position < len(self.source) and self.source[self.position] != quote_char:
            char = self.source[self.position]
            if char == '\\':
                self.advance()
                if self.position < len(self.source):
                    esc_char = self.source[self.position]
                    if esc_char == 'n':
                        value += '\n'
                    elif esc_char == 't':
                        value += '\t'
                    elif esc_char == 'r':
                        value += '\r'
                    elif esc_char == '\\':
                        value += '\\'
                    elif esc_char == quote_char:
                        value += quote_char
                    else:
                        value += esc_char
            else:
                value += char
            
            if char == '\n':
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            
            self.advance()
        
        if self.position >= len(self.source):
            raise SyntaxError(f"Unterminated string at line {start_line}")
        
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, value, start_line, start_col)
    
    def read_number(self) -> Token:
        start_line = self.line
        start_col = self.col
        
        value = ""
        has_dot = False
        
        while self.position < len(self.source):
            char = self.source[self.position]
            if char.isdigit():
                value += char
            elif char == '.' and not has_dot:
                value += char
                has_dot = True
            else:
                break
            self.advance()
        
        return Token(
            TokenType.NUMBER,
            float(value) if has_dot else int(value),
            start_line,
            start_col
        )
    
    def read_identifier(self) -> Token:
        start_line = self.line
        start_col = self.col
        
        value = ""
        while self.position < len(self.source):
            char = self.source[self.position]
            if char.isalnum() or char == '_':
                value += char
            else:
                break
            self.advance()
        
        # Check if it's a keyword
        token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, start_line, start_col)

# ==================== PARSER ====================
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def parse(self) -> ASTNode:
        statements = []
        while not self.is_at_end():
            statements.append(self.statement())
        return ASTNode(NodeType.BLOCK, children=statements)
    
    def statement(self) -> ASTNode:
        if self.match(TokenType.IF):
            return self.if_statement()
        elif self.match(TokenType.WHILE):
            return self.while_statement()
        elif self.match(TokenType.FOR):
            return self.for_statement()
        elif self.match(TokenType.FUNCTION):
            return self.function_statement()
        elif self.match(TokenType.RETURN):
            return self.return_statement()
        elif self.match(TokenType.IMPORT):
            return self.import_statement()
        elif self.match(TokenType.TRY):
            return self.try_catch_statement()
        elif self.match(TokenType.THROW):
            return self.throw_statement()
        elif self.match(TokenType.CLASS):
            return self.class_statement()
        elif self.check(TokenType.IDENTIFIER) and self.check_next(TokenType.ASSIGN):
            return self.assignment_statement()
        else:
            return self.expression_statement()
    
    def if_statement(self) -> ASTNode:
        condition = self.expression()
        self.consume(TokenType.LBRACE, "Expect '{' after if condition")
        
        then_branch = self.block()
        
        elif_branches = []
        while self.match(TokenType.ELIF):
            elif_cond = self.expression()
            self.consume(TokenType.LBRACE, "Expect '{' after elif condition")
            elif_branch = self.block()
            elif_branches.extend([elif_cond, elif_branch])
        
        else_branch = None
        if self.match(TokenType.ELSE):
            self.consume(TokenType.LBRACE, "Expect '{' after else")
            else_branch = self.block()
        
        # Create a chain of if-elif-else nodes
        result = ASTNode(NodeType.IF, children=[condition, then_branch, else_branch])
        if elif_branches:
            result.children.extend(elif_branches)
        return result
    
    def while_statement(self) -> ASTNode:
        condition = self.expression()
        self.consume(TokenType.LBRACE, "Expect '{' after while condition")
        body = self.block()
        return ASTNode(NodeType.WHILE, children=[condition, body])
    
    def for_statement(self) -> ASTNode:
        self.consume(TokenType.LPAREN, "Expect '(' after 'for'")
        
        # Initializer
        if self.match(TokenType.SEMICOLON):
            initializer = None
        else:
            initializer = self.expression_statement()
        
        # Condition
        condition = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after loop condition")
        
        # Increment
        increment = None
        if not self.check(TokenType.RPAREN):
            increment = self.expression()
        self.consume(TokenType.RPAREN, "Expect ')' after for clauses")
        
        self.consume(TokenType.LBRACE, "Expect '{' after for")
        body = self.block()
        
        return ASTNode(NodeType.FOR, children=[initializer, condition, increment, body])
    
    def function_statement(self) -> ASTNode:
        name = self.consume(TokenType.IDENTIFIER, "Expect function name").value
        
        self.consume(TokenType.LPAREN, "Expect '(' after function name")
        params = []
        if not self.check(TokenType.RPAREN):
            params.append(self.consume(TokenType.IDENTIFIER, "Expect parameter name").value)
            while self.match(TokenType.COMMA):
                params.append(self.consume(TokenType.IDENTIFIER, "Expect parameter name").value)
        self.consume(TokenType.RPAREN, "Expect ')' after parameters")
        
        self.consume(TokenType.LBRACE, "Expect '{' before function body")
        body = self.block()
        
        return ASTNode(NodeType.FUNCTION, value=name, children=[ASTNode(NodeType.BLOCK, children=body.children)] + [
            ASTNode(NodeType.IDENTIFIER, value=param) for param in params
        ])
    
    def block(self) -> ASTNode:
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.statement())
        self.consume(TokenType.RBRACE, "Expect '}' after block")
        return ASTNode(NodeType.BLOCK, children=statements)
    
    def expression(self) -> ASTNode:
        return self.assignment()
    
    def assignment(self) -> ASTNode:
        expr = self.logical_or()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_EQ, TokenType.MINUS_EQ, TokenType.MULT_EQ, TokenType.DIV_EQ):
            operator = self.previous()
            value = self.assignment()
            
            if expr.type != NodeType.IDENTIFIER:
                raise SyntaxError("Invalid assignment target")
            
            return ASTNode(NodeType.ASSIGNMENT, value=operator.type, children=[expr, value])
        
        return expr
    
    def logical_or(self) -> ASTNode:
        expr = self.logical_and()
        
        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.logical_and()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def logical_and(self) -> ASTNode:
        expr = self.equality()
        
        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def equality(self) -> ASTNode:
        expr = self.comparison()
        
        while self.match(TokenType.EQ, TokenType.NEQ):
            operator = self.previous()
            right = self.comparison()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def comparison(self) -> ASTNode:
        expr = self.term()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            operator = self.previous()
            right = self.term()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def term(self) -> ASTNode:
        expr = self.factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.factor()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def factor(self) -> ASTNode:
        expr = self.power()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.previous()
            right = self.power()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def power(self) -> ASTNode:
        expr = self.unary()
        
        while self.match(TokenType.POWER):
            operator = self.previous()
            right = self.unary()
            expr = ASTNode(NodeType.BINARY_OP, value=operator.type, children=[expr, right])
        
        return expr
    
    def unary(self) -> ASTNode:
        if self.match(TokenType.NOT, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return ASTNode(NodeType.UNARY_OP, value=operator.type, children=[right])
        
        return self.call()
    
    def call(self) -> ASTNode:
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expect property name after '.'")
                expr = ASTNode(NodeType.INDEX, value=name.value, children=[expr])
            elif self.match(TokenType.LBRACKET):
                index = self.expression()
                self.consume(TokenType.RBRACKET, "Expect ']' after index")
                expr = ASTNode(NodeType.INDEX, children=[expr, index])
            else:
                break
        
        return expr
    
    def finish_call(self, callee: ASTNode) -> ASTNode:
        arguments = []
        if not self.check(TokenType.RPAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
        
        self.consume(TokenType.RPAREN, "Expect ')' after arguments")
        return ASTNode(NodeType.CALL, children=[callee] + arguments)
    
    def primary(self) -> ASTNode:
        if self.match(TokenType.TRUE):
            return ASTNode(NodeType.BOOLEAN, value=True)
        elif self.match(TokenType.FALSE):
            return ASTNode(NodeType.BOOLEAN, value=False)
        elif self.match(TokenType.NULL):
            return ASTNode(NodeType.IDENTIFIER, value=None)
        elif self.match(TokenType.NUMBER, TokenType.STRING):
            token_type = self.previous().type
            value = self.previous().value
            node_type = NodeType.NUMBER if token_type == TokenType.NUMBER else NodeType.STRING
            return ASTNode(node_type, value=value)
        elif self.match(TokenType.IDENTIFIER):
            return ASTNode(NodeType.IDENTIFIER, value=self.previous().value)
        elif self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expect ')' after expression")
            return expr
        elif self.match(TokenType.LBRACKET):
            return self.list_literal()
        elif self.match(TokenType.LBRACE):
            return self.dict_literal()
        
        raise SyntaxError(f"Expect expression, got {self.peek().type}")
    
    def list_literal(self) -> ASTNode:
        elements = []
        if not self.check(TokenType.RBRACKET):
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                elements.append(self.expression())
        
        self.consume(TokenType.RBRACKET, "Expect ']' after list elements")
        return ASTNode(NodeType.LIST, children=elements)
    
    def dict_literal(self) -> ASTNode:
        elements = []
        if not self.check(TokenType.RBRACE):
            key = self.expression()
            self.consume(TokenType.COLON, "Expect ':' after key")
            value = self.expression()
            elements.extend([key, value])
            
            while self.match(TokenType.COMMA):
                key = self.expression()
                self.consume(TokenType.COLON, "Expect ':' after key")
                value = self.expression()
                elements.extend([key, value])
        
        self.consume(TokenType.RBRACE, "Expect '}' after dictionary elements")
        return ASTNode(NodeType.DICT, children=elements)
    
    def match(self, *types: TokenType) -> bool:
        for t in types:
            if self.check(t):
                self.advance()
                return True
        return False
    
    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def check_next(self, token_type: TokenType) -> bool:
        if self.position + 1 >= len(self.tokens):
            return False
        return self.tokens[self.position + 1].type == token_type
    
    def advance(self) -> Token:
        if not self.is_at_end():
            self.position += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        return self.tokens[self.position]
    
    def previous(self) -> Token:
        return self.tokens[self.position - 1]
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.check(token_type):
            return self.advance()
        raise SyntaxError(f"{message} at line {self.peek().line}")
    
    def expression_statement(self) -> ASTNode:
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after expression")
        return expr
    
    def return_statement(self) -> ASTNode:
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after return value")
        return ASTNode(NodeType.RETURN, children=[value] if value else [])
    
    def import_statement(self) -> ASTNode:
        module = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after import")
        return ASTNode(NodeType.IMPORT, children=[module])
    
    def try_catch_statement(self) -> ASTNode:
        self.consume(TokenType.LBRACE, "Expect '{' after 'try'")
        try_block = self.block()
        
        self.consume(TokenType.CATCH, "Expect 'catch' after try block")
        error_var = self.consume(TokenType.IDENTIFIER, "Expect error variable name").value
        
        self.consume(TokenType.LBRACE, "Expect '{' after catch")
        catch_block = self.block()
        
        return ASTNode(NodeType.TRY_CATCH, value=error_var, children=[try_block, catch_block])
    
    def throw_statement(self) -> ASTNode:
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after throw")
        return ASTNode(NodeType.CALL, value="throw", children=[value])
    
    def class_statement(self) -> ASTNode:
        name = self.consume(TokenType.IDENTIFIER, "Expect class name").value
        
        self.consume(TokenType.LBRACE, "Expect '{' after class name")
        
        methods = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            if not self.match(TokenType.FUNCTION):
                raise SyntaxError("Expect method declaration in class")
            methods.append(self.function_statement())
        
        self.consume(TokenType.RBRACE, "Expect '}' after class body")
        
        # Create a dictionary of methods
        methods_dict = {}
        for method in methods:
            methods_dict[method.value] = method
        
        return ASTNode(NodeType.ASSIGNMENT, children=[
            ASTNode(NodeType.IDENTIFIER, value=name),
            ASTNode(NodeType.DICT, children=[
                ASTNode(NodeType.STRING, value=key) if i % 2 == 0 else value
                for i, (key, value) in enumerate(methods_dict.items())
                for item in [key, value]
            ])
        ])
    
    def assignment_statement(self) -> ASTNode:
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name").value
        token = self.consume(TokenType.ASSIGN, "Expect '=' after variable name")
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after assignment")
        
        return ASTNode(NodeType.ASSIGNMENT, value=token.type, children=[
            ASTNode(NodeType.IDENTIFIER, value=name),
            value
        ])

# ==================== RUNTIME ====================
class NouhaRuntimeError(Exception):
    def __init__(self, message: str, line: int = None):
        super().__init__(message)
        self.message = message
        self.line = line

class Environment:
    def __init__(self, parent: 'Environment' = None):
        self.values = {}
        self.parent = parent
    
    def define(self, name: str, value: Any):
        self.values[name] = value
    
    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NouhaRuntimeError(f"Undefined variable '{name}'")
    
    def assign(self, name: str, value: Any):
        if name in self.values:
            self.values[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise NouhaRuntimeError(f"Undefined variable '{name}'")
    
    def get_at(self, distance: int, name: str) -> Any:
        return self.ancestor(distance).values.get(name)
    
    def assign_at(self, distance: int, name: str, value: Any):
        self.ancestor(distance).values[name] = value
    
    def ancestor(self, distance: int) -> 'Environment':
        env = self
        for _ in range(distance):
            env = env.parent
        return env

class NouhaFunction:
    def __init__(self, declaration: ASTNode, closure: Environment, is_initializer: bool = False):
        self.declaration = declaration
        self.closure = closure
        self.is_initializer = is_initializer
    
    def call(self, interpreter: 'Interpreter', arguments: List[Any]) -> Any:
        env = Environment(self.closure)
        
        # Bind parameters
        for i, param in enumerate(self.declaration.children[1:]):
            env.define(param.value, arguments[i] if i < len(arguments) else None)
        
        try:
            interpreter.execute_block(self.declaration.children[0].children, env)
        except ReturnException as ret:
            if self.is_initializer:
                return self.closure.get_at(0, "this")
            return ret.value
        
        if self.is_initializer:
            return self.closure.get_at(0, "this")
        return None
    
    def arity(self) -> int:
        return len(self.declaration.children) - 1

class ReturnException(Exception):
    def __init__(self, value: Any):
        self.value = value

class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        self.locals = {}
        
        # Built-in functions
        self.globals.define("print", self.make_native_fn(self._print, 1))
        self.globals.define("input", self.make_native_fn(self._input, 0))
        self.globals.define("len", self.make_native_fn(self._len, 1))
        self.globals.define("type", self.make_native_fn(self._type, 1))
        self.globals.define("int", self.make_native_fn(self._to_int, 1))
        self.globals.define("float", self.make_native_fn(self._to_float, 1))
        self.globals.define("str", self.make_native_fn(self._to_string, 1))
        self.globals.define("list", self.make_native_fn(self._list, -1))
        self.globals.define("dict", self.make_native_fn(self._dict, -1))
        self.globals.define("append", self.make_native_fn(self._append, 2))
        self.globals.define("pop", self.make_native_fn(self._pop, 1))
        self.globals.define("insert", self.make_native_fn(self._insert, 3))
        self.globals.define("range", self.make_native_fn(self._range, 3))
        
        # Math functions
        self.globals.define("math", {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
        })
    
    def make_native_fn(self, func: Callable, arity: int) -> Any:
        def wrapper(*args):
            return func(*args)
        wrapper.arity = arity
        wrapper.call = lambda interpreter, args: func(*args)
        return wrapper
    
    def _print(self, value: Any) -> None:
        print(self.stringify(value))
        return None
    
    def _input(self, prompt: str = "") -> str:
        return input(prompt)
    
    def _len(self, value: Any) -> int:
        if isinstance(value, (list, dict, str)):
            return len(value)
        raise NouhaRuntimeError(f"Object of type '{type(value).__name__}' has no length")
    
    def _type(self, value: Any) -> str:
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, NouhaFunction):
            return "function"
        else:
            return "object"
    
    def _to_int(self, value: Any) -> int:
        try:
            return int(value)
        except ValueError:
            raise NouhaRuntimeError(f"Cannot convert '{value}' to integer")
    
    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except ValueError:
            raise NouhaRuntimeError(f"Cannot convert '{value}' to float")
    
    def _to_string(self, value: Any) -> str:
        return self.stringify(value)
    
    def _list(self, *args) -> list:
        return list(args)
    
    def _dict(self, *args) -> dict:
        if len(args) % 2 != 0:
            raise NouhaRuntimeError("Dictionary requires an even number of arguments")
        result = {}
        for i in range(0, len(args), 2):
            key = args[i]
            if not isinstance(key, (str, int, float, bool)):
                raise NouhaRuntimeError("Dictionary keys must be strings, numbers, or booleans")
            result[key] = args[i + 1]
        return result
    
    def _append(self, lst: list, value: Any) -> list:
        if not isinstance(lst, list):
            raise NouhaRuntimeError("First argument must be a list")
        lst.append(value)
        return lst
    
    def _pop(self, lst: list, index: int = -1) -> Any:
        if not isinstance(lst, list):
            raise NouhaRuntimeError("Argument must be a list")
        return lst.pop(index)
    
    def _insert(self, lst: list, index: int, value: Any) -> list:
        if not isinstance(lst, list):
            raise NouhaRuntimeError("First argument must be a list")
        lst.insert(index, value)
        return lst
    
    def _range(self, start: int, stop: int, step: int = 1) -> list:
        return list(range(start, stop, step))
    
    def interpret(self, source: str):
        try:
            # Tokenize
            tokenizer = Tokenizer(source)
            tokens = tokenizer.tokenize()
            
            # Parse
            parser = Parser(tokens)
            statements = parser.parse()
            
            # Execute
            self.execute(statements)
        except SyntaxError as e:
            self.report_error(str(e))
        except NouhaRuntimeError as e:
            self.report_error(f"Runtime error: {e.message}")
        except Exception as e:
            self.report_error(f"Unexpected error: {e}")
    
    def execute(self, stmt: ASTNode):
        if stmt.type == NodeType.BLOCK:
            self.execute_block(stmt.children, Environment(self.environment))
        elif stmt.type == NodeType.IF:
            self.execute_if(stmt)
        elif stmt.type == NodeType.WHILE:
            self.execute_while(stmt)
        elif stmt.type == NodeType.FOR:
            self.execute_for(stmt)
        elif stmt.type == NodeType.FUNCTION:
            self.execute_function(stmt)
        elif stmt.type == NodeType.RETURN:
            self.execute_return(stmt)
        elif stmt.type == NodeType.IMPORT:
            self.execute_import(stmt)
        elif stmt.type == NodeType.TRY_CATCH:
            self.execute_try_catch(stmt)
        elif stmt.type == NodeType.ASSIGNMENT:
            self.execute_assignment(stmt)
        elif stmt.type == NodeType.CALL and stmt.value == "throw":
            self.execute_throw(stmt)
        else:
            self.evaluate(stmt)
    
    def execute_block(self, statements: List[ASTNode], env: Environment):
        previous = self.environment
        try:
            self.environment = env
            for stmt in statements:
                self.execute(stmt)
        finally:
            self.environment = previous
    
    def execute_if(self, stmt: ASTNode):
        # stmt.children: [condition, then_branch, else_branch, elif_cond1, elif_branch1, ...]
        condition = self.evaluate(stmt.children[0])
        
        if self.is_truthy(condition):
            self.execute(stmt.children[1])
            return
        
        # Check elif branches
        i = 3  # Start at first elif condition
        while i < len(stmt.children):
            elif_cond = self.evaluate(stmt.children[i])
            if self.is_truthy(elif_cond):
                self.execute(stmt.children[i + 1])
                return
            i += 2
        
        # Execute else branch if exists
        if stmt.children[2]:
            self.execute(stmt.children[2])
    
    def execute_while(self, stmt: ASTNode):
        while self.is_truthy(self.evaluate(stmt.children[0])):
            self.execute(stmt.children[1])
    
    def execute_for(self, stmt: ASTNode):
        if stmt.children[0]:
            self.execute(stmt.children[0])
        
        while True:
            if stmt.children[1]:
                condition = self.evaluate(stmt.children[1])
                if not self.is_truthy(condition):
                    break
            
            self.execute(stmt.children[3])
            
            if stmt.children[2]:
                self.evaluate(stmt.children[2])
    
    def execute_function(self, stmt: ASTNode):
        func = NouhaFunction(stmt, self.environment)
        self.environment.define(stmt.value, func)
    
    def execute_return(self, stmt: ASTNode):
        value = None
        if stmt.children:
            value = self.evaluate(stmt.children[0])
        raise ReturnException(value)
    
    def execute_import(self, stmt: ASTNode):
        module_name = self.evaluate(stmt.children[0])
        
        # Built-in modules
        if module_name == "math":
            self.environment.define("math", self.globals.get("math"))
        elif module_name == "time":
            self.environment.define("time", {
                "now": time.time,
                "sleep": time.sleep,
                "ctime": time.ctime,
                "strftime": time.strftime,
            })
        elif module_name == "random":
            self.environment.define("random", {
                "randint": random.randint,
                "choice": random.choice,
                "random": random.random,
                "shuffle": random.shuffle,
                "uniform": random.uniform,
            })
        elif module_name == "os":
            self.environment.define("os", {
                "getcwd": os.getcwd,
                "listdir": os.listdir,
                "isfile": os.path.isfile,
                "isdir": os.path.isdir,
                "exists": os.path.exists,
            })
        elif module_name == "json":
            self.environment.define("json", {
                "loads": json.loads,
                "dumps": json.dumps,
            })
        else:
            # Try to load from file
            filename = f"{module_name}.nh"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    source = f.read()
                # Execute in a new environment
                previous_env = self.environment
                module_env = Environment(self.globals)
                self.environment = module_env
                self.interpret(source)
                self.environment = previous_env
                
                # Export all defined symbols
                for name in module_env.values:
                    self.environment.define(name, module_env.values[name])
            else:
                raise NouhaRuntimeError(f"Module '{module_name}' not found")
    
    def execute_try_catch(self, stmt: ASTNode):
        try:
            self.execute(stmt.children[0])
        except Exception as e:
            error_env = Environment(self.environment)
            error_env.define(stmt.value, str(e))
            previous = self.environment
            self.environment = error_env
            self.execute(stmt.children[1])
            self.environment = previous
    
    def execute_assignment(self, stmt: ASTNode):
        value = self.evaluate(stmt.children[1])
        
        if stmt.children[0].type == NodeType.IDENTIFIER:
            var_name = stmt.children[0].value
            
            if stmt.value == TokenType.PLUS_EQ:
                current = self.environment.get(var_name)
                value = self.add(current, value)
            elif stmt.value == TokenType.MINUS_EQ:
                current = self.environment.get(var_name)
                value = self.subtract(current, value)
            elif stmt.value == TokenType.MULT_EQ:
                current = self.environment.get(var_name)
                value = self.multiply(current, value)
            elif stmt.value == TokenType.DIV_EQ:
                current = self.environment.get(var_name)
                value = self.divide(current, value)
            
            self.environment.assign(var_name, value)
        else:
            raise NouhaRuntimeError("Invalid assignment target")
    
    def execute_throw(self, stmt: ASTNode):
        value = self.evaluate(stmt.children[0])
        raise NouhaRuntimeError(str(value))
    
    def evaluate(self, expr: ASTNode) -> Any:
        if expr is None:
            return None
        
        if expr.type == NodeType.NUMBER:
            return expr.value
        elif expr.type == NodeType.STRING:
            return expr.value
        elif expr.type == NodeType.BOOLEAN:
            return expr.value
        elif expr.type == NodeType.IDENTIFIER:
            if expr.value is None:
                return None
            return self.environment.get(expr.value)
        elif expr.type == NodeType.BINARY_OP:
            left = self.evaluate(expr.children[0])
            right = self.evaluate(expr.children[1])
            
            if expr.value == TokenType.PLUS:
                return self.add(left, right)
            elif expr.value == TokenType.MINUS:
                return self.subtract(left, right)
            elif expr.value == TokenType.MULTIPLY:
                return self.multiply(left, right)
            elif expr.value == TokenType.DIVIDE:
                return self.divide(left, right)
            elif expr.value == TokenType.POWER:
                return self.power(left, right)
            elif expr.value == TokenType.MODULO:
                return self.modulo(left, right)
            elif expr.value == TokenType.EQ:
                return self.is_equal(left, right)
            elif expr.value == TokenType.NEQ:
                return not self.is_equal(left, right)
            elif expr.value == TokenType.LT:
                return self.less_than(left, right)
            elif expr.value == TokenType.GT:
                return self.greater_than(left, right)
            elif expr.value == TokenType.LTE:
                return self.less_than_equal(left, right)
            elif expr.value == TokenType.GTE:
                return self.greater_than_equal(left, right)
            elif expr.value == TokenType.AND:
                return self.is_truthy(left) and self.is_truthy(right)
            elif expr.value == TokenType.OR:
                return self.is_truthy(left) or self.is_truthy(right)
        elif expr.type == NodeType.UNARY_OP:
            right = self.evaluate(expr.children[0])
            
            if expr.value == TokenType.NOT:
                return not self.is_truthy(right)
            elif expr.value == TokenType.MINUS:
                if not isinstance(right, (int, float)):
                    raise NouhaRuntimeError("Operand must be a number")
                return -right
        elif expr.type == NodeType.CALL:
            callee = self.evaluate(expr.children[0])
            
            arguments = []
            for arg in expr.children[1:]:
                arguments.append(self.evaluate(arg))
            
            if not callable(callee) and not hasattr(callee, 'call'):
                if isinstance(callee, dict) and len(arguments) == 1 and arguments[0] in callee:
                    return callee[arguments[0]]
                raise NouhaRuntimeError("Can only call functions and classes")
            
            if hasattr(callee, 'call'):
                if callee.arity() >= 0 and len(arguments) != callee.arity():
                    raise NouhaRuntimeError(f"Expected {callee.arity()} arguments but got {len(arguments)}")
                return callee.call(self, arguments)
            else:
                return callee(*arguments)
        elif expr.type == NodeType.LIST:
            elements = []
            for child in expr.children:
                elements.append(self.evaluate(child))
            return elements
        elif expr.type == NodeType.DICT:
            result = {}
            for i in range(0, len(expr.children), 2):
                key = self.evaluate(expr.children[i])
                value = self.evaluate(expr.children[i + 1])
                if not isinstance(key, (str, int, float, bool)):
                    raise NouhaRuntimeError("Dictionary keys must be strings, numbers, or booleans")
                result[key] = value
            return result
        elif expr.type == NodeType.INDEX:
            obj = self.evaluate(expr.children[0])
            
            if len(expr.children) == 2:
                # List/Dict indexing
                index = self.evaluate(expr.children[1])
                
                if isinstance(obj, list):
                    if not isinstance(index, int):
                        raise NouhaRuntimeError("List index must be an integer")
                    if index < 0 or index >= len(obj):
                        raise NouhaRuntimeError("List index out of bounds")
                    return obj[index]
                elif isinstance(obj, dict):
                    if index not in obj:
                        return None
                    return obj[index]
                elif isinstance(obj, str):
                    if not isinstance(index, int):
                        raise NouhaRuntimeError("String index must be an integer")
                    if index < 0 or index >= len(obj):
                        raise NouhaRuntimeError("String index out of bounds")
                    return obj[index]
                else:
                    raise NouhaRuntimeError("Cannot index this object type")
            else:
                # Object property access
                if not isinstance(obj, dict):
                    raise NouhaRuntimeError("Cannot access properties on non-object")
                if expr.value not in obj:
                    return None
                return obj[expr.value]
        
        return None
    
    def add(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right
        elif isinstance(left, str) or isinstance(right, str):
            return str(left) + str(right)
        elif isinstance(left, list) and isinstance(right, list):
            return left + right
        else:
            raise NouhaRuntimeError(f"Cannot add {type(left).__name__} and {type(right).__name__}")
    
    def subtract(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left - right
        else:
            raise NouhaRuntimeError(f"Cannot subtract {type(left).__name__} and {type(right).__name__}")
    
    def multiply(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left * right
        elif isinstance(left, str) and isinstance(right, int):
            return left * right
        elif isinstance(left, int) and isinstance(right, str):
            return right * left
        elif isinstance(left, list) and isinstance(right, int):
            return left * right
        elif isinstance(left, int) and isinstance(right, list):
            return right * left
        else:
            raise NouhaRuntimeError(f"Cannot multiply {type(left).__name__} and {type(right).__name__}")
    
    def divide(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                raise NouhaRuntimeError("Division by zero")
            return left / right
        else:
            raise NouhaRuntimeError(f"Cannot divide {type(left).__name__} and {type(right).__name__}")
    
    def power(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left ** right
        else:
            raise NouhaRuntimeError(f"Cannot exponentiate {type(left).__name__} and {type(right).__name__}")
    
    def modulo(self, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                raise NouhaRuntimeError("Modulo by zero")
            return left % right
        else:
            raise NouhaRuntimeError(f"Cannot modulo {type(left).__name__} and {type(right).__name__}")
    
    def is_equal(self, left: Any, right: Any) -> bool:
        return left == right
    
    def less_than(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left < right
        elif isinstance(left, str) and isinstance(right, str):
            return left < right
        else:
            raise NouhaRuntimeError(f"Cannot compare {type(left).__name__} and {type(right).__name__}")
    
    def greater_than(self, left: Any, right: Any) -> bool:
        return self.less_than(right, left)
    
    def less_than_equal(self, left: Any, right: Any) -> bool:
        return not self.greater_than(left, right)
    
    def greater_than_equal(self, left: Any, right: Any) -> bool:
        return not self.less_than(left, right)
    
    def is_truthy(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True
    
    def stringify(self, value: Any) -> str:
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return str(value)
        elif isinstance(value, list):
            items = [self.stringify(item) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            items = [f"{self.stringify(k)}: {self.stringify(v)}" for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return str(value)
    
    def report_error(self, message: str):
        print(f"Error: {message}", file=sys.stderr)

# ==================== REPL & FILE EXECUTION ====================
class Nouha:
    def __init__(self):
        self.interpreter = Interpreter()
    
    def run_file(self, filename: str):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                source = f.read()
            self.interpreter.interpret(source)
        except FileNotFoundError:
            print(f"File '{filename}' not found")
        except Exception as e:
            print(f"Error: {e}")
    
    def run_prompt(self):
        print("Nouha Language v1.0 - Type 'exit' to quit")
        print("Help: Type 'help' to see available commands")
        
        while True:
            try:
                line = input(">>> ")
                if line.strip() == "exit":
                    break
                elif line.strip() == "help":
                    self.show_help()
                elif line.strip() == "clear":
                    os.system('clear' if os.name == 'posix' else 'cls')
                else:
                    self.interpreter.interpret(line)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_help(self):
        help_text = """
        ======== Nouha Language - Commands ========
        
        Code Examples:
        
        # Variables and Operations
        x = 10;
        y = 20;
        sum = x + y;
        print(sum);
        
        # Conditionals
        if (x > y) {
          print("x is greater than y");
        } else {
          print("x is less than y");
        }
        
        # Loops
        i = 0;
        while (i < 5) {
          print(i);
          i = i + 1;
        }
        
        # Functions
        func sum(a, b) {
          return a + b;
        }
        result = sum(10, 20);
        print(result);
        
        # Lists
        lst = [1, 2, 3, 4, 5];
        print(len(lst));
        
        # Dictionaries
        person = {"name": "Alice", "age": 25};
        print(person["name"]);
        
        # Try-Catch
        try {
          result = 10 / 0;
        } catch (error) {
          print("Error: " + error);
        }
        
        # Special Commands:
        exit    - Quit the program
        clear   - Clear the screen
        help    - Show this message
        
        ===========================================
        """
        print(help_text)

# ==================== MAIN ====================
def main():
    nouha = Nouha()
    
    if len(sys.argv) == 1:
        nouha.run_prompt()
    elif len(sys.argv) == 2:
        nouha.run_file(sys.argv[1])
    else:
        print("Usage: python nouha.py [filename]")

if __name__ == "__main__":
    main()
