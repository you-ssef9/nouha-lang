#!/usr/bin/env python3
"""
Nouha Programming Language - Advanced Interpreter
Version 2.0 - Enterprise Edition
"""

import sys
import os
import re
import math
import json
import time
import random
import datetime
import hashlib
import inspect
import threading
import multiprocessing
import asyncio
import collections
import typing
import decimal
import fractions
import itertools
import statistics
import textwrap
import string
import csv
import pickle
import sqlite3
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path

# ==================== TYPE SYSTEM ====================
class NouhaType(Enum):
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    FUNCTION = auto()
    CLASS = auto()
    OBJECT = auto()
    NULL = auto()
    MODULE = auto()
    ITERATOR = auto()
    COROUTINE = auto()

@dataclass
class NouhaValue:
    type: NouhaType
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"NouhaValue({self.type.name}: {repr(self.value)})"
    
    def copy(self):
        return NouhaValue(self.type, self.value, self.metadata.copy())

# ==================== AST & PARSER ====================
class NodeType(Enum):
    PROGRAM = auto()
    BLOCK = auto()
    STATEMENT = auto()
    EXPRESSION = auto()
    LITERAL = auto()
    IDENTIFIER = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    ASSIGNMENT = auto()
    FUNCTION_DEF = auto()
    FUNCTION_CALL = auto()
    CLASS_DEF = auto()
    METHOD_CALL = auto()
    IF_STATEMENT = auto()
    WHILE_LOOP = auto()
    FOR_LOOP = auto()
    TRY_CATCH = auto()
    IMPORT = auto()
    EXPORT = auto()
    AWAIT_EXPR = auto()
    YIELD_EXPR = auto()
    LAMBDA = auto()
    LIST_COMP = auto()
    DICT_COMP = auto()
    GENERATOR = auto()
    SLICE = auto()
    TUPLE = auto()
    SET = auto()

@dataclass
class ASTNode:
    node_type: NodeType
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    line: int = 0
    column: int = 0
    annotations: Dict[str, Any] = field(default_factory=dict)

class TokenType(Enum):
    # Keywords
    LET = auto()
    CONST = auto()
    FUNC = auto()
    CLASS = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    YIELD = auto()
    ASYNC = auto()
    AWAIT = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()
    THROW = auto()
    IMPORT = auto()
    EXPORT = auto()
    FROM = auto()
    AS = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    UNDEFINED = auto()
    TYPE = auto()
    INTERFACE = auto()
    ENUM = auto()
    MATCH = auto()
    CASE = auto()
    DEFAULT = auto()
    
    # Types
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    
    # Comparison
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    
    # Assignment
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    MULT_ASSIGN = auto()
    DIV_ASSIGN = auto()
    MOD_ASSIGN = auto()
    AND_ASSIGN = auto()
    OR_ASSIGN = auto()
    XOR_ASSIGN = auto()
    SHIFT_LEFT_ASSIGN = auto()
    SHIFT_RIGHT_ASSIGN = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()
    SPREAD = auto()
    QUESTION = auto()
    DOUBLE_COLON = auto()
    AT = auto()
    
    # Special
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

class AdvancedLexer:
    """Advanced lexer with Unicode support and error recovery"""
    
    KEYWORDS = {
        'let': TokenType.LET,
        'const': TokenType.CONST,
        'func': TokenType.FUNC,
        'class': TokenType.CLASS,
        'if': TokenType.IF,
        'elif': TokenType.ELIF,
        'else': TokenType.ELSE,
        'while': TokenType.WHILE,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'return': TokenType.RETURN,
        'yield': TokenType.YIELD,
        'async': TokenType.ASYNC,
        'await': TokenType.AWAIT,
        'try': TokenType.TRY,
        'catch': TokenType.CATCH,
        'finally': TokenType.FINALLY,
        'throw': TokenType.THROW,
        'import': TokenType.IMPORT,
        'export': TokenType.EXPORT,
        'from': TokenType.FROM,
        'as': TokenType.AS,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'null': TokenType.NULL,
        'undefined': TokenType.UNDEFINED,
        'type': TokenType.TYPE,
        'interface': TokenType.INTERFACE,
        'enum': TokenType.ENUM,
        'match': TokenType.MATCH,
        'case': TokenType.CASE,
        'default': TokenType.DEFAULT,
    }
    
    OPERATORS = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.MULTIPLY,
        '/': TokenType.DIVIDE,
        '%': TokenType.MODULO,
        '**': TokenType.POWER,
        '++': TokenType.INCREMENT,
        '--': TokenType.DECREMENT,
        '==': TokenType.EQ,
        '!=': TokenType.NEQ,
        '<': TokenType.LT,
        '>': TokenType.GT,
        '<=': TokenType.LTE,
        '>=': TokenType.GTE,
        '&&': TokenType.AND,
        '||': TokenType.OR,
        '!': TokenType.NOT,
        '&': TokenType.BIT_AND,
        '|': TokenType.BIT_OR,
        '^': TokenType.BIT_XOR,
        '~': TokenType.BIT_NOT,
        '<<': TokenType.LEFT_SHIFT,
        '>>': TokenType.RIGHT_SHIFT,
        '=': TokenType.ASSIGN,
        '+=': TokenType.PLUS_ASSIGN,
        '-=': TokenType.MINUS_ASSIGN,
        '*=': TokenType.MULT_ASSIGN,
        '/=': TokenType.DIV_ASSIGN,
        '%=': TokenType.MOD_ASSIGN,
        '&=': TokenType.AND_ASSIGN,
        '|=': TokenType.OR_ASSIGN,
        '^=': TokenType.XOR_ASSIGN,
        '<<=': TokenType.SHIFT_LEFT_ASSIGN,
        '>>=': TokenType.SHIFT_RIGHT_ASSIGN,
        '=>': TokenType.ARROW,
        '...': TokenType.SPREAD,
        '?': TokenType.QUESTION,
        '::': TokenType.DOUBLE_COLON,
        '@': TokenType.AT,
    }
    
    def __init__(self, source: str, filename: str = "<string>"):
        self.source = source
        self.filename = filename
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.errors = []
    
    def tokenize(self) -> List[Token]:
        while not self.is_at_end():
            self.scan_token()
        self.add_token(TokenType.EOF, None)
        return self.tokens
    
    def scan_token(self):
        char = self.advance()
        
        # Whitespace
        if char in ' \t\r':
            self.column += 1
            return
        
        # Newline
        if char == '\n':
            self.line += 1
            self.column = 1
            return
        
        # Comments
        if char == '#':
            while self.peek() != '\n' and not self.is_at_end():
                self.advance()
            return
        
        if char == '/' and self.peek() == '/':
            while self.peek() != '\n' and not self.is_at_end():
                self.advance()
            return
        
        if char == '/' and self.peek() == '*':
            self.advance()  # Skip '*'
            while not (self.peek() == '*' and self.peek_next() == '/'):
                if self.is_at_end():
                    self.error("Unterminated multi-line comment")
                    return
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 0
                self.advance()
            self.advance()  # Skip '*'
            self.advance()  # Skip '/'
            return
        
        # Strings
        if char in ('"', "'", '`'):
            self.string(char)
            return
        
        # Numbers
        if char.isdigit() or (char == '.' and self.peek().isdigit()):
            self.number()
            return
        
        # Identifiers
        if char.isalpha() or char == '_' or char.isdigit():
            self.identifier()
            return
        
        # Operators
        if char in self.OPERATORS:
            # Check for multi-character operators
            if char + self.peek() in self.OPERATORS:
                double_op = char + self.peek()
                if double_op + self.peek_next() in self.OPERATORS:
                    triple_op = double_op + self.peek_next()
                    self.add_token(self.OPERATORS[triple_op], triple_op)
                    self.advance()
                    self.advance()
                else:
                    self.add_token(self.OPERATORS[double_op], double_op)
                    self.advance()
            else:
                self.add_token(self.OPERATORS[char], char)
            return
        
        # Delimiters
        delimiter_map = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
        }
        
        if char in delimiter_map:
            self.add_token(delimiter_map[char], char)
            return
        
        # Unknown character
        self.error(f"Unexpected character: '{char}'")
    
    def string(self, quote: str):
        start_line = self.line
        start_column = self.column - 1
        
        value = ''
        is_template = (quote == '`')
        
        while not (self.peek() == quote and not self.is_escaped()):
            if self.is_at_end():
                self.error("Unterminated string", start_line, start_column)
                return
            
            char = self.advance()
            
            if char == '\\':
                escape_char = self.advance()
                value += self.escape_sequence(escape_char)
            elif is_template and char == '$' and self.peek() == '{':
                # Template interpolation
                self.advance()  # Skip '{'
                value += char  # Keep '$' for now
            else:
                value += char
            
            if char == '\n':
                if not is_template:
                    self.error("Newline in string literal", start_line, start_column)
                    return
                self.line += 1
                self.column = 1
        
        self.advance()  # Skip closing quote
        self.add_token(TokenType.STRING, value)
    
    def number(self):
        start_column = self.column - 1
        value = ''
        is_float = False
        is_hex = False
        is_binary = False
        is_octal = False
        
        # Handle different number bases
        if self.peek(-1) == '0':
            if self.peek() in 'xX':
                is_hex = True
                self.advance()  # Skip 'x'
            elif self.peek() in 'bB':
                is_binary = True
                self.advance()  # Skip 'b'
            elif self.peek() in 'oO':
                is_octal = True
                self.advance()  # Skip 'o'
        
        while self.peek().isdigit() or \
              (is_hex and self.peek().lower() in 'abcdef') or \
              (self.peek() == '.' and not is_float) or \
              (self.peek().lower() == 'e'):
            
            char = self.advance()
            
            if char == '.':
                is_float = True
            elif char.lower() == 'e':
                is_float = True
                if self.peek() in '+-':
                    self.advance()
            
            value += char
        
        # Parse the number
        try:
            if is_hex:
                num = int(value, 16)
            elif is_binary:
                num = int(value, 2)
            elif is_octal:
                num = int(value, 8)
            elif is_float:
                num = float(value)
            else:
                num = int(value)
        except ValueError:
            self.error(f"Invalid number: {value}", self.line, start_column)
            return
        
        self.add_token(TokenType.NUMBER, num)
    
    def identifier(self):
        start_column = self.column - 1
        value = self.peek(-1)
        
        while self.peek().isalnum() or self.peek() == '_':
            value += self.advance()
        
        # Check if it's a keyword
        token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
        self.add_token(token_type, value)
    
    def escape_sequence(self, char: str) -> str:
        escapes = {
            'n': '\n',
            'r': '\r',
            't': '\t',
            'b': '\b',
            'f': '\f',
            'v': '\v',
            '0': '\0',
            "'": "'",
            '"': '"',
            '\\': '\\',
            'u': self.unicode_escape(4),
            'U': self.unicode_escape(8),
            'x': self.hex_escape(2),
        }
        return escapes.get(char, char)
    
    def unicode_escape(self, length: int) -> str:
        hex_str = ''
        for _ in range(length):
            hex_str += self.advance()
        try:
            return chr(int(hex_str, 16))
        except ValueError:
            self.error(f"Invalid Unicode escape: \\u{hex_str}")
            return ''
    
    def hex_escape(self, length: int) -> str:
        hex_str = ''
        for _ in range(length):
            hex_str += self.advance()
        try:
            return chr(int(hex_str, 16))
        except ValueError:
            self.error(f"Invalid hex escape: \\x{hex_str}")
            return ''
    
    def add_token(self, token_type: TokenType, value: Any):
        self.tokens.append(Token(token_type, value, self.line, self.column))
    
    def error(self, message: str, line: int = None, column: int = None):
        line = line or self.line
        column = column or self.column
        self.errors.append(f"{self.filename}:{line}:{column}: {message}")
    
    def advance(self) -> str:
        char = self.source[self.position]
        self.position += 1
        self.column += 1
        return char
    
    def peek(self, offset: int = 0) -> str:
        pos = self.position + offset
        if pos >= len(self.source):
            return '\0'
        return self.source[pos]
    
    def peek_next(self) -> str:
        return self.peek(1)
    
    def is_at_end(self) -> bool:
        return self.position >= len(self.source)
    
    def is_escaped(self) -> bool:
        count = 0
        pos = self.position - 1
        while pos >= 0 and self.source[pos] == '\\':
            count += 1
            pos -= 1
        return count % 2 == 1

# ==================== ADVANCED PARSER ====================
class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<string>"):
        self.tokens = tokens
        self.filename = filename
        self.current = 0
        self.errors = []
        self.ast = None
    
    def parse(self) -> ASTNode:
        try:
            statements = []
            while not self.is_at_end():
                statements.append(self.declaration())
            
            self.ast = ASTNode(NodeType.PROGRAM, children=statements)
            return self.ast
        except ParseError as e:
            self.errors.append(str(e))
            return None
    
    def declaration(self) -> ASTNode:
        try:
            if self.match(TokenType.LET, TokenType.CONST):
                return self.variable_declaration()
            elif self.match(TokenType.FUNC):
                return self.function_declaration()
            elif self.match(TokenType.CLASS):
                return self.class_declaration()
            elif self.match(TokenType.IMPORT):
                return self.import_declaration()
            elif self.match(TokenType.EXPORT):
                return self.export_declaration()
            elif self.match(TokenType.TYPE):
                return self.type_alias_declaration()
            elif self.match(TokenType.INTERFACE):
                return self.interface_declaration()
            elif self.match(TokenType.ENUM):
                return self.enum_declaration()
            else:
                return self.statement()
        except ParseError as e:
            self.synchronize()
            return ASTNode(NodeType.STATEMENT, value="error")
    
    def variable_declaration(self) -> ASTNode:
        is_const = self.previous().type == TokenType.CONST
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name")
        
        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.expression()
        
        self.consume(TokenType.SEMICOLON, "Expect ';' after variable declaration")
        
        return ASTNode(
            NodeType.ASSIGNMENT,
            value={"name": name.value, "const": is_const},
            children=[initializer] if initializer else []
        )
    
    def function_declaration(self) -> ASTNode:
        is_async = self.match(TokenType.ASYNC)
        name = self.consume(TokenType.IDENTIFIER, "Expect function name")
        
        self.consume(TokenType.LPAREN, "Expect '(' after function name")
        params = self.parameters()
        self.consume(TokenType.RPAREN, "Expect ')' after parameters")
        
        return_type = None
        if self.match(TokenType.COLON):
            return_type = self.type_annotation()
        
        self.consume(TokenType.LBRACE, "Expect '{' before function body")
        body = self.block()
        
        return ASTNode(
            NodeType.FUNCTION_DEF,
            value={
                "name": name.value,
                "async": is_async,
                "params": params,
                "return_type": return_type.value if return_type else None
            },
            children=[body]
        )
    
    def class_declaration(self) -> ASTNode:
        name = self.consume(TokenType.IDENTIFIER, "Expect class name")
        
        superclass = None
        if self.match(TokenType.COLON):
            superclass = self.consume(TokenType.IDENTIFIER, "Expect superclass name")
        
        interfaces = []
        if self.match(TokenType.IMPLEMENTS):
            interfaces.append(self.consume(TokenType.IDENTIFIER, "Expect interface name").value)
            while self.match(TokenType.COMMA):
                interfaces.append(self.consume(TokenType.IDENTIFIER, "Expect interface name").value)
        
        self.consume(TokenType.LBRACE, "Expect '{' before class body")
        
        methods = []
        fields = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            if self.match(TokenType.FUNC):
                methods.append(self.function_declaration())
            else:
                fields.append(self.variable_declaration())
        
        self.consume(TokenType.RBRACE, "Expect '}' after class body")
        
        return ASTNode(
            NodeType.CLASS_DEF,
            value={
                "name": name.value,
                "superclass": superclass.value if superclass else None,
                "interfaces": interfaces
            },
            children=fields + methods
        )
    
    def block(self) -> ASTNode:
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.declaration())
        
        self.consume(TokenType.RBRACE, "Expect '}' after block")
        return ASTNode(NodeType.BLOCK, children=statements)
    
    def expression(self) -> ASTNode:
        return self.assignment()
    
    def assignment(self) -> ASTNode:
        expr = self.or_expression()
        
        if self.match(*[
            TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.MULT_ASSIGN, TokenType.DIV_ASSIGN, TokenType.MOD_ASSIGN,
            TokenType.AND_ASSIGN, TokenType.OR_ASSIGN, TokenType.XOR_ASSIGN,
            TokenType.SHIFT_LEFT_ASSIGN, TokenType.SHIFT_RIGHT_ASSIGN
        ]):
            equals = self.previous()
            value = self.assignment()
            
            if expr.node_type == NodeType.IDENTIFIER:
                return ASTNode(
                    NodeType.ASSIGNMENT,
                    value={"operator": equals.type, "name": expr.value},
                    children=[value]
                )
            elif expr.node_type == NodeType.METHOD_CALL:
                # Property assignment
                return ASTNode(
                    NodeType.ASSIGNMENT,
                    value={"operator": equals.type, "property": True},
                    children=[expr, value]
                )
            else:
                self.error(equals, "Invalid assignment target")
        
        return expr
    
    def or_expression(self) -> ASTNode:
        expr = self.and_expression()
        
        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.and_expression()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def and_expression(self) -> ASTNode:
        expr = self.equality()
        
        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def equality(self) -> ASTNode:
        expr = self.comparison()
        
        while self.match(TokenType.EQ, TokenType.NEQ):
            operator = self.previous()
            right = self.comparison()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def comparison(self) -> ASTNode:
        expr = self.bitwise_or()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            operator = self.previous()
            right = self.bitwise_or()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def bitwise_or(self) -> ASTNode:
        expr = self.bitwise_xor()
        
        while self.match(TokenType.BIT_OR):
            operator = self.previous()
            right = self.bitwise_xor()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def bitwise_xor(self) -> ASTNode:
        expr = self.bitwise_and()
        
        while self.match(TokenType.BIT_XOR):
            operator = self.previous()
            right = self.bitwise_and()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def bitwise_and(self) -> ASTNode:
        expr = self.shift()
        
        while self.match(TokenType.BIT_AND):
            operator = self.previous()
            right = self.shift()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def shift(self) -> ASTNode:
        expr = self.addition()
        
        while self.match(TokenType.LEFT_SHIFT, TokenType.RIGHT_SHIFT):
            operator = self.previous()
            right = self.addition()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def addition(self) -> ASTNode:
        expr = self.multiplication()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.multiplication()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def multiplication(self) -> ASTNode:
        expr = self.unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.previous()
            right = self.unary()
            expr = ASTNode(
                NodeType.BINARY_OP,
                value=operator.type,
                children=[expr, right]
            )
        
        return expr
    
    def unary(self) -> ASTNode:
        if self.match(TokenType.MINUS, TokenType.NOT, TokenType.BIT_NOT, TokenType.INCREMENT, TokenType.DECREMENT):
            operator = self.previous()
            right = self.unary()
            return ASTNode(
                NodeType.UNARY_OP,
                value=operator.type,
                children=[right]
            )
        
        return self.call()
    
    def call(self) -> ASTNode:
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expect property name after '.'")
                expr = ASTNode(
                    NodeType.METHOD_CALL,
                    value=name.value,
                    children=[expr]
                )
            elif self.match(TokenType.LBRACKET):
                index = self.expression()
                self.consume(TokenType.RBRACKET, "Expect ']' after index")
                expr = ASTNode(
                    NodeType.METHOD_CALL,
                    value="[]",
                    children=[expr, index]
                )
            elif self.match(TokenType.INCREMENT, TokenType.DECREMENT):
                # Postfix increment/decrement
                expr = ASTNode(
                    NodeType.UNARY_OP,
                    value=self.previous().type,
                    children=[expr],
                    annotations={"postfix": True}
                )
            else:
                break
        
        return expr
    
    def primary(self) -> ASTNode:
        if self.match(TokenType.TRUE):
            return ASTNode(NodeType.LITERAL, value=True)
        if self.match(TokenType.FALSE):
            return ASTNode(NodeType.LITERAL, value=False)
        if self.match(TokenType.NULL):
            return ASTNode(NodeType.LITERAL, value=None)
        if self.match(TokenType.UNDEFINED):
            return ASTNode(NodeType.LITERAL, value=Ellipsis)  # Using Ellipsis as undefined
        
        if self.match(TokenType.NUMBER, TokenType.STRING):
            return ASTNode(NodeType.LITERAL, value=self.previous().value)
        
        if self.match(TokenType.IDENTIFIER):
            return ASTNode(NodeType.IDENTIFIER, value=self.previous().value)
        
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expect ')' after expression")
            
            # Check if it's a tuple
            if self.match(TokenType.COMMA):
                elements = [expr]
                while not self.check(TokenType.RPAREN) and not self.is_at_end():
                    elements.append(self.expression())
                    if not self.match(TokenType.COMMA):
                        break
                self.consume(TokenType.RPAREN, "Expect ')' after tuple")
                return ASTNode(NodeType.TUPLE, children=elements)
            
            return expr
        
        if self.match(TokenType.LBRACKET):
            return self.list_or_comprehension()
        
        if self.match(TokenType.LBRACE):
            return self.dict_or_comprehension()
        
        if self.match(TokenType.FUNC):
            return self.lambda_expression()
        
        if self.match(TokenType.AWAIT):
            expr = self.primary()
            return ASTNode(NodeType.AWAIT_EXPR, children=[expr])
        
        if self.match(TokenType.YIELD):
            expr = None
            if not self.check(TokenType.SEMICOLON):
                expr = self.expression()
            return ASTNode(NodeType.YIELD_EXPR, children=[expr] if expr else [])
        
        raise self.error(self.peek(), "Expect expression")
    
    def finish_call(self, callee: ASTNode) -> ASTNode:
        arguments = []
        
        if not self.check(TokenType.RPAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
        
        paren = self.consume(TokenType.RPAREN, "Expect ')' after arguments")
        
        return ASTNode(
            NodeType.FUNCTION_CALL,
            value={"line": paren.line, "column": paren.column},
            children=[callee] + arguments
        )
    
    def parameters(self) -> List[Dict]:
        params = []
        
        if not self.check(TokenType.RPAREN):
            param = self.consume(TokenType.IDENTIFIER, "Expect parameter name").value
            
            param_info = {"name": param}
            
            if self.match(TokenType.COLON):
                param_info["type"] = self.type_annotation().value
            
            if self.match(TokenType.ASSIGN):
                param_info["default"] = self.expression()
            
            params.append(param_info)
            
            while self.match(TokenType.COMMA):
                param = self.consume(TokenType.IDENTIFIER, "Expect parameter name").value
                
                param_info = {"name": param}
                
                if self.match(TokenType.COLON):
                    param_info["type"] = self.type_annotation().value
                
                if self.match(TokenType.ASSIGN):
                    param_info["default"] = self.expression()
                
                params.append(param_info)
        
        return params
    
    def type_annotation(self) -> ASTNode:
        # Simplified type annotation parsing
        if self.match(TokenType.IDENTIFIER):
            type_name = self.previous().value
            
            # Check for generic types
            if self.match(TokenType.LT):
                generic_args = [self.type_annotation()]
                while self.match(TokenType.COMMA):
                    generic_args.append(self.type_annotation())
                self.consume(TokenType.GT, "Expect '>' after generic arguments")
                
                return ASTNode(
                    NodeType.EXPRESSION,
                    value=f"{type_name}<{','.join([arg.value for arg in generic_args])}>"
                )
            
            return ASTNode(NodeType.EXPRESSION, value=type_name)
        
        # Function type
        if self.match(TokenType.LPAREN):
            param_types = []
            if not self.check(TokenType.RPAREN):
                param_types.append(self.type_annotation().value)
                while self.match(TokenType.COMMA):
                    param_types.append(self.type_annotation().value)
            self.consume(TokenType.RPAREN, "Expect ')' after parameters")
            
            self.consume(TokenType.ARROW, "Expect '=>' in function type")
            return_type = self.type_annotation().value
            
            return ASTNode(
                NodeType.EXPRESSION,
                value=f"({','.join(param_types)}) => {return_type}"
            )
        
        # Array type
        if self.match(TokenType.LBRACKET):
            self.consume(TokenType.RBRACKET, "Expect ']' in array type")
            element_type = self.type_annotation().value
            return ASTNode(NodeType.EXPRESSION, value=f"{element_type}[]")
        
        raise self.error(self.peek(), "Expect type annotation")
    
    def list_or_comprehension(self) -> ASTNode:
        if self.check(TokenType.FOR):
            return self.list_comprehension()
        
        elements = []
        if not self.check(TokenType.RBRACKET):
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                elements.append(self.expression())
        
        self.consume(TokenType.RBRACKET, "Expect ']' after list elements")
        return ASTNode(NodeType.LIST_COMP, children=elements)
    
    def list_comprehension(self) -> ASTNode:
        expr = self.expression()
        self.consume(TokenType.FOR, "Expect 'for' in list comprehension")
        
        var = self.consume(TokenType.IDENTIFIER, "Expect variable name in comprehension")
        self.consume(TokenType.IN, "Expect 'in' in comprehension")
        
        iterable = self.expression()
        
        conditions = []
        while self.match(TokenType.IF):
            conditions.append(self.expression())
        
        self.consume(TokenType.RBRACKET, "Expect ']' after comprehension")
        
        return ASTNode(
            NodeType.LIST_COMP,
            value={"var": var.value},
            children=[expr, iterable] + conditions
        )
    
    def dict_or_comprehension(self) -> ASTNode:
        if self.check(TokenType.FOR):
            return self.dict_comprehension()
        
        elements = []
        if not self.check(TokenType.RBRACE):
            key = self.expression()
            self.consume(TokenType.COLON, "Expect ':' after dictionary key")
            value = self.expression()
            elements.extend([key, value])
            
            while self.match(TokenType.COMMA):
                key = self.expression()
                self.consume(TokenType.COLON, "Expect ':' after dictionary key")
                value = self.expression()
                elements.extend([key, value])
        
        self.consume(TokenType.RBRACE, "Expect '}' after dictionary")
        return ASTNode(NodeType.DICT_COMP, children=elements)
    
    def dict_comprehension(self) -> ASTNode:
        key = self.expression()
        self.consume(TokenType.COLON, "Expect ':' in dict comprehension")
        value = self.expression()
        
        self.consume(TokenType.FOR, "Expect 'for' in dict comprehension")
        
        var = self.consume(TokenType.IDENTIFIER, "Expect variable name in comprehension")
        self.consume(TokenType.IN, "Expect 'in' in comprehension")
        
        iterable = self.expression()
        
        conditions = []
        while self.match(TokenType.IF):
            conditions.append(self.expression())
        
        self.consume(TokenType.RBRACE, "Expect '}' after comprehension")
        
        return ASTNode(
            NodeType.DICT_COMP,
            value={"var": var.value},
            children=[key, value, iterable] + conditions
        )
    
    def lambda_expression(self) -> ASTNode:
        self.consume(TokenType.LPAREN, "Expect '(' after lambda")
        params = self.parameters()
        self.consume(TokenType.RPAREN, "Expect ')' after lambda parameters")
        
        self.consume(TokenType.ARROW, "Expect '=>' in lambda")
        
        body = self.expression()
        
        return ASTNode(
            NodeType.LAMBDA,
            value={"params": params},
            children=[body]
        )
    
    def match(self, *types: TokenType) -> bool:
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        return self.tokens[self.current - 1]
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.check(token_type):
            return self.advance()
        
        raise self.error(self.peek(), message)
    
    def error(self, token: Token, message: str):
        error_msg = f"{self.filename}:{token.line}:{token.column}: {message}"
        self.errors.append(error_msg)
        return ParseError(error_msg)
    
    def synchronize(self):
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMICOLON:
                return
            
            if self.peek().type in [
                TokenType.CLASS, TokenType.FUNC, TokenType.LET,
                TokenType.CONST, TokenType.FOR, TokenType.IF,
                TokenType.WHILE, TokenType.RETURN
            ]:
                return
            
            self.advance()

class ParseError(Exception):
    pass

# ==================== ADVANCED INTERPRETER ====================
class Scope:
    def __init__(self, parent=None, name="global"):
        self.parent = parent
        self.name = name
        self.variables = {}
        self.constants = set()
        self.types = {}
        self.functions = {}
        self.classes = {}
    
    def declare(self, name: str, value, is_const=False):
        if name in self.variables:
            raise NouhaRuntimeError(f"Variable '{name}' already declared in this scope")
        
        self.variables[name] = value
        if is_const:
            self.constants.add(name)
    
    def assign(self, name: str, value):
        if name in self.constants:
            raise NouhaRuntimeError(f"Cannot reassign constant '{name}'")
        
        if name in self.variables:
            self.variables[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise NouhaRuntimeError(f"Undefined variable '{name}'")
    
    def get(self, name: str):
        if name in self.variables:
            return self.variables[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NouhaRuntimeError(f"Undefined variable '{name}'")
    
    def has(self, name: str) -> bool:
        if name in self.variables:
            return True
        elif self.parent:
            return self.parent.has(name)
        return False

class NouhaFunction:
    def __init__(self, name: str, params: List[Dict], body: ASTNode, closure: Scope, is_async=False):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
        self.is_async = is_async
    
    def arity(self) -> int:
        return len([p for p in self.params if "default" not in p])
    
    def call(self, interpreter, arguments: List, this=None):
        # Create new scope for function execution
        function_scope = Scope(parent=self.closure, name=f"function:{self.name}")
        
        # Bind 'this' if provided
        if this is not None:
            function_scope.declare("this", this)
        
        # Bind arguments to parameters
        for i, param in enumerate(self.params):
            param_name = param["name"]
            
            if i < len(arguments):
                value = arguments[i]
            elif "default" in param:
                # Evaluate default value
                value = interpreter.evaluate(param["default"], function_scope)
            else:
                raise NouhaRuntimeError(f"Missing argument for parameter '{param_name}'")
            
            function_scope.declare(param_name, value)
        
        # Execute function body
        try:
            result = interpreter.execute(self.body, function_scope)
            
            # Handle async functions
            if self.is_async:
                async def async_wrapper():
                    return result
                return async_wrapper()
            
            return result
        except ReturnException as ret:
            return ret.value
    
    def __repr__(self):
        return f"<NouhaFunction {self.name}>"

class NouhaClass:
    def __init__(self, name: str, superclass=None, interfaces=None):
        self.name = name
        self.superclass = superclass
        self.interfaces = interfaces or []
        self.methods = {}
        self.static_methods = {}
        self.fields = {}
    
    def add_method(self, name: str, method: NouhaFunction):
        self.methods[name] = method
    
    def add_static_method(self, name: str, method: NouhaFunction):
        self.static_methods[name] = method
    
    def add_field(self, name: str, value):
        self.fields[name] = value
    
    def instantiate(self, interpreter, arguments: List):
        instance = NouhaObject(self)
        
        # Initialize fields
        for name, value in self.fields.items():
            instance.fields[name] = value
        
        # Call constructor if exists
        if "constructor" in self.methods:
            constructor = self.methods["constructor"]
            constructor.call(interpreter, arguments, this=instance)
        
        return instance
    
    def __repr__(self):
        return f"<NouhaClass {self.name}>"

class NouhaObject:
    def __init__(self, klass: NouhaClass):
        self.klass = klass
        self.fields = {}
    
    def get(self, name: str):
        if name in self.fields:
            return self.fields[name]
        
        if name in self.klass.methods:
            # Bind method to this instance
            method = self.klass.methods[name]
            return BoundMethod(self, method)
        
        if name in self.klass.static_methods:
            return self.klass.static_methods[name]
        
        raise NouhaRuntimeError(f"Property '{name}' not found on object of type {self.klass.name}")
    
    def set(self, name: str, value):
        self.fields[name] = value
    
    def __repr__(self):
        return f"<{self.klass.name} object at {id(self)}>"

class BoundMethod:
    def __init__(self, instance: NouhaObject, method: NouhaFunction):
        self.instance = instance
        self.method = method
    
    def call(self, interpreter, arguments: List):
        return self.method.call(interpreter, arguments, this=self.instance)
    
    def __repr__(self):
        return f"<BoundMethod {self.method.name} of {self.instance}>"

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class AdvancedInterpreter:
    def __init__(self):
        self.global_scope = Scope(name="global")
        self.current_scope = self.global_scope
        self.modules = {}
        self.builtins = self._init_builtins()
        
        # Runtime state
        self.debug_mode = False
        self.max_recursion_depth = 1000
        self.recursion_depth = 0
        self.call_stack = []
        
        # Initialize standard library
        self._init_stdlib()
    
    def _init_builtins(self):
        """Initialize built-in functions and objects"""
        builtins = {}
        
        # Console I/O
        builtins["print"] = self._builtin_print
        builtins["println"] = self._builtin_println
        builtins["input"] = self._builtin_input
        builtins["format"] = self._builtin_format
        
        # Type conversion
        builtins["int"] = self._builtin_int
        builtins["float"] = self._builtin_float
        builtins["str"] = self._builtin_str
        builtins["bool"] = self._builtin_bool
        builtins["list"] = self._builtin_list
        builtins["dict"] = self._builtin_dict
        builtins["tuple"] = self._builtin_tuple
        builtins["set"] = self._builtin_set
        
        # Type checking
        builtins["type"] = self._builtin_type
        builtins["isinstance"] = self._builtin_isinstance
        
        # Math
        builtins["abs"] = abs
        builtins["round"] = round
        builtins["min"] = min
        builtins["max"] = max
        builtins["sum"] = sum
        builtins["len"] = len
        
        # String operations
        builtins["lower"] = lambda s: s.lower()
        builtins["upper"] = lambda s: s.upper()
        builtins["strip"] = lambda s: s.strip()
        builtins["split"] = lambda s, sep=None: s.split(sep)
        builtins["join"] = lambda sep, items: sep.join(items)
        builtins["replace"] = lambda s, old, new: s.replace(old, new)
        
        # List operations
        builtins["append"] = lambda lst, item: lst.append(item) or lst
        builtins["pop"] = lambda lst, idx=-1: lst.pop(idx)
        builtins["insert"] = lambda lst, idx, item: lst.insert(idx, item) or lst
        builtins["remove"] = lambda lst, item: lst.remove(item) or lst
        builtins["sort"] = lambda lst: sorted(lst)
        builtins["reverse"] = lambda lst: list(reversed(lst))
        
        # Dictionary operations
        builtins["keys"] = lambda d: list(d.keys())
        builtins["values"] = lambda d: list(d.values())
        builtins["items"] = lambda d: list(d.items())
        
        # Functional programming
        builtins["map"] = self._builtin_map
        builtins["filter"] = self._builtin_filter
        builtins["reduce"] = self._builtin_reduce
        
        # File I/O
        builtins["open"] = self._builtin_open
        builtins["read"] = self._builtin_read
        builtins["write"] = self._builtin_write
        
        # System
        builtins["exit"] = self._builtin_exit
        builtins["time"] = time.time
        builtins["sleep"] = time.sleep
        
        # Advanced
        builtins["range"] = range
        builtins["enumerate"] = enumerate
        builtins["zip"] = zip
        builtins["any"] = any
        builtins["all"] = all
        builtins["sorted"] = sorted
        
        return builtins
    
    def _init_stdlib(self):
        """Initialize standard library modules"""
        
        # Math module
        math_module = {
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "sqrt": math.sqrt,
            "ceil": math.ceil,
            "floor": math.floor,
            "trunc": math.trunc,
            "degrees": math.degrees,
            "radians": math.radians,
            "hypot": math.hypot,
            "pow": math.pow,
            "fabs": math.fabs,
            "factorial": math.factorial,
            "gcd": math.gcd,
            "isclose": math.isclose,
            "isfinite": math.isfinite,
            "isinf": math.isinf,
            "isnan": math.isnan,
            "modf": math.modf,
        }
        self.modules["math"] = math_module
        
        # Random module
        random_module = {
            "random": random.random,
            "uniform": random.uniform,
            "randint": random.randint,
            "randrange": random.randrange,
            "choice": random.choice,
            "choices": random.choices,
            "shuffle": random.shuffle,
            "sample": random.sample,
            "gauss": random.gauss,
            "normalvariate": random.normalvariate,
            "seed": random.seed,
        }
        self.modules["random"] = random_module
        
        # Time module
        time_module = {
            "time": time.time,
            "sleep": time.sleep,
            "ctime": time.ctime,
            "gmtime": time.gmtime,
            "localtime": time.localtime,
            "mktime": time.mktime,
            "strftime": time.strftime,
            "strptime": time.strptime,
            "monotonic": time.monotonic,
            "perf_counter": time.perf_counter,
            "process_time": time.process_time,
        }
        self.modules["time"] = time_module
        
        # OS module
        os_module = {
            "name": os.name,
            "getcwd": os.getcwd,
            "listdir": os.listdir,
            "mkdir": os.mkdir,
            "rmdir": os.rmdir,
            "remove": os.remove,
            "rename": os.rename,
            "path": {
                "exists": os.path.exists,
                "isfile": os.path.isfile,
                "isdir": os.path.isdir,
                "join": os.path.join,
                "split": os.path.split,
                "splitext": os.path.splitext,
                "basename": os.path.basename,
                "dirname": os.path.dirname,
                "abspath": os.path.abspath,
                "realpath": os.path.realpath,
                "getsize": os.path.getsize,
                "getmtime": os.path.getmtime,
            }
        }
        self.modules["os"] = os_module
        
        # JSON module
        self.modules["json"] = {
            "loads": json.loads,
            "dumps": json.dumps,
        }
        
        # Collections module
        self.modules["collections"] = {
            "Counter": collections.Counter,
            "defaultdict": collections.defaultdict,
            "OrderedDict": collections.OrderedDict,
            "deque": collections.deque,
            "namedtuple": collections.namedtuple,
        }
        
        # Itertools module
        self.modules["itertools"] = {
            "chain": itertools.chain,
            "combinations": itertools.combinations,
            "permutations": itertools.permutations,
            "product": itertools.product,
            "cycle": itertools.cycle,
            "repeat": itertools.repeat,
        }
        
        # Statistics module
        self.modules["statistics"] = {
            "mean": statistics.mean,
            "median": statistics.median,
            "mode": statistics.mode,
            "stdev": statistics.stdev,
            "variance": statistics.variance,
        }
        
        # CSV module
        self.modules["csv"] = {
            "reader": csv.reader,
            "writer": csv.writer,
            "DictReader": csv.DictReader,
            "DictWriter": csv.DictWriter,
        }
        
        # Hashlib module
        self.modules["hashlib"] = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }
        
        # Datetime module
        self.modules["datetime"] = {
            "datetime": datetime.datetime,
            "date": datetime.date,
            "time": datetime.time,
            "timedelta": datetime.timedelta,
        }
        
        # SQLite module
        self.modules["sqlite3"] = {
            "connect": sqlite3.connect,
            "Connection": sqlite3.Connection,
            "Cursor": sqlite3.Cursor,
        }
        
        # HTTP module
        self.modules["http"] = {
            "request": urllib.request,
            "parse": urllib.parse,
        }
    
    def interpret(self, source: str, filename: str = "<string>"):
        """Main interpretation entry point"""
        try:
            # Lexical analysis
            lexer = AdvancedLexer(source, filename)
            tokens = lexer.tokenize()
            
            if lexer.errors:
                for error in lexer.errors:
                    print(f"Lexer Error: {error}", file=sys.stderr)
                return False
            
            # Parsing
            parser = Parser(tokens, filename)
            ast = parser.parse()
            
            if parser.errors:
                for error in parser.errors:
                    print(f"Parser Error: {error}", file=sys.stderr)
                return False
            
            # Execution
            result = self.execute(ast, self.current_scope)
            
            if self.debug_mode:
                print(f"Execution completed. Result: {result}")
            
            return True
            
        except NouhaRuntimeError as e:
            print(f"Runtime Error: {e}", file=sys.stderr)
            if self.debug_mode and self.call_stack:
                print("Call stack:", file=sys.stderr)
                for frame in reversed(self.call_stack):
                    print(f"  at {frame}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Internal Error: {e}", file=sys.stderr)
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def execute(self, node: ASTNode, scope: Scope):
        """Execute an AST node"""
        if self.recursion_depth > self.max_recursion_depth:
            raise NouhaRuntimeError("Maximum recursion depth exceeded")
        
        self.recursion_depth += 1
        self.call_stack.append(f"{node.node_type.name} at line {node.line}")
        
        try:
            result = self._execute_node(node, scope)
            return result
        finally:
            self.call_stack.pop()
            self.recursion_depth -= 1
    
    def _execute_node(self, node: ASTNode, scope: Scope):
        """Dispatch execution based on node type"""
        handler_name = f"_execute_{node.node_type.name.lower()}"
        handler = getattr(self, handler_name, None)
        
        if handler:
            return handler(node, scope)
        else:
            raise NouhaRuntimeError(f"No handler for node type: {node.node_type.name}")
    
    def _execute_program(self, node: ASTNode, scope: Scope):
        """Execute a program (list of statements)"""
        result = None
        for child in node.children:
            result = self.execute(child, scope)
        return result
    
    def _execute_block(self, node: ASTNode, scope: Scope):
        """Execute a block of statements"""
        block_scope = Scope(parent=scope, name="block")
        result = None
        
        for child in node.children:
            result = self.execute(child, block_scope)
        
        return result
    
    def _execute_statement(self, node: ASTNode, scope: Scope):
        """Execute a single statement"""
        if node.value == "error":
            return None
        
        if node.children:
            return self.execute(node.children[0], scope)
        
        return None
    
    def _execute_assignment(self, node: ASTNode, scope: Scope):
        """Execute variable assignment"""
        if isinstance(node.value, dict) and "name" in node.value:
            # Variable declaration/assignment
            name = node.value["name"]
            is_const = node.value.get("const", False)
            
            if node.children:
                value = self.execute(node.children[0], scope)
            else:
                value = None
            
            scope.declare(name, value, is_const=is_const)
            return value
        
        elif isinstance(node.value, dict) and "operator" in node.value:
            # Assignment with operator (e.g., +=, -=)
            operator = node.value["operator"]
            target_name = node.value.get("name")
            
            if target_name:
                # Simple variable assignment
                old_value = scope.get(target_name)
                new_value = self.execute(node.children[0], scope)
                
                if operator == TokenType.ASSIGN:
                    result = new_value
                elif operator == TokenType.PLUS_ASSIGN:
                    result = self._add(old_value, new_value)
                elif operator == TokenType.MINUS_ASSIGN:
                    result = self._subtract(old_value, new_value)
                elif operator == TokenType.MULT_ASSIGN:
                    result = self._multiply(old_value, new_value)
                elif operator == TokenType.DIV_ASSIGN:
                    result = self._divide(old_value, new_value)
                elif operator == TokenType.MOD_ASSIGN:
                    result = self._modulo(old_value, new_value)
                else:
                    raise NouhaRuntimeError(f"Unsupported assignment operator: {operator}")
                
                scope.assign(target_name, result)
                return result
        
        raise NouhaRuntimeError("Invalid assignment")
    
    def _execute_identifier(self, node: ASTNode, scope: Scope):
        """Evaluate an identifier"""
        name = node.value
        
        # Check in scope first
        if scope.has(name):
            return scope.get(name)
        
        # Check builtins
        if name in self.builtins:
            return self.builtins[name]
        
        # Check if it's a module
        if name in self.modules:
            return self.modules[name]
        
        raise NouhaRuntimeError(f"Undefined identifier: '{name}'")
    
    def _execute_literal(self, node: ASTNode, scope: Scope):
        """Evaluate a literal value"""
        return node.value
    
    def _execute_binary_op(self, node: ASTNode, scope: Scope):
        """Evaluate a binary operation"""
        left = self.execute(node.children[0], scope)
        right = self.execute(node.children[1], scope)
        
        operator = node.value
        
        if operator == TokenType.PLUS:
            return self._add(left, right)
        elif operator == TokenType.MINUS:
            return self._subtract(left, right)
        elif operator == TokenType.MULTIPLY:
            return self._multiply(left, right)
        elif operator == TokenType.DIVIDE:
            return self._divide(left, right)
        elif operator == TokenType.MODULO:
            return self._modulo(left, right)
        elif operator == TokenType.POWER:
            return self._power(left, right)
        elif operator == TokenType.EQ:
            return self._equal(left, right)
        elif operator == TokenType.NEQ:
            return not self._equal(left, right)
        elif operator == TokenType.LT:
            return self._less_than(left, right)
        elif operator == TokenType.GT:
            return self._greater_than(left, right)
        elif operator == TokenType.LTE:
            return self._less_than_equal(left, right)
        elif operator == TokenType.GTE:
            return self._greater_than_equal(left, right)
        elif operator == TokenType.AND:
            return self._logical_and(left, right)
        elif operator == TokenType.OR:
            return self._logical_or(left, right)
        elif operator == TokenType.BIT_AND:
            return self._bitwise_and(left, right)
        elif operator == TokenType.BIT_OR:
            return self._bitwise_or(left, right)
        elif operator == TokenType.BIT_XOR:
            return self._bitwise_xor(left, right)
        elif operator == TokenType.LEFT_SHIFT:
            return self._left_shift(left, right)
        elif operator == TokenType.RIGHT_SHIFT:
            return self._right_shift(left, right)
        
        raise NouhaRuntimeError(f"Unsupported binary operator: {operator}")
    
    def _execute_unary_op(self, node: ASTNode, scope: Scope):
        """Evaluate a unary operation"""
        operand = self.execute(node.children[0], scope)
        operator = node.value
        is_postfix = node.annotations.get("postfix", False)
        
        if operator == TokenType.MINUS:
            return self._negate(operand)
        elif operator == TokenType.NOT:
            return self._logical_not(operand)
        elif operator == TokenType.BIT_NOT:
            return self._bitwise_not(operand)
        elif operator == TokenType.INCREMENT:
            # TODO: Handle increment with variable assignment
            if is_postfix:
                result = operand
                # This would need to modify the variable
            else:
                result = self._add(operand, 1)
                # This would need to modify the variable
            return result
        elif operator == TokenType.DECREMENT:
            # TODO: Handle decrement with variable assignment
            if is_postfix:
                result = operand
            else:
                result = self._subtract(operand, 1)
            return result
        
        raise NouhaRuntimeError(f"Unsupported unary operator: {operator}")
    
    def _execute_function_def(self, node: ASTNode, scope: Scope):
        """Define a function"""
        func_info = node.value
        name = func_info["name"]
        params = func_info["params"]
        is_async = func_info["async"]
        body = node.children[0]
        
        function = NouhaFunction(name, params, body, scope, is_async)
        scope.functions[name] = function
        
        # Also make it available as a variable
        scope.declare(name, function)
        
        return function
    
    def _execute_function_call(self, node: ASTNode, scope: Scope):
        """Call a function"""
        callee = self.execute(node.children[0], scope)
        arguments = [self.execute(arg, scope) for arg in node.children[1:]]
        
        if callable(callee):
            # Python function
            try:
                return callee(*arguments)
            except Exception as e:
                raise NouhaRuntimeError(f"Error calling function: {e}")
        
        elif isinstance(callee, NouhaFunction):
            # Nouha function
            return callee.call(self, arguments)
        
        elif isinstance(callee, BoundMethod):
            # Bound method
            return callee.call(self, arguments)
        
        elif hasattr(callee, '__call__'):
            # Callable object
            return callee(*arguments)
        
        else:
            raise NouhaRuntimeError(f"'{callee}' is not callable")
    
    def _execute_class_def(self, node: ASTNode, scope: Scope):
        """Define a class"""
        class_info = node.value
        name = class_info["name"]
        superclass_name = class_info.get("superclass")
        
        # Get superclass
        superclass = None
        if superclass_name:
            superclass = scope.get(superclass_name)
            if not isinstance(superclass, NouhaClass):
                raise NouhaRuntimeError(f"Superclass '{superclass_name}' is not a class")
        
        # Create class
        klass = NouhaClass(name, superclass)
        
        # Execute class body
        class_scope = Scope(parent=scope, name=f"class:{name}")
        class_scope.declare("self", klass)  # For static methods
        
        for child in node.children:
            self.execute(child, class_scope)
        
        # Register class
        scope.classes[name] = klass
        scope.declare(name, klass)
        
        return klass
    
    # ==================== HELPER METHODS ====================
    
    def _add(self, left, right):
        """Addition with type checking"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right
        elif isinstance(left, str) or isinstance(right, str):
            return str(left) + str(right)
        elif isinstance(left, list) and isinstance(right, list):
            return left + right
        else:
            raise NouhaRuntimeError(f"Cannot add {type(left)} and {type(right)}")
    
    def _subtract(self, left, right):
        """Subtraction with type checking"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left - right
        else:
            raise NouhaRuntimeError(f"Cannot subtract {type(left)} and {type(right)}")
    
    def _multiply(self, left, right):
        """Multiplication with type checking"""
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
            raise NouhaRuntimeError(f"Cannot multiply {type(left)} and {type(right)}")
    
    def _divide(self, left, right):
        """Division with type checking"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                raise NouhaRuntimeError("Division by zero")
            return left / right
        else:
            raise NouhaRuntimeError(f"Cannot divide {type(left)} and {type(right)}")
    
    def _modulo(self, left, right):
        """Modulo with type checking"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                raise NouhaRuntimeError("Modulo by zero")
            return left % right
        else:
            raise NouhaRuntimeError(f"Cannot modulo {type(left)} and {type(right)}")
    
    def _power(self, left, right):
        """Power with type checking"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left ** right
        else:
            raise NouhaRuntimeError(f"Cannot raise {type(left)} to power {type(right)}")
    
    def _equal(self, left, right):
        """Equality comparison"""
        return left == right
    
    def _less_than(self, left, right):
        """Less than comparison"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left < right
        elif isinstance(left, str) and isinstance(right, str):
            return left < right
        else:
            raise NouhaRuntimeError(f"Cannot compare {type(left)} and {type(right)} with <")
    
    def _greater_than(self, left, right):
        """Greater than comparison"""
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left > right
        elif isinstance(left, str) and isinstance(right, str):
            return left > right
        else:
            raise NouhaRuntimeError(f"Cannot compare {type(left)} and {type(right)} with >")
    
    def _less_than_equal(self, left, right):
        """Less than or equal comparison"""
        return not self._greater_than(left, right)
    
    def _greater_than_equal(self, left, right):
        """Greater than or equal comparison"""
        return not self._less_than(left, right)
    
    def _logical_and(self, left, right):
        """Logical AND"""
        return bool(left) and bool(right)
    
    def _logical_or(self, left, right):
        """Logical OR"""
        return bool(left) or bool(right)
    
    def _logical_not(self, operand):
        """Logical NOT"""
        return not bool(operand)
    
    def _bitwise_and(self, left, right):
        """Bitwise AND"""
        if isinstance(left, int) and isinstance(right, int):
            return left & right
        else:
            raise NouhaRuntimeError(f"Cannot perform bitwise AND on {type(left)} and {type(right)}")
    
    def _bitwise_or(self, left, right):
        """Bitwise OR"""
        if isinstance(left, int) and isinstance(right, int):
            return left | right
        else:
            raise NouhaRuntimeError(f"Cannot perform bitwise OR on {type(left)} and {type(right)}")
    
    def _bitwise_xor(self, left, right):
        """Bitwise XOR"""
        if isinstance(left, int) and isinstance(right, int):
            return left ^ right
        else:
            raise NouhaRuntimeError(f"Cannot perform bitwise XOR on {type(left)} and {type(right)}")
    
    def _bitwise_not(self, operand):
        """Bitwise NOT"""
        if isinstance(operand, int):
            return ~operand
        else:
            raise NouhaRuntimeError(f"Cannot perform bitwise NOT on {type(operand)}")
    
    def _left_shift(self, left, right):
        """Left shift"""
        if isinstance(left, int) and isinstance(right, int):
            return left << right
        else:
            raise NouhaRuntimeError(f"Cannot left shift {type(left)} by {type(right)}")
    
    def _right_shift(self, left, right):
        """Right shift"""
        if isinstance(left, int) and isinstance(right, int):
            return left >> right
        else:
            raise NouhaRuntimeError(f"Cannot right shift {type(left)} by {type(right)}")
    
    def _negate(self, operand):
        """Negation"""
        if isinstance(operand, (int, float)):
            return -operand
        else:
            raise NouhaRuntimeError(f"Cannot negate {type(operand)}")
    
    # ==================== BUILTIN FUNCTIONS ====================
    
    def _builtin_print(self, *args, **kwargs):
        """Built-in print function"""
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        file = kwargs.get('file', sys.stdout)
        
        print(*args, sep=sep, end=end, file=file)
    
    def _builtin_println(self, *args):
        """Built-in println function (always newline)"""
        print(*args)
    
    def _builtin_input(self, prompt=""):
        """Built-in input function"""
        return input(prompt)
    
    def _builtin_format(self, value, format_spec=""):
        """Built-in format function"""
        return format(value, format_spec)
    
    def _builtin_int(self, value, base=10):
        """Built-in int conversion"""
        try:
            return int(value, base)
        except (ValueError, TypeError):
            raise NouhaRuntimeError(f"Cannot convert '{value}' to integer")
    
    def _builtin_float(self, value):
        """Built-in float conversion"""
        try:
            return float(value)
        except (ValueError, TypeError):
            raise NouhaRuntimeError(f"Cannot convert '{value}' to float")
    
    def _builtin_str(self, value):
        """Built-in string conversion"""
        return str(value)
    
    def _builtin_bool(self, value):
        """Built-in boolean conversion"""
        return bool(value)
    
    def _builtin_list(self, iterable=None):
        """Built-in list creation"""
        if iterable is None:
            return []
        return list(iterable)
    
    def _builtin_dict(self, iterable=None):
        """Built-in dictionary creation"""
        if iterable is None:
            return {}
        return dict(iterable)
    
    def _builtin_tuple(self, iterable=None):
        """Built-in tuple creation"""
        if iterable is None:
            return ()
        return tuple(iterable)
    
    def _builtin_set(self, iterable=None):
        """Built-in set creation"""
        if iterable is None:
            return set()
        return set(iterable)
    
    def _builtin_type(self, obj):
        """Built-in type function"""
        type_map = {
            int: "int",
            float: "float",
            str: "str",
            bool: "bool",
            list: "list",
            dict: "dict",
            tuple: "tuple",
            set: "set",
            type(None): "null",
            NouhaFunction: "function",
            NouhaClass: "class",
            NouhaObject: "object",
        }
        
        obj_type = type(obj)
        return type_map.get(obj_type, str(obj_type))
    
    def _builtin_isinstance(self, obj, class_or_type):
        """Built-in isinstance function"""
        if isinstance(class_or_type, str):
            # Type name as string
            type_name = self._builtin_type(obj)
            return type_name == class_or_type
        else:
            # Actual type/class
            return isinstance(obj, class_or_type)
    
    def _builtin_map(self, func, iterable):
        """Built-in map function"""
        if not callable(func):
            raise NouhaRuntimeError("First argument to map must be callable")
        
        return [func(item) for item in iterable]
    
    def _builtin_filter(self, func, iterable):
        """Built-in filter function"""
        if not callable(func):
            raise NouhaRuntimeError("First argument to filter must be callable")
        
        return [item for item in iterable if func(item)]
    
    def _builtin_reduce(self, func, iterable, initial=None):
        """Built-in reduce function"""
        if not callable(func):
            raise NouhaRuntimeError("First argument to reduce must be callable")
        
        if not iterable:
            if initial is not None:
                return initial
            raise NouhaRuntimeError("reduce() of empty sequence with no initial value")
        
        iterator = iter(iterable)
        
        if initial is None:
            try:
                value = next(iterator)
            except StopIteration:
                raise NouhaRuntimeError("reduce() of empty sequence with no initial value")
        else:
            value = initial
        
        for item in iterator:
            value = func(value, item)
        
        return value
    
    def _builtin_open(self, filename, mode='r', encoding='utf-8'):
        """Built-in file open function"""
        try:
            return open(filename, mode, encoding=encoding)
        except Exception as e:
            raise NouhaRuntimeError(f"Cannot open file '{filename}': {e}")
    
    def _builtin_read(self, fileobj, size=-1):
        """Built-in file read function"""
        try:
            return fileobj.read(size)
        except Exception as e:
            raise NouhaRuntimeError(f"Cannot read file: {e}")
    
    def _builtin_write(self, fileobj, data):
        """Built-in file write function"""
        try:
            return fileobj.write(data)
        except Exception as e:
            raise NouhaRuntimeError(f"Cannot write to file: {e}")
    
    def _builtin_exit(self, code=0):
        """Built-in exit function"""
        sys.exit(code)

# ==================== COMMAND LINE INTERFACE ====================
class NouhaCLI:
    def __init__(self):
        self.interpreter = AdvancedInterpreter()
        self.history = []
        self.context = {}
    
    def run_file(self, filename: str):
        """Run a Nouha script file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                source = f.read()
            
            print(f"Running {filename}...")
            success = self.interpreter.interpret(source, filename)
            
            if success:
                print(f"\n Execution completed successfully.")
            else:
                print(f"\n Execution failed.")
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    
    def run_repl(self):
        """Start the REPL (Read-Eval-Print Loop)"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║                 Nouha Programming Language               ║
        ║                    Advanced Interpreter                  ║
        ║                     Version 2.0.0                        ║
        ╚══════════════════════════════════════════════════════════╝
        
        Type '.help' for help, '.exit' to quit.
        """)
        
        while True:
            try:
                # Get input
                try:
                    line = input("nouha> ").strip()
                except EOFError:
                    print("\nGoodbye!")
                    break
                
                # Handle empty input
                if not line:
                    continue
                
                # Handle commands
                if line.startswith('.'):
                    self._handle_command(line[1:])
                    continue
                
                # Add to history
                self.history.append(line)
                
                # Try to interpret
                try:
                    result = self.interpreter.interpret(line, "<repl>")
                    if result is not None:
                        print(f"Result: {result}")
                except NouhaRuntimeError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Internal error: {e}")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type '.exit' to quit.")
    
    def _handle_command(self, command: str):
        """Handle REPL commands"""
        parts = command.split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            self._show_help()
        elif cmd == "exit" or cmd == "quit":
            print("Goodbye!")
            sys.exit(0)
        elif cmd == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif cmd == "history":
            self._show_history()
        elif cmd == "reset":
            self.interpreter = AdvancedInterpreter()
            print("Interpreter reset.")
        elif cmd == "debug":
            self.interpreter.debug_mode = not self.interpreter.debug_mode
            print(f"Debug mode: {'ON' if self.interpreter.debug_mode else 'OFF'}")
        elif cmd == "modules":
            self._show_modules()
        elif cmd == "vars":
            self._show_variables()
        elif cmd == "load":
            if args:
                self.run_file(args[0])
            else:
                print("Usage: .load <filename>")
        else:
            print(f"Unknown command: '{cmd}'. Type '.help' for available commands.")
    
    def _show_help(self):
        """Show help message"""
        help_text = """
        Available Commands:
          .help                    Show this help message
          .exit/.quit              Exit the REPL
          .clear                   Clear the screen
          .history                 Show command history
          .reset                   Reset the interpreter
          .debug                   Toggle debug mode
          .modules                 Show loaded modules
          .vars                    Show current variables
          .load <file>             Load and run a Nouha file
        
        Language Features:
          - Variables: let x = 10; const pi = 3.14;
          - Functions: func add(a, b) { return a + b; }
          - Classes: class Person { constructor(name) { this.name = name; } }
          - Control flow: if/else, while, for, match/case
          - Exception handling: try/catch/finally
          - Modules: import "math"; from "os" import path;
          - Async/await: async func fetch() { await request(); }
          - Generators: func* range(n) { for (i in 0..n) { yield i; } }
          - Comprehensions: [x*2 for x in 1..10 if x % 2 == 0]
        
        Built-in Types:
          - Numbers: 42, 3.14, 0xFF, 0b1010
          - Strings: "hello", 'world', `template ${var}`
          - Booleans: true, false
          - Collections: [1, 2, 3], {"key": "value"}, (1, 2), {1, 2, 3}
          - Null/Undefined: null, undefined
        """
        print(help_text)
    
    def _show_history(self):
        """Show command history"""
        if not self.history:
            print("No history yet.")
            return
        
        for i, cmd in enumerate(self.history[-20:], 1):  # Show last 20 commands
            print(f"{i:3}: {cmd}")
    
    def _show_modules(self):
        """Show loaded modules"""
        modules = list(self.interpreter.modules.keys())
        if modules:
            print("Loaded modules:", ", ".join(sorted(modules)))
        else:
            print("No modules loaded.")
    
    def _show_variables(self):
        """Show current variables"""
        # This would need access to the current scope
        print("Variable inspection not fully implemented yet.")

# ==================== MAIN ENTRY POINT ====================
def main():
    """Main entry point for the Nouha interpreter"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nouha Programming Language Interpreter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)
