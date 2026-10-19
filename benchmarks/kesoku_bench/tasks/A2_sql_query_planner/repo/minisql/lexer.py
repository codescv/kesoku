"""SQL tokenizer for minisql."""

from dataclasses import dataclass
from enum import StrEnum


class TokenKind(StrEnum):
    """Token categories recognized by the minisql lexer."""

    SELECT = "SELECT"
    FROM = "FROM"
    WHERE = "WHERE"
    AS = "AS"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    GROUP = "GROUP"
    BY = "BY"
    ORDER = "ORDER"
    ASC = "ASC"
    DESC = "DESC"
    LIMIT = "LIMIT"
    COUNT = "COUNT"
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    OP = "OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    DOT = "DOT"
    STAR = "STAR"
    SEMICOLON = "SEMICOLON"
    EOF = "EOF"


KEYWORDS: dict[str, TokenKind] = {
    "SELECT": TokenKind.SELECT,
    "FROM": TokenKind.FROM,
    "WHERE": TokenKind.WHERE,
    "AS": TokenKind.AS,
    "AND": TokenKind.AND,
    "OR": TokenKind.OR,
    "NOT": TokenKind.NOT,
    "GROUP": TokenKind.GROUP,
    "BY": TokenKind.BY,
    "ORDER": TokenKind.ORDER,
    "ASC": TokenKind.ASC,
    "DESC": TokenKind.DESC,
    "LIMIT": TokenKind.LIMIT,
    "COUNT": TokenKind.COUNT,
    "SUM": TokenKind.SUM,
    "MIN": TokenKind.MIN,
    "MAX": TokenKind.MAX,
}


class LexError(ValueError):
    """Raised when the SQL input contains an unrecognized token or unterminated literal."""


@dataclass(frozen=True)
class Token:
    """A single lexical token with its kind, text, and 0-based byte offset."""

    kind: TokenKind
    text: str
    pos: int


def tokenize(sql: str) -> list[Token]:
    """Tokenize a SQL string into a sequence of Token instances ending with EOF.

    Supports line comments (`-- ...`), single-quoted strings, integer and float
    literals, comparison operators, punctuation, and SQL keywords.
    """
    tokens: list[Token] = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        # Skip whitespace characters.
        if ch.isspace():
            i += 1
            continue

        # Skip single-line SQL comments starting with `--`.
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            continue

        # Single-character punctuation tokens.
        if ch == "(":
            tokens.append(Token(TokenKind.LPAREN, ch, i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TokenKind.RPAREN, ch, i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token(TokenKind.COMMA, ch, i))
            i += 1
            continue
        if ch == ".":
            tokens.append(Token(TokenKind.DOT, ch, i))
            i += 1
            continue
        if ch == "*":
            tokens.append(Token(TokenKind.STAR, ch, i))
            i += 1
            continue
        if ch == ";":
            tokens.append(Token(TokenKind.SEMICOLON, ch, i))
            i += 1
            continue

        # Comparison operators: <=, >=, !=, <>, <, >, =
        if ch in ("<", ">", "!", "="):
            start = i
            if i + 1 < n and sql[i : i + 2] in ("<=", ">=", "!=", "<>"):
                op_text = sql[i : i + 2]
                if op_text == "<>":
                    op_text = "!="
                tokens.append(Token(TokenKind.OP, op_text, start))
                i += 2
                continue
            if ch in ("<", ">", "="):
                tokens.append(Token(TokenKind.OP, ch, start))
                i += 1
                continue
            raise LexError(f"Unexpected character {ch!r} at position {i}")

        # Single-quoted string literals.
        if ch == "'":
            start = i
            i += 1
            chars: list[str] = []
            while i < n:
                if sql[i] == "'":
                    # Support escaped doubled single quotes `''`.
                    if i + 1 < n and sql[i + 1] == "'":
                        chars.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                chars.append(sql[i])
                i += 1
            else:
                raise LexError(f"Unterminated string literal starting at position {start}")
            tokens.append(Token(TokenKind.STRING, "".join(chars), start))
            continue

        # Numeric literals (integers and decimals, including optional leading minus).
        if ch.isdigit() or (ch == "-" and i + 1 < n and sql[i + 1].isdigit()):
            start = i
            if ch == "-":
                i += 1
            while i < n and sql[i].isdigit():
                i += 1
            if i < n and sql[i] == "." and i + 1 < n and sql[i + 1].isdigit():
                i += 1
                while i < n and sql[i].isdigit():
                    i += 1
            tokens.append(Token(TokenKind.NUMBER, sql[start:i], start))
            continue

        # Identifiers and reserved keywords.
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            word = sql[start:i]
            upper = word.upper()
            kind = KEYWORDS.get(upper, TokenKind.IDENT)
            tokens.append(Token(kind, word if kind is TokenKind.IDENT else upper, start))
            continue

        raise LexError(f"Unexpected character {ch!r} at position {i}")

    tokens.append(Token(TokenKind.EOF, "", n))
    return tokens
