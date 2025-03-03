package plc.project;

import java.util.List;
import java.util.ArrayList;

/**
 * The lexer works through three main functions:
 *
 *  - {@link #lex()}, which repeatedly calls lexToken() and skips whitespace
 *  - {@link #lexToken()}, which lexes the next token
 *  - {@link CharStream}, which manages the state of the lexer and literals
 *
 * If the lexer fails to parse something (such as an unterminated string) you
 * should throw a {@link ParseException} with an index at the invalid character.
 *
 * The {@link #peek(String...)} and {@link #match(String...)} functions are
 * helpers you need to use, they will make the implementation easier.
 */
public final class Lexer {

    private final CharStream chars;

    public Lexer(String input) {
        chars = new CharStream(input);
    }

    /**
     * Repeatedly lexes the input using {@link #lexToken()}, also skipping over
     * whitespace where appropriate.
     */
    public List<Token> lex() {
        List<Token> tokenList = new ArrayList<Token>();
        while (chars.has(0)) {
            // Skip Whitespace
            if (match("[ \b\n\t\r]")) {
                chars.skip();
                continue;
            }
            Token newToken = lexToken();
            tokenList.add(newToken);
        }
        return tokenList;
    }

    /**
     * This method determines the type of the next token, delegating to the
     * appropriate lex method. As such, it is best for this method to not change
     * the state of the char stream (thus, use peek not match).
     *
     * The next character should start a valid token since whitespace is handled
     * by {@link #lex()}
     */
    public Token lexToken() {
        Token newToken;
        if (peek("[A-Za-z_]")) {
            newToken = lexIdentifier();
        }
        else if (peek("([-+]|[0-9])")) {
            newToken = lexNumber();
        }
        else if (peek("\\'")) {
            newToken = lexCharacter();
        }
        else if (peek("\\\"")) {
            newToken = lexString();
        }
        else {
            newToken = lexOperator();
        }
        return newToken;
    }

    public Token lexIdentifier() {
        chars.advance();
        while(match("[A-Za-z0-9_-]"));
        return chars.emit(Token.Type.IDENTIFIER);
    }

    public Token lexNumber() {
        if (peek("[\\+-]")) {
            match("[\\+-]");
        }

        // 0 followed by decimal (0.5)
        if (peek("0")) {
            match("0");
            if (peek("\\.")) {
                match("\\.");
                if (peek("[0-9]")) {
                    while (match("[0-9]"));
                    return chars.emit(Token.Type.DECIMAL);
                }
                // Trailing decimal or non-numeric
                else {
                    return chars.emit(Token.Type.OPERATOR);
                }
            }
            else if (!peek("[1-9]")) {
                return chars.emit(Token.Type.INTEGER);
            }
        }

        else if (peek("[1-9]")) {
            // Consume all numbers until decimal point
            while (match("[0-9]"));

            // Find decimal and continue consuming numbers
            if (peek("\\.")) {
                match("\\.");
                if (peek("[0-9]")) {
                    while (match("[0-9]"));
                    return chars.emit(Token.Type.DECIMAL);
                }
                // Trailing decimal or non-numeric
                else {
                    return chars.emit(Token.Type.OPERATOR);
                }
            } else {
                return chars.emit(Token.Type.INTEGER);
            }
        }
        // If just + or -
        return chars.emit(Token.Type.OPERATOR);
    }

    public Token lexCharacter() {
        chars.advance();
        if (match("\\\\")) {
            lexEscape();
        }
        else {
            chars.advance();
        }
        if (!match("\\'")) {
            throw new ParseException("Invalid use of character literal", chars.index);
        }
        return chars.emit(Token.Type.CHARACTER);
    }

    public Token lexString() {
        chars.advance();
        // Consume until "
        while (!match("\\\"")) {
            if (chars.index >= chars.input.length()) {
                throw new ParseException("Reached EOF in string literal", chars.index);
            }
            else if (match("\\\\")) {
                lexEscape();
            }
            else {
                chars.advance();
            }
        }
        return chars.emit(Token.Type.STRING);
    }

    public void lexEscape() {
        if (!match("[bnrt'\\\\\"]")) {
            throw new ParseException("Invalid escape sequence", chars.index);
        }
    }

    public Token lexOperator() {
        if (match("[!=<>]")) {
            match("=");
        }
        else if (match("[&]")) {
            match("[&]");
        }
        else if (match("[|]")) {
            match("[|]");
        }
        else {
            chars.advance();
        }
        return chars.emit(Token.Type.OPERATOR);
    }

    /**
     * Returns true if the next sequence of characters match the given patterns,
     * which should be a regex. For example, {@code peek("a", "b", "c")} would
     * return true if the next characters are {@code 'a', 'b', 'c'}.
     */
    public boolean peek(String... patterns) {
        for (int i=0; i<patterns.length; i++) {
            if (!chars.has(i) || !String.valueOf(chars.get(i)).matches((patterns[i]))) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns true in the same way as {@link #peek(String...)}, but also
     * advances the character stream past all matched characters if peek returns
     * true. Hint - it's easiest to have this method simply call peek.
     */
    public boolean match(String... patterns) {
        boolean peek = peek(patterns);
        if (peek) {
            for (int i=0; i<patterns.length; i++) {
                chars.advance();
            }
        }
        return peek;
    }

    /**
     * A helper class maintaining the input string, current index of the char
     * stream, and the current length of the token being matched.
     *
     * You should rely on peek/match for state management in nearly all cases.
     * The only field you need to access is {@link #index} for any {@link
     * ParseException} which is thrown.
     */
    public static final class CharStream {

        private final String input;
        private int index = 0;
        private int length = 0;

        public CharStream(String input) {
            this.input = input;
        }

        public boolean has(int offset) {
            return index + offset < input.length();
        }

        public char get(int offset) {
            return input.charAt(index + offset);
        }

        public void advance() {
            index++;
            length++;
        }

        public void skip() {
            length = 0;
        }

        public Token emit(Token.Type type) {
            int start = index - length;
            skip();
            return new Token(type, input.substring(start, index), start);
        }

    }

}
