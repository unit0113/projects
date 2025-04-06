package plc.project;

import java.util.*;
import java.math.BigInteger;
import java.math.BigDecimal;

/**
 * The parser takes the sequence of tokens emitted by the lexer and turns that
 * into a structured representation of the program, called the Abstract Syntax
 * Tree (AST).
 *
 * The parser has a similar architecture to the lexer, just with {@link Token}s
 * instead of characters. As before, {@link #peek(Object...)} and {@link
 * #match(Object...)} are helpers to make the implementation easier.
 *
 * This type of parser is called <em>recursive descent</em>. Each rule in our
 * grammar will have it's own function, and reference to other rules correspond
 * to calling that functions.
 */
public final class Parser {

    private final TokenStream tokens;

    public Parser(List<Token> tokens) {
        this.tokens = new TokenStream(tokens);
    }

    /**
     * Parses the {@code source} rule.
     */
    public Ast.Source parseSource() throws ParseException {
        //source ::= field* method*
        try {
            List<Ast.Field> fields = new ArrayList<>();
            List<Ast.Method> methods = new ArrayList<>();
            while (tokens.has(0)) {
                if (match("LET")) {
                    fields.add(parseField());
                } else if (match("DEF")) {
                    methods.add(parseMethod());
                }
            }
            return new Ast.Source(fields, methods);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code field} rule. This method should only be called if the
     * next tokens start a field, aka {@code LET}.
     */
    public Ast.Field parseField() throws ParseException {
        //field ::= 'LET' 'CONST'? identifier ('=' expression)? ';'
        //Updated for Analyser: field ::= field ::= 'LET' 'CONST'? identifier ':' identifier ('=' expression)? ';'
        /* For original Parser
        try {
            boolean constant = match("CONST");
            Ast.Statement.Declaration declaration = parseDeclarationStatement();
            return new Ast.Field(declaration.getName(), constant, declaration.getValue());
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }*/
        try {
            boolean constant = match("CONST");

            if (!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Invalid Identifier", index);
            }

            String identifier = tokens.get(-1).getLiteral();

            // Get Type
            if (!match(":")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected ':'", index);
            }

            if (!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Invalid Identifier", index);
            }
            String typeName = tokens.get(-1).getLiteral();

            if (match("=")) {
                Ast.Expression rightExpr = parseExpression();
                if (match(";")) {
                    return new Ast.Field(identifier, typeName, constant, Optional.of(rightExpr));
                }
            } else {
                if (match(";")) {
                    return new Ast.Field(identifier, typeName, constant,Optional.empty());
                }
            }
            int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
            throw new ParseException("Expected Semicolon", index);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code method} rule. This method should only be called if the
     * next tokens start a method, aka {@code DEF}.
     */
    public Ast.Method parseMethod() throws ParseException {
        //method ::= 'DEF' identifier '(' (identifier (',' identifier)*)? ')' 'DO' statement* 'END'
        //Update for Analyzer: 'DEF' identifier '(' (identifier ':' identifier (',' identifier ':' identifier)*)? ')' (':' identifier)? 'DO' statement* 'END'
        /* Previous version
        try {
            if (!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Identifier", index);
            }

            String functionName = tokens.get(-1).getLiteral();
            if (!match("(")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Parenthesis", index);
            }

            List<String> params = new ArrayList<>();
            //Loop through args and store
            while (match(Token.Type.IDENTIFIER)) {
                params.add(tokens.get(-1).getLiteral());
                if (!match(",")) {
                    if (!peek(")")) {
                        int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                        throw new ParseException("Invalid Syntax: Function arguments must be comma separated", index);
                    }
                }
            }

            if (!match(")")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Parenthesis", index);
            }
            if (!match("DO")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected DO", index);
            }

            //Loop through all statements
            List<Ast.Statement> statements = new ArrayList<>();
            while (!match("END") && tokens.has(0)) {
                statements.add(parseStatement());
            }

            if(!tokens.get(-1).getLiteral().equals("END")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("missing END", index);
            }
            return new Ast.Method(functionName, params, statements);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }*/
        try {
            if (!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Identifier", index);
            }

            String functionName = tokens.get(-1).getLiteral();
            if (!match("(")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Parenthesis", index);
            }

            List<String> params = new ArrayList<>();
            List<String> paramTypes = new ArrayList<>();
            //Loop through args and store
            while (match(Token.Type.IDENTIFIER)) {
                params.add(tokens.get(-1).getLiteral());
                if (!match(":")) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Expected ':'", index);
                }
                if (!match(Token.Type.IDENTIFIER)) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Invalid Identifier", index);
                }
                paramTypes.add(tokens.get(-1).getLiteral());

                if (!match(",")) {
                    if (!peek(")")) {
                        int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                        throw new ParseException("Invalid Syntax: Function arguments must be comma separated", index);
                    }
                }
            }

            if (!match(")")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Parenthesis", index);
            }

            // Return type
            Optional<String> returnType = Optional.empty();
            if (match(":")) {
                if (!match(Token.Type.IDENTIFIER)) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Invalid Identifier", index);
                }
                returnType = Optional.of(tokens.get(-1).getLiteral());
            }

            if (!match("DO")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected DO", index);
            }

            //Loop through all statements
            List<Ast.Statement> statements = new ArrayList<>();
            while (!match("END") && tokens.has(0)) {
                statements.add(parseStatement());
            }

            if(!tokens.get(-1).getLiteral().equals("END")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("missing END", index);
            }
            return new Ast.Method(functionName, params, paramTypes, returnType, statements);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code statement} rule and delegates to the necessary method.
     * If the next tokens do not start a declaration, if, while, or return
     * statement, then it is an expression/assignment statement.
     */
    public Ast.Statement parseStatement() throws ParseException {
        /** statement ::=
        *    //'LET' identifier ('=' expression)? ';' |
        *    Update for Analyzer: 'LET' identifier (':' identifier)? ('=' expression)? ';' |
        *    'IF' expression 'DO' statement* ('ELSE' statement*)? 'END' |
        *    'FOR' identifier 'IN' expression 'DO' statement* 'END' |
        *    'WHILE' expression 'DO' statement* 'END' |
        *    'RETURN' expression ';' |
        *    expression ('=' expression)? ';'
        */
        try {
            //Switch case (sort of)
            if (match("LET")) {
                return parseDeclarationStatement();
            } else if (match("IF")) {
                return parseIfStatement();
            } else if (match("FOR")) {
                return parseForStatement();
            } else if (match("WHILE")) {
                return parseWhileStatement();
            } else if (match("RETURN")) {
                return parseReturnStatement();
            } else {
                Ast.Expression lhs = parseExpression();
                if (!match("=")) {
                    if (!match(";")) {
                        int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                        throw new ParseException("Expected semicolon", index);
                    }
                    return new Ast.Statement.Expression(lhs);
                }

                Ast.Expression rightExpr = parseExpression();
                if (!match(";")) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Expected semicolon", index);
                }
                return new Ast.Statement.Assignment(lhs, rightExpr);
            }
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses a declaration statement from the {@code statement} rule. This
     * method should only be called if the next tokens start a declaration
     * statement, aka {@code LET}.
     */
    public Ast.Statement.Declaration parseDeclarationStatement() throws ParseException {
        //'LET' identifier ('=' expression)? ';'
        //Update for Analyzer: 'LET' identifier (':' identifier)? ('=' expression)? ';'
        try {
            if (!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Invalid Identifier", index);
            }

            String identifier = tokens.get(-1).getLiteral();

            // Get Type
            Optional<String> typeName = Optional.empty();
            if (match(":")) {
                if (!match(Token.Type.IDENTIFIER)) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Invalid Identifier", index);
                }
                typeName = Optional.of(tokens.get(-1).getLiteral());
            }

            if (match("=")) {
                Ast.Expression rightExpr = parseExpression();
                if (match(";")) {
                    return new Ast.Statement.Declaration(identifier, typeName, Optional.of(rightExpr));
                }
            } else {
                if (match(";")) {
                    return new Ast.Statement.Declaration(identifier, typeName, Optional.empty());
                }
            }
            int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
            throw new ParseException("Expected Semicolon", index);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses an if statement from the {@code statement} rule. This method
     * should only be called if the next tokens start an if statement, aka
     * {@code IF}.
     */
    public Ast.Statement.If parseIfStatement() throws ParseException {
        //'IF' expression 'DO' statement* ('ELSE' statement*)? 'END'
        try {
            Ast.Expression expression = parseExpression();
            if (!match("DO")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected DO", index);
            }
            boolean elseSeen = false;
            List<Ast.Statement> thenStatements = new ArrayList<>();
            List<Ast.Statement> elseStatements = new ArrayList<>();

            while (!match("END") && tokens.has(0)) {
                if (match("ELSE")) {
                    //Ensure only 1 else block
                    if (!elseSeen) {
                        elseSeen = true;
                    } else {
                        int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                        throw new ParseException("Invalid Syntax: ELSE statement without matching IF", index);
                    }
                }
                if (elseSeen) {
                    elseStatements.add(parseStatement());
                } else {
                    thenStatements.add(parseStatement());
                }
            }
            if(!tokens.get(-1).getLiteral().equals("END")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected END", index);
            }
            return new Ast.Statement.If(expression, thenStatements, elseStatements);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses a for statement from the {@code statement} rule. This method
     * should only be called if the next tokens start a for statement, aka
     * {@code FOR}.
     */
    public Ast.Statement.For parseForStatement() throws ParseException {
        //'FOR' '(' (identifier '=' expression)? ';' expression ';' (identifier '=' expression)? ')' statement* 'END'
        //TODO Awaiting determination on missing optionals in Ast.java
        /*try {
            if (!match("(")) {
                throw new ParseException("Expected Parentheses", tokens.get(-1).getIndex());
            }

            Optional<Ast.Statement> initialization = Optional.empty();
            if (match(Token.Type.IDENTIFIER)) {
                initialization = Optional.of(parseStatement());
            }
            if (!match(";")) {
                throw new ParseException("Expected Semicolon", tokens.get(-1).getIndex());
            }

            Ast.Expression condition = parseExpression();
            if (!match(";")) {
                throw new ParseException("Expected Semicolon", tokens.get(-1).getIndex());
            }

            Optional<Ast.Statement> increment = Optional.empty();
            if (match(Token.Type.IDENTIFIER)) {
                increment = Optional.of(parseStatement());
            }

            if (!match(")")) {
                throw new ParseException("Expected Parentheses", tokens.get(-1).getIndex());
            }

            List<Ast.Statement> statements = new ArrayList<>();
            while (!match("END") && tokens.has(0)) {
                statements.add(parseStatement());
            }
            if(!tokens.get(-1).getLiteral().equals("END")) {
                throw new ParseException("Expected END", tokens.get(-1).getIndex());
            }
            return new Ast.Statement.For(initialization, condition, increment, statements);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }*/
        //TODO Optional-less version, remove when ast fixed
        try {
            if (!match("(")) {
                throw new ParseException("Expected Parentheses", tokens.get(-1).getIndex());
            }

            if(!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Identifier", index);
            }
            Ast.Statement initialization = parseStatement();

            if (!match(";")) {
                throw new ParseException("Expected Semicolon", tokens.get(-1).getIndex());
            }

            Ast.Expression condition = parseExpression();
            if (!match(";")) {
                throw new ParseException("Expected Semicolon", tokens.get(-1).getIndex());
            }

            if(!match(Token.Type.IDENTIFIER)) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Identifier", index);
            }
            Ast.Statement increment = parseStatement();

            if (!match(")")) {
                throw new ParseException("Expected Parentheses", tokens.get(-1).getIndex());
            }

            List<Ast.Statement> statements = new ArrayList<>();
            while (!match("END") && tokens.has(0)) {
                statements.add(parseStatement());
            }
            if(!tokens.get(-1).getLiteral().equals("END")) {
                throw new ParseException("Expected END", tokens.get(-1).getIndex());
            }
            return new Ast.Statement.For(initialization, condition, increment, statements);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses a while statement from the {@code statement} rule. This method
     * should only be called if the next tokens start a while statement, aka
     * {@code WHILE}.
     */
    public Ast.Statement.While parseWhileStatement() throws ParseException {
        //'WHILE' expression 'DO' statement* 'END'
        try {
            Ast.Expression expression = parseExpression();
            if (!match("DO")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected DO", index);
            }

            List<Ast.Statement> statements = new ArrayList<>();
            while (!match("END") && tokens.has(0)) {
                statements.add(parseStatement());
            }
            if(!tokens.get(-1).getLiteral().equals("END")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected END", index);
            }
            return new Ast.Statement.While(expression, statements);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses a return statement from the {@code statement} rule. This method
     * should only be called if the next tokens start a return statement, aka
     * {@code RETURN}.
     */
    public Ast.Statement.Return parseReturnStatement() throws ParseException {
        //'RETURN' expression ';'
        try {
            Ast.Expression expression = parseExpression();
            if (!match(";")) {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Expected Semicolon", index);
            }
            return new Ast.Statement.Return(expression);
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code expression} rule.
     */
    public Ast.Expression parseExpression() throws ParseException {
        //expression ::= logical_expression
        try {
            return parseLogicalExpression();
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code logical-expression} rule.
     */
    public Ast.Expression parseLogicalExpression() throws ParseException {
        //logical_expression ::= comparison_expression (('AND' | 'OR') comparison_expression)*
        try {
            Ast.Expression leftExpr = parseEqualityExpression();
            while (match("&&") || match("||") || match("AND") || match("OR")) {
                String operation = tokens.get(-1).getLiteral();
                Ast.Expression rightExpr = parseEqualityExpression();
                leftExpr = new Ast.Expression.Binary(operation, leftExpr, rightExpr);
            }
            return leftExpr;
        } catch(ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code equality-expression} rule.
     */
    public Ast.Expression parseEqualityExpression() throws ParseException {
        //comparison_expression ::= additive_expression (('<' | '<=' | '>' | '>=' | '==' | '!=') additive_expression)*
        try {
            Ast.Expression leftExpr = parseAdditiveExpression();
            while (match("<")
                    || match("<=")
                    || match(">")
                    || match(">=")
                    || match("==")
                    || match("!=")) {
                String operation = tokens.get(-1).getLiteral();
                Ast.Expression rightExpr = parseAdditiveExpression();
                leftExpr = new Ast.Expression.Binary(operation, leftExpr, rightExpr);
            }
            return leftExpr;
        } catch(ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code additive-expression} rule.
     */
    public Ast.Expression parseAdditiveExpression() throws ParseException {
        //additive_expression ::= multiplicative_expression (('+' | '-') multiplicative_expression)*
        try {
            Ast.Expression leftExpr = parseMultiplicativeExpression();
            while (match("+") || match("-")) {
                String operation = tokens.get(-1).getLiteral();
                Ast.Expression rightExpr = parseMultiplicativeExpression();
                leftExpr = new Ast.Expression.Binary(operation, leftExpr, rightExpr);
            }
            return leftExpr;

        } catch(ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code multiplicative-expression} rule.
     */
    public Ast.Expression parseMultiplicativeExpression() throws ParseException {
        //multiplicative_expression ::= secondary_expression (('*' | '/') secondary_expression)*
        try {
            Ast.Expression leftExpr = parseSecondaryExpression();
            while (match("/") || match("*")) {
                String operation = tokens.get(-1).getLiteral();
                Ast.Expression rightExpr = parseSecondaryExpression();
                leftExpr = new Ast.Expression.Binary(operation, leftExpr, rightExpr);
            }
            return leftExpr;
        } catch(ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code secondary-expression} rule.
     */
    public Ast.Expression parseSecondaryExpression() throws ParseException {
        //secondary_expression ::= primary_expression ('.' identifier ('(' (expression (',' expression)*)? ')')?)*
        try {
            Ast.Expression leftExpr = parsePrimaryExpression();
            //Loop through chained methods
            while (match(".")) {
                if (!match(Token.Type.IDENTIFIER)) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Invalid Identifier", index);
                }
                String receiver = tokens.get(-1).getLiteral();
                if (!match("(")) {
                    leftExpr = new Ast.Expression.Access(Optional.of(leftExpr), receiver);
                } else {
                    List<Ast.Expression> args = new ArrayList<>();
                    if(!match(")")) {
                        args.add(parseExpression());
                        while (match(",")) {
                            args.add(parseExpression());
                        }
                        if(!match(")")) {
                            int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                            throw new ParseException("Missing ')'", index);
                        }
                    }
                    leftExpr = new Ast.Expression.Function(Optional.of(leftExpr), receiver, args);
                }
            }
            return leftExpr;
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * Parses the {@code primary-expression} rule. This is the top-level rule
     * for expressions and includes literal values, grouping, variables, and
     * functions. It may be helpful to break these up into other methods but is
     * not strictly necessary.
     */
    public Ast.Expression parsePrimaryExpression() throws ParseException {
        /** primary_expression ::=
            'NIL' | 'TRUE' | 'FALSE' |
            integer | decimal | character | string |
            '(' expression ')' |
            identifier ('(' (expression (',' expression)*)? ')')?
         */
        try {
            //Switch case (sort of) (again)
            if (match("NIL")) {
                return new Ast.Expression.Literal(null);
            }
            else if (match("TRUE")) {
                return new Ast.Expression.Literal(true);
            }
            else if (match("FALSE")) {
                return new Ast.Expression.Literal(false);
            }
            else if (match(Token.Type.INTEGER)) {
                return new Ast.Expression.Literal(new BigInteger(tokens.get(-1).getLiteral()));
            }
            else if (match(Token.Type.DECIMAL)) {
                return new Ast.Expression.Literal(new BigDecimal(tokens.get(-1).getLiteral()));
            }
            else if (match(Token.Type.CHARACTER)) {
                String str = tokens.get(-1).getLiteral();
                return new Ast.Expression.Literal(str.charAt(1));
            }
            else if (match(Token.Type.STRING)) {
                String str = tokens.get(-1).getLiteral();
                str = str.substring(1, str.length() - 1);
                if(str.contains("\\")) {
                    str = str.replace("\\n", "\n")
                            .replace("\\t", "\t")
                            .replace("\\b", "\b")
                            .replace("\\r", "\r")
                            .replace("\\'", "'")
                            .replace("\\\\", "\\")
                            .replace("\\\"", "\"");
                }
                return new Ast.Expression.Literal(str);
            }
            else if (match(Token.Type.IDENTIFIER)) {
                String name = tokens.get(-1).getLiteral();
                if (!match("(")) {
                    return new Ast.Expression.Access(Optional.empty(), name);
                }
                else {
                    if (!match(")")) {
                        Ast.Expression initalExpr = parseExpression();
                        List<Ast.Expression> args = new ArrayList<>();
                        args.add(initalExpr);

                        while (match(",")) {
                            args.add(parseExpression());
                        }

                        if (match(")")) {
                            return new Ast.Expression.Function(Optional.empty(), name, args);
                        } else {
                            int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                            throw new ParseException("Closing parentheses expected", index);
                        }
                    } else {
                        return new Ast.Expression.Function(Optional.empty(), name, Collections.emptyList());
                    }
                }

            } else if (match("(")) {
                Ast.Expression expr = parseExpression();
                if (!match(")")) {
                    int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                    throw new ParseException("Expected closing parenthesis", index);
                }
                return new Ast.Expression.Group(expr);
            } else {
                int index = (tokens.has(0)) ? tokens.get(0).getIndex() : tokens.get(-1).getIndex() + tokens.get(-1).getLiteral().length();
                throw new ParseException("Invalid Primary Expression", index);
            }
        } catch (ParseException p) {
            throw new ParseException(p.getMessage(), p.getIndex());
        }
    }

    /**
     * As in the lexer, returns {@code true} if the current sequence of tokens
     * matches the given patterns. Unlike the lexer, the pattern is not a regex;
     * instead it is either a {@link Token.Type}, which matches if the token's
     * type is the same, or a {@link String}, which matches if the token's
     * literal is the same.
     *
     * In other words, {@code Token(IDENTIFIER, "literal")} is matched by both
     * {@code peek(Token.Type.IDENTIFIER)} and {@code peek("literal")}.
     */
    private boolean peek(Object... patterns) {
        for (int i = 0; i < patterns.length; i++) {
            if (!tokens.has(i)) {
                return false;
            } else if (patterns[i] instanceof Token.Type) {
                if (patterns[i] != tokens.get(i).getType()) {
                    return false;
                }
            } else if (patterns[i] instanceof String) {
                if (!patterns[i].equals(tokens.get(i).getLiteral())) {
                    return false;
                }
            } else {
                throw new AssertionError("Invalid pattern object: " + patterns[i].getClass());
            }
        }
        return true;
    }

    /**
     * As in the lexer, returns {@code true} if {@link #peek(Object...)} is true
     * and advances the token stream.
     */
    private boolean match(Object... patterns) {
        boolean peek = peek(patterns);
        if (peek) {
            for (int i = 0; i < patterns.length; i++)
                tokens.advance();
        }
        return peek;
    }

    private static final class TokenStream {

        private final List<Token> tokens;
        private int index = 0;

        private TokenStream(List<Token> tokens) {
            this.tokens = tokens;
        }

        /**
         * Returns true if there is a token at index + offset.
         */
        public boolean has(int offset) {
            return index + offset < tokens.size();
        }

        /**
         * Gets the token at index + offset.
         */
        public Token get(int offset) {
            return tokens.get(index + offset);
        }

        /**
         * Advances to the next token, incrementing the index.
         */
        public void advance() {
            index++;
        }

    }

}
