package plc.project;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * See the specification for information about what the different visit
 * methods should do.
 */
public final class Analyzer implements Ast.Visitor<Void> {
    public Scope scope;
    private Ast.Method method;

    public Analyzer(Scope parent) {
        scope = new Scope(parent);
        scope.defineFunction("print", "System.out.println", Arrays.asList(Environment.Type.ANY), Environment.Type.NIL, args -> Environment.NIL);
    }

    public Scope getScope() {
        return scope;
    }

    @Override
    public Void visit(Ast.Source ast) {
        if(scope.lookupFunction("main", 0) != null && scope.lookupFunction("main", 0).getReturnType() == Environment.Type.INTEGER) {
            for(Ast.Field field : ast.getFields()) {
                visit(field);
            }
            for(Ast.Method method : ast.getMethods()) {
                visit(method);
            }
        } else {
            throw new RuntimeException();
        }
        return null;
    }

    @Override
    public Void visit(Ast.Field ast) {
        try {
            if (ast.getValue().isPresent()) {
                visit(ast.getValue().get());
                requireAssignable(Environment.getType(ast.getTypeName()), ast.getValue().get().getType());
                scope.defineVariable(ast.getName(), ast.getName(), ast.getValue().get().getType(), scope.lookupVariable(ast.getName()).getConstant(), Environment.NIL);
                ast.setVariable(scope.lookupVariable(ast.getName()));
            }
            else {
                scope.defineVariable(ast.getName(), ast.getName(), Environment.getType(ast.getTypeName()), scope.lookupVariable(ast.getName()).getConstant(), Environment.NIL);
                ast.setVariable(scope.lookupVariable(ast.getName()));
            }
        } catch (RuntimeException r) {
            throw new RuntimeException(r);
        }
        return null;
    }

    @Override
    public Void visit(Ast.Method ast) {
        List<Environment.Type> paramTypes = new ArrayList<>();
        ast.getParameterTypeNames().forEach(s -> {
            paramTypes.add(Environment.getType(s));
        });

        Environment.Type returnType = Environment.Type.NIL;
        if (ast.getReturnTypeName().isPresent()) {
            returnType = Environment.getType(ast.getReturnTypeName().get());
        }

        ast.setFunction(scope.defineFunction(ast.getName(), ast.getName(), paramTypes, returnType, args -> Environment.NIL));
        try {
            scope = new Scope(scope);

            for (int i = 0; i < ast.getParameters().size(); i++) {
                scope.defineVariable(ast.getParameters().get(i), ast.getParameters().get(i), paramTypes.get(i), scope.lookupVariable(ast.getParameters().get(i)).getConstant(), Environment.NIL);
            }

            for (Ast.Statement stmt : ast.getStatements()) {
                visit(stmt);
            }
        } finally {
            scope = scope.getParent();
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Expression ast) {
        visit(ast.getExpression());
        try {
            if (ast.getExpression().getClass() != Ast.Expression.Function.class) {
                throw new RuntimeException("Must be function");
            }
        }
        catch (RuntimeException r) {
            throw new RuntimeException(r);
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Declaration ast) {
        try {
            Environment.Type typeName = Environment.Type.ANY;
            if (ast.getValue().isPresent()) {
                visit(ast.getValue().get());
                typeName = ast.getValue().get().getType();
            }
            if (ast.getTypeName().isPresent())
                typeName = Environment.getType(ast.getTypeName().get());

            if (ast.getValue().isEmpty() && ast.getTypeName().isEmpty())
                throw new RuntimeException();

            scope.defineVariable(ast.getName(), ast.getName(), typeName, false, Environment.NIL);
            ast.setVariable(scope.lookupVariable(ast.getName()));
        } catch (RuntimeException r) {
            throw new RuntimeException(r);
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Assignment ast) {
        try {
            if (!(ast.getReceiver() instanceof Ast.Expression.Access)) {
                throw new RuntimeException("Expected Access Expression");
            }

            visit(ast.getReceiver());
            visit(ast.getValue());
            requireAssignable(ast.getReceiver().getType(), ast.getValue().getType());
        } catch (RuntimeException r) {
            throw new RuntimeException(r);
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.If ast) {
        visit(ast.getCondition());
        if (ast.getCondition().getType() != Environment.Type.BOOLEAN || ast.getThenStatements().isEmpty()) {
            throw new RuntimeException();
        }

        for (Ast.Statement then : ast.getThenStatements()) {
            try {
                scope = new Scope(scope);
                visit(then);
            } finally {
                scope = scope.getParent();
            }
        }
        for (Ast.Statement elseStmt : ast.getElseStatements()) {
            try {
                scope = new Scope(scope);
                visit(elseStmt);
            } finally {
                scope = scope.getParent();
            }
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.For ast) {
        try {
            visit(ast.getCondition());
            visit(ast.getInitialization());
            visit(ast.getIncrement());

            if (ast.getCondition().getType() != Environment.Type.BOOLEAN || ast.getStatements().isEmpty()) {
                throw new RuntimeException();
            }

            if (!(ast.getInitialization().getClass().equals(ast.getIncrement().getClass()))) {
                throw new RuntimeException("Initialization and Increment must be same expression type");
            }

            try {
                scope = new Scope(scope);
                for (Ast.Statement stmt : ast.getStatements()) {
                    visit(stmt);
                }
            } finally {
                scope = scope.getParent();
            }
        } catch (RuntimeException r) {
            throw new RuntimeException(r);
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.While ast) {
        try {
            visit(ast.getCondition());
            requireAssignable(Environment.Type.BOOLEAN, ast.getCondition().getType());
            try {
                scope = new Scope(scope);
                for (Ast.Statement stmt : ast.getStatements()) {
                    visit(stmt);
                }
            } finally {
                scope = scope.getParent();
            }
        } catch (RuntimeException r) {
            throw new RuntimeException(r);
        }
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Return ast) {
        if (method.getReturnTypeName().isPresent())
            requireAssignable(Environment.getType(method.getReturnTypeName().get()), ast.getValue().getType());
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Literal ast) {
        Object literal = ast.getLiteral();

        if (literal == null)
            ast.setType(Environment.Type.NIL);

        if (literal instanceof Boolean)
            ast.setType(Environment.Type.BOOLEAN);

        if (literal instanceof Character)
            ast.setType(Environment.Type.CHARACTER);

        if (literal instanceof String)
            ast.setType(Environment.Type.STRING);

        if (literal instanceof BigInteger value) {
            // Check size
            if (value.compareTo(BigInteger.valueOf(Integer.MAX_VALUE)) <= 0 &&
                    value.compareTo(BigInteger.valueOf(Integer.MIN_VALUE)) >= 0)
                ast.setType(Environment.Type.INTEGER);
            else
                throw new RuntimeException("Value of Integer is not in the range of and Integer.");
            return null;
        }

        if (literal instanceof BigDecimal value) {
            // Check size
            if (value.compareTo(BigDecimal.valueOf(Double.MAX_VALUE)) <= 0 &&
                    value.compareTo(BigDecimal.valueOf(Double.MIN_VALUE)) >= 0)
                ast.setType(Environment.Type.DECIMAL);
            else
                throw new RuntimeException("Value of Decimal is not in the range of and Double.");
            return null;
        }
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Group ast) {
        if (ast.getExpression() instanceof Ast.Expression.Binary) {
            visit(ast.getExpression());
            ast.setType(ast.getExpression().getType());
            return null;
        }
        throw new RuntimeException("Expected an Ast.Expr.Binary");
    }

    @Override
    public Void visit(Ast.Expression.Binary ast) {
        Ast.Expression left = ast.getLeft();
        visit(left);
        Ast.Expression right = ast.getRight();
        visit(right);

        switch (ast.getOperator()) {
            case "AND":
            case "OR":
            case "&&":
            case "||":
                if (left.getType() == Environment.Type.BOOLEAN &&
                        right.getType() == Environment.Type.BOOLEAN) {
                    ast.setType(Environment.Type.BOOLEAN);
                    return null;
                }
                throw new RuntimeException("Boolean Type Expected");
            case "<":
            case "<=":
            case ">":
            case ">=":
            case "==":
            case "!=":
                requireAssignable(Environment.Type.COMPARABLE, left.getType());
                requireAssignable(Environment.Type.COMPARABLE, right.getType());
                requireAssignable(left.getType(), right.getType());
                ast.setType(Environment.Type.BOOLEAN);
                break;
            case "+":
                // Check for string concat
                if (left.getType() == Environment.Type.STRING ||
                        right.getType() == Environment.Type.STRING) {
                    ast.setType(Environment.Type.STRING);
                    break;
                }
            case "-":
            case "*":
            case "/":
                if (left.getType() == Environment.Type.INTEGER ||
                        left.getType() == Environment.Type.DECIMAL) {
                    if (left.getType() == right.getType()) {
                        ast.setType(left.getType());
                        return null;
                    }
                }
                throw new RuntimeException("Expected Integer or Decimal");
            default:
                return null;
        }
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Access ast) {
        // Check if field
        if (ast.getReceiver().isPresent()) {
            Ast.Expression expr = ast.getReceiver().get();
            visit(expr);
            ast.setVariable(expr.getType().getField(ast.getName()));
        } else {
            ast.setVariable(scope.lookupVariable(ast.getName()));
        }
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Function ast) {
        //Check if Method or function
        if (ast.getReceiver().isPresent()) {
            Ast.Expression expr = ast.getReceiver().get();
            visit(expr);

            Environment.Function func = expr.getType().getFunction(ast.getName(), ast.getArguments().size());
            List<Ast.Expression> args = ast.getArguments();
            List<Environment.Type> argTypes = func.getParameterTypes();

            for (int i = 1; i < args.size(); i++) {
                visit(args.get(i));
                requireAssignable(argTypes.get(i), args.get(i).getType());
            }

            ast.setFunction(func);
        } else {
            Environment.Function func = scope.lookupFunction(ast.getName(), ast.getArguments().size());
            List<Ast.Expression> args = ast.getArguments();
            List<Environment.Type> argTypes = func.getParameterTypes();

            for (int i = 0; i < args.size(); i++) {
                visit(args.get(i));
                requireAssignable(argTypes.get(i), args.get(i).getType());
            }
            ast.setFunction(func);
        }
        return null;
    }

    public static void requireAssignable(Environment.Type target, Environment.Type type) {
        if (target.getName().equals(type.getName()))
            return;

        switch (target.getName()) {
            case "Any":
                return;
            case "Comparable":
                if (type.getName().equals("Integer") ||
                        type.getName().equals("Decimal") ||
                        type.getName().equals("Character") ||
                        type.getName().equals("String")
                )
                    return;
        }
        throw new RuntimeException("Invalid Type");
    }
}
