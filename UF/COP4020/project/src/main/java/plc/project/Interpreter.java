package plc.project;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class Interpreter implements Ast.Visitor<Environment.PlcObject> {

    private Scope scope = new Scope(null);

    public Interpreter(Scope parent) {
        scope = new Scope(parent);
        scope.defineFunction("print", 1, args -> {
            System.out.println(args.get(0).getValue());
            return Environment.NIL;
        });
    }

    public Scope getScope() {
        return scope;
    }

    @Override
    public Environment.PlcObject visit(Ast.Source ast) {
        for(Ast.Field field : ast.getFields()) {visit(field);}
        for(Ast.Method method : ast.getMethods()) {visit(method);}
        return scope.lookupFunction("main", 0).invoke(Collections.emptyList());
    }

    @Override
    public Environment.PlcObject visit(Ast.Field ast) {
        Environment.PlcObject value = (ast.getValue().isPresent()) ? visit(ast.getValue().get()) : Environment.NIL;
        scope.defineVariable(ast.getName(), ast.getConstant(), value);
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Method ast) {
        scope.defineFunction(ast.getName(), ast.getParameters().size(), args -> {
            try {
                scope = new Scope(scope);
                ast.getParameters().forEach( param -> {
                    args.forEach(arg -> {
                        scope.defineVariable(param, false, arg);
                    });
                });
                ast.getStatements().forEach(this::visit);
            }
            catch (Return r) {
                return r.value;
            }
            finally {
                scope = scope.getParent();
            }
            return Environment.NIL;
        });
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.Expression ast) {
        visit(ast.getExpression());
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.Declaration ast) {
        Environment.PlcObject value = (ast.getValue().isPresent()) ? visit(ast.getValue().get()) : Environment.NIL;
        scope.defineVariable(ast.getName(), false, value);
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.Assignment ast) {
        Ast.Expression access = ast.getReceiver();
        if(ast.getReceiver() instanceof Ast.Expression.Access) {
            if(((Ast.Expression.Access) access).getReceiver().isPresent()) {
                visit(((Ast.Expression.Access) access).getReceiver().get()).setField(((Ast.Expression.Access) access).getName(), visit(ast.getValue()));
            } else {
                Environment.Variable variable = scope.lookupVariable(((Ast.Expression.Access) access).getName());
                variable.setValue(visit(ast.getValue()));
            }
        }
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.If ast) {
        // True
        if (requireType(Boolean.class, visit(ast.getCondition()))) {
            try {
                scope = new Scope(scope);
                for (Ast.Statement stmt : ast.getThenStatements()) {
                    visit(stmt);
                }
            } finally {
                scope = scope.getParent();
            }
        }
        // False
        else if (!requireType(Boolean.class, visit(ast.getCondition()))) {
            try {
                scope = new Scope(scope);
                for (Ast.Statement stmt : ast.getElseStatements()) {
                    visit(stmt);
                }
            } finally {
                scope = scope.getParent();
            }
        }
        else {
            throw new RuntimeException();
        }
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.For ast) {
        // Initialization
        visit(ast.getInitialization());

        // Loop
        while (requireType(Boolean.class, visit(ast.getCondition()))) {
            try {
                scope = new Scope(scope);
                ast.getStatements().forEach(this::visit);
                // Increment
                visit(ast.getIncrement());
            } finally {
                scope = scope.getParent();
            }
        }
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.While ast) {
        while (requireType(Boolean.class, visit(ast.getCondition()))) {
            try {
                scope = new Scope(scope);
                ast.getStatements().forEach(this::visit);
            } finally {
                scope = scope.getParent();
            }
        }
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Statement.Return ast) {
        throw new Return(visit(ast.getValue()));
    }

    @Override
    public Environment.PlcObject visit(Ast.Expression.Literal ast) {
        return (ast.getLiteral() == null) ? Environment.NIL : Environment.create(ast.getLiteral());
    }

    @Override
    public Environment.PlcObject visit(Ast.Expression.Group ast) {
        return visit(ast.getExpression());
    }

    @Override
    public Environment.PlcObject visit(Ast.Expression.Binary ast) {
        Environment.PlcObject left = visit(ast.getLeft());

        switch (ast.getOperator()) {
            case "AND":
            case "&&":
                if (!(left.getValue() instanceof Boolean) || !(visit(ast.getRight()).getValue() instanceof Boolean)) {
                    throw new RuntimeException();
                }
                if(!(Boolean)left.getValue() || !(Boolean)visit(ast.getRight()).getValue()) {return Environment.create(false);}
                else {return Environment.create(true);}

            case "OR":
            case "||":
                if(left.getValue() instanceof Boolean) {
                    if ((Boolean)left.getValue()) {
                        return Environment.create(true);
                    }
                    if(visit(ast.getRight()).getValue() instanceof Boolean) {
                        return Environment.create(visit(ast.getRight()).getValue());
                    }
                }
                throw new RuntimeException();

            case "<":
                if(left.getValue() instanceof Comparable) {
                    Environment.PlcObject right = visit(ast.getRight());
                    if(requireType(left.getValue().getClass(), right) != null) {
                        return Environment.create(((Comparable) left.getValue()).compareTo(right.getValue()) < 0);
                    }
                }
                break;

            case "<=":
                if(left.getValue() instanceof Comparable) {
                    Environment.PlcObject right = visit(ast.getRight());
                    if(requireType(left.getValue().getClass(), right) != null) {
                        return Environment.create(((Comparable) left.getValue()).compareTo(right.getValue()) <= 0);
                    }
                }
                break;

            case ">":
                if(left.getValue() instanceof Comparable) {
                    Environment.PlcObject right = visit(ast.getRight());
                    if(requireType(left.getValue().getClass(), right) != null) {
                        return Environment.create(((Comparable) left.getValue()).compareTo(right.getValue()) > 0);
                    }
                }
                break;

            case ">=":
                if(left.getValue() instanceof Comparable) {
                    Environment.PlcObject right = visit(ast.getRight());
                    if(requireType(left.getValue().getClass(), right) != null) {
                        return Environment.create(((Comparable) left.getValue()).compareTo(right.getValue()) >= 0);
                    }
                }
                break;

            case "==":
                return Environment.create(Objects.equals(left.getValue(), visit(ast.getRight()).getValue()));

            case "!=":
                return Environment.create(!Objects.equals(left.getValue(), visit(ast.getRight()).getValue()));

            case "+":
                if(left.getValue() instanceof BigInteger && visit(ast.getRight()).getValue() instanceof BigInteger) {
                    return Environment.create(requireType(BigInteger.class, left).add(requireType(BigInteger.class, visit(ast.getRight()))));
                }
                if(left.getValue() instanceof BigDecimal && visit(ast.getRight()).getValue() instanceof BigDecimal) {
                    return Environment.create(requireType(BigDecimal.class, left).add(requireType(BigDecimal.class, visit(ast.getRight()))));
                }
                if(left.getValue() instanceof String || visit(ast.getRight()).getValue() instanceof String) {
                    return Environment.create(left.getValue().toString() + visit(ast.getRight()).getValue().toString());
                }
                throw new RuntimeException();

            case "-":
                if(left.getValue() instanceof BigInteger && visit(ast.getRight()).getValue() instanceof BigInteger) {
                    return Environment.create(requireType(BigInteger.class, left).subtract(requireType(BigInteger.class, visit(ast.getRight()))));
                }
                if(left.getValue() instanceof BigDecimal && visit(ast.getRight()).getValue() instanceof BigDecimal) {
                    return Environment.create(requireType(BigDecimal.class, left).subtract(requireType(BigDecimal.class, visit(ast.getRight()))));
                }
                throw new RuntimeException();

            case "*":
                if(left.getValue() instanceof BigInteger && visit(ast.getRight()).getValue() instanceof BigInteger) {
                    return Environment.create(requireType(BigInteger.class, left).multiply(requireType(BigInteger.class, visit(ast.getRight()))));
                }
                if(left.getValue() instanceof BigDecimal && visit(ast.getRight()).getValue() instanceof BigDecimal) {
                    return Environment.create(requireType(BigDecimal.class, left).multiply(requireType(BigDecimal.class, visit(ast.getRight()))));
                }
                throw new RuntimeException();

            case "/":
                if(left.getValue() instanceof BigInteger && visit(ast.getRight()).getValue() instanceof BigInteger) {
                    if(((BigInteger) visit(ast.getRight()).getValue()).intValue() == 0) {
                        throw new RuntimeException();
                    }
                    return Environment.create(requireType(BigInteger.class, left).divide(requireType(BigInteger.class, visit(ast.getRight()))));
                }
                if(left.getValue() instanceof BigDecimal && visit(ast.getRight()).getValue() instanceof BigDecimal) {
                    if(((BigDecimal) visit(ast.getRight()).getValue()).doubleValue() == 0) {
                        throw new RuntimeException();
                    }
                    return Environment.create(requireType(BigDecimal.class, left).divide(requireType(BigDecimal.class, visit(ast.getRight())), RoundingMode.HALF_EVEN));
                }
                throw new RuntimeException();
        }
        return Environment.NIL;
    }

    @Override
    public Environment.PlcObject visit(Ast.Expression.Access ast) {
        return (ast.getReceiver().isPresent()) ? visit(ast.getReceiver().get()).getField(ast.getName()).getValue() : scope.lookupVariable(ast.getName()).getValue();
    }

    @Override
    public Environment.PlcObject visit(Ast.Expression.Function ast) {
        List<Environment.PlcObject> arguments = new ArrayList<>();
        for (Ast.Expression argument : ast.getArguments()) {
            arguments.add(visit(argument));
        }
        // Function
        if (!ast.getReceiver().isPresent()) {
            Environment.Function function = scope.lookupFunction(ast.getName(), ast.getArguments().size());
            return function.invoke(arguments);
        }
        // Method
        else {
            Environment.PlcObject obj = visit(ast.getReceiver().get());
            return obj.callMethod(ast.getName(), arguments);
        }
    }

    /**
     * Helper function to ensure an object is of the appropriate type.
     */
    private static <T> T requireType(Class<T> type, Environment.PlcObject object) {
        if (type.isInstance(object.getValue())) {
            return type.cast(object.getValue());
        } else {
            throw new RuntimeException("Expected type " + type.getName() + ", received " + object.getValue().getClass().getName() + ".");
        }
    }

    /**
     * Exception class for returning values.
     */
    private static class Return extends RuntimeException {

        private final Environment.PlcObject value;

        private Return(Environment.PlcObject value) {
            this.value = value;
        }

    }

}
