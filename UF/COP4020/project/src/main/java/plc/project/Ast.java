package plc.project;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * See each project assignment specification for specific notes on the AST classes
 * and how to use this hierarchy.
 */
public abstract class Ast {

    public static final class Source extends Ast {

        private final List<Field> fields;
        private final List<Method> methods;

        public Source(List<Field> fields, List<Method> methods) {
            this.fields = fields;
            this.methods = methods;
        }

        public List<Ast.Field> getFields() {
            return fields;
        }

        public List<Method> getMethods() {
            return methods;
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof Source &&
                    fields.equals(((Source) obj).fields) &&
                    methods.equals(((Source) obj).methods);
        }

        @Override
        public String toString() {
            return "Ast.Source{" +
                    "fields=" + fields +
                    ", methods=" + methods +
                    '}';
        }

    }

    public static final class Field extends Ast {

        private final String name;
        private final String typeName;
        private final boolean constant;
        private final Optional<Ast.Expression> value;
        private Environment.Variable variable = null;

        public Field(String name, boolean constant, Optional<Expression> value) {
            this(name, "Any", constant, value);
		}

        public Field(String name, String typeName, boolean constant, Optional<Ast.Expression> value) {
            this.name = name;
            this.typeName = typeName;
            this.constant = constant;
            this.value = value;
        }


        public String getName() {
            return name;
        }

        public String getTypeName() {
            return typeName;
        }

        public boolean getConstant() {
            return constant;
        }

        public Optional<Ast.Expression> getValue() {
            return value;
        }

        public Environment.Variable getVariable() {
            if (variable == null) {
                throw new IllegalStateException("variable is uninitialized");
            }
            return variable;
        }

        public void setVariable(Environment.Variable variable) {
            this.variable = variable;
        }


        @Override
        public boolean equals(Object obj) {
            return obj instanceof Field &&
                    name.equals(((Field) obj).name) &&
                    typeName.equals(((Field) obj).typeName) &&
                    constant == ((Field) obj).constant &&
                    value.equals(((Field) obj).value) &&
                    Objects.equals(variable, ((Field) obj).variable);
        }

        @Override
        public String toString() {
            return "Ast.Field{" +
                    "name='" + name + '\'' +
                    ", typeName=" + typeName +
                    ", constant=" + constant +
                    ", value=" + value +
                    ", variable=" + variable +
                    '}';
        }

    }

    public static final class Method extends Ast {

        private final String name;
        private final List<String> parameters;
        private final List<String> parameterTypeNames;
        private final Optional<String> returnTypeName;
        private final List<Statement> statements;
        private Environment.Function function = null;
        
        public Method(String name, List<String> parameters, List<Statement> statements) {
            this(name, parameters, new ArrayList<>(), Optional.of("Any"), statements);
            for (int i = 0; i < parameters.size(); i++) {
                parameterTypeNames.add("Any");
            }
        }

        public Method(String name, List<String> parameters, List<String> parameterTypeNames, Optional<String> returnTypeName, List<Statement> statements) {

            this.name = name;
            this.parameters = parameters;
            this.parameterTypeNames = parameterTypeNames;
            this.returnTypeName = returnTypeName;
            this.statements = statements;
        }

        public String getName() {
            return name;
        }

        public List<String> getParameters() {
            return parameters;
        }

        public List<String> getParameterTypeNames() {
            return parameterTypeNames;
        }

        public Optional<String> getReturnTypeName() {
            return returnTypeName;
        }

        public List<Statement> getStatements() {
            return statements;
        }

        public Environment.Function getFunction() {
            if (function == null) {
                throw new IllegalStateException("function is uninitialized");
            }
            return function;
        }

        public void setFunction(Environment.Function function) {
            this.function = function;
        }


        @Override
        public boolean equals(Object obj) {
            return obj instanceof Ast.Method &&
                    name.equals(((Ast.Method) obj).name) &&
                    parameters.equals(((Ast.Method) obj).parameters) &&
                    parameterTypeNames.equals(((Ast.Method) obj).parameterTypeNames) &&
                    returnTypeName.equals(((Ast.Method) obj).returnTypeName) &&
                    statements.equals(((Ast.Method) obj).statements) &&
                    Objects.equals(function, ((Ast.Method) obj).function);
        }


        @Override
        public String toString() {
            return "Method{" +
                    "name='" + name + '\'' +
                    ", parameters=" + parameters +
                    ", parameterTypeNames=" + parameterTypeNames +
                    ", returnTypeName='" + returnTypeName + '\'' +
                    ", statements=" + statements +
                    ", function=" + function +
                    '}';
        }

    }

    public static abstract class Statement extends Ast {

        public static final class Expression extends Statement {

            private final Ast.Expression expression;

            public Expression(Ast.Expression expression) {
                this.expression = expression;
            }

            public Ast.Expression getExpression() {
                return expression;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Ast.Statement.Expression &&
                        expression.equals(((Ast.Statement.Expression) obj).expression);
            }

            @Override
            public String toString() {
                return "Ast.Statement.Expression{" +
                        "expression=" + expression +
                        '}';
            }

        }

        public static final class Declaration extends Statement {

            private String name;
            private final Optional<String> typeName;
            private Optional<Ast.Expression> value;
            private Environment.Variable variable = null;

            public Declaration(String name, Optional<Ast.Expression> value) {
                this(name, Optional.empty(), value);
            }

            public Declaration(String name, Optional<String> typeName, Optional<Ast.Expression> value) {
                this.name = name;
                this.typeName = typeName;
                this.value = value;
            }

            public String getName() {
                return name;
            }

            public Optional<String> getTypeName() {
                return typeName;
            }

            public Optional<Ast.Expression> getValue() {
                return value;
            }

            public Environment.Variable getVariable() {
                if (variable == null) {
                    throw new IllegalStateException("variable is uninitialized");
                }
                return variable;
            }

            public void setVariable(Environment.Variable variable) {
                this.variable = variable;
            }
            
            
            @Override
            public boolean equals(Object obj) {
                return obj instanceof Declaration &&
                        name.equals(((Declaration) obj).name) &&
                        typeName.equals(((Declaration) obj).typeName) &&
                        value.equals(((Declaration) obj).value) &&
                        Objects.equals(variable, ((Declaration) obj).variable);
            }

            @Override
            public String toString() {
                return "Ast.Statement.Declaration{" +
                        "name='" + name + '\'' +
                        ", typeName=" + typeName +
                        ", value=" + value +
                        ", variable=" + variable +
                        '}';
            }

        }

        public static final class Assignment extends Statement {

            private final Ast.Expression receiver;
            private final Ast.Expression value;

            public Assignment(Ast.Expression receiver, Ast.Expression value) {
                this.receiver = receiver;
                this.value = value;
            }

            public Ast.Expression getReceiver() {
                return receiver;
            }

            public Ast.Expression getValue() {
                return value;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Assignment &&
                        receiver.equals(((Assignment) obj).receiver) &&
                        value.equals(((Assignment) obj).value);
            }

            @Override
            public final String toString() {
                return "Ast.Statement.Assignment{" +
                        "receiver=" + receiver +
                        ", value=" + value +
                        '}';
            }

        }

        public static final class If extends Statement {

            private final Ast.Expression condition;
            private final List<Statement> thenStatements;
            private final List<Statement> elseStatements;


            public If(Ast.Expression condition, List<Statement> thenStatements, List<Statement> elseStatements) {
                this.condition = condition;
                this.thenStatements = thenStatements;
                this.elseStatements = elseStatements;
            }

            public Ast.Expression getCondition() {
                return condition;
            }

            public List<Statement> getThenStatements() {
                return thenStatements;
            }

            public List<Statement> getElseStatements() {
                return elseStatements;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof If &&
                        condition.equals(((If) obj).condition) &&
                        thenStatements.equals(((If) obj).thenStatements) &&
                        elseStatements.equals(((If) obj).elseStatements);
            }

            @Override
            public String toString() {
                return "Ast.Statement.If{" +
                        "condition=" + condition +
                        ", thenStatements=" + thenStatements +
                        ", elseStatements=" + elseStatements +
                        '}';
            }

        }

        public static final class For extends Statement {

            private final Statement initialization;
            private final Ast.Expression condition;
            private final Statement increment;
            private final List<Statement> statements;

            public For(Statement initialization, Ast.Expression condition, Statement increment, List<Statement> statements) {
                this.initialization = initialization;
                this.condition = condition;
                this.increment = increment;
                this.statements = statements;
            }

            public Ast.Statement getInitialization() {
                return initialization;
            }

            public Ast.Expression getCondition() {
                return condition;
            }

            public Ast.Statement getIncrement() {
                return increment;
            }

            public List<Statement> getStatements() {
                return statements;
            }

            @Override
            public boolean equals(Object obj) {

                boolean init, incr;
                For myFor = null;

                if (obj instanceof For) {
                    myFor = (For) obj;
                } else {
                    return false;
                }

                if (initialization == null || myFor.initialization == null) {
                    init = initialization == myFor.initialization;
                } else {
                    init = initialization.equals(myFor.initialization);
                }

                if (increment == null || myFor.increment == null) {
                    incr = increment == myFor.increment;
                } else {
                    incr = increment.equals(myFor.increment);
                }

                return  init &&
                        condition.equals(myFor.condition) &&
                        incr &&
                        statements.equals(myFor.statements);
            }

            @Override
            public String toString() {
                return "For{" +
                        "initialization=" + initialization +
                        ", condition=" + condition +
                        ", increment=" + increment +
                        ", statements=" + statements +
                        '}';
            }

        }

        public static final class While extends Statement {

            private final Ast.Expression condition;
            private final List<Statement> statements;

            public While(Ast.Expression condition, List<Statement> statements) {
                this.condition = condition;
                this.statements = statements;
            }

            public Ast.Expression getCondition() {
                return condition;
            }

            public List<Statement> getStatements() {
                return statements;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof While &&
                        condition.equals(((While) obj).condition) &&
                        statements.equals(((While) obj).statements);
            }

            @Override
            public String toString() {
                return "Ast.Statement.While{" +
                        "condition=" + condition +
                        ", statements=" + statements +
                        '}';
            }

        }

        public static final class Return extends Statement {

            private final Ast.Expression value;

            public Return(Ast.Expression value) {
                this.value = value;
            }

            public Ast.Expression getValue() {
                return value;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Return &&
                        value.equals(((Return) obj).value);
            }

            @Override
            public String toString() {
                return "Ast.Statement.Return{" +
                        "value=" + value +
                        '}';
            }

        }

    }

    public static abstract class Expression extends Ast {

        public abstract Environment.Type getType();

        public static final class Literal extends Ast.Expression {

            private final Object literal;
            private Environment.Type type = null;
            
            public Literal(Object literal) {
                this.literal = literal;
            }

            public Object getLiteral() {
                return literal;
            }

            @Override
            public Environment.Type getType() {
                if (type == null) {
                    throw new IllegalStateException("type is uninitialized");
                }
                return type;
            }

            public void setType(Environment.Type type) {
                this.type = type;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Literal &&
                        Objects.equals(literal, ((Literal) obj).literal) &&
                        Objects.equals(type, ((Literal) obj).type);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Literal{" +
                        "literal=" + literal +
                        ", type=" + type +
                        '}';
            }

        }

        public static final class Group extends Ast.Expression {

            private final Ast.Expression expression;
            private Environment.Type type = null;

            public Group(Ast.Expression expression) {
                this.expression = expression;
            }

            public Ast.Expression getExpression() {
                return expression;
            }

            @Override
            public Environment.Type getType() {
                if (type == null) {
                    throw new IllegalStateException("type is uninitialized");
                }
                return type;
            }

            public void setType(Environment.Type type) {
                this.type = type;
            }


            @Override
            public boolean equals(Object obj) {
                return obj instanceof Group &&
                        expression.equals(((Group) obj).expression) &&
                        Objects.equals(type, ((Group) obj).type);
            }


            @Override
            public String toString() {
                return "Ast.Expression.Group{" +
                        "expression=" + expression +
                        ", type=" + type +
                        '}';
            }

        }

        public static final class Binary extends Ast.Expression {

            private final String operator;
            private final Ast.Expression left;
            private final Ast.Expression right;
            private Environment.Type type = null;

            public Binary(String operator, Ast.Expression left, Ast.Expression right) {
                this.operator = operator;
                this.left = left;
                this.right = right;
            }

            public String getOperator() {
                return operator;
            }

            public Ast.Expression getLeft() {
                return left;
            }

            public Ast.Expression getRight() {
                return right;
            }

            @Override
            public Environment.Type getType() {
                if (type == null) {
                    throw new IllegalStateException("type is uninitialized");
                }
                return type;
            }

            public void setType(Environment.Type type) {
                this.type = type;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Binary &&
                        operator.equals(((Binary) obj).operator) &&
                        left.equals(((Binary) obj).left) &&
                        right.equals(((Binary) obj).right) &&
                        Objects.equals(type, ((Binary) obj).type);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Binary{" +
                        "operator='" + operator + '\'' +
                        ", left=" + left +
                        ", right=" + right +
                        ", type=" + type +
                        '}';
            }

        }


        public static final class Access extends Ast.Expression {

            private final Optional<Ast.Expression> receiver;
            private final String name;
            private Environment.Variable variable = null;

            public Access(Optional<Ast.Expression> receiver, String name) {
                this.receiver = receiver;
                this.name = name;
            }

            public Optional<Ast.Expression> getReceiver() {
                return receiver;
            }

            public String getName() {
                return name;
            }

            public Environment.Variable getVariable() {
                if (variable == null) {
                    throw new IllegalStateException("variable is uninitialized");
                }
                return variable;
            }

            public void setVariable(Environment.Variable variable) {
                this.variable = variable;
            }

            @Override
            public Environment.Type getType() {
                return getVariable().getType();
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Access &&
                        receiver.equals(((Access) obj).receiver) &&
                        name.equals(((Access) obj).name) &&
                        Objects.equals(variable, ((Access) obj).variable);
            }


            @Override
            public String toString() {
                return "Ast.Expression.Access{" +
                        "receiver=" + receiver +
                        ", name='" + name + '\'' +
                        ", variable=" + variable +
                        '}';
            }

        }

        public static final class Function extends Ast.Expression {

            private final Optional<Ast.Expression> receiver;
            private final String name;
            private final List<Ast.Expression> arguments;
            private Environment.Function function = null;

            public Function(Optional<Ast.Expression> receiver, String name, List<Ast.Expression> arguments) {
                this.receiver = receiver;
                this.name = name;
                this.arguments = arguments;
            }

            public Optional<Ast.Expression> getReceiver() {
                return receiver;
            }

            public String getName() {
                return name;
            }

            public List<Ast.Expression> getArguments() {
                return arguments;
            }

            public Environment.Function getFunction() {
                if (function == null) {
                    throw new IllegalStateException("function is uninitialized");
                }
                return function;
            }

            public void setFunction(Environment.Function function) {
                this.function = function;
            }

            @Override
            public Environment.Type getType() {
                return getFunction().getReturnType();
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Ast.Expression.Function &&
                        receiver.equals(((Ast.Expression.Function) obj).receiver) &&
                        name.equals(((Ast.Expression.Function) obj).name) &&
                        arguments.equals(((Ast.Expression.Function) obj).arguments) &&
                        Objects.equals(function, ((Ast.Expression.Function) obj).function);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Function{" +
                        "receiver=" + receiver +
                        "name='" + name + '\'' +
                        ", arguments=" + arguments +
                        ", function=" + function +
                        '}';
            }

        }

    }

    public interface Visitor<T> {

        default T visit(Ast ast) {
            if (ast instanceof Ast.Source) {
                return visit((Ast.Source) ast);
            } else if (ast instanceof Ast.Field) {
                return visit((Ast.Field) ast);
            } else if (ast instanceof Ast.Method) {
                return visit((Ast.Method) ast);
            } else if (ast instanceof Ast.Statement.Expression) {
                return visit((Ast.Statement.Expression) ast);
            } else if (ast instanceof Ast.Statement.Declaration) {
                return visit((Ast.Statement.Declaration) ast);
            } else if (ast instanceof Ast.Statement.Assignment) {
                return visit((Ast.Statement.Assignment) ast);
            } else if (ast instanceof Ast.Statement.If) {
                return visit((Ast.Statement.If) ast);
            } else if (ast instanceof Ast.Statement.For) {
                return visit((Ast.Statement.For) ast);
            } else if (ast instanceof Ast.Statement.While) {
                return visit((Ast.Statement.While) ast);
            } else if (ast instanceof Ast.Statement.Return) {
                return visit((Ast.Statement.Return) ast);
            } else if (ast instanceof Ast.Expression.Literal) {
                return visit((Ast.Expression.Literal) ast);
            } else if (ast instanceof Ast.Expression.Group) {
                return visit((Ast.Expression.Group) ast);
            } else if (ast instanceof Ast.Expression.Binary) {
                return visit((Ast.Expression.Binary) ast);
            } else if (ast instanceof Ast.Expression.Access) {
                return visit((Ast.Expression.Access) ast);
            } else if (ast instanceof Ast.Expression.Function) {
                return visit((Ast.Expression.Function) ast);
            } else {
                throw new AssertionError("Unimplemented AST type: " + ast.getClass().getName() + ".");
            }
        }

        T visit(Ast.Source ast);

        T visit(Ast.Field ast);

        T visit(Ast.Method ast);

        T visit(Ast.Statement.Expression ast);

        T visit(Ast.Statement.Declaration ast);

        T visit(Ast.Statement.Assignment ast);

        T visit(Ast.Statement.If ast);

        T visit(Ast.Statement.For ast);

        T visit(Ast.Statement.While ast);

        T visit(Ast.Statement.Return ast);

        T visit(Ast.Expression.Literal ast);

        T visit(Ast.Expression.Group ast);

        T visit(Ast.Expression.Binary ast);

        T visit(Ast.Expression.Access ast);

        T visit(Ast.Expression.Function ast);
    }

}
