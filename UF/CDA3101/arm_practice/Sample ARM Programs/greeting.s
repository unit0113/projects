# the data section, for declaring constants and variables
.section .data

# format specifiers for the printf() calls
welcome:    .asciz "Welcome! Please enter your name: "
greeting:   .asciz "Hello, %s! Hope you're having a fantastic day!\n"

# format specifier for the scanf() call
input:      .asciz "%s"

# variable to hold the user's name as read by scanf()
name:       .space 8

# the text section, for program content
.section .text

# make the main symbol global; that is, visible to whatever is calling the program
.global main

# main symbol
main:
    # print the welcome message #

    # to call a C function, we load the registers with whatever arguments we wish to pass
    # to print a string, we call printf() with the first argument containing the string we want to print

    # load the x0 register (first argument) with the string we want to print
    ldr x0, =welcome
    # call printf() by branching to the printf() symbol
    bl printf
    # since we branch-with-link, the printf() call will return here after execution is finished


    # receive the user's input #

    # scanf() requires two parameters for this use-case:
    # 1. a format specifier indicating what we expect
    # 2. a pointer to the memory location we want to store the result in

    # load the x0 register (first argument) with the format specifier, indicating we are expecting a string
    ldr x0, =input
    # load the x1 register (second argument) with the variable we want to store the result (a pointer to the data) in
    ldr x1, =name
    # call scanf() by branching to the scanf() symbol
    bl scanf
    # again, since we branch-with-link, program executin will resume here once scanf() is done


    # print the greeting #

    # here we need to call printf() again, this time providing two arguments:
    # 1. a format specifier indicating what we are passing
    # 2. the variable we want to print

    # load the format specifier for the greeting
    ldr x0, =greeting
    # load the variable which will be read by printf()
    ldr x1, =name
    # call printf()
    bl printf


    # end the program #

    # while not strictly necessary, there is a standard way to gracefully exit the program
    # as we would in a C++ program, it is best to separate this logic into its own function (or in this case, symbol)

    # branch to the exit symbol to gracefully close
    b exit

# here we define the exit symbol, which starts a series of commands which gracefully exit the program
exit:
    # load registers with data indicating successful execution
    mov x0, 0
    mov x8, 93
    # supervisor call indicating program completion
    svc 0
    # return execution to whatever called the program
    ret
