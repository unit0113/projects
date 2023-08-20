.section .data

single_output   :       .asciz  "%d\n"
pair_output     :       .asciz  "%d\n"

.section .text

.global main

# int doubleIt(int x)
double_it:
    # multiply
    mov x9, 2
    mul x0, x0, x9

    # create stack frame for two variables
    sub sp, sp, 16
    # store return address (x30) and argument x (x0)
    str x30, [sp, 0]
    str x0, [sp, 8]

    # set up arguments and call printf
    mov x1, x0
    ldr x0, =single_output
    bl printf

    # restore variables from stack frame
    ldr x30, [sp, 0]
    ldr x0, [sp, 8]
    # deallocate stack frame
    add sp, sp, 16

    # return x
    mov x1, x0
    ret

# int addThem(int x, int y)
add_them:
    # add and store in variable val (x9)
    add x9, x0, x1

    # create stack frame for four variables
    sub sp, sp, 32
    # store return address (x30), argument x (x0), argument y (x1), and local variable val (x9)
    str x30, [sp, 0]
    str x0, [sp, 8]
    str x1, [sp, 16]
    str x9, [sp, 24]

    # set up arguments and call printf
    mov x2, x1
    mov x1, x0
    ldr x0, =pair_output
    bl printf

    # restore variables from stack frame
    ldr x30, [sp, 0]
    ldr x0, [sp, 8]
    ldr x1, [sp, 16]
    ldr x9, [sp, 24]
    # deallocate stack frame
    add sp, sp, 32

    # return x
    mov x2, x9
    ret

# int halfIt(int x)
half_it:
    # halve
    mov x9, 2
    sdiv x0, x0, x9

    # create stack frame for two variables
    sub sp, sp, 16
    # store return address (x30) and argument x (x0)
    str x30, [sp, 0]
    str x0, [sp, 8]

    # set up arguments and call printf
    mov x1, x0
    ldr x0, =single_output
    bl printf

    # restore variables from stack frame
    ldr x30, [sp, 0]
    ldr x0, [sp, 8]
    # deallocate stack frame
    add sp, sp, 16

    # return x
    mov x1, x0
    ret

# int main()
main:
    # create x and y
    mov x9, 2
    mov x10, 4

    # create stack frame and call doubleIt(x)
    sub sp, sp, 24
    str x30, [sp, 0]
    str x9, [sp, 8]
    str x10, [sp, 16]
    mov x0, x9
    bl double_it
    # store result in first
    mov x11, x1
    # restore variables from stack frame and deallocate
    ldr x30, [sp, 0]
    ldr x9, [sp, 8]
    ldr x10, [sp, 16]
    add sp, sp, 24

    # create stack frame and call addThem(x, y)
    sub sp, sp, 32
    str x30, [sp, 0]
    str x9, [sp, 8]
    str x10, [sp, 16]
    str x11, [sp, 24]
    mov x0, x9
    mov x1, x10
    bl add_them
    # store result in second
    mov x12, x2
    # restore variables from stack frame and deallocate
    ldr x30, [sp, 0]
    ldr x9, [sp, 8]
    ldr x10, [sp, 16]
    ldr x11, [sp, 24]
    add sp, sp, 32

    # create stack frame and call halfIt(x)
    sub sp, sp, 48
    str x30, [sp, 0]
    str x9, [sp, 8]
    str x10, [sp, 16]
    str x11, [sp, 24]
    str x12, [sp, 32]
    mov x0, x9
    bl half_it
    # store result in third
    mov x13, x1
    # restore variables from stack frame and deallocate
    ldr x30, [sp, 0]
    ldr x9, [sp, 8]
    ldr x10, [sp, 16]
    ldr x11, [sp, 24]
    ldr x12, [sp, 32]
    add sp, sp, 48

    # return 0
    mov x0, 0
    ret
