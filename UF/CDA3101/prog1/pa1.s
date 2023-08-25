.section .data

input_x_prompt  :   .asciz  "Please enter x: "
input_y_prompt  :   .asciz  "Please enter y: "
input_spec  :   .asciz  "%d"
result      :   .asciz  "x*y = %d\n"

.section .text

.global main

main:

# Input examples from sample code
# Create room on stack for x (and eventually y, stored at same address)
sub sp, sp, 8
# Print prompt for x, X0 arg is text string
ldr x0, = input_x_prompt
bl printf	
#get input x value
# Get input for x, X0 arg is type, X1 is stack address return will be stored in
ldr x0, = input_spec
mov x1, sp
bl scanf
# Retrieve X from stack
ldr x19, [sp]


# Print prompt for y, X0 arg is text string
ldr x0, = input_y_prompt
bl printf
# Get input for y, X0 arg is type, X1 is stack address return will be stored in
ldr x0, = input_spec
mov x1, sp
bl scanf
# Retrieve y from stack
ldr x20, [sp]


# X stored in X19
# Y stored in X20


# Check if Y is negative
CMP X20, XZR
B.GE start_loop
# Negate X and Y if Y is negative
SUB X19, XZR, X19
SUB X20, XZR, X20


start_loop:
# Initialize i to 0
MOV X10, XZR
# Initilize sum
MOV X11, XZR

loop:
    # Compare i and y
    CMP X10, X20
    # Exit loop if equal
    B.EQ print
    # Else, add X to itself, increment i
    ADD X11, X11, X19
    ADD X10, X10, 1
    B loop


# Print results
print:
    LDR X0, =result
    MOV X1, x11
    BL printf	
    B exit


# branch to this label on program completion
exit:
    mov x0, 0
    mov x8, 93
    svc 0
    ret

