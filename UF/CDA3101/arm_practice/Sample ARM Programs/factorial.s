.data
    readformat: .asciz "%d"

.text
    inputrequest: .string "Enter an integer: \n"
    outformat: .asciz "factorial = %d    \n"
    error: .string "The input cannot be negative\n"


.balign 4
.global main

main:
STP X29, X30, [SP, -16]!
MOV X29, SP

// Print input prompt
getinput:
LDR X0, =inputrequest
BL printf

// Get input, store on stack
SUB SP, SP, #16
LDR X0, =readformat
MOV X1, SP
BL scanf

//save the value from the stack
LDR X9, [SP]
//restore the stack
ADD SP, SP, #16

// Check if input is not negative
CMP X9, 0
B.GE run
LDR X0, =error
BL printf
B getinput

// Run main recursive function
run:
MOV X0, X9
BL fact

// Print results
LDR X0, =outformat
BL printf
B exit

fact:
SUB SP, SP, #16 // adjust stack for 2 items
MOV X9, X30
STUR X9, [SP,#8] // save the return address
STUR X0, [SP,#0] // save the argument

CMP X0, #1       // test for n < 2
B.GE  L1               // if n >= 1, go to L1

MOV  X1, #1
ADD  SP,SP,#16        // pop 2 items off stack
BR   X30               // return to caller

//recursive case
L1:
SUB X0,X0,#1      // n >= 1: argument gets (n − 1)
BL fact            // call fact with (n − 1)

//restore stack after return from base case
LDUR X0, [SP, #0]       // return from BL: restore argument n
LDUR X9, [SP, #8]       // restore the return address
MOV X30, X9
ADD SP, SP, #16       // adjust stack pointer to pop 2 items

//x0 has n, x1 has result of recursive call
MUL X1,X0,X1           // return n * fact (n − 1)

BR X30  //now this recursive call is done






exit:
    MOV X0, #0
    LDP X29, X30, [SP], #16
    ret

