
.data

readformat:     .asciz "%d"

.text
enter:           .string "enter an integer \n"
outformat:     .asciz "factorial = %d    \n"


.balign 4
.global main

main:           stp     x29,    x30,    [sp,    -16]!
mov     x29,    sp

//print prompt
ldr x0, =enter
bl printf

afterprint:
//get value
//have scanf place its result on the stack
//second argument to scanf will be an address on the stack (not a static address)
sub sp, sp, #16
ldr x0, =readformat
mov x1, sp
bl scanf
//save the value from the stack
afterscan:
ldr x9, [sp]
//restore the stack
add sp, sp, #16

//put the read value into an argument
mov x0, x9
bl fact

//print the result
ldr x0, =outformat
bl      printf

b done

fact:    //get ready for recursive call
SUB SP, SP, #16 // adjust stack for 2 items
MOV X9, X30
STUR X9, [SP,#8] // save the return address
STUR X0, [SP,#0] // save the argument

SUBS X9,X0, #1       // test for n < 2
B.GE  L1               // if n >= 1, go to L1

MOV  X1, #1
ADD  SP,SP,#16        // pop 2 items off stack
BR   X30               // return to caller

//recursive case
L1:   SUB X0,X0,#1      // n >= 1: argument gets (n − 1)
BL fact            // call fact with (n − 1)

//restore stack after return from base case
LDUR X0, [SP,#0]       // return from BL: restore argument n
LDUR X9, [SP,#8]       // restore the return address
MOV X30, X9
ADD SP, SP, #16       // adjust stack pointer to pop 2 items

//x0 has n, x1 has result of recursive call
MUL X1,X0,X1           // return n * fact (n − 1)

BR X30  //now this recursive call is done






done:           mov x0, 0

end:            ldp     x29,   x30,   [sp],   16
ret





