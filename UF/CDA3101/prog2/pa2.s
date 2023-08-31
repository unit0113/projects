.section .data

    input_prompt:    .asciz  "Please enter a string: \n"
    input_format:    .asciz "%[^\n]"
    output_format:   .asciz "%c"
    new_line_format: .ascii "\n"

.section .text


.global main

main:

    // Print call for input
    LDR X0, =input_prompt
    BL printf

    // Get input, store on stack
    SUB SP, SP, #16
    LDR X0, =input_format
    MOV X1, SP
    BL scanf

    // Save SP as first arg
    MOV X0, SP
    BL revstr
    LDR X0, =new_line_format
    BL printf                   // Print new line
    B exit

revstr:
    // Save return address and arg
    SUB SP, SP, #16
    STUR X30, [SP,#8]
    STUR X0, [SP,#0]

    // Base case
    LDRB W1, [X0]               // Get char
    CMP X1, #0                  // Check against null char
    B.NE recursive

    // Return
    LDR X0, =new_line_format
    BL printf                   // Print new line
    LDUR X30, [SP,#8]           // Restore the return address
    ADD SP, SP, #16             // Pop from stack
    BR X30                      // Return to caller

recursive:
    LDR X0, =output_format
    BL printf                   // Print char first time, X1 still loaded from base case check
    LDUR X0, [SP,#0]            // Restore argument n
    ADD X0, X0, #1              // Increment starting address
    
    BL revstr                   // Recursive call

    LDUR X9, [SP,#0]            // Restore argument n
    LDRB W1, [X9]               // Get char
    LDR X0, =output_format
    BL printf                   // Print char second time

    LDUR X30, [SP,#8]           // Restore the return address
    ADD SP, SP, #16             // Pop from stack
    BR X30                      // Return to caller


# Exit commands
exit:
    MOV x0, 0
    MOV x8, 93
    SVC 0
    RET
