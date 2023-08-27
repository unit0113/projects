.section .data

input_prompt  :   .asciz  "Please enter a string: \n"


.section .text

.global main

main:


# branch to this label on program completion
exit:
    mov x0, 0
    mov x8, 93
    svc 0
    ret

