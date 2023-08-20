.section .data

helloMessage:	.asciz "Hello, World!\n"

.section .text

.global main

main:
	ldr x0, =helloMessage
	bl printf

	b exit

exit:
	mov x0, 0
	mov x8, 93
	svc 0
	ret
