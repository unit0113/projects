.section .data

input_format:	.asciz	"Enter a string: "
input_spec:		.asciz	"%[^\n]"
output_spec:	.asciz	"First two characters: '%c', '%c'\n"
input:			.space	255

.section .text

.global main

main:
	ldr x0, =input_format
	bl printf

	ldr x0, =input_spec
	ldr x1, =input
	bl scanf

	ldr x0, =output_spec
	ldr x9, =input
	ldrb w1, [x9, 0]
	ldrb w2, [x9, 1]
	bl printf

	b exit

exit:
	mov x0, 0
	mov x8, 93
	svc 0
	ret
