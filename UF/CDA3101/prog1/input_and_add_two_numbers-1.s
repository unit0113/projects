.section .data

input_x_prompt	:	.asciz	"Please enter x: "
input_y_prompt	:	.asciz	"Please enter y: "
input_spec	:	.asciz	"%d"
result		:	.asciz	"x + y = %d\n"

.section .text

.global main

#main
main:

#create room on stack for x
sub sp, sp, 8
# input x prompt
ldr x0, = input_x_prompt
bl printf	
#get input x value
# spec input
ldr x0, = input_spec
mov x1, sp
bl scanf
ldr x19, [sp]


#get y input value
# enter y output
ldr x0, = input_y_prompt
bl printf
# spec input
ldr x0, = input_spec
mov x1, sp
bl scanf
ldr x20, [sp]

add x21, x19, x20
mov x1, x21
ldr x0, =result
bl printf

# branch to this label on program completion
exit:
	mov x0, 0
	mov x8, 93
	svc 0
	ret

