.global _start
_start:

	MOV R0,#5
	MOV R1,#7
	
	ADD R2,R0,R1 // R2 = R0 + R1
	MUL R3,R0,R1 // R3 = R0 * R1
	SUB R4,R0,R1 // R4 = R0 - R1
	SUBS R5,R0,R1 // subtract, but also set cpsr flag if negative (safer), use if not guaranteed to be positive
	
	MOV R10,#0xFFFFFFFF
	ADDS R11,R10,R1 // Sets carry flag due to overflow (overwrites negative flag)
	ADC R12,R10,R1 // R12 = R10 + R1 + carry. Adds carry to operation (Just adds 1 to value)
	
	LDR R0,=list // load address of list into R0
	LDR R1,[R0] // load item at address that R0 points to (first item in list)
	LDR R2,[R0,#4] // load item 4 bytes after address in R0
	LDR R3,[R0,#4]! // preincrements addresss of R0 by 4 bytes and loads item
	LDR R4,[R0],#4 // Loads item at the address pointed to by R0 and postincrements address of R0
	
	
	MOV R7,#1
	SWI 0
	
.data
list:
	.word 4,5,6,-9,1,0