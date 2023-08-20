.global _start

.equ endlist, 0xaaaaaaaa	// constant to indicate end of list (default value in memory)

_start:
	
	LDR R0,=list // load address of list in R0
	LDR R3,=endlist
	LDR R1,[R0]
	MOV R2,R1	// Initialize sum to first element
	
loop:
	LDR R1,[R0,#4]!
	CMP R1,R3	// compare retrieved value to end of list value
	BEQ exit	// exit if end of list
	ADD R2,R2,R1	// else add value to running sum
	BAL loop	// Return to start of loop
	
exit:
	MOV R7,#1
	SWI 0
	
.data
list:
	.word 1,2,3,4,5,6,7,8,9,10