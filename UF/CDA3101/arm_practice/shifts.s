.global _start
_start:
	
	MOV R0,#10
	LSL R0,#1 // shift left 1 (multiply by 2)
	LSR R0,#1 // shift right 1 (divide by 2)
	
	MOV R1,R0,LSL #1 // Store R0 in R1 and immediately multiply by 2
	
	
	MOV R7,#1
	SWI 0