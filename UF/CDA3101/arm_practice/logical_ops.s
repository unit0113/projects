.global _start
_start:

	MOV R0,#0xFF
	MOV R1,#22
	AND R2,R0,R1 // bitwise and stored in R2
	ORR R3,R0,R1 // bitwise or
	ANDS R4,R0,R1 // And but sets flags
	EOR R5,R0,R1 // XOR
	
	MVN R6,R1 // Moves the negation of R1 into R6, negates entire word (0xFF->0xFFFFFF00)
	AND R6,R6,#0xFF // restores leading zeroes to negated number
	
	
	
	MOV R7,#1
	SWI 0
