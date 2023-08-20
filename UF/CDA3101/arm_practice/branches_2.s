.global _start
_start:
	
	MOV R0,#2
	MOV R1,#4
	CMP R0,R1
	
	ADDLT R2,R0,R1 // Conditional add, only trigger if previous cmp was less than
	MOVGE R3,#8