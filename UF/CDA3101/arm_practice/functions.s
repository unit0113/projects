.global _start
_start:
	
	MOV R0,#1
	MOV R1,#3
	BL add2	//branch late, call function add2, execution continues on following line
	MOV R3,#4
	
add2:
	ADD R2,R0,R1
	bx lr // Return to link register (lr) stored by BL