.global _start
_start:
	
	MOV R0,#1
	MOV R1,#2
	MOV R2,#0
	
	CMP R0,R1 // R0-R1, cpsr set based on result. N get set if negative, C if positive, Z and C if 0
	
	BGT greater // branch greater than
	// BNE branch not equal
	// BEQ branch equal
	// BGE branch greater equal
	// BLT branch less than
	// BLE branch less equal
	
	BAL end // branch always



	
greater:
	MOV R2,#5
	
end:
	MOV R7,#1
	SWI 0