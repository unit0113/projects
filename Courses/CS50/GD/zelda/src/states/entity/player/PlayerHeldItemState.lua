--[[
	GD50
	Legend of Zelda
	-- PlayerHeldWalkState --

	Author: Macarthur Inbody
	133794m3r@gmail.com

	This State is for when the player is holding something above their head. It is called HeldItem solely b/c it's shorter
	to type out but in reality it can be any object that the player can "hold" that doesn't go into their inventory.

	When they throw something they will do the "throw item" state. This will be our projectile system also.
]]

PlayerItemWalkState = Class{__includes = EntityWalkState}
