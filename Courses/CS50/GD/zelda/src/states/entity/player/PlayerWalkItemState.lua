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

PlayerWalkItemState = Class{__includes = EntityWalkState}

function PlayerWalkItemState:init(entity,dungeon)
	self.entity = entity
	self.dungeon = dungeon
	self.entity:changeAnimation('walk-item-'.. self.entity.direction)
end

function PlayerWalkItemState:update(dt)

	if love.keyboard.isDown('left') then
		self.entity.direction = 'left'
		self.entity:changeAnimation('walk-item-left')
	elseif love.keyboard.isDown('right') then
		self.entity.direction = 'right'
		self.entity:changeAnimation('walk-item-right')
	elseif love.keyboard.isDown('up') then
		self.entity.direction = 'up'
		self.entity:changeAnimation('walk-item-up')
	elseif love.keyboard.isDown('down') then
		self.entity.direction = 'down'
		self.entity:changeAnimation('walk-item-down')
	else
		self.entity:changeState('idle-item')
	end

	if love.keyboard.wasPressed('enter') or love.keyboard.wasPressed('return') then
		self.entity.projectile:throw()
		self.entity:changeState('idle')
	end

	-- perform base collision detection against walls
	EntityWalkState.update(self, dt)
end
