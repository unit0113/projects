--[[
	GD50
	Legend of Zelda
	-- PlayerHeldIdleState --

	Author: Macarthur Inbody
	133794m3r@gmail.com

	This State is for when the player is holding something above their head. It is called HeldItem solely b/c it's shorter
	to type out but in reality it can be any object that the player can "hold" that doesn't go into their inventory.

	When they throw something they will do the "throw item" state. This will be our projectile system also.
]]

PlayerIdleItemState = Class{__includes = PlayerIdleState}

function PlayerIdleItemState:init(entity,dungeon)
	self.entity = entity
	self.dungeon = dungeon
	self.entity:changeAnimation('idle-item-'.. self.entity.direction)
end

function PlayerIdleItemState:update(dt)
	if love.keyboard.isDown('left') or love.keyboard.isDown('right') or
			love.keyboard.isDown('up') or love.keyboard.isDown('down') then
		self.entity:changeState('walk-item')
	elseif love.keyboard.wasPressed('return') then
		self.entity.projectile:throw()
		self.entity:changeState('idle')
	end
end
function PlayerIdleItemState:render()
	--print(self.entity.currentAnimation.texture)
	love.graphics.draw(gTextures[self.entity.currentAnimation.texture],
			gFrames[self.entity.currentAnimation.texture]
			[
			self.entity.currentAnimation:getCurrentFrame()
			],
			math.floor(self.entity.x - self.entity.offsetX),
			math.floor(self.entity.y - self.entity.offsetY))

	--love.setColor(0, 255, 255, 255)
	--love.graphics.rectangle('line', self.pickupBox.x, self.pickupBox.y, self.pickupBox.width, self.pickupBox.height)
	--love.setColor(255, 255, 255, 255)
end
