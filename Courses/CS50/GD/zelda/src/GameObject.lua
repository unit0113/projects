--[[
	GD50
	Legend of Zelda

	Author: Colton Ogden
	cogden@cs50.harvard.edu
]]

GameObject = Class{}

function GameObject:init(def, x, y)
	-- string identifying this object type
	self.type = def.type
	self.texture = def.texture
	self.frame = def.frame or 1

	-- whether it acts as an obstacle or not
	self.solid = def.solid

	self.defaultState = def.defaultState
	self.state = self.defaultState
	self.states = def.states

	-- dimensions
	self.x = x
	self.y = y
	self.width = def.width
	self.height = def.height

	self.collided = false
	-- whether they can pickup this item and throw it.
	self.canPickup = def.canPickup == nil and false or def.canPickup
	-- default empty collision callback
	self.onCollide = def.onCollide == nil and function() end or def.onCollide
	-- for hearts and the like.
	self.onConsume = function() end
end

function GameObject:update(dt)

end

function GameObject:collides(target)
	local selfY, selfHeight = self.y + self.height / 2, self.height - self.height / 2

	return not (self.x + self.width < target.x or self.x > target.x + target.width or
			selfY + selfHeight < target.y or selfY > target.y + target.height)
end

function GameObject:render(adjacentOffsetX, adjacentOffsetY)
	love.graphics.draw(gTextures[self.texture],
			gFrames[self.texture][ self.states ~= nil and self.states[self.state].frame
					or self.frame],
		self.x + adjacentOffsetX, self.y + adjacentOffsetY)
	-- debug for player and hurtbox collision rects
	--love.setColor(255, 0, 255, 255)
	-- love.graphics.rectangle('line', self.x, self.y, self.width, self.height)
	-- love.setColor(255, 255, 255, 255)
end