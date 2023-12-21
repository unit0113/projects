--[[
	GD50
	Legend of Zelda

	Author: Colton Ogden
	cogden@cs50.harvard.edu
]]

EntityIdleState = Class{__includes = BaseState}

function EntityIdleState:init(entity)
	self.entity = entity == nil and nil or entity
	if self.entity ~= nil then
		self.entity:changeAnimation('idle-' .. self.entity.direction)
	end

	-- used for AI waiting
	self.waitDuration = 0
	self.waitTimer = 0
end

function EntityIdleState:enter(params)
	self.entity = params
	self.entity:changeAnimation('idle-' .. self.entity.direction)
end

function EntityIdleState:processAI(params, dt)
	if self.waitDuration == 0 then
		self.waitDuration = math.random(5)
	else
		self.waitTimer = self.waitTimer + dt

		if self.waitTimer > self.waitDuration then
		  self.entity:changeState('walk')
		end
	end
end

function EntityIdleState:render()
	local anim = self.entity.currentAnimation
	love.graphics.draw(gTextures[anim.texture],
			gFrames[anim.texture][anim:getCurrentFrame()],
		math.floor(self.entity.x - self.entity.offsetX), math.floor(self.entity.y - self.entity.offsetY))

	-- love.setColor(255, 0, 255, 255)
	-- love.graphics.rectangle('line', self.entity.x, self.entity.y, self.entity.width, self.entity.height)
	-- love.setColor(255, 255, 255, 255)
end