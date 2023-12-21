--[[
	GD50
	Legend of Zelda

	Author: Colton Ogden
	cogden@cs50.harvard.edu
]]

Projectile = Class{__includes = GameObject}

function Projectile:init(def,x,y,player)
	GameObject.init(self,def)
	self.dx = 0
	self.dy = 0
	self.x = x
	self.y = y
	self.speed = def.speed == nil and 64 or def.speed
	self.player = player
	self.thrown = false
	self.areaTravelled = 0
	self.maxAreaTravel = TILE_SIZE * 4
end
function Projectile:throw(dy,dx)
	if self.player.direction == 'left' then
		self.dx = -self.speed
	elseif self.player.direction == 'right' then
		self.dx = self.speed
	elseif self.player.direction == 'up' then
		self.dy = -self.speed
	elseif self.player.direction == 'down' then
		self.dy = self.speed
	end
end
function Projectile:update(dt)
	if self.dx ~= 0 or self.dy ~= 0 then
		self.x = self.x + self.dx * dt
		self.y = self.y + self.dy * dt
		self.areaTravelled = self.areaTravelled + math.abs((self.dx * dt) + (self.dy * dt))
	else
		self.x = self.player.x
		self.y = self.player.y - (self.player.height/2 - 2)
	end
end

function Projectile:doneMoving()
	return self.areaTravelled >= self.maxAreaTravel
end

function Projectile:outOfBounds(room)
	--if self.dy > 0 or self.dx > 0 then
		-- returns true if the projectile has went out of bounds.
		--return (self.x >MAP_RENDER_OFFSET_X + TILE_SIZE) or (self.x + self.width < VIRTUAL_WIDTH - TILE_SIZE * 2) or (self.y > MAP_RENDER_OFFSET_Y + TILE_SIZE - self.height /2) or (self.y + self.height < (VIRTUAL_HEIGHT - (VIRTUAL_HEIGHT - MAP_HEIGHT * TILE_SIZE)))
		return (self.x < 1 + TILE_SIZE) or (self.x > VIRTUAL_WIDTH - (2 * TILE_SIZE)) or (self.y < 1+TILE_SIZE) or (self.y > VIRTUAL_HEIGHT - (TILE_SIZE*2))
	--end
end

function Projectile:render(offsetX,offsetY)
	love.graphics.draw(gTextures[self.texture],gFrames[self.texture][self.states[self.state].frame or self.frame],self.x + offsetX, self.y+offsetY)

	-- debug for the hitbox.
	--love.setColor(128, 128, 128, 255)
	--love.graphics.rectangle('line', self.x, self.y, self.width, self.height)
	--love.setColor(255,255,255,255)
end