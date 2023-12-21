---
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

PlayerLiftState = Class{__includes = BaseState}

function PlayerLiftState:init(player,dungeon)
	self.player = player
	self.dungeon = dungeon

	self.player.offsetY = 5
	self.player.offsetX = 0
	local width = 0
	local height = 0
	local boxX = 0
	local boxY = 0
	local hitBox = {}
	if self.player.direction == 'left' then
		width = 8
		height = 16
		boxX = player.x - 8
		boxY = player.y + 2
	elseif self.player.direction == 'right' then
		width = 8
		height = 16
		boxX = player.x + player.width
		boxY = player.y + 2
	elseif self.player.direction == 'up' then
		width = 16
		height = 16
		boxX = player.x
		boxY = player.y - 4
	else
		width = 16
		height = 8
		boxX = player.x
		boxY = player.y + player.height+1
	end
	self.pickupBox = {
		['width'] = width,
		['height'] = height,
		['x'] = boxX,
		['y'] = boxY
	}

	self.player:changeAnimation('lift-'..self.player.direction)
end
function PlayerLiftState:enter(params)
	self.player.currentAnimation:refresh()
end
function PlayerLiftState:update(dt)
	local type = ''
	local broke = false
	local obj = {}
	if self.player.currentAnimation.timesPlayed > 0  then
		self.player.currentAnimation.timesPlayed = 0
		for i,object in pairs(self.dungeon.currentRoom.objects) do
			if object:collides(self.pickupBox) then
				if object.canPickup == true then
					type = object.type
					table.remove(self.dungeon.currentRoom.objects,i)
					obj = Projectile(
							--GAME_OBJECT_DEFS[type],
							object,
							self.player.x,
							self.player.y - ((self.player.height/2)+3),
							self.player
					)
					table.insert(self.dungeon.currentRoom.projectiles,obj)
					self.player:changeState('idle-item')
					self.player.projectile = self.dungeon.currentRoom.projectiles[#self.dungeon.currentRoom.projectiles]
					broke = true
					break;
				end
			end
		end
		if broke == false then
			self.player:changeState('idle')
		end
	end

end

function PlayerLiftState:render()
	love.graphics.draw(gTextures[self.player.currentAnimation.texture],
			gFrames[self.player.currentAnimation.texture]
			[
			self.player.currentAnimation:getCurrentFrame()
			],
			math.floor(self.player.x - self.player.offsetX),
			math.floor(self.player.y - self.player.offsetY))

	--love.setColor(0, 255, 255, 255)
	--love.graphics.rectangle('line', self.pickupBox.x, self.pickupBox.y, self.pickupBox.width, self.pickupBox.height)
	--love.setColor(255, 255, 255, 255)
end