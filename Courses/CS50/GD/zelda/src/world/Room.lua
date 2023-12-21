--[[
	GD50
	Legend of Zelda

	Author: Colton Ogden
	cogden@cs50.harvard.edu
]]

Room = Class{}

function Room:init(player)
	self.width = MAP_WIDTH
	self.height = MAP_HEIGHT


	self.tiles = {}
	self:generateWallsAndFloors()

	-- entities in the room
	self.entities = {}
	self:generateEntities()

	-- game objects in the room
	self.objects = {}
	self:generateObjects()

	self.projectiles = {}
	--self:generateProjectiles()
	-- doorways that lead to other dungeon rooms
	self.doorways = {}
	table.insert(self.doorways, Doorway('top', false, self))
	table.insert(self.doorways, Doorway('bottom', false, self))
	table.insert(self.doorways, Doorway('left', false, self))
	table.insert(self.doorways, Doorway('right', false, self))

	-- reference to player for collisions, etc.
	self.player = player

	-- used for centering the dungeon rendering
	self.renderOffsetX = MAP_RENDER_OFFSET_X
	self.renderOffsetY = MAP_RENDER_OFFSET_Y

	-- used for drawing when this room is the next room, adjacent to the active
	self.adjacentOffsetX = 0
	self.adjacentOffsetY = 0
end

--[[
	Randomly creates an assortment of enemies for the player to fight.
]]
function Room:generateEntities()
	local types = {'skeleton', 'slime', 'bat', 'ghost', 'spider'}
	local x,y = 0,0

	for i = 1, 10 do
		local type = types[math.random(#types)]
		table.insert(self.entities, Entity {
		  animations = ENTITY_DEFS[type].animations,
		  walkSpeed = ENTITY_DEFS[type].walkSpeed or 20,

		  -- ensure X and Y are within bounds of the map
		  x = math.random(MAP_RENDER_OFFSET_X + TILE_SIZE,
			 VIRTUAL_WIDTH - (TILE_SIZE * 2) - 40),
		  y = math.random(MAP_RENDER_OFFSET_Y + TILE_SIZE,
			 VIRTUAL_HEIGHT - (VIRTUAL_HEIGHT - MAP_HEIGHT * TILE_SIZE) + MAP_RENDER_OFFSET_Y - TILE_SIZE - 40),

		  width = 16,
		  height = 16,

		  health = 1
		})
		--print('i',i,'x,y',self.entities[i].x,self.entities[i].y)

		self.entities[i].stateMachine = StateMachine {
		  ['walk'] = function() return EntityWalkState(self.entities[i]) end,
		  ['idle'] = function() return EntityIdleState(self.entities[i]) end
		}

		self.entities[i]:changeState('walk')
	end
end

--[[
	Randomly creates an assortment of obstacles for the player to navigate around.
]]
function Room:generateObjects()
	table.insert(self.objects, GameObject(
			GAME_OBJECT_DEFS['switch'],
			math.random(MAP_RENDER_OFFSET_X + TILE_SIZE,
					VIRTUAL_WIDTH - TILE_SIZE * 2 - 16),
			math.random(MAP_RENDER_OFFSET_Y + TILE_SIZE,
					VIRTUAL_HEIGHT - (VIRTUAL_HEIGHT - MAP_HEIGHT * TILE_SIZE) + MAP_RENDER_OFFSET_Y - TILE_SIZE - 16)
	))

	-- get a reference to the switch
	local switch = self.objects[1]

	-- define a function for the switch that will open all doors in the room
	switch.onCollide = function()
		if switch.state == 'unpressed' then
			switch.state = 'pressed'

			-- open every door in the room if we press the switch
			for k, doorway in pairs(self.doorways) do
				doorway.open = true
			end

			gSounds['door']:play()
		end
	end
	local pot = GAME_OBJECT_DEFS['pot']
	local potFrame = math.random(#pot.normal_frames)
	pot.frame = pot.normal_frames[potFrame]
	pot.states['normal'] = {frame = pot.frame}
	pot.states['broke'] = {frame = pot.broke_frames[potFrame]}
	pot.onCollide = function(self,player,dt) end
	local potFrame = math.random(#pot.normal_frames)
	local x = math.random(MAP_RENDER_OFFSET_X+TILE_SIZE,VIRTUAL_WIDTH - TILE_SIZE * 2 - 16)
	local y = math.random(MAP_RENDER_OFFSET_Y+TILE_SIZE,VIRTUAL_HEIGHT - (VIRTUAL_HEIGHT - MAP_HEIGHT * TILE_SIZE)+MAP_RENDER_OFFSET_Y - TILE_SIZE - 16)
	local minX = MAP_RENDER_OFFSET_X+TILE_SIZE,VIRTUAL_WIDTH - TILE_SIZE * 2 - 16
	local maxX = VIRTUAL_WIDTH - TILE_SIZE * 2 - 16
	local minY = MAP_RENDER_OFFSET_Y+TILE_SIZE
	local maxY = VIRTUAL_HEIGHT - (VIRTUAL_HEIGHT - MAP_HEIGHT * TILE_SIZE)+MAP_RENDER_OFFSET_Y - TILE_SIZE*2
	local X = 0
	local Y = 0
	local entityBlocked = true
	local broke = false
	local entityY = 0
	local entityHeight = 0
	local selfY, selfHeight = Y + 8, 4
	local potHitbox = {
		['height'] = 16,
		['width'] = 16,
		['x'] = 0,
		['y'] = 0
	}
	local num_pots = math.random(4)
	--local num_pots = 10
--	num_pots = 10
	for i=1,num_pots do
		X = 0
		Y = 0
		entityBlocked = true
		while entityBlocked do
			X = math.random(minX,maxX)
			Y = math.random(minY,maxY)
			broke = false
			entityBlocked = false
			potHitbox.x,potHitbox.y = X,Y
			for _,entity in pairs(self.entities) do
				entityY = entity.y + entity.height / 2
				entityHeight = entity.height - entity.height / 2
				if entity:collides(potHitbox) or
					((entity.x-2 < X and entity.x+entity.width+2 > X) and ( entity.y-2 < Y and entity.y + entity.height+2 > Y) )
				then
					entityBlocked = true
				else
					entityBlocked = false
				end
			end
			selfY, selfHeight = Y + 8, 4
			for __,object in pairs(self.objects) do
				if object:collides(potHitbox) or
					((object.x-2 < X and object.x+object.width+2 > X) and ( object.y-2 < Y and object.y + object.height+2 > Y) )
				then
					entityBlocked = true
				else
					entityBlocked = false
				end
			end
		end
	table.insert(self.objects,GameObject(pot,X,Y))
	end
end

--[[
	Generates the walls and floors of the room, randomizing the various varieties
	of said tiles for visual variety.
]]
function Room:generateWallsAndFloors()
	for y = 1, self.height do
		table.insert(self.tiles, {})

		for x = 1, self.width do
		  local id = TILE_EMPTY

		  if x == 1 and y == 1 then
			 id = TILE_TOP_LEFT_CORNER
		  elseif x == 1 and y == self.height then
			 id = TILE_BOTTOM_LEFT_CORNER
		  elseif x == self.width and y == 1 then
			 id = TILE_TOP_RIGHT_CORNER
		  elseif x == self.width and y == self.height then
			 id = TILE_BOTTOM_RIGHT_CORNER

		  -- random left-hand walls, right walls, top, bottom, and floors
		  elseif x == 1 then
			 id = TILE_LEFT_WALLS[math.random(#TILE_LEFT_WALLS)]
		  elseif x == self.width then
			 id = TILE_RIGHT_WALLS[math.random(#TILE_RIGHT_WALLS)]
		  elseif y == 1 then
			 id = TILE_TOP_WALLS[math.random(#TILE_TOP_WALLS)]
		  elseif y == self.height then
			 id = TILE_BOTTOM_WALLS[math.random(#TILE_BOTTOM_WALLS)]
		  else
			 id = TILE_FLOORS[math.random(#TILE_FLOORS)]
		  end

		  table.insert(self.tiles[y], {
			 id = id
		  })
		end
	end
end

function Room:update(dt)

	-- don't update anything if we are sliding to another room (we have offsets)
	if self.adjacentOffsetX ~= 0 or self.adjacentOffsetY ~= 0 then return end
	self.player:update(dt)
	--local remove_entities = {}
	for i = #self.entities, 1, -1 do
		local entity = self.entities[i]

		-- remove entity from the table if health is <= 0
		if entity.health <= 0 then
			if math.random(10) == 1 and not entity.dead then
				self:spawnHeart(entity)
			end
			entity.dead = true
		elseif not entity.dead then
			entity:processAI({room = self}, dt)
			entity:update(dt)
		end

		-- collision between the player and entities in the room
		if not entity.dead then
			if self.player:collides(entity) and not self.player.invulnerable then
				gSounds['hit-player']:play()
				self.player:damage(1)
				self.player:goInvulnerable(1.5)

				if self.player.health == 0 then
					gStateMachine:change('game-over')
				end
			else
				if entity.bumped == false then
					for _,object in pairs(self.objects) do
						if entity:collides(object) and object.solid then
							-- top side
							if (entity.y+entity.height == object.y) and (entity.x >= object.x and entity.x+entity.width <= object.x+object.width) then
								entity.y = entity.y - entity.walkSpeed * dt
								-- left side
							elseif (entity.y >= object.y and entity.y <= object.y+object.height) and (entity.x+entity.width >= object.x) then
								entity.x = entity.x - entity.walkSpeed * dt
								-- right side
							elseif	(entity.y >= object.y and entity.y <= object.y+object.height) and (entity.x >= object.x+object.width) then
								entity.x = entity.x + entity.walkSpeed * dt
								-- bottom side
							elseif (entity.y == object.y + object.height) and (entity.x >= object.x and entity.x+entity.width <= object.x+object.width) then
								entity.y = entity.y + entity.walkSpeed * dt
							end
							Event.dispatch('bumped',entity,dt)
							--entity.bumped = true
							break
						end
					end
				end
			end
		end
	end

	for k, object in pairs(self.objects) do
		object:update(dt)
		-- check for collision for players and objects.
		if self.player:collides(object) then
			if object.solid then
				Event.dispatch('bumped',self.player,dt)
				break
			end
			if object.onCollide(object,self.player,dt) then
				table.remove(self.objects,k)
			end
		end
	end
	for _,projectile in pairs(self.projectiles) do
		projectile:update(dt)
		if projectile.dy ~= 0  or projectile.dx ~= 0 then
			if projectile:doneMoving() then
				table.remove(self.projectiles,_)
			elseif projectile:outOfBounds(self) then
				table.remove(self.projectiles,_)
			else
				for __,entity in pairs(self.entities) do
					if projectile:collides(entity) then
						entity:damage(1)
						table.remove(self.projectiles,_)
						break;
					end
				end
			end

		end
	end
end

function Room:render()
	for y = 1, self.height do
		for x = 1, self.width do
		  local tile = self.tiles[y][x]
		  love.graphics.draw(gTextures['tiles'], gFrames['tiles'][tile.id],
			 (x - 1) * TILE_SIZE + self.renderOffsetX + self.adjacentOffsetX,
			 (y - 1) * TILE_SIZE + self.renderOffsetY + self.adjacentOffsetY)
		end

	end

	-- render doorways; stencils are placed where the arches are after so the player can
	-- move through them convincingly
	for k, doorway in pairs(self.doorways) do
		doorway:render(self.adjacentOffsetX, self.adjacentOffsetY)
	end

	for k, object in pairs(self.objects) do
		object:render(self.adjacentOffsetX, self.adjacentOffsetY)
	end

	for k, entity in pairs(self.entities) do
		if not entity.dead then entity:render(self.adjacentOffsetX, self.adjacentOffsetY) end
	end


	-- stencil out the door arches so it looks like the player is going through
	love.graphics.stencil(function()
	   -- left
	   love.graphics.rectangle('fill', -TILE_SIZE - 6, MAP_RENDER_OFFSET_Y + (MAP_HEIGHT / 2) * TILE_SIZE - TILE_SIZE,
		  TILE_SIZE * 2 + 6, TILE_SIZE * 2)

	   -- right
	   love.graphics.rectangle('fill', MAP_RENDER_OFFSET_X + (MAP_WIDTH * TILE_SIZE) - 6,
		  MAP_RENDER_OFFSET_Y + (MAP_HEIGHT / 2) * TILE_SIZE - TILE_SIZE, TILE_SIZE * 2 + 6, TILE_SIZE * 2)

	   -- top
	   love.graphics.rectangle('fill', MAP_RENDER_OFFSET_X + (MAP_WIDTH / 2) * TILE_SIZE - TILE_SIZE,
		  -TILE_SIZE - 6, TILE_SIZE * 2, TILE_SIZE * 2 + 12)

	   --bottom
	   love.graphics.rectangle('fill', MAP_RENDER_OFFSET_X + (MAP_WIDTH / 2) * TILE_SIZE - TILE_SIZE,
		  VIRTUAL_HEIGHT - TILE_SIZE - 6, TILE_SIZE * 2, TILE_SIZE * 2 + 12)
	end, 'replace', 1)

	love.graphics.setStencilTest('less', 1)

	if self.player then
	   self.player:render()
	end

	love.graphics.setStencilTest()
	for k, projectile in pairs(self.projectiles) do
		projectile:render(self.adjacentOffsetX, self.adjacentOffsetY)
	end
end

function Room:spawnHeart(entity)
	local heart = GameObject(GAME_OBJECT_DEFS['heart'],entity.x,entity.y)
	heart.onCollide = function(obj,player,dt)
		if not obj.collided then
			obj.collided = true
			player:damage(-2)
			return true
		end
	end
	table.insert(self.objects,heart)
end