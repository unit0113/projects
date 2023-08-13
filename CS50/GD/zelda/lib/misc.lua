

---Sets the color used for drawing.
---@param red number @The amount of red.
---@param green number @The amount of green.
---@param blue number @The amount of blue.
---@param alpha number @The amount of alpha. The alpha value will be applied to all subsequent draw operations, even the drawing of an image.
---@overload fun(rgba:table):void
function love.setColor(red,green,blue,alpha)
	if alpha == nil then
		alpha = 255
	end
	if LOVE_VERSION_11 then
		if type(red) == 'table' then
		  r=red[1]/255
		  g=red[2]/255
		  b=red[3]/255
		  a=red[4]/255
		  love.graphics.setColor(r,g,b,a)
		else
		--end
		  love.graphics.setColor(red/255,green/255,blue/255,alpha/255)
		end
	else
		if type(red) == 'table' then
		  love.graphics.setColor(red)
		else
		  love.graphics.setColor(red,green,blue,alpha)

		end
	end
end
LOVE_VERSION_11 = love.getVersion()
--print(LOVE_VERSION_11)
printf = function(s,...)
	return io.write(s:format(...))
end
Event.on('bumped',function(entity,dt)
	entity.bumped = true
	if entity.direction == 'left' then
		entity.x = entity.x + entity.walkSpeed * dt
	elseif entity.direction == 'right' then
		entity.x = entity.x - entity.walkSpeed * dt
	elseif entity.direction == 'up' then
		entity.y = entity.y + entity.walkSpeed * dt
	elseif entity.direction == 'down' then
		entity.y = entity.y - entity.walkSpeed * dt
	end
end)
function deepcopy(orig, copies)
	copies = copies or {}
	local orig_type = type(orig)
	local copy
	if orig_type == 'table' then
		if copies[orig] then
		  copy = copies[orig]
		else
		  copy = {}
		  copies[orig] = copy
		  for orig_key, orig_value in next, orig, nil do
			 copy[deepcopy(orig_key, copies)] = deepcopy(orig_value, copies)
		  end
		  setmetatable(copy, deepcopy(getmetatable(orig), copies))
		end
	else -- number, string, boolean, etc
		copy = orig
	end
	return copy
end
