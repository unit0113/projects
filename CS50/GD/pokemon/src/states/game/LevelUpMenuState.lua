--[[
    GD50
    Pokemon

    Author: Macarthur Inbody
   	133794m3r@gmail.com
]]

--[[
Because the assignment wants a freaking menu? Why in the hell. This makes _zero_ sense. 
I got it working  with a textbox which is all this needs but screw it. I'll waste time redowing it as a menu.
Yeah no I'm not dealing with a freaking menu it makes _zero_ sense to make it as part of a menu as it's just a textbox.
]]

LevelUpMenuState = Class{__includes = BaseState}

function LevelUpMenuState:init(pokemon,stat_increase)

	local old_hp = pokemon.HP - stat_increase[1]
	local oldattack = pokemon.attack - stat_increase[2]
	local olddefense = pokemon.defense - stat_increase[3]
	local oldspeed = pokemon.speed - stat_increase[4]
	local hp = string.format("HP: %d + %d = %d",old_hp,stat_increase[1],pokemon.HP)
	local atk = string.format("Attack: %d + %d = %d",oldattack,stat_increase[2],pokemon.attack)
	local def = string.format("Defense: %d + %d = %d",olddefense,stat_increase[3],pokemon.defense)
	local spd = string.format("Speed: %d + %d = %d",oldspeed,stat_increase[4],pokemon.speed)
	self.battleMenu = Menu {
		x = VIRTUAL_WIDTH - 190,
		y = 64,
		width = 196,
		height =VIRTUAL_HEIGHT - 128,
		items = {
			{
				text = hp,
				onSelect = function()
					gStateStack:pop()
				end
			},
			{
				text = atk,
				onSelect = function()
					gStateStack:pop()
				end
			},
			{
				text = def,
				onSelect = function()
					gStateStack:pop()
				end
			},
			{
				text = spd,
				onSelect = function()
					-- this state
					gStateStack:pop()

					-- battle
					self:fadeOutWhite()
				end
			}
		},
		showCursor = false
	}
end

function LevelUpMenuState:update(dt)
	self.battleMenu:update(dt)
end

function LevelUpMenuState:render()
	self.battleMenu:render()
end

function LevelUpMenuState:fadeOutWhite()
	-- pop message state
	gStateStack:pop()
	-- pop the battle state
	gStateStack:pop()
	-- pop message state
	gStateStack:pop()
	-- fade in
	gStateStack:push(FadeInState({
		r = 255, g = 255, b = 255
	}, 1,
			function()

				-- resume field music
				gSounds['victory-music']:stop()
				gSounds['field-music']:play()

				-- pop off the battle state
				gStateStack:pop()
				gStateStack:push(FadeOutState({
					r = 255, g = 255, b = 255
				}, 1, function() end))
			end))
end