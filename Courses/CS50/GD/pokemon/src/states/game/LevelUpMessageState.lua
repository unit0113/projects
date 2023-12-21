--[[
    GD50
    Pokemon

    Author: Macarthur Inbody
   	133794m3r@gmail.com
]]

LevelUpMessageState = Class{__includes = BaseState}

function LevelUpMessageState:init(pokemon,stat_increase, onClose, canInput)
	local msg = self:makeMsg(pokemon,stat_increase)
	self.textbox = Textbox(VIRTUAL_WIDTH - 190, 64, 188, VIRTUAL_HEIGHT-128, msg, gFonts['medium'],4)

	-- function to be called once this message is popped
	self.onClose = onClose or function() end

	-- whether we can detect input with this or not; true by default
	self.canInput = canInput

	-- default input to true if nothing was passed in
	if self.canInput == nil then self.canInput = true end
end

function LevelUpMessageState:update(dt)
	if self.canInput then
		self.textbox:update(dt)

		if self.textbox:isClosed() then
			gStateStack:pop()
			self.onClose()
		end
	end
end

function LevelUpMessageState:render()
	self.textbox:render()
end

function LevelUpMessageState:makeMsg(pokemon,stat_increase)
	local old_hp = pokemon.HP - stat_increase[1]
	local oldattack = pokemon.attack - stat_increase[2]
	local olddefense = pokemon.defense - stat_increase[3]
	local oldspeed = pokemon.speed - stat_increase[4]
	local hp = string.format("HP: %d + %d = %d",old_hp,stat_increase[1],pokemon.HP)
	local atk = string.format("Attack: %d + %d = %d",oldattack,stat_increase[2],pokemon.attack)
	local def = string.format("Defense: %d + %d = %d",olddefense,stat_increase[3],pokemon.defense)
	local spd = string.format("Speed: %d + %d = %d",oldspeed,stat_increase[4],pokemon.speed)
	local msg = ''
	--HPIncrease, attackIncrease, defenseIncrease, speedIncrease
	msg = string.format("%s\n%s\n%s\n%s",hp,atk,def,spd)
	return msg
end