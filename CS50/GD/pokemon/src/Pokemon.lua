--[[
    GD50
    Pokemon

    Authors: Colton Ogden, Macarthur Inbody
    cogden@cs50.harvard.edu, 133794m3r@gmail.com
]]

Pokemon = Class{}

function Pokemon:init(def, level,player)
    self.name = def.name

    self.battleSpriteFront = def.battleSpriteFront
    self.battleSpriteBack = def.battleSpriteBack

    self.baseHP = def.baseHP
    self.baseAttack = def.baseAttack
    self.baseDefense = def.baseDefense
    self.baseSpeed = def.baseSpeed
    self.HPIV = def.HPIV
    self.attackIV = def.attackIV
    self.defenseIV = def.defenseIV
    self.speedIV = def.speedIV



    self.HP = self.baseHP
    self.attack = self.baseAttack
    self.defense = self.baseDefense
    self.speed = self.baseSpeed

    self.level = level
    self.currentExp = 0
    self.expToLevel = self.level * self.level * 5 * 0.75
    self:calculateStats(player)

    self.currentHP = self.HP
end

function Pokemon:calculateStats(player)
    local HPIncrease, attackIncrease, defenseIncrease, speedIncrease
    for i = 1, self.level do
        HPIncrease, attackIncrease, defenseIncrease, speedIncrease =self:statsLevelUp()
        -- to make sure that the player's Pokemon is stronger than wild ones we make sure that it always gets 1 extra
        -- stat during each of it's "levelups" during it's initial state. So its total stats will always be higher than wild
        -- ones at the same level 5.
        if player then
            self.HP = self.HP + (HPIncrease == 0 and 1 or 0)
            self.attack = self.attack + (attackIncrease == 0 and 1 or 0)
            self.defense = self.defense + (defenseIncrease == 0 and 1 or 0)
            self.speed = self.speed + (speedIncrease == 0 and 1 or 0)
        end

    end

end

function Pokemon.getRandomDef()
    return POKEMON_DEFS[POKEMON_IDS[math.random(#POKEMON_IDS)]]
end

--[[
    Takes the IV (individual value) for each stat into consideration and rolls
    the dice 3 times to see if that number is less than or equal to the IV (capped at 5).
    The dice is capped at 6 just so no stat ever increases by 3 each time, but
    higher IVs will on average give higher stat increases per level. Returns all of
    the increases so they can be displayed in the TakeTurnState on level up.
]]
function Pokemon:statsLevelUp()
    local HPIncrease = 0

    for j = 1, 3 do
        if math.random(6) <= self.HPIV then
            self.HP = self.HP + 1
            HPIncrease = HPIncrease + 1
        end
    end

    local attackIncrease = 0

    for j = 1, 3 do
        if math.random(6) <= self.attackIV then
            self.attack = self.attack + 1
            attackIncrease = attackIncrease + 1
        end
    end

    local defenseIncrease = 0

    for j = 1, 3 do
        if math.random(6) <= self.defenseIV then
            self.defense = self.defense + 1
            defenseIncrease = defenseIncrease + 1
        end
    end

    local speedIncrease = 0

    for j = 1, 3 do
        if math.random(6) <= self.speedIV then
            self.speed = self.speed + 1
            speedIncrease = speedIncrease + 1
        end
    end

    return HPIncrease, attackIncrease, defenseIncrease, speedIncrease
end

function Pokemon:levelUp()
    self.level = self.level + 1
    self.expToLevel = self.level * self.level * 5 * 0.75

    return self:statsLevelUp()
end