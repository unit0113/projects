Powerup = Class{}

function Powerup:init(type)
    -- simple positional and dimensional variables
    self.width = 16
    self.height = 16
    self.type = type

    -- Downward velocity
    self.dy = 1

    self.y = -5
    self.x = math.random(10, VIRTUAL_WIDTH - 25)
end

function Powerup:update(dt)
    self.y = self.y + self.dy
end

function Powerup:render()
    love.graphics.draw(gTextures['main'], gFrames['powerups'][self.type], self.x, self.y)
end

function Powerup:is_out_of_play()
    return self.y > VIRTUAL_HEIGHT
end

function Powerup:collides(target)
    -- Check X direction
    if self.x > target.x + target.width or target.x > self.x + self.width then
        return false
    end

    -- Check Y direction
    if self.y > target.y + target.height or target.y > self.y + self.height then
        return false
    end 

    return true
end