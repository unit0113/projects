Bird = Class {};
local GRAVITY = 20
function Bird:init()
    self.image = love.graphics.newImage('imgs/bird.png')
    self.x = VIRTUAL_WIDTH / 2 - 8
    self.y = VIRTUAL_HEIGHT / 2 - 8

    self.width = self.image:getWidth()
    self.height = self.image:getHeight()

    self.dy = 0
end
function Bird:update(dt)
    --[[
        this should be updated on a value based upon the ticks not a raw frame as frame-rate decides how quickly
    the bird starts to fall and reaches terminal velocity.
    ]]
    self.dy = self.dy + GRAVITY * dt
    self.y = self.y + self.dy
    if love.keyboard.was_pressed('space') or love.mouse.was_pressed(1) then
        self.dy = -5
    end
end
function Bird:collides(pipe)
    -- the 2's are left and top offsets
    -- the 4's are right and bottom offsets
    -- both offsets are used to shrink the bounding box to give the player
    -- a little bit of leeway with the collision

    --on hard mode the standard leeway is used. On easy mode they get an extra pixel.
    local leeway_pixels = (gCurrentDifficulty) and 2 or 3
    local leeway_width = (gCurrentDifficulty) and 4 or 6

    if (self.x + leeway_pixels) + (self.width - leeway_width) >= pipe.x and self.x + leeway_pixels <= pipe.x + PIPE_WIDTH then
        if (self.y + leeway_pixels) + (self.height - leeway_width) >= pipe.y and self.y + leeway_pixels <= pipe.y + PIPE_HEIGHT then
            return true
        end
    end

    return false

end
function Bird:render()
    love.graphics.draw(self.image, self.x, self.y)
end