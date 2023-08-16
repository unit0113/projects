Pipe = Class {}
local PIPE_IMAGE = love.graphics.newImage('imgs/pipe.png')
local PIPE_SCROLL = -60
PIPE_SPEED = 60
PIPE_WIDTH = 70
PIPE_HEIGHT = 288
function Pipe:init(orientation, y)
    self.x = VIRTUAL_WIDTH + 64
    self.height = PIPE_HEIGHT
    self.width = PIPE_WIDTH
    self.y = y
    self.orientation = orientation
end

function Pipe:update(dt)
    --self.x = self.x + PIPE_SCROLL * dt
end

function Pipe:render()
    love.graphics.draw(PIPE_IMAGE, self.x, (self.orientation == "top" and self.y + self.height or self.y),
            0, 1, self.orientation == "top" and -1 or 1)
end