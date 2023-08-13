--[[
    GD50
    Angry Birds

    Author: Colton Ogden
    cogden@cs50.harvard.edu

    An Obstacle is any physics-based construction that forms the current level,
    usually shielding the aliens the player is trying to kill; they can form houses,
    boxes, anything the developer wishes. Depending on what kind they are, they are
    typically rectangular or polygonal.
]]

Obstacle = Class{}

function Obstacle:init(world, shape, x, y)
    self.shape = shape or 'horizontal'

    if self.shape == 'horizontal' then
        self.frame = 2
    elseif self.shape == 'vertical' then
        self.frame = 4
    end

    self.startX = x
    self.startY = y
    local health = health or 1
    self.world = world

    self.body = love.physics.newBody(self.world, 
        self.startX or math.random(VIRTUAL_WIDTH), self.startY or math.random(VIRTUAL_HEIGHT - 35), 'dynamic')

    -- assign width and height based on shape to create physics shape
    if self.shape == 'horizontal' then
        self.width = 110
        self.height = 35
    elseif self.shape == 'vertical' then
        self.width = 35
        self.height = 110
    end

    self.shape = love.physics.newRectangleShape(self.width, self.height)

    self.fixture = love.physics.newFixture(self.body, self.shape)

    self.fixture:setUserData({
        type = 'Obstacle',
        health = 2
    })
end

function Obstacle:update(dt)
    --print_r(self.fixture:getUserData())
    if self.fixture:getUserData().health == 1 then
        if self.frame == 2 or self.frame == 4 then
            self.frame = self.frame - 1
        end
    end
end

function Obstacle:render()
    love.graphics.draw(gTextures['wood'], gFrames['wood'][self.frame],
        self.body:getX(), self.body:getY(), self.body:getAngle(), 1, 1,
        self.width / 2, self.height / 2)
end