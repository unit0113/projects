--[[
    GD50
    Match-3 Remake

    -- Tile Class --

    Author: Colton Ogden
    cogden@cs50.harvard.edu

    The individual tiles that make up our game board. Each Tile can have a
    color and a variety, with the varietes adding extra points to the matches.
]]

SHINY_PERCENTAGE = 0.1

Tile = Class{}

function Tile:init(x, y, color, variety)
    
    -- board positions
    self.gridX = x
    self.gridY = y

    -- coordinate positions
    self.x = (self.gridX - 1) * 32
    self.y = (self.gridY - 1) * 32

    -- tile appearance/points
    self.color = color
    self.variety = variety
    self.is_shiny = math.random() <= SHINY_PERCENTAGE
    self.shiny_table = {timer = nil,  factor = 0.1} 
end

function Tile:render(x, y)
    
    -- draw shadow
    love.graphics.setColor(34/255, 32/255, 52/255, 1)
    love.graphics.draw(gTextures['main'], gFrames['tiles'][self.color][self.variety], self.x + x + 2, self.y + y + 2)

    -- draw tile itself
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(gTextures['main'], gFrames['tiles'][self.color][self.variety], self.x + x, self.y + y)

    -- Draw shiny
    if self.is_shiny then
        love.graphics.setColor(1, 1, 1, self.shiny_table.factor)
        love.graphics.rectangle('fill', self.x + x, self.y + y, 32, 32, 4)
       
        if not self.shiny_table.timer then         
            self.shiny_table.timer = Timer.tween(1, {[self.shiny_table] = {factor = 0.5}}):finish(function()
                Timer.tween(1, {[self.shiny_table] = {factor = 0}}):finish(function() self.shiny_table.timer = nil end)
            end)
        end

        love.graphics.setColor(1, 1, 1, 1)
    
    end
end