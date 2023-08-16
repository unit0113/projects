--[[
    PauseState Class
    Author: Macarthur Inbody
    mdi2455@email.vccs.edu
    A simple pause-state class that allows the user to pause the action and then return
    to the main game.
]]
PauseState = Class { __includes = BaseState }
function PauseState:init()
    -- nothing
end
function PauseState:enter(params)
    --pause the music
    sounds['music']:pause();
    self.score = params.score
end
function PauseState:update(dt)
    if love.keyboard.was_pressed("enter") or love.keyboard.was_pressed("p") then
        sounds['pause']:play()
        --you can't resume as it's set to loop. so we just set it to play again.
        sounds['music']:play()
        gStateMachine:change('countdown')
    end
end

function PauseState:render()
    love.graphics.setFont(medium_font)
    love.graphics.printf("Press Enter to return to the game.", 0, 50, VIRTUAL_WIDTH, 'center')
    love.graphics.printf("Current score: " .. tostring(self.score), 0, 80, VIRTUAL_WIDTH, 'center')
end