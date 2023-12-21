--[[
    TitleScreenState Class
    
    Author: Colton Ogden
    cogden@cs50.harvard.edu

    The TitleScreenState is the starting screen of the game, shown on startup. It should
    display "Press Enter" and also our highest score.
]]

TitleScreenState = Class { __includes = BaseState }

function TitleScreenState:init()
    -- nothing
end

function TitleScreenState:update(dt)
    if love.keyboard.was_pressed('enter') or love.keyboard.was_pressed('return') then
        -- reseed the PRNG again.
        math.randomseed(os.time() + love.timer.getAverageDelta())
        gStateMachine:change('countdown')
    elseif love.keyboard.was_pressed("d") then
        gCurrentDifficulty = not gCurrentDifficulty
    end
end

function TitleScreenState:render()
    love.graphics.setFont(flappy_font)
    love.graphics.printf('Fifty Bird', 0, 64, VIRTUAL_WIDTH, 'center')

    love.graphics.setFont(medium_font)
    love.graphics.printf('Press Enter', 0, 100, VIRTUAL_WIDTH, 'center')
    love.graphics.printf("To toggle FPS limit(60),physics work best <=60\n hit the letter \"t\"", 0, 170, VIRTUAL_WIDTH, 'center')
    love.graphics.printf("To toggle difficulty, press the letter \"d\".", 0, 210, VIRTUAL_WIDTH, "center");
    local current_difficulty = (gCurrentDifficulty) and "hard" or "easy";

    love.graphics.printf("Difficulty: " .. current_difficulty, 0, 130, VIRTUAL_WIDTH, 'center')
end