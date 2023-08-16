--[[
    Countdown State
    Author: Colton Ogden
    cogden@cs50.harvard.edu

    Counts down visually on the screen (3,2,1) so that the player knows the
    game is about to begin. Transitions to the PlayState as soon as the
    countdown is complete.
]]

CountdownState = Class { __includes = BaseState }

-- takes 1 second to count down each time
COUNTDOWN_TIME = 0.75

function CountdownState:init()
    gCurrentPlayTime = 0
    self.count = 3
    self.timer = 0
end

--[[
    Keeps track of how much time has passed and decreases count if the
    timer has exceeded our countdown time. If we have gone down to 0,
    we should transition to our PlayState.
]]
function CountdownState:update(dt)
    self.timer = self.timer + dt

    if self.timer > COUNTDOWN_TIME then
        self.timer = self.timer % COUNTDOWN_TIME
        self.count = self.count - 1

        if self.count == 0 then
            gStateMachine:change('play')
        end
    end
end

function CountdownState:render()
    love.graphics.setFont(huge_font)
    local string_table = {}
    if self.count == 3 then
        table.insert(string_table, { 1, 0, 0 })
    elseif self.count == 2 then
        table.insert(string_table, { 1, 1, 0 })
    elseif self.count == 1 then
        table.insert(string_table, { 0, 1, 0 })
    end
    table.insert(string_table, tostring(self.count))
    love.graphics.printf(string_table, 0, 120, VIRTUAL_WIDTH, 'center')
end