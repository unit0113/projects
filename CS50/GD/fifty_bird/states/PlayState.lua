--[[
    PlayState Class
    Author: Colton Ogden
    cogden@cs50.harvard.edu

    The PlayState class is the bulk of the game, where the player actually controls the bird and
    avoids pipes. When the player collides with a pipe, we should go to the GameOver state, where
    we then go back to the main menu.
]]

PlayState = Class { __includes = BaseState }

PIPE_SPEED = 60
PIPE_WIDTH = 70
PIPE_HEIGHT = 288

BIRD_WIDTH = 38
BIRD_HEIGHT = 24

function PlayState:init()
    self.bird = Bird()
    self.pipePairs = {}
    self.timer = 0
    self.score = 0
    -- difficulty changes.
    self.wait_time = (gCurrentDifficulty) and 1.9 or 2.25
    -- initialize our last recorded Y value for a gap placement to base other gaps off of
    self.lastY = -PIPE_HEIGHT + math.random(80) + 20
end

function PlayState:update(dt)
    gCurrentPlayTime = gCurrentPlayTime + dt;
    -- update timer for pipe spawning
    self.timer = self.timer + dt
    --[[
     Difficulty modifies how big the gaps are. The easier difficulty.
     If the person is playing on a harder difficulty the max gap height is 105, but if they are playing on easier
     difficulty the max is 111. Basically it randomizes it some but not a ton to make it too hard to play with.
     ]]

    local PIPE_GAP_HEIGHT = (math.random(20) + ((gCurrentDifficulty) and 85 or 91))
    -- spawn a new pipe pair for the randomized time
    if self.timer > self.wait_time then
        -- modify the last Y coordinate we placed so pipe gaps aren't too far apart
        -- no higher than 10 pixels below the top edge of the screen,
        -- and no lower than a gap length (90 pixels) from the) bottom
        local y = math.max(-PIPE_HEIGHT + 10,
                math.min(self.lastY + math.random(-20, 20), VIRTUAL_HEIGHT - PIPE_GAP_HEIGHT - PIPE_HEIGHT))
        self.lastY = y

        -- add a new pipe pair at the end of the screen at our new Y
        table.insert(self.pipePairs, PipePair(y, PIPE_GAP_HEIGHT))

        -- reset timer
        self.timer = 0
        --[[
            Sets the spawn timer based on some random value in seconds. Below are the ranges for this.
            Hard mode: ~1.25-2.5
            Easy Mode: ~1.5-2.5
        ]]
        self.wait_time = (math.random() + ((gCurrentDifficulty) and 1.25 or 1.5))
    end
    if love.keyboard.was_pressed("p") then
        gStateMachine:change("pause",
                {
                    score = self.score
                })
    end
    -- for every pair of pipes..
    for k, pair in pairs(self.pipePairs) do
        -- score a point if the pipe has gone past the bird to the left all the way
        -- be sure to ignore it if it's already been scored
        if not pair.scored then
            if pair.x + PIPE_WIDTH < self.bird.x then
                self.score = self.score + 1
                pair.scored = true
                sounds['score']:play()
            end
        end

        -- update position of pair
        pair:update(dt)
    end

    -- we need this second loop, rather than deleting in the previous loop, because
    -- modifying the table in-place without explicit keys will result in skipping the
    -- next pipe, since all implicit keys (numerical indices) are automatically shifted
    -- down after a table removal
    for k, pair in pairs(self.pipePairs) do
        if pair.remove then
            table.remove(self.pipePairs, k)
        end
    end

    -- simple collision between bird and all pipes in pairs
    for k, pair in pairs(self.pipePairs) do
        for l, pipe in pairs(pair.pipes) do
            if self.bird:collides(pipe) then
                sounds['explosion']:play()
                sounds['hurt']:play()

                gStateMachine:change('score', {
                    score = self.score
                })
            end
        end
    end

    -- update bird based on gravity and input
    self.bird:update(dt)

    -- reset if we get to the ground
    if self.bird.y > VIRTUAL_HEIGHT - 15 then
        sounds['explosion']:play()
        sounds['hurt']:play()

        gStateMachine:change('score', {
            score = self.score
        })
    end
end

function PlayState:render()
    for k, pair in pairs(self.pipePairs) do
        pair:render()
    end

    love.graphics.setFont(flappy_font)
    love.graphics.print('Score: ' .. tostring(self.score), 8, 8)

    self.bird:render()
end

--[[
    Called when this state is transitioned to from another state.
]]
function PlayState:enter()
    -- if we're coming from death, restart scrolling
    scrolling = true
end

--[[
    Called when this state changes to another state.
]]
function PlayState:exit()
    -- stop scrolling for the death/score screen
    scrolling = false
end