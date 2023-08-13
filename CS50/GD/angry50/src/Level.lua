--[[
    GD50
    Angry Birds

    Author: Colton Ogden
    cogden@cs50.harvard.edu
]]

Level = Class{}

function Level:init()
    -- create a new "world" (where physics take place), with no x gravity
    -- and 30 units of Y gravity (for downward force)
    self.world = love.physics.newWorld(0, 300)

    -- bodies we will destroy after the world update cycle; destroying these in the
    -- actual collision callbacks can cause stack overflow and other errors
    self.destroyedBodies = {}

    -- define collision callbacks for our world; the World object expects four,
    -- one for different stages of any given collision
    function beginContact(a, b, coll)
        local types = {}
        local health = 0

        local data_a = a:getUserData()
        local data_b = b:getUserData()
        types[data_a.type] = true
        types[data_b.type] = true

        -- if we collided between both an alien and an obstacle...
        if types['Obstacle'] and types['Player'] then

            self.launchMarker.hasCollided = true
            -- destroy the obstacle if player's combined velocity is high enough
            if data_a.type == 'Obstacle' then
                health = data_a.health
                local velX, velY = b:getBody():getLinearVelocity()
                local sumVel = math.abs(velX) + math.abs(velY)

                if sumVel > 20 then
                    health = health -1
                    a:setUserData({type='Obstacle',health = health})
                    if health <= 0 then
                        table.insert(self.destroyedBodies, a:getBody())
                    end
                end
            else
                health = data_b.health
                local velX, velY = a:getBody():getLinearVelocity()
                local sumVel = math.abs(velX) + math.abs(velY)

                if sumVel > 20 then
                    health = health -1
                    b:setUserData({type='Obstacle',health = health})
                    if health <= 0 then
                        table.insert(self.destroyedBodies, b:getBody())
                    end
                end
            end
        end

        -- if we collided between an obstacle and an alien, as by debris falling...
        if types['Obstacle'] and types['Alien'] then
            -- destroy the alien if falling debris is falling fast enough
            if a:getUserData() == 'Obstacle' then
                local velX, velY = a:getBody():getLinearVelocity()
                local sumVel = math.abs(velX) + math.abs(velY)

                if sumVel > 20 then
                    table.insert(self.destroyedBodies, b:getBody())
                end
            else
                local velX, velY = b:getBody():getLinearVelocity()
                local sumVel = math.abs(velX) + math.abs(velY)

                if sumVel > 20 then
                    table.insert(self.destroyedBodies, a:getBody())
                end
            end
        end

        -- if we collided between the player and the alien...
        if types['Player'] and types['Alien'] then
            self.launchMarker.hasCollided = true

            -- destroy the alien if player is traveling fast enough
            if a:getUserData() == 'Player' then
                local velX, velY = a:getBody():getLinearVelocity()
                local sumVel = math.abs(velX) + math.abs(velY)
                
                if sumVel > 20 then
                    table.insert(self.destroyedBodies, b:getBody())
                end
            else
                local velX, velY = b:getBody():getLinearVelocity()
                local sumVel = math.abs(velX) + math.abs(velY)

                if sumVel > 20 then
                    table.insert(self.destroyedBodies, a:getBody())
                end
            end
        end

        -- if we hit the ground, play a bounce sound
        if types['Player'] and types['Ground'] then
            self.launchMarker.hasCollided = true        
            gSounds['bounce']:stop()
            gSounds['bounce']:play()
        end
    end

    -- the remaining three functions here are sample definitions, but we are not
    -- implementing any functionality with them in this demo; use-case specific
    function endContact(a, b, coll)
        
    end

    function preSolve(a, b, coll)

    end

    function postSolve(a, b, coll, normalImpulse, tangentImpulse)

    end

    -- register just-defined functions as collision callbacks for world
    self.world:setCallbacks(beginContact, endContact, preSolve, postSolve)

    -- shows alien before being launched and its trajectory arrow
    self.launchMarker = AlienLaunchMarker(self.world)

    -- aliens in our scene
    self.aliens = {}

    -- obstacles guarding aliens that we can destroy
    self.obstacles = {}

    -- simple edge shape to represent collision for ground
    self.edgeShape = love.physics.newEdgeShape(0, 0, VIRTUAL_WIDTH * 3, 0)

    self:generate()

    -- ground data
    self.groundBody = love.physics.newBody(self.world, -VIRTUAL_WIDTH, VIRTUAL_HEIGHT - 35, 'static')
    self.groundFixture = love.physics.newFixture(self.groundBody, self.edgeShape)
    self.groundFixture:setFriction(0.5)
    self.groundFixture:setUserData({type = 'Ground'})

    -- background graphics
    self.background = Background()
end

function Level:generate()
    -- which of my designed levels to show.
    local level = math.random(2)
    -- what level of health the blocks will have. has a 25% chance of having 2 health aka 2 hits to destory.
    local health = math.random(4) == 1 and 2 or 1
    if level == 1 then

        table.insert(self.aliens, Alien(self.world, 'square', VIRTUAL_WIDTH - 90, VIRTUAL_HEIGHT - TILE_SIZE - ALIEN_SIZE / 2, 'Alien'))
        -- spawn a few obstacles
        --[[ Makes a level like this
           ___
         || . ||
            _______
         |||| . ||||

         Where || is a vertical block. ___ is a horizontal block. and . is an alien.
        ]]
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
                VIRTUAL_WIDTH - 135, VIRTUAL_HEIGHT - 35 - 110 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
                VIRTUAL_WIDTH - 175, VIRTUAL_HEIGHT - 35 - 110 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
            VIRTUAL_WIDTH - 205, VIRTUAL_HEIGHT - 35 - 110 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
                VIRTUAL_WIDTH - 50, VIRTUAL_HEIGHT - 35 - 110 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
                VIRTUAL_WIDTH-10, VIRTUAL_HEIGHT - 35 - 110 / 2,health))


        table.insert(self.obstacles, Obstacle(self.world, 'horizontal',
                VIRTUAL_WIDTH - 145, VIRTUAL_HEIGHT - 35 - 110 - 35 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'horizontal',
                VIRTUAL_WIDTH - 35, VIRTUAL_HEIGHT - 35 - 110 - 35 / 2,health))
        table.insert(self.aliens, Alien(self.world, 'square', VIRTUAL_WIDTH - 95, VIRTUAL_HEIGHT - TILE_SIZE - 145 - ALIEN_SIZE / 2, 'Alien'))

        table.insert(self.obstacles,Obstacle(self.world, 'horizontal',
                VIRTUAL_WIDTH - 175, VIRTUAL_HEIGHT - 180 - 35 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'horizontal',
                VIRTUAL_WIDTH - 20, VIRTUAL_HEIGHT - 180 - 35 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'horizontal',
            VIRTUAL_WIDTH - 90, VIRTUAL_HEIGHT - 35 - 180 - 35 / 2,health))
    else
        -- spawn an alien to try and destroy
        table.insert(self.aliens, Alien(self.world, 'square', VIRTUAL_WIDTH - 80, VIRTUAL_HEIGHT - TILE_SIZE - ALIEN_SIZE / 2, 'Alien'))

        -- spawn a few obstacles
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
                VIRTUAL_WIDTH - 120, VIRTUAL_HEIGHT - 35 - 110 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'vertical',
                VIRTUAL_WIDTH - 35, VIRTUAL_HEIGHT - 35 - 110 / 2,health))
        table.insert(self.obstacles, Obstacle(self.world, 'horizontal',
                VIRTUAL_WIDTH - 80, VIRTUAL_HEIGHT - 35 - 110 - 35 / 2,health))

    end
end

function Level:update(dt)
    -- update launch marker, which shows trajectory
    self.launchMarker:update(dt)

    -- Box2D world update code; resolves collisions and processes callbacks
    self.world:update(dt)

    -- destroy all bodies we calculated to destroy during the update call
    for k, body in pairs(self.destroyedBodies) do
        if not body:isDestroyed() then 
            body:destroy()
        end
    end

    -- reset destroyed bodies to empty table for next update phase
    self.destroyedBodies = {}

    -- remove all destroyed obstacles from level
    for i = #self.obstacles, 1, -1 do
        if self.obstacles[i].body:isDestroyed() then
            table.remove(self.obstacles, i)

            -- play random wood sound effect
            local soundNum = math.random(5)
            gSounds['break' .. tostring(soundNum)]:stop()
            gSounds['break' .. tostring(soundNum)]:play()
        end
    end

    for _,obstacle in pairs(self.obstacles) do
        obstacle:update()
    end
    -- remove all destroyed aliens from level
    for i = #self.aliens, 1, -1 do
        if self.aliens[i].body:isDestroyed() then
            table.remove(self.aliens, i)
            gSounds['kill']:stop()
            gSounds['kill']:play()
        end
    end

    -- replace launch marker if original alien stopped moving
    if self.launchMarker.launched then
        local xPos,yPos = 0,0
        local xVel,yVel = 0,0
        if not self.launchMarker.hasSplit then
            xPos,yPos = self.launchMarker.alien.body:getPosition()
            xVel, yVel = self.launchMarker.alien.body:getLinearVelocity()
            if xPos < 0 or (math.abs(xVel) + math.abs(yVel) < 1.5) then
                self.launchMarker.alien.body:destroy()
                self.launchMarker = AlienLaunchMarker(self.world)
                -- re-initialize level if we have no more aliens
                if #self.aliens == 0 then
                    gStateMachine:change('start')
                end
            end
            if love.keyboard.wasPressed('space') and not self.launchMarker.hasCollided then
                self.launchMarker:split()
            end
        else

            for i,alien in pairs(self.launchMarker.alien) do
                xPos,yPos = alien.body:getPosition()
                xVel, yVel = alien.body:getLinearVelocity()

                if xPos < 0 or (math.abs(xVel)+math.abs(yVel) < 1.5) then
                    alien.body:destroy()
                    table.remove(self.launchMarker.alien,i)
                end
            end

            if #self.launchMarker.alien == 0 then
                self.launchMarker.alien = nil
                self.launchMarker = AlienLaunchMarker(self.world)

                if #self.aliens == 0 then
                    gStateMachine:change('start')
                end
            end
        end

    end
end

function Level:render()
    -- render ground tiles across full scrollable width of the screen
    for x = -VIRTUAL_WIDTH, VIRTUAL_WIDTH * 2, 35 do
        love.graphics.draw(gTextures['tiles'], gFrames['tiles'][12], x, VIRTUAL_HEIGHT - 35)
    end

    self.launchMarker:render()

    for k, alien in pairs(self.aliens) do
        alien:render()
    end

    for k, obstacle in pairs(self.obstacles) do
        obstacle:render()
    end

    -- render instruction text if we haven't launched bird
    if not self.launchMarker.launched then
        love.graphics.setFont(gFonts['medium'])
        love.setColor(0, 0, 0, 255)
        love.graphics.printf('Click and drag circular alien to shoot!',
            0, 64, VIRTUAL_WIDTH, 'center')
        love.setColor(255, 255, 255, 255)
    end

    -- render victory text if all aliens are dead
    if #self.aliens == 0 then
        love.graphics.setFont(gFonts['huge'])
        love.setColor(0, 0, 0, 255)
        love.graphics.printf('VICTORY', 0, VIRTUAL_HEIGHT / 2 - 32, VIRTUAL_WIDTH, 'center')
        love.setColor(255, 255, 255, 255)
    end
end
