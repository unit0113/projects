--[[
    GD50
    Angry Birds

    Author: Colton Ogden
    cogden@cs50.harvard.edu
]]

AlienLaunchMarker = Class{}

function AlienLaunchMarker:init(world)
    self.world = world

    -- starting coordinates for launcher used to calculate launch vector
    self.baseX = 90
    self.baseY = VIRTUAL_HEIGHT - 100

    -- shifted coordinates when clicking and dragging launch alien
    self.shiftedX = self.baseX
    self.shiftedY = self.baseY

    -- rotation for the trajectory arrow
    self.rotation = 0

    -- whether our arrow is showing where we're aiming
    self.aiming = false

    -- whether we launched the alien and should stop rendering the preview
    self.launched = false

    -- our alien we will eventually spawn
    self.alien = nil

    -- if they've collided with anything..
    self.hasCollided = false

    -- if they've split
    self.hasSplit = false
end

function AlienLaunchMarker:update(dt)
    
    -- perform everything here as long as we haven't launched yet
    if not self.launched then

        -- grab mouse coordinates
        local x, y = push:toGame(love.mouse.getPosition())

        -- if we click the mouse and haven't launched, show arrow preview
        if love.mouse.wasPressed(1) and not self.launched then
            self.aiming = true

            -- if we release the mouse, launch an Alien
        elseif love.mouse.wasReleased(1) and self.aiming then
            self.launched = true

            -- spawn new alien in the world, passing in user data of player
            self.alien = Alien(self.world, 'round', self.shiftedX, self.shiftedY, 'Player')

            -- apply the difference between current X,Y and base X,Y as launch vector impulse
            self.alien.body:setLinearVelocity((self.baseX - self.shiftedX) * 10, (self.baseY - self.shiftedY) * 10)

            -- make the alien pretty bouncy
            self.alien.fixture:setRestitution(0.4)
            self.alien.body:setAngularDamping(1)

            -- we're no longer aiming
            self.aiming = false
            -- re-render trajectory
        elseif self.aiming then
            self.rotation = self.baseY - self.shiftedY * 0.9
            self.shiftedX = math.min(self.baseX + 30, math.max(x, self.baseX - 30))
            self.shiftedY = math.min(self.baseY + 30, math.max(y, self.baseY - 30))
        end
    end
end

function AlienLaunchMarker:render()
    if not self.launched then
        
        -- render base alien, non physics based
        love.graphics.draw(gTextures['aliens'], gFrames['aliens'][9], 
            self.shiftedX - 17.5, self.shiftedY - 17.5)

        if self.aiming then
            
            -- render arrow if we're aiming, with transparency based on slingshot distance
            local impulseX = (self.baseX - self.shiftedX) * 10
            local impulseY = (self.baseY - self.shiftedY) * 10

            -- draw 6 circles simulating trajectory of estimated impulse
            local trajX, trajY = self.shiftedX, self.shiftedY
            local gravX, gravY = self.world:getGravity()

            -- http://www.iforce2d.net/b2dtut/projected-trajectory
            for i = 1, 90 do
                
                -- magenta color that starts off slightly transparent
                love.setColor(255, 80, 255, (255 / 12) * i)
                
                -- trajectory X and Y for this iteration of the simulation
                trajX = self.shiftedX + i * 1/60 * impulseX
                trajY = self.shiftedY + i * 1/60 * impulseY + 0.5 * (i * i + i) * gravY * 1/60 * 1/60

                -- render every fifth calculation as a circle
                if i % 5 == 0 then
                    love.graphics.circle('fill', trajX, trajY, 3)
                end
            end
        end
        
        love.setColor(255, 255, 255, 255)
    else
        if self.hasSplit then
            for _,alien in pairs(self.alien) do
                alien:render()
            end
        else
            self.alien:render()
        end
    end
end

function AlienLaunchMarker:split()

    local x = self.alien.body:getX()
    local y = self.alien.body:getY()
    local velX, velY = self.alien.body:getLinearVelocity()
    local original_alien = Alien(self.world,'round',x,y,'Player')
    -- move it by the center of the sphere of the alien.
    local bottom_alien = Alien(self.world,'round',x,y + 17.5, 'Player')
    -- move it down by teh center of the sphere of the alien.
    local top_alien = Alien(self.world, 'round', x, y - 17.5, 'Player')
    local velX2 = velX /10
    local velY2 = velY / 10
    top_alien.body:setLinearVelocity(velX-velX2,velY-velY2)
    top_alien.fixture:setRestitution(0.4)
    top_alien.body:setAngularDamping(1)

    original_alien.body:setLinearVelocity(velX, velY)
    original_alien.fixture:setRestitution(0.4)
    original_alien.body:setAngularDamping(1)

    bottom_alien.body:setLinearVelocity(velX+(-velX2),velY+(-velY2))
    bottom_alien.fixture:setRestitution(0.4)
    bottom_alien.body:setAngularDamping(1)

    print('alien has split')

    self.hasSplit = true
    self.alien.body:destroy()
    self.alien = {}
    table.insert(self.alien,top_alien)
    table.insert(self.alien,bottom_alien)
    table.insert(self.alien,original_alien)
end