---
---
--- Created by macarthur.


push = require "push"
Class = require 'class'
--state machine

require 'StateMachine'
require 'states/BaseState'
require 'states/PlayState'
require 'states/ScoreState'
require 'states/CountdownState'
require 'states/TitleScreenState'
require 'states/PauseState'
--objects
require 'bird'
require 'pipe'
require 'pipe_pair'

--constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VIRTUAL_WIDTH = 512
VIRTUAL_HEIGHT = 288
-- local files.
local background = love.graphics.newImage('imgs/background.png')
local background_scroll = 0
local ground_scroll = 0
local BACKGROUND_SPEED = 30
local GROUND_SCROLL_SPEED = 60
local BACKGROUND_LOOP = 413;
local ground = love.graphics.newImage('imgs/ground.png')
--[[
When the guy designed the physics system of this game he was designing it for a 60fps system. Thus each frame tick that
is higher will screw with the physics and cause the terminal velocity to be hit even faster. To get around having to modify
this to use a _true_ timer like it should I am simply capping the FPS to 60 to simplify my work.VIRTUAL_WIDTH
]]
local fps_capped = true
gCurrentPlayTime = 0


--some globals.
gCurrentDifficulty = false
--commented the lines below out to avoid declaring variables that aren't needed.
--[[
--Awards table. Currently empty.
gAwards = {}
--TODO:Track awards won.
gAwardsWon = {}
]]
--seed the PRNG with the current time and also the average delta over the last second prior to the game. The PRNG will also be reseeded each time a game is started.
math.randomseed(os.time() + love.timer.getAverageDelta())
function love.load()
    love.graphics.setDefaultFilter('nearest', 'nearest')
    love.window.setTitle('Flappy Bird Clone')
    --fonts
    small_font = love.graphics.newFont("fonts/font.ttf", 8)
    medium_font = love.graphics.newFont('fonts/flappy.ttf', 14)
    flappy_font = love.graphics.newFont('fonts/flappy.ttf', 28)
    huge_font = love.graphics.newFont('fonts/flappy.ttf', 56)
    --setting fonts.
    love.graphics.setFont(flappy_font)

    --sound table
    sounds = {
        ['jump'] = love.audio.newSource('sfx/jump.wav', 'static'),
        ['explosion'] = love.audio.newSource('sfx/explosion.wav', 'static'),
        ['hurt'] = love.audio.newSource('sfx/hurt.wav', 'static'),
        ['score'] = love.audio.newSource('sfx/score.wav', 'static'),
        -- https://freesound.org/people/xsgianni/sounds/388079/
        ['music'] = love.audio.newSource('sfx/marios_way.mp3', 'static'),
        ['pause'] = love.audio.newSource('sfx/pause.wav', 'static')
    }

    --kick off music
    sounds['music']:setLooping(true)
    sounds['music']:play()

    push:setupScreen(VIRTUAL_WIDTH, VIRTUAL_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, {
        vsync = true,
        fullscreen = false,
        resizeable = true
    })
    gStateMachine = StateMachine {
        ['title'] = function()
            return TitleScreenState()
        end,
        ['countdown'] = function()
            return CountdownState()
        end,
        ['play'] = function()
            return PlayState()
        end,
        ['score'] = function()
            return ScoreState()
        end,
        ['pause'] = function()
            return PauseState()
        end
    }
    gStateMachine:change('title')

    --input table
    love.keyboard.keys_pressed = {}
    --mouse input table
    love.mouse.buttons_pressed = {}

    min_dt = 1 / 60
    next_time = love.timer.getTime()
end

function love.resize(w, h)
    push:resize(w, h)
end

function love.keypressed(key)
    love.keyboard.keys_pressed[key] = true
    if key == 'escape' then
        love.event.quit()
    elseif key == "t" then
        next_time = 0
        fps_capped = not fps_capped
    end
end

function love.mousepressed(x, y, button)
    love.mouse.buttons_pressed[button] = true
end
function love.keyboard.was_pressed(key)
    return love.keyboard.keys_pressed[key]
end
function love.mouse.was_pressed(button)
    return love.mouse.buttons_pressed[button]
end

function love.update(dt)
    next_time = next_time + min_dt;
    background_scroll = (background_scroll + BACKGROUND_SPEED * dt) % BACKGROUND_LOOP
    ground_scroll = (ground_scroll + GROUND_SCROLL_SPEED * dt) % VIRTUAL_WIDTH
    --call gStateMachine
    gStateMachine:update(dt)

    love.keyboard.keys_pressed = {}
    love.mouse.buttons_pressed = {}
end

function fps_cap()

    local cur_time = love.timer.getTime()
    if next_time <= cur_time then
        next_time = cur_time
    else
        love.timer.sleep(next_time - cur_time)
    end

end

function love.draw()
    push:start()
    love.graphics.draw(background, -background_scroll, 0);
    gStateMachine:render()
    love.graphics.draw(ground, -ground_scroll, VIRTUAL_HEIGHT - 16)
    if fps_capped then
        fps_cap()
    end
    display_ui()
    push:finish();


end

function display_ui()
    love.graphics.setFont(medium_font)
    love.graphics.print("FPS: " .. tostring(love.timer.getFPS()), VIRTUAL_WIDTH - 75, 5)

end