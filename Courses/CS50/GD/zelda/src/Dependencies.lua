--
-- libraries
--

Class = require 'lib/class'
Event = require 'lib/knife.event'
push = require 'lib/push'
Timer = require 'lib/knife.timer'

require 'src/Animation'
require 'src/constants'
require 'src/Entity'
require 'src/entity_defs'
require 'src/GameObject'
require 'src/game_objects'
require 'src/Hitbox'
require 'src/Player'
require 'src/Projectile'
require 'src/StateMachine'
require 'src/Util'

-- all of my common utility functions I've written thus far.
require 'lib/misc'
require 'src/world/Doorway'
require 'src/world/Dungeon'
require 'src/world/Room'

require 'src/states/BaseState'

require 'src/states/entity/EntityIdleState'
require 'src/states/entity/EntityWalkState'

-- the player's states
require 'src/states/entity/player/PlayerIdleState'
require 'src/states/entity/player/PlayerSwingSwordState'
require 'src/states/entity/player/PlayerWalkState'
require 'src/states/entity/player/PlayerLiftState'
require 'src/states/entity/player/PlayerIdleItemState'
require 'src/states/entity/player/PlayerWalkItemState'

require 'src/states/game/GameOverState'
require 'src/states/game/PlayState'
require 'src/states/game/StartState'

gTextures = {
	['tiles'] = love.graphics.newImage('graphics/tilesheet.png'),
	['background'] = love.graphics.newImage('graphics/background.png'),
	['character-walk'] = love.graphics.newImage('graphics/character_walk.png'),
	['character-swing-sword'] = love.graphics.newImage('graphics/character_swing_sword.png'),
	['hearts'] = love.graphics.newImage('graphics/hearts.png'),
	['switches'] = love.graphics.newImage('graphics/switches.png'),
	['entities'] = love.graphics.newImage('graphics/entities.png'),
	['character-pot-lift'] = love.graphics.newImage('graphics/character_pot_lift.png'),
	['character-pot-walk'] = love.graphics.newImage('graphics/character_pot_walk.png')
}

gFrames = {
	['tiles'] = GenerateQuads(gTextures['tiles'], 16, 16),
	['character-walk'] = GenerateQuads(gTextures['character-walk'], 16, 32),
	['character-swing-sword'] = GenerateQuads(gTextures['character-swing-sword'], 32, 32),
	['entities'] = GenerateQuads(gTextures['entities'], 16, 16),
	['hearts'] = GenerateQuads(gTextures['hearts'], 16, 16),
	['switches'] = GenerateQuads(gTextures['switches'], 16, 18),
	['character-pot-lift'] = GenerateQuads(gTextures['character-pot-lift'],16,32),
	['character-pot-walk'] = GenerateQuads(gTextures['character-pot-walk'],16,32)
}

gFonts = {
	['small'] = love.graphics.newFont('fonts/font.ttf', 8),
	['medium'] = love.graphics.newFont('fonts/font.ttf', 16),
	['large'] = love.graphics.newFont('fonts/font.ttf', 32),
	['gothic-medium'] = love.graphics.newFont('fonts/GothicPixels.ttf', 16),
	['gothic-large'] = love.graphics.newFont('fonts/GothicPixels.ttf', 32),
	['zelda'] = love.graphics.newFont('fonts/zelda.otf', 64),
	['zelda-small'] = love.graphics.newFont('fonts/zelda.otf', 32)
}

gSounds = {
	--[[for love 11 we have to say what type of source they are whether it's a stream or static.
		a static stream is one that has the whole file decoded and stored into memory. Whereas a stream only loads a buffer
		and the stream should be used if the file size is very large and it's an mp3 or an ogg file that is many MBs in size.
	]]
	['music'] = love.audio.newSource('sounds/music.mp3','static'),
	['sword'] = love.audio.newSource('sounds/sword.wav','static'),
	['hit-enemy'] = love.audio.newSource('sounds/hit_enemy.wav','static'),
	['hit-player'] = love.audio.newSource('sounds/hit_player.wav','static'),
	['door'] = love.audio.newSource('sounds/door.wav','static')
}