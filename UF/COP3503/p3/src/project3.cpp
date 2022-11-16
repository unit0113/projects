#include <SFML\Graphics.hpp>
#include <memory>
#include "textureManager.h"
#include "board.h"

Board reset(sf::RenderWindow& window);  // reset if face is clicked

int main()
{
    sf::RenderWindow window(sf::VideoMode(200, 200), "Minesweeper");
    window.setFramerateLimit(60);
    //TextureManager textureManager;
    //Board board(textureManager, window);
    //sf::CircleShape shape(100.f);
    //shape.setFillColor(sf::Color::Green);
    Tile tile(0, 0, window);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::MouseButtonReleased) {
                auto mousePosition = sf::Mouse::getPosition(window);
            }
        }

        window.clear();
        //window.draw(shape);
        tile.draw();
        window.display();
    }

    TextureManager::clear();
}

Board reset(sf::RenderWindow& window) {
    Board newBoard(window);
    return newBoard;
}