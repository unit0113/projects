#include <SFML\Graphics.hpp>
#include <memory>
#include "textureManager.h"
#include "board.h"
#include "boardConfig.h"

int main()
{
    BoardConfig boardConfig;
    sf::RenderWindow window(sf::VideoMode(boardConfig.m_columns * 32, boardConfig.m_rows * 32 + 100), "Minesweeper");
    window.setFramerateLimit(60);
    Board board(window, boardConfig);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::MouseButtonReleased) {
                sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                // If Tile
                if (mousePosition.y < board.getNumRows() * 32) {
                    if (event.mouseButton.button == sf::Mouse::Left) {
                        board.reveal(mousePosition);
                    }
                    else if (event.mouseButton.button == sf::Mouse::Right) {
                        board.toggleFlag(mousePosition);
                    }
                }
                // If Button
                else {
                    board.checkButtonSelection(mousePosition);
                }
            }
        }
        window.clear(sf::Color(255, 255, 255, 0));
        board.draw();
        window.display();
    }
    TextureManager::clear();
}