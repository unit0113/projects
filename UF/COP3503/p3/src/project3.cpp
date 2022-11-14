#include <SFML\Graphics.hpp>
#include "textureManager.h"
#include "board.h"

int main()
{
    sf::RenderWindow window(sf::VideoMode(200, 200), "Minesweeper");
    TextureManager textureManager();
    //sf::CircleShape shape(100.f);
    //shape.setFillColor(sf::Color::Green);

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
        window.display();
    }
}