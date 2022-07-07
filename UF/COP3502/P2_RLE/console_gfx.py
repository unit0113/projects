class ConsoleGfx:

    default_top = "═"
    default_up_left = "╔"
    default_up_right = "╗"
    default_start = "║"
    default_end = "║"
    default_bottom = "═"
    default_low_left = "╚"
    default_low_right = "╝"

    COLOR_RESET = '\033[0m'
    fg_palette = ['']*16
    em_palette = ['']*16
    ul_palette = ['']*16
    bg_palette = ['']*16

    for i in range(8):
        fg_palette[i] = '\033[3' + str(i) + 'm'
        fg_palette[i+8] = '\033[9' + str(i) + 'm'
        em_palette[i] = '\033[1;3' + str(i) + 'm'
        em_palette[i+8] = '\033[1;9' + str(i) + 'm'
        ul_palette[i] = '\033[4;3' + str(i) + 'm'
        ul_palette[i+8] = '\033[4;9' + str(i) + 'm'
        bg_palette[i] = '\033[4' + str(i) + 'm'
        bg_palette[i+8] = '\033[10' + str(i) + 'm'

    BLACK = 0
    RED = 1
    DARK_GREEN = 2
    GOLD = 3
    BLUE = 4
    GARNETT = 5
    ORANGE = 6
    LIGHT_GRAY = 7
    GRAY = 8
    PEACH = 9
    GREEN = 10
    BRIGHT_GOLD = 11
    CYAN = 12
    MAGENTA = 13
    BRIGHT_ORANGE = 14
    WHITE = 15

    CLEAR = MAGENTA
    TRANS_DISPLAY = BLACK

    test_rainbow = [16, 2,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    test_image = [14, 6,
        CLEAR, CLEAR, GREEN, GREEN, GREEN, CLEAR, CLEAR, CLEAR,
        CLEAR, CLEAR, CLEAR, GREEN, GREEN, CLEAR, CLEAR, GREEN,
        WHITE, BLACK, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN,
        GREEN, DARK_GREEN, GREEN, GREEN, GREEN, GREEN, GREEN,
        GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN,
        GREEN, GREEN, CLEAR, GREEN, GREEN, GREEN, GREEN, GREEN,
        GREEN, GREEN, GREEN, GREEN, BLACK, BLACK, BLACK, GREEN,
        CLEAR, GREEN, GREEN, GREEN, BLACK, BLACK, BLACK, BLACK,
        BLACK, BLACK, GREEN, GREEN, GREEN, CLEAR, CLEAR, CLEAR,
        GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN,
        CLEAR, CLEAR, CLEAR, CLEAR, CLEAR
        ]

    def display_image(image_data):
        ConsoleGfx.display_image2(image_data, ConsoleGfx.default_top, ConsoleGfx.default_up_left, ConsoleGfx.default_up_right, ConsoleGfx.default_start,
                          ConsoleGfx.default_end, ConsoleGfx.default_bottom, ConsoleGfx.default_low_left, ConsoleGfx.default_low_right)

    def display_image2(image_data, top, up_left, up_right, start, end, bottom, low_left, low_right):
        width = image_data[0]
        height = image_data[1]
        data_index = 2

        print(up_left, end='')
        for x_index in range(width):
            print(top, end='')
        print(up_right)

        for y_index in range(0, height, 2):
            output_str = start
            for x_index in range(width):
                output_color = image_data[data_index]
                output_str += ConsoleGfx.fg_palette[ConsoleGfx.TRANS_DISPLAY if output_color == ConsoleGfx.CLEAR else output_color]
                output_color = image_data[data_index + width] if y_index + 1 < height else ConsoleGfx.CLEAR
                output_str += ConsoleGfx.bg_palette[ConsoleGfx.TRANS_DISPLAY if output_color == ConsoleGfx.CLEAR else output_color]
                output_str += '▀'
                data_index += 1
            data_index += width
            print(output_str + ConsoleGfx.COLOR_RESET + end)

        print(low_left, end='')
        for x_index in range(width):
            print(bottom, end='')
        print(low_right)

    def load_file(filename):
        file_data = []
        with open(filename, 'rb') as my_file:

            contents = my_file.read()

            for c in contents:
                file_data += [c]

            my_file.close()

        return file_data
