from typing import final
import colorama


def progress_bar(progress, total, *, percentage=True):
    percent = 100 * progress / total
    bar = '█' * int(percent) + '-' * int(100 - percent)

    try:
        color = colorama.Fore.YELLOW

        if percentage:
            part_2 = f'{percent:.2f}%'
        else:
            part_2 = f'{progress}//{total}'
        
        print(color + f'|{bar}| {part_2}', end = '\r')

        if percent == 100:
            color = colorama.Fore.GREEN
            print(color + f'|{bar}| {part_2}', end = '\r')
            print(colorama.Fore.RESET)

    except:
        print(colorama.Fore.RESET)







# Tests
total = 25000
for i in range(total + 1):
    progress_bar(i, total)