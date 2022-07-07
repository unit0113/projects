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


class ProgressBar:
    def __init__(self, total, *, percentage=True):
        self.total = total
        self.current_progress = 0
        self.percentage = percentage

    def progress(self, progress):
        if progress > self.total:
            raise ValueError('Usage: Current progress greater than expected total')

        self.current_progress = progress
        try:
            self._print_progress_bar()
        except:
            print(colorama.Fore.RESET)

    def increment(self, increment):
        self.current_progress += increment
        
        if self.current_progress > self.total:
            raise ValueError('Usage: Current progress greater than expected total')

        try:
            self._print_progress_bar()
        except:
            print(colorama.Fore.RESET)

    def _print_progress_bar(self):
        percent = 100 * self.current_progress / total
        bar = '█' * int(percent) + '-' * int(100 - percent)

        color = colorama.Fore.YELLOW

        if self.percentage:
            part_2 = f'{percent:.2f}%'
        else:
            part_2 = f'{self.current_progress}//{total}'
        
        print(color + f'|{bar}| {part_2}', end = '\r')

        if percent == 100:
            color = colorama.Fore.GREEN
            print(color + f'|{bar}| {part_2}', end = '\r')
            print(colorama.Fore.RESET)




# Tests
total = 25000
pb = ProgressBar(total)
for i in range(total+1):
    pb.increment(1)