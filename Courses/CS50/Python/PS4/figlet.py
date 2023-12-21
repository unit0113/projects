import sys
from pyfiglet import Figlet
import random


figlet = Figlet()
fonts = figlet.getFonts()

if (len(sys.argv) == 2
    or (len(sys.argv) == 3 and (sys.argv[1] != '-f' or sys.argv[2] not in fonts))
    or len(sys.argv) > 3
    ):
    sys.exit("Invalid usage")

if len(sys.argv) == 1:
    font = random.choice(fonts)
else:
    font = sys.argv[2]

figlet.setFont(font=font)
print(figlet.renderText(input()))
