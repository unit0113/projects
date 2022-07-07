red = int(input())
green = int(input())
blue = int(input())
rgb = [red, green, blue]
grey = min(rgb)
for index, color in enumerate(rgb.copy()):
    rgb[index] = color - grey

print(*rgb, sep=' ')