Grid = []
with open(r'AoC\2021\Day_15\input.txt', 'r') as data:
    for t in data:
        FirstRead = list(map(int, t.strip()))
        Grid.append(FirstRead)


Height = len(Grid)
Width = len(Grid[0])
New_Height = Height*5
New_Width = Width*5
closed = set()
open = [(0, (0,0))]
Directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
cycles = 0
while open:
    cycles += 1
    next = open.pop(0)
    CurDistance = next[0]
    Y = next[1][0]
    X = next[1][1]
    closed.add((Y, X))

    if Y == New_Height - 1 and X == New_Width - 1:
        print(CurDistance)
        break

    for t in Directions:
        NY = Y + t[0]
        NX = X + t[1]
        if 0 <= NY < New_Height and 0 <= NX < New_Width and (NY, NX) not in closed:
            GY = NY % Height
            GX = NX % Width
            Repeat = NY//Height + NX//Width
            NewDistance = Grid[GY][GX] + Repeat
            NewDistance = (NewDistance - 1) % 9 + 1
            NewTotalDist = NewDistance + CurDistance
            open.append((NewTotalDist, (NY, NX)))
            closed.add((NY, NX))

    open = sorted(open)

    if cycles % 5000 == 0:
        print(f"Cycle: {cycles}. Processing grid [{next[1][1]}, {next[1][0]}]. Current cost: {next[0]}")