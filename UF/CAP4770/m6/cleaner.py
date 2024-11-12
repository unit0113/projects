remove_chars = ["'", "(", ")", ",", "[", "]"]
vertices = 0
edges = 0
with open("raw.txt", "r", encoding="utf-8") as read_file:
    with open("adj.txt", "w", encoding="utf-8") as write_file:
        for line in read_file.readlines():
            line = [
                entry
                for entry in "".join(
                    char for char in line.strip() if char not in remove_chars
                ).split(" ")
                if entry
            ]
            line = list(set(line))
            vertices += 1
            edges += len(line) - 1
            write_file.write("\t".join(line) + "\n")

sparsisity = (2 * edges) / (vertices * (vertices - 1))
print(sparsisity)
