import pygame
import json

NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
AUTOTILE_NEIGHBORS = [(1, 0), (-1, 0), (0, -1), (0, 1)]
AUTOTILE_TYPES = {"grass", "stone"}
AUTOTILE_MAP = {
    tuple(sorted([(1, 0), (0, 1)])): 0,
    tuple(sorted([(1, 0), (0, 1), (-1, 0)])): 1,
    tuple(sorted([(-1, 0), (0, 1)])): 2,
    tuple(sorted([(-1, 0), (0, -1), (0, 1)])): 3,
    tuple(sorted([(-1, 0), (0, -1)])): 4,
    tuple(sorted([(-1, 0), (0, -1), (1, 0)])): 5,
    tuple(sorted([(1, 0), (0, -1)])): 6,
    tuple(sorted([(1, 0), (0, -1), (0, 1)])): 7,
    tuple(sorted([(1, 0), (-1, 0), (0, 1), (0, -1)])): 8,
}
PHYSICS_TILES = {"grass", "stone"}


class Tilemap:
    def __init__(self, game, tile_size: int = 16) -> None:
        self.game = game
        self.tile_size = tile_size
        self.tile_map = {}
        self.offgrid_tiles = []

    def extract(self, id_pairs, keep=False):
        # I Id pair is a type + variant
        matches = []
        for tile in self.offgrid_tiles.copy():
            if (tile["type"], tile["variant"]) in id_pairs:
                matches.append(tile.copy())
                if not keep:
                    self.offgrid_tiles.remove(tile)

        for loc, tile in self.tile_map.items():
            if (tile["type"], tile["variant"]) in id_pairs:
                matches.append(tile.copy())
                matches[-1]["pos"] = matches[-1]["pos"].copy()
                matches[-1]["pos"][0] *= self.tile_size
                matches[-1]["pos"][1] *= self.tile_size
                if not keep:
                    del self.tile_map[loc]

        return matches

    def solid_tile_check(self, pos):
        tile_loc = f"{int(pos[0] // self.tile_size)};{int(pos[1] // self.tile_size)}"
        if (
            tile_loc in self.tile_map
            and self.tile_map[tile_loc]["type"] in PHYSICS_TILES
        ):
            return self.tile_map[tile_loc]

    def tiles_around(self, pos):
        """Find neighboring tiles to check collisions"""
        tile_loc = (int(pos[0] // self.tile_size), int(pos[1] // self.tile_size))
        tiles = []
        for offset in NEIGHBOR_OFFSETS:
            check_loc = f"{tile_loc[0] + offset[0]};{tile_loc[1] + offset[1]}"
            if check_loc in self.tile_map:
                tiles.append(self.tile_map[check_loc])
        return tiles

    def physics_rects_around(self, pos):
        rects = []
        for tile in self.tiles_around(pos):
            if tile["type"] in PHYSICS_TILES:
                rects.append(
                    pygame.Rect(
                        tile["pos"][0] * self.tile_size,
                        tile["pos"][1] * self.tile_size,
                        self.tile_size,
                        self.tile_size,
                    )
                )
        return rects

    def draw(self, window, offset) -> None:
        for tile in self.offgrid_tiles:
            window.blit(
                self.game.assets[tile["type"]][tile["variant"]],
                (tile["pos"][0] - offset[0], tile["pos"][1] - offset[1]),
            )

        # Calculate visible tiles
        for x in range(
            offset[0] // self.tile_size,
            (offset[0] + window.get_width()) // self.tile_size + 1,
        ):
            for y in range(
                offset[1] // self.tile_size,
                (offset[1] + window.get_height()) // self.tile_size + 1,
            ):
                loc = f"{x};{y}"
                if loc in self.tile_map:
                    tile = self.tile_map[loc]
                    window.blit(
                        self.game.assets[tile["type"]][tile["variant"]],
                        (
                            tile["pos"][0] * self.tile_size - offset[0],
                            tile["pos"][1] * self.tile_size - offset[1],
                        ),
                    )

    def autotile(self):
        for loc, tile in self.tile_map.items():
            neighbors = set()
            for shift in AUTOTILE_NEIGHBORS:
                check_loc = f'{tile["pos"][0] + shift[0]};{tile["pos"][1] + shift[1]}'
                if (
                    check_loc in self.tile_map
                    and self.tile_map[check_loc]["type"] == tile["type"]
                ):
                    neighbors.add(shift)
            neighbors = tuple(sorted(neighbors))
            if (tile["type"] in AUTOTILE_TYPES) and (neighbors in AUTOTILE_MAP):
                tile["variant"] = AUTOTILE_MAP[neighbors]

    def save(self, file_path):
        with open(file_path, "w") as file:
            json.dump(
                {
                    "tilemap": self.tile_map,
                    "tile_size": self.tile_size,
                    "offgrid": self.offgrid_tiles,
                },
                file,
            )

    def load(self, file_path):
        with open(file_path, "r") as file:
            map_data = json.load(file)

        self.tile_map = map_data["tilemap"]
        self.tile_size = map_data["tile_size"]
        self.offgrid_tiles = map_data["offgrid"]
