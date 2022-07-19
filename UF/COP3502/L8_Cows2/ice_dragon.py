from dragon import Dragon


class IceDragon(Dragon):
    def __init__(self, name, image) -> None:
        super().__init__(name, image)

    def can_breathe_fire(self):
        return False