class Node:
    def __init__(self, key, parent=None, l_child=None, r_child=None) -> None:
        self._key = key
        self._parent = parent
        self._l_child = l_child
        self._r_child = r_child
        self._count = 1

    # Getters
    @property
    def key(self):
        return self._key
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def l_child(self):
        return self._l_child
    
    @property
    def r_child(self):
        return self._r_child
    
    @property
    def count(self):
        return self._count
    
    # Setters
    @key.setter
    def key(self, new_key):
        self._key = new_key

    @parent.setter
    def parent(self, new_parent):
        self._parent = new_parent

    @l_child.setter
    def l_child(self, new_l_child):
        self._l_child = new_l_child

    @r_child.setter
    def r_child(self, new_r_child):
        self._r_child = new_r_child

    def increment(self):
        self._count += 1

    def decrement(self):
        self._count -= 1

    def __bool__(self):
        return self._count > 0
    