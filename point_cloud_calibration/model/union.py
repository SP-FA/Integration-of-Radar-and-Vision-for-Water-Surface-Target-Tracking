class DisjointSetUnion:
    def __init__(self, parents):
        self.parent = parents
        self.hasParent = [0] * len(parents)

    @property
    def unioned(self): return bool(sum(self.hasParent))

    def find(self, x):
        if self.hasParent[x] == 1: return self.parent[x]
        else: return False

    def union(self, x, y):
        self.parent[y] = x
        self.hasParent[y] = 1
