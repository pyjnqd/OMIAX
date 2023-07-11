
def preserve_key(state, preserve_prefix: str):
    """Preserve part of model weights based on the
       prefix of the preserved module name.
    """
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if preserve_prefix + "." in key:
            newkey = key.replace(preserve_prefix + '.', "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    return state

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0