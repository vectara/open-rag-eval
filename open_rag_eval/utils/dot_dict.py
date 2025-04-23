"""A recursive dictionary class that allows dot notation access """

class DotDict(dict):
    """Dot notation access to dictionary attributes, with recursive support for nested dicts"""

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self._convert_nested_dicts()

    def _convert_nested_dicts(self):
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, DotDict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = [DotDict(item) if isinstance(item, dict) else item
                             for item in value]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]