class MCPContext:
    def __init__(self):
        self.state = {}
        self.logs = []

    def update(self, key, value):
        self.state[key] = value
        self.logs.append({
            "action": key,
            "value": value
        })

    def get(self, key):
        return self.state.get(key)

    def get_logs(self):
        return self.logs