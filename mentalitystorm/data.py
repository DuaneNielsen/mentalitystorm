from mentalitystorm import Selector


class AutoEncodeSelect(Selector):
    def get_input(self, package, device):
        return package[0].to(device),

    def get_target(self, package, device):
        return package[0].to(device),


class StandardSelect(Selector):
    def __init__(self, source_index=0, target_index=1):
        self.source_index = source_index
        self.target_index = target_index

    def get_input(self, package, device):
        return package[self.source_index].to(device),

    def get_target(self, package, device):
        return package[self.target_index].to(device),