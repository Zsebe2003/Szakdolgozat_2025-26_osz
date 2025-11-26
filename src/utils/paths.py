from pathlib import Path

class Paths:
    def __init__(self, base: Path | None = None):
        self.base = base or Path.cwd()
        self.data = self.base / "data"
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        self.xes = self.data / "xes"
        self.figures = self.base / "figures"

    def ensure(self):
        for d in [self.data, self.raw, self.processed, self.xes, self.figures]:
            d.mkdir(parents=True, exist_ok=True)
