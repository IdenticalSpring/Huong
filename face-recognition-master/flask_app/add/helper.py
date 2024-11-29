import random

class Helper:
    def __init__(self):
        self.labels = [
            "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
            "Normal Videos", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
            "Stealing", "Vandalism"
        ]

    def classify_frame(self):
        return random.choices(
            ["Burglary", "Stealing"] + [label for label in self.labels if label not in ["Burglary", "Stealing"]],
            weights=[0.4, 0.3] + [0.3 / (len(self.labels) - 2)] * (len(self.labels) - 2),
            k=1
        )[0]
