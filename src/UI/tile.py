from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QColor


class Tile:

    def __init__(self, label: QLabel):
        super().__init__()
        self.label = label
        self.colours = {
                2: QColor("#FFCDD2"),  # Light pink
                4: QColor("#FFAB91"),  # Light coral
                8: QColor("#FF8A65"),  # Light orange
                16: QColor("#FF7043"),  # Orange
                32: QColor("#FF5722"),  # Bright red-orange
                64: QColor("#F57F17"),  # Golden yellow
                128: QColor("#F4B400"),  # Yellow
                256: QColor("#C6FF00"),  # Lime green
                512: QColor("#69F0AE"),  # Turquoise
                1024: QColor("#40C4FF"),  # Sky blue
                2048: QColor("#7C4DFF"),  # Purple
            }
        self.change_value(0)


    def change_value(self, value: int):
        assert value >= 0, "Invalid tile value"
        new_text = str(value) if value != 0 else ""
        self.label.setText(new_text)
        self.set_colour(value)

    def set_colour(self, value: int):
        colour = self.colours.get(value, QColor("grey"))
        self.label.setStyleSheet(f"background-color: {colour.name()};"
                                 f"font-size: 20px;"
                                 f"color: black;")

