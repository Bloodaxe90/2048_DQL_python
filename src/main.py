from PySide6.QtWidgets import QApplication

from src.UI.application import Application


def main():
    print("Started")
    app = QApplication([])
    window = Application()
    window.show()
    window.resize(400,400)
    app.exec()

if __name__ == "__main__":
    main()