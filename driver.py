import google.colab

class Driver:
    "Fancy yet stupid interface for Google Drive"
    def __init__(self, dirpath: str) -> None:
        self.dirpath = None
        self.open(dirpath)

    def _assert_open(self)
        if self.dirpath is None:
            raise RuntimeError("Driver instance has been closed before")

    def _assert_closed(self)
        if self.dirpath is not None:
            raise RuntimeError("Driver instance has been opened before")

    def _mount(self) -> None:
        self._assert_open()
        google.colab.drive.mount('/content/drive/')

    def _unmount(self) -> None:
        self._assert_open()
        google.colab.drive.flush_and_unmount()

    def reload(self) -> None:
        self._unmount()
        self._mount()

    def close(self) -> None:
        self._unmount()
        self.dirpath = None

    def open(self, dirpath: str) -> None:
        self._assert_closed()
        self.dirpath = os.path.join("drive/MyDrive", dirpath)
        self._mount()

    def file(self, filepath: str) -> str:
        self._assert_open()
        return os.path.join(self.dirpath, filepath)
