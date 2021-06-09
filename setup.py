from setuptools import setup

setup(
    name = "soundsep",
    version = "0.1",
    packages = [
        "soundsep",
        "soundsep.examples"
    ],
    zip_safe = False,
    description = "GUI for manual separation of audio data on multiple mic channels",
    author = "Kevin Yu",
    author_email = "thekevinyu@gmail.com",
    install_requires = [
        "click",
        "numpy",
        "pyqtgraph",
        "PyQt5",
        "scipy",
        "SoundFile",
        "qasync",
    ]
)
