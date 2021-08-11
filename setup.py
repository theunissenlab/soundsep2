from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name = "soundsep",
    version = "0.1.2",
    packages = find_packages(exclude=["examples*", "test*"]),
    scripts = ["manage.py"],
    include_package_data = True,
    zip_safe = False,
    description = "GUI for manual separation of audio data on multiple mic channels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = "Kevin Yu",
    author_email = "thekevinyu@gmail.com",
    url = "https://github.com/kevinyu/soundsep2",
    keywords = "spectrogram stft audio visualization sound source-separation segmentation labeling annotation",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "click",
        "numpy",
        "matplotlib",
        "pandas",
        "PyQt5",
        "pyqtgraph",
        "qasync",
        "scipy",
        "SoundFile",
        "parse",
        "pyyaml",
    ],
    entry_points="""
        [console_scripts]
        sep=manage:cli
    """,
)
