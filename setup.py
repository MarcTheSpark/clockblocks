import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clockblocks",
    version="0.2.1",
    author="Marc Evanstein",
    author_email="marc@marcevanstein.com",
    description="A python library for controlling the flow of musical time.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcTheSpark/clockblocks",
    packages=setuptools.find_packages(),
    install_requires=['expenvelope >= 0.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
