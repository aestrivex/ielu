import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name="ielu",
    version="0.2.4",
    maintainer="Roan LaPlante",
    maintainer_email="rlaplant@nmr.mgh.harvard.edu",
    description=("Interactive Electrode localization Utility"),
    license="Visuddhimagga Sutta; GPLv3+",
    datafiles=[('', ['readme.md', 'LICENSE'])],
    long_description=read('readme.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: X11 Applications",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    url="https://github.com/aestrivex/ielu",
    platforms=['any'],
    packages=['ielu'],
    package_data={'ielu':['make_dural_surface.csh']},
    scripts=['bin/ielu', 'ielu/mkoutersurf.py'],
    install_requires=["numpy", "scipy", "pymcubes", "pysurfer", "nibabel", "mne"]
)
