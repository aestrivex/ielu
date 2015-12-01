# Grids and Strips Electrode Localization Utility (gselu)
This is a GUI utility to assist in preprocessing tasks for electrocorticography (ecog) and stereo eeg (seeg) data analysis. It includes:

1. CT-to-MRI co-registration using Mutual Information (via Freesurfer)
1. Automatic electrode extraction and sorting into grids via pre-defined grid geometries
1. A GUI for manually tweaking electrode locations and grid information
1. Snapping electrode locations to a freesurfer cortical surface model
1. Exporting to montage files suitable for use in MNE-python

For more information about this package and how it works, see the [wiki](https://github.com/aestrivex/gselu/wiki).