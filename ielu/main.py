from traitsui.api import toolkit
from traits.trait_base import ETSConfig

from gselu import iEEGCoregistrationFrame
from utils import crash_if_freesurfer_is_not_sourced
import signal


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    crash_if_freesurfer_is_not_sourced()

    iEEGCoregistrationFrame().configure_traits()
