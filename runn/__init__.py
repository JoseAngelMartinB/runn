"""RUNN: Random Utility Neural Network"""

import tensorflow as tf

__author__ = """José Ángel Martín Baos"""
__email__ = "joseangel.martin@uclm.es"
__version__ = "0.1.0"


def display_info():
    """Display runn module information."""
    print("\n" + "-" * 34 + " RUNN info " + "-" * 35)
    print(__doc__)
    print("Version: " + __version__)
    print("Author: " + __author__)
    print("")
    print("System information:")
    print("TensorFlow version: " + tf.__version__)
    print("Number of CPUs available: " + str(len(tf.config.experimental.list_physical_devices("CPU"))))
    print("Number of GPUs available: " + str(len(tf.config.experimental.list_physical_devices("GPU"))))
    print("-" * 80)
