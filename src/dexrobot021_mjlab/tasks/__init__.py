"""Register the DexRobot021 mjlab tasks."""

from mjlab.utils.lab_api.tasks.importer import import_packages

from dexrobot021_mjlab import checkpoint_guard

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)

# Every script that loads a checkpoint imports this package first to register the
# tasks, which makes it the one place the checkpoint guard can be installed.
checkpoint_guard.install()
