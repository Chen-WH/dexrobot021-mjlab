"""List the registered DexRobot021 tasks."""

from __future__ import annotations

from prettytable import PrettyTable

from dexrobot021_mjlab.scripts._tasks import dexrobot021_tasks


def main() -> None:
  table = PrettyTable()
  table.title = "DexRobot021 mjlab tasks"
  table.field_names = ["Task ID"]
  table.align["Task ID"] = "l"
  for task in dexrobot021_tasks():
    table.add_row([task])
  print(table.get_string())


if __name__ == "__main__":
  main()
