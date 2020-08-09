#!/bin/python
import json
import glob

"""
Clears notebooks output cells and other difficult to merge data
"""

nbs = glob.glob("./**/*.ipynb", recursive=True)

delete_keys = ["data"]

for nb_path in nbs:
    with open(nb_path) as f:
        nb = json.loads(f.read())
        for c in nb["cells"]:
            c["metadata"] = {}
            c["outputs"] = []
            c["execution_count"] = None
            for k in delete_keys:
                if k in c:
                    del c[k]
    with open(nb_path, "w") as f:
        f.write(json.dumps(nb, indent="  "))
