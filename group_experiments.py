import os
import json
import shutil

for i in os.listdir("my_runs"):
    if not i.isdigit():
        continue

    with open(os.path.join("my_runs", str(i), "run.json"), 'r') as f:
        j = json.load(f)
        for source in j['experiment']['sources']:
            # os.remove(os.path.join("my_runs", str(i), source[1][9:]))
            shutil.copy(os.path.join("my_runs", source[1]), os.path.join(
                "my_runs", str(i), source[0]))
