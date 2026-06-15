---
name: run-circle-packing-smoke
description: "Use when a fast end-to-end smoke run is needed for the Eve loop development, will run on circle packing task."
---

If you want to test with the circle packing task (usually when you are implementing Eve),
run the following command from the repository root. 
You don't need to look into the py or yaml file, just run it.

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke
```

If you want to test with your own task (usually when you are using Eve loop for downstream tasks),
you should change the config name accordingly. You may refer to other skills or confirm with the user.
