You may edit exactly these files:

- `configs/experiment/evolve_base.yaml`
- `configs/model/icon_evolve.yaml`
- `src/models/icon/icon_evolve.py`
- `src/models/base/transformer_evolve.py`
- `src/models/icon/pe_evolve.py`

They start as a no-op shadow of vanilla ICON.

`configs/experiment/evolve_base.yaml` wires the evolve experiment and can switch
dataset or runtime-facing defaults. `configs/model/icon_evolve.yaml` is the
Hydra entrypoint for model composition. `src/models/icon/icon_evolve.py` is the
top-level model hook. `src/models/base/transformer_evolve.py` is the natural
place for attention-path positional changes. `src/models/icon/pe_evolve.py` is
the scratch space for new positional modules or helpers.

Prefer edits that keep responsibility clear across these files instead of
smearing one idea everywhere.
