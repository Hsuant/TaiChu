# Data Architecture

Canonical design:

- Common schema: `dataset/schema/`
- Versioned storage: `dataset/storage/`
- Offline preparation: `dataset/offline/`
- Task selection: `dataset/tasks.py`
- Runtime: `dataset/runtime/`

Layer rules:

1. Schema does not import Storage or Runtime.
2. Storage does not import Sampling, Packing or Trainer.
3. Preprocessing Adapter owns task semantics and emits typed Samples.
4. Runtime selects components through the Task Registry.
5. Dataset and DataLoader do not contain per-task `if/else` expansion.

Full lifecycle analysis, diagrams and extension guidance:
[`docs/UNIFIED_DATA_INFRASTRUCTURE.md`](../docs/UNIFIED_DATA_INFRASTRUCTURE.md).
