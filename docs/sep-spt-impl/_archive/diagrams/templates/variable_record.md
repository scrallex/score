<!-- Copy this block into the subsystem catalogue for each variable under audit -->
### `<variable_name>`
- **File / Symbol**: `path/to/file.py::VARIABLE`
- **Type**: (`env` | `arg` | `const` | `Valkey key` | `context state` | ...)
- **Declared in**: code snippet, link to line
- **Purpose**: _short description_
- **Default / Range**: _document defaults, units, valid ranges_
- **Read by**: services / functions consuming the value
- **Mutated by**: sources that change it (scripts, API endpoints, live updates)
- **External Effects**: downstream systems, trades, alerts
- **Related Concepts**: duplicates or aliases in other subsystems
- **Open Questions**: if behaviour unclear, record here
