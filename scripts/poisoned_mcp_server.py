"""
Run the MCP server with one injection payload planted in the Inception record.

    python scripts/poisoned_mcp_server.py <payload_id>

The tools run in the server process, so the in-process patch used by the injection test has to be
applied here, not in the agent's process. The database is not modified. Used by
`scripts/indirect_injection.py --mcp`; src/mcp_server/server.py itself has no test code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcp_server.server import main  # noqa: E402
from src.safety.indirect_injection import PAYLOADS, PAYLOADS_V2, PAYLOADS_V3, poisoned  # noqa: E402

if __name__ == "__main__":
    by_id = {p.id: p for p in (*PAYLOADS, *PAYLOADS_V2, *PAYLOADS_V3)}
    if len(sys.argv) != 2 or sys.argv[1] not in by_id:
        sys.exit(f"usage: poisoned_mcp_server.py <payload_id>  one of: {sorted(by_id)}")
    with poisoned(by_id[sys.argv[1]]):
        main()
