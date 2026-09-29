"""Dev helper: convert a saved CDP Page.captureScreenshot JSON response to a PNG.

python scripts/cdp_to_png.py <cdp-response.json> <out.png>
"""

import base64
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
obj = json.loads(src.read_text(encoding="utf-8"))
data = obj.get("data") or obj.get("result", {}).get("data")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(base64.b64decode(data))
print("wrote", dst)
