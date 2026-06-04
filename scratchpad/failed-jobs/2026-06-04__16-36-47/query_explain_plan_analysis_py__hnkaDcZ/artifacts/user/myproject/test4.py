import re

txt = """
Vector Search Plan:
ProjectionExec: expr=...
"""

for line in txt.splitlines():
    if not line.strip(): continue
    m = re.match(r'^\s*([A-Za-z0-9_]+):', line)
    if m:
        print("MATCH:", m.group(1))
    else:
        print("NO MATCH:", line)
