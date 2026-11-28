import os, re
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(G, "预览", "web_pet.html")
s = open(p, encoding="utf-8").read()
paths = re.findall(r'(?:sheet|anim):\s*"([^"]+)"', s)
bad = []
for q in paths:
    ap = os.path.normpath(os.path.join(os.path.join(G, "预览"), q.replace("../", "../")))
    if not os.path.exists(ap):
        bad.append(q)
print("引用总数", len(paths), "缺失", len(bad))
for b in bad:
    print("  缺失:", b)
