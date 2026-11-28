import os
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(G, "预览", "web_pet.html")
blocks = open(os.path.join(G, "待审", "_diag0506", "js_blocks.txt"), encoding="utf-8").read()
s = open(p, encoding="utf-8").read()
anchor = '      {\n        id: "10happy_jump",'
assert anchor in s, "anchor missing"
assert 'id: "05thinking"' not in s, "already inserted"
s = s.replace(anchor, blocks + "\n" + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("inserted", len(blocks), "chars; total", len(s))
