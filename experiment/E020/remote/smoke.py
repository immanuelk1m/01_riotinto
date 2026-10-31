import sys
sys.path.insert(0, '/content/E020')
import E020 as m
import test_e020 as t
for direction in m.DIRECTIONS:
    m.configure(direction)
    assert m.DIRECTION == direction
if m.DIRECTIONS != {"base": 0.0, "recent": 0.0, "tta": 0.0}:
    raise SystemExit('direction table changed')
t.test_snapshots_and_splits()
for commodity in m.TARGETS:
    m.load(commodity)
print('PASS six input archives, split causality and three E020 directions')
