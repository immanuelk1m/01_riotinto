import sys
sys.path.insert(0, '/content/E019')
import E019 as m
import test_e019 as t
frames = t.synthetic()
for direction in m.DIRECTIONS:
    panels = {c: m.weekly_panel(f) for c, f in frames.items()}
    origins = sorted(set().union(*[set(f.origin) for f in frames.values()]))
    m.replay(panels, origins, direction)
t.Contracts().test_future_labels_do_not_change_past_decisions()
t.Contracts().test_ebh_rejects_the_obvious_winners()
for commodity in m.TARGETS:
    m.load(commodity)
print('PASS six input archives and three cross-sectional directions')
