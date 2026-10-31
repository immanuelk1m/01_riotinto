import sys
sys.path.insert(0, '/content/E018')
import E018 as m
import test_e018 as t
for direction in m.DIRECTIONS:
    out, trace = m.replay(t.synthetic(), direction)
    m.validate(out)
t.Contracts().test_future_labels_do_not_change_past_predictions_or_state()
for commodity in m.TARGETS:
    m.load(commodity)
print('PASS six input archives and three anytime-valid gate directions')
