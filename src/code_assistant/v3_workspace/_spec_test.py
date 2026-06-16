import sys
sys.path.insert(0, '/home/bmartins/dev/agentic_patterns/src/code_assistant/v3_workspace/agent_code')
from counter import BoundedCounter

def test_add_and_top():
    assert (lambda c=BoundedCounter(3): (c.add('a'), c.add('a'), c.add('b'), c.top()))()[-1] == 'a'

def test_keys():
    assert set((lambda c=BoundedCounter(3): (c.add('a'), c.add('b'), c.keys()))()[-1]) == {'a','b'}

def test_evicts():
    assert (lambda c=BoundedCounter(2): (c.add('a'), c.add('b'), c.add('c'), 'a' not in c.keys()))()[-1] is True

def test_capacity():
    assert (lambda c=BoundedCounter(2): (c.add('a'), c.add('b'), c.add('c'), len(c.keys()) <= 2))()[-1] is True
