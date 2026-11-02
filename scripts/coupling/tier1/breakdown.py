"""Mean cost per phase of a coupled bridge step, from Torch's timestamped log.

    python breakdown.py <run_raw.txt> [first_step]"""
import re
import sys
from collections import defaultdict
from datetime import datetime

LOG = sys.argv[1]
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 3

STAMP = re.compile(r"^\[torch \d\d-\d\d-\d{4} (\d\d:\d\d:\d\d\.\d+)\] (.*)$")

# (phase name, line that opens it, line that closes it)
PHASES = [
    ("bridge kick 1 -> gas", "First bridge kick", "... grid kicked"),
    ("bridge kick 1 -> stars", "... grid kicked", "... stars kicked"),
    ("stellar evolution", "Do stellar evolution", "... dt from stellar evol"),
    ("N-body (PeTar)", "... dt from stellar evol", "Advance hydro"),
    ("hydro (LUNA)", "Advance hydro", "Star formation check"),
    ("star formation", "Star formation check", "Second bridge kick"),
    ("bridge kick 2 -> gas", "Second bridge kick", "... grid kicked"),
    ("bridge kick 2 -> stars", "... grid kicked", "... stars kicked"),
]


def parse(path):
    out = []
    for line in open(path, errors="replace"):
        m = STAMP.match(line.rstrip("\n"))
        if m:
            t = datetime.strptime(m.group(1), "%H:%M:%S.%f")
            out.append((t, m.group(2).strip()))
    return out


events = parse(LOG)
steps, cur = [], None
for t, msg in events:
    if msg.startswith("Bridge step: it="):
        if cur:
            steps.append(cur)
        cur = [(t, msg)]
    elif cur is not None:
        cur.append((t, msg))
if cur:
    steps.append(cur)

print("{} bridge steps in {}".format(len(steps), LOG))
if len(steps) <= FIRST:
    sys.exit("not enough steps to average")

totals, counts = defaultdict(float), defaultdict(int)
wall = 0.0
for st in steps[FIRST:-1]:
    wall += (st[-1][0] - st[0][0]).total_seconds()
    used = set()
    for name, a, b in PHASES:
        ia = next((i for i, (_, m) in enumerate(st)
                   if m.startswith(a) and i not in used), None)
        if ia is None:
            continue
        ib = next((i for i, (_, m) in enumerate(st) if m.startswith(b) and i > ia), None)
        if ib is None:
            continue
        used.add(ia)
        totals[name] += (st[ib][0] - st[ia][0]).total_seconds()
        counts[name] += 1

n = len(steps[FIRST:-1])
print("averaged over steps {}..{}\n".format(FIRST, len(steps) - 1))
print("{:<24}{:>10}{:>10}".format("phase", "s/step", "% step"))
mean_step = wall / n
for name, _, _ in PHASES:
    if counts[name]:
        v = totals[name] / n
        print("{:<24}{:>10.3f}{:>9.1f}%".format(name, v, 100 * v / mean_step))
acc = sum(totals[k] for k in totals) / n
print("{:<24}{:>10.3f}{:>9.1f}%".format("(unattributed)", mean_step - acc,
                                        100 * (mean_step - acc) / mean_step))
print("{:<24}{:>10.3f}".format("TOTAL", mean_step))
