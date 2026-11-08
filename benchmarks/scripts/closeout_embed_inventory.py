"""Embed the sized lookup inventory in the single closeout report; no data copies."""
import csv
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parents[1]
W = B.parent
report = W/'CLOSEOUT.md'
rows = list(csv.DictReader((B/'results/closeout/data_availability_candidates.csv').open()))
unique = {}
for row in rows:
    path = Path(row['path'])
    if not path.is_file():
        continue
    if path not in unique:
        unique[path] = dict(groups=set(), items=set())
    unique[path]['groups'].add(row['group'])
    if not row['paper_item'].startswith('unassigned'):
        unique[path]['items'].add(row['paper_item'])
lines = ['', '<!-- sized-inventory:start -->', '',
    f"File-size snapshot {datetime.now(timezone.utc).isoformat(timespec='seconds')}: "
    f"{len(unique)} existing unique files. Sizes are logical file bytes; aliases and embedded "
    'archive members should not be added again to estimate a deposit total. Unassigned assets '
    'are organized by content, without inventing manuscript numbering.', '',
    '| Component / dataset | Existing file | Bytes |', '|---|---|---:|']
for path, info in unique.items():
    label = str(path.relative_to(W))
    component = ', '.join(sorted(info['items'])) if info['items'] else ', '.join(sorted(info['groups']))+' (unassigned)'
    lines.append(f'| {component} | [{label}]({path}) | {path.stat().st_size:,} |')
lines += ['', '<!-- sized-inventory:end -->', '']
text = report.read_text()
start, end = '<!-- sized-inventory:start -->', '<!-- sized-inventory:end -->'
if start in text:
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    text = prefix.rstrip()+'\n'+'\n'.join(lines)+suffix.lstrip('\n')
else:
    text = text.rstrip()+'\n'+'\n'.join(lines)
report.write_text(text)
print('Embedded', len(unique), 'existing unique files in', report)
