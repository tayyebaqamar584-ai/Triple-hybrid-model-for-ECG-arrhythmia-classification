import os

BASE = r'D:/ECG_Project_Complete'
raw = os.path.join(BASE, 'raw_data', 'mit-bih-arrhythmia-database-1.0.0')

DS1_BASE = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205,
    207, 208, 209, 215, 220, 223, 230,
]
DS2_BASE = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
    219, 221, 222, 228, 231, 232, 233, 234,
]

DS1 = sorted(DS1_BASE + [102, 104, 107])
DS2 = sorted(DS2_BASE + [217])
AVAILABLE = sorted(DS1 + DS2)

missing = []
for r in AVAILABLE:
    try:
        found = [f for f in os.listdir(raw) if f.startswith(str(r) + '.')]
    except FileNotFoundError:
        print('Raw signal directory not found:', raw)
        raise

    has_hea = any(f.endswith('.hea') for f in found)
    has_atr = any(f.endswith('.atr') for f in found)
    has_dat = any(f.endswith('.dat') for f in found)
    if not (has_hea and has_atr):
        missing.append((r, found))

print('Checked', len(AVAILABLE), 'records')
if not missing:
    print('All records have .hea and .atr present.')
else:
    print('Missing files for records:')
    for r, found in missing:
        print(r, '->', found)
