from pathlib import Path
import re

files = [
    Path('src/03b_train_one_model.py'),
    Path('src/03_train_base_models.py'),
    Path('src/archive/_comparison_report.py'),
    Path('src/archive/_tune_proposed_optimized.py'),
]

for path in files:
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'[ \t]+(\r?\n)', r'\1', text)
    text = re.sub(r'(?m)^\s+$', '', text)
    text = text.rstrip() + '\n'
    path.write_text(text, encoding='utf-8')
