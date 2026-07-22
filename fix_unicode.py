import re

filepath = r"C:\Users\LEE\Documents\Keta Intelligence system\keta_flood_prediction_v3_2_3.js"

with open(filepath, encoding='utf-8') as f:
    content = f.read()

# Replace all problematic Unicode characters with ASCII equivalents
replacements = {
    '\u2014': '--',    # em dash
    '\u2013': '-',     # en dash  
    '\u00b0': ' deg',  # degree sign
    '\u2265': '>=',    # greater than or equal
    '\u2192': '->',    # right arrow
    '\u00d7': 'x',     # multiplication sign
    '\u2018': "'",     # left single quote
    '\u2019': "'",     # right single quote
    '\u201c': '"',     # left double quote
    '\u201d': '"',     # right double quote
}

for old, new in replacements.items():
    if old in content:
        count = content.count(old)
        content = content.replace(old, new)
        print(f"Replaced '{old}' (U+{ord(old):04X}) -> '{new}' ({count} occurrences)")

# Verify no non-ASCII remains
remaining = [(i+1, line.rstrip()) for i, line in enumerate(content.split('\n')) 
             if any(ord(c) > 127 for c in line)]
if remaining:
    print(f"\nWARNING: {len(remaining)} lines still have non-ASCII:")
    for ln, text in remaining:
        print(f"  L{ln}: {text}")
else:
    print("\nAll non-ASCII characters removed successfully.")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("File saved.")
