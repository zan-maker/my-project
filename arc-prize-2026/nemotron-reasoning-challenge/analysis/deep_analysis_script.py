#!/usr/bin/env python3
"""
Deep analysis of Nemotron Reasoning Challenge training data.
Analyzes all 6 puzzle categories to find patterns for building solvers.
"""

import pandas as pd
import re
import math
import random
from collections import Counter, defaultdict

random.seed(42)

df = pd.read_csv('data/train.csv')
prompts = df['prompt'].astype(str)

# ==============================================================================
# CATEGORIZATION
# ==============================================================================
def categorize(df):
    p = df['prompt'].astype(str)
    masks = {
        'bit_manipulation': p.str.contains('bit manipulation'),
        'cryptarithm': p.str.contains('transformation rules is applied to equations') & ~p.str.contains('cipher|encrypt'),
        'encryption': p.str.contains('cipher|encrypt|decrypt', case=False),
        'base_conversion': p.str.contains('numeral system|numeral.*converted'),
        'unit_conversion': p.str.contains('unit conversion.*measurement|secret unit conversion'),
        'gravitational': p.str.contains('gravitational constant'),
    }
    
    categories = {}
    for name, mask in masks.items():
        categories[name] = df[mask].copy()
    
    # Check for uncategorized
    all_mask = pd.concat([v for v in masks.values()], axis=1).any(axis=1)
    uncategorized = df[~all_mask]
    print(f"Total: {len(df)}, Uncategorized: {len(uncategorized)}")
    if len(uncategorized) > 0:
        print("Uncategorized prompt prefixes:")
        for i, row in uncategorized.head(5).iterrows():
            print(f"  {row['prompt'][:100]}")
    
    return categories

cats = categorize(df)
for name, subset in cats.items():
    print(f"{name}: {len(subset)} examples")

output_parts = []

def section(title):
    output_parts.append(f"\n{'='*80}\n{title}\n{'='*80}\n")

def add(text):
    output_parts.append(text)

# ==============================================================================
# 1. BIT MANIPULATION ANALYSIS
# ==============================================================================
section("1. BIT MANIPULATION ANALYSIS")

add(f"Total examples: {len(cats['bit_manipulation'])}\n")

def parse_bit_puzzle(prompt):
    """Extract input->output pairs from bit manipulation prompt."""
    lines = prompt.split('\n')
    pairs = []
    for line in lines:
        match = re.match(r'\s*([01]+)\s*->\s*([01]+)', line)
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs

def try_operations(inp, out):
    """Try to identify the bit transformation."""
    # Pad to 8 bits
    inp_bits = inp.zfill(8)
    out_bits = out.zfill(8)
    
    results = {}
    
    if len(inp_bits) != len(out_bits):
        results['length_mismatch'] = f"{len(inp_bits)} -> {len(out_bits)}"
        return results
    
    n = len(inp_bits)
    
    # XOR
    xor_result = ''.join('1' if inp_bits[i] != out_bits[i] else '0' for i in range(n))
    results['xor_mask'] = xor_result
    
    # AND
    and_result = ''.join('1' if inp_bits[i] == '1' and out_bits[i] == '1' else '0' for i in range(n))
    results['and_result'] = and_result
    
    # OR
    or_result = ''.join('1' if inp_bits[i] == '1' or out_bits[i] == '1' else '0' for i in range(n))
    results['or_result'] = or_result
    
    # Bit reversal
    rev = inp_bits[::-1]
    results['reverse'] = rev
    
    # NOT
    not_result = ''.join('1' if b == '0' else '0' for b in inp_bits)
    results['not'] = not_result
    
    # Left rotation by 1
    left_rot1 = inp_bits[1:] + inp_bits[0]
    results['left_rot_1'] = left_rot1
    left_rot2 = inp_bits[2:] + inp_bits[:2]
    results['left_rot_2'] = left_rot2
    
    # Right rotation by 1
    right_rot1 = inp_bits[-1] + inp_bits[:-1]
    results['right_rot_1'] = right_rot1
    right_rot2 = inp_bits[-2:] + inp_bits[:-2]
    results['right_rot_2'] = right_rot2
    
    # Left shift by 1
    left_shift1 = inp_bits[1:] + '0'
    results['left_shift_1'] = left_shift1
    
    # Right shift by 1
    right_shift1 = '0' + inp_bits[:-1]
    results['right_shift_1'] = right_shift1
    
    # NOT then XOR with constant
    for xor_val in range(256):
        xor_str = format(xor_val, f'0{n}b')
        candidate = ''.join('1' if (not_result[i] != xor_str[i]) else '0' for i in range(n))
        # not actually doing NOT XOR, let me just check all possible single ops combined
    
    # Check: input XOR mask = output?
    if xor_result == out_bits:
        results['DETECTED'] = f'XOR with mask {xor_result}'
    
    # Check: reverse = output?
    if rev == out_bits:
        results['DETECTED'] = 'Bit reversal'
    
    # Check: NOT = output?
    if not_result == out_bits:
        results['DETECTED'] = 'Bit NOT'
    
    # Check rotations
    for k in range(1, n):
        lr = inp_bits[k:] + inp_bits[:k]
        if lr == out_bits:
            results['DETECTED'] = f'Left rotation by {k}'
            break
        rr = inp_bits[-k:] + inp_bits[:-k]
        if rr == out_bits:
            results['DETECTED'] = f'Right rotation by {k}'
            break
    
    # Check NOT + rotation
    for k in range(1, n):
        lr = not_result[k:] + not_result[:k]
        if lr == out_bits:
            results['DETECTED'] = f'NOT then left rotation by {k}'
            break
        rr = not_result[-k:] + not_result[:-k]
        if rr == out_bits:
            results['DETECTED'] = f'NOT then right rotation by {k}'
            break
    
    # Check swap nibbles
    if inp_bits[:4] == out_bits[4:] and inp_bits[4:] == out_bits[:4]:
        results['DETECTED'] = 'Swap nibbles'
    
    # Check reverse + NOT
    rev_not = ''.join('1' if b == '0' else '0' for b in rev)
    if rev_not == out_bits:
        results['DETECTED'] = 'Reverse then NOT'
    
    # Check NOT then reverse
    not_rev = not_result[::-1]
    if not_rev == out_bits:
        results['DETECTED'] = 'NOT then reverse'
    
    # Check swap halves (4-bit)
    if n == 8:
        # Check byte-swap with some XOR
        pass
    
    # Two-input operations: AND, OR with constant
    for const in range(256):
        c_str = format(const, f'0{n}b')
        and_c = ''.join('1' if inp_bits[i] == '1' and c_str[i] == '1' else '0' for i in range(n))
        if and_c == out_bits:
            results['DETECTED'] = f'AND with {c_str}'
            break
        or_c = ''.join('1' if inp_bits[i] == '1' or c_str[i] == '1' else '0' for i in range(n))
        if or_c == out_bits:
            results['DETECTED'] = f'OR with {c_str}'
            break
    
    return results

# Analyze 15 examples
add("### Sample Analysis of 15 Bit Manipulation Puzzles\n")

detected_ops = Counter()
total_analyzed = 0
total_detected = 0

bit_samples = cats['bit_manipulation'].sample(min(15, len(cats['bit_manipulation'])), random_state=42)

for idx, row in bit_samples.iterrows():
    pairs = parse_bit_puzzle(row['prompt'])
    if not pairs:
        continue
    
    add(f"#### Puzzle {row['id']} (answer: {row['answer']})\n")
    
    # Check consistency of XOR mask across pairs
    xor_masks = []
    all_ops_match = {}
    
    for inp, out in pairs:
        ops = try_operations(inp, out)
        if 'xor_mask' in ops:
            xor_masks.append(ops['xor_mask'])
        if 'DETECTED' in ops:
            all_ops_match[ops['DETECTED']] = all_ops_match.get(ops['DETECTED'], 0) + 1
    
    add(f"  Example pairs: {pairs[:5]}")
    add(f"  XOR masks: {xor_masks[:5]}")
    
    # Check if XOR mask is consistent
    if len(set(xor_masks)) == 1:
        add(f"  **CONSISTENT XOR mask: {xor_masks[0]}**")
        detected_ops['consistent_xor'] += 1
        total_detected += 1
    elif len(set(xor_masks)) <= 2:
        add(f"  Near-consistent XOR masks: {set(xor_masks)}")
    
    if all_ops_match:
        best_op = max(all_ops_match, key=all_ops_match.get)
        add(f"  Detected operation: {best_op} (in {all_ops_match[best_op]}/{len(pairs)} pairs)")
        detected_ops[best_op] += 1
        if all_ops_match[best_op] == len(pairs):
            total_detected += 1
    else:
        add(f"  No single operation detected across all pairs")
    
    total_analyzed += 1
    add("")

# Broader statistical analysis
add("### Broader Bit Manipulation Statistics\n")
add(f"Analyzed {total_analyzed} puzzles, detected consistent operations in {total_detected}\n")
add("Operation detection summary:")
for op, count in detected_ops.most_common(10):
    add(f"  {op}: {count}")

# Check all puzzles for consistent XOR
xor_consistent = 0
xor_near_consistent = 0
total_checked = 0

for idx, row in cats['bit_manipulation'].iterrows():
    pairs = parse_bit_puzzle(row['prompt'])
    if not pairs:
        continue
    total_checked += 1
    xor_masks = []
    for inp, out in pairs:
        inp8 = inp.zfill(8)
        out8 = out.zfill(8)
        if len(inp8) == len(out8):
            xor_mask = ''.join('1' if inp8[i] != out8[i] else '0' for i in range(len(inp8)))
            xor_masks.append(xor_mask)
    if len(set(xor_masks)) == 1:
        xor_consistent += 1
    elif len(set(xor_masks)) <= 2:
        xor_near_consistent += 1

add(f"\nXOR consistency across all {total_checked} bit puzzles:")
add(f"  Fully consistent XOR: {xor_consistent} ({100*xor_consistent/total_checked:.1f}%)")
add(f"  Near-consistent (≤2 masks): {xor_near_consistent} ({100*xor_near_consistent/total_checked:.1f}%)")

# Analyze non-XOR puzzles
add("\n### Non-XOR Puzzle Analysis\n")
non_xor_puzzles = []
for idx, row in cats['bit_manipulation'].iterrows():
    pairs = parse_bit_puzzle(row['prompt'])
    if not pairs:
        continue
    xor_masks = []
    for inp, out in pairs:
        inp8 = inp.zfill(8)
        out8 = out.zfill(8)
        if len(inp8) == len(out8):
            xor_mask = ''.join('1' if inp8[i] != out8[i] else '0' for i in range(len(inp8)))
            xor_masks.append(xor_mask)
    if len(set(xor_masks)) > 1:
        non_xor_puzzles.append((idx, row, pairs, xor_masks))

add(f"Found {len(non_xor_puzzles)} puzzles where XOR is NOT consistent\n")

# Analyze a few non-XOR puzzles in detail
for idx, row, pairs, xor_masks in non_xor_puzzles[:10]:
    add(f"\nPuzzle {row['id']} (answer: {row['answer']}):")
    add(f"  Pairs: {pairs[:6]}")
    add(f"  XOR masks: {set(xor_masks)}")
    
    # Check for consistent NOT
    not_matches = 0
    for inp, out in pairs:
        inp8 = inp.zfill(8)
        out8 = out.zfill(8)
        if len(inp8) == len(out8):
            not_result = ''.join('1' if b == '0' else '0' for b in inp8)
            if not_result == out8:
                not_matches += 1
    if not_matches > 0:
        add(f"  NOT matches: {not_matches}/{len(pairs)}")
    
    # Check for consistent reversal
    rev_matches = 0
    for inp, out in pairs:
        inp8 = inp.zfill(8)
        out8 = out.zfill(8)
        if len(inp8) == len(out8) and inp8[::-1] == out8:
            rev_matches += 1
    if rev_matches > 0:
        add(f"  Reverse matches: {rev_matches}/{len(pairs)}")
    
    # Check rotations
    for k in range(1, 8):
        rot_matches = 0
        for inp, out in pairs:
            inp8 = inp.zfill(8)
            out8 = out.zfill(8)
            if len(inp8) == len(out8) and inp8[k:] + inp8[:k] == out8:
                rot_matches += 1
        if rot_matches > 0:
            add(f"  Left rot by {k} matches: {rot_matches}/{len(pairs)}")
    
    for k in range(1, 8):
        rot_matches = 0
        for inp, out in pairs:
            inp8 = inp.zfill(8)
            out8 = out.zfill(8)
            if len(inp8) == len(out8) and inp8[-k:] + inp8[:-k] == out8:
                rot_matches += 1
        if rot_matches > 0:
            add(f"  Right rot by {k} matches: {rot_matches}/{len(pairs)}")
    
    # Check NOT + rotations
    for k in range(1, 8):
        rot_matches = 0
        for inp, out in pairs:
            inp8 = inp.zfill(8)
            out8 = out.zfill(8)
            if len(inp8) == len(out8):
                not8 = ''.join('1' if b == '0' else '0' for b in inp8)
                if not8[k:] + not8[:k] == out8:
                    rot_matches += 1
        if rot_matches > 0:
            add(f"  NOT + left rot by {k} matches: {rot_matches}/{len(pairs)}")

# Check output bit lengths
add("\n### Output Bit Length Analysis\n")
bit_lengths = Counter()
for idx, row in cats['bit_manipulation'].iterrows():
    pairs = parse_bit_puzzle(row['prompt'])
    if pairs:
        for inp, out in pairs:
            bit_lengths[len(out)] += 1
            bit_lengths[len(inp)] += 1

add(f"Bit lengths in input/output: {dict(bit_lengths.most_common())}")

# ==============================================================================
# 2. ENCRYPTION ANALYSIS
# ==============================================================================
section("2. ENCRYPTION ANALYSIS")

add(f"Total examples: {len(cats['encryption'])}\n")

def parse_encryption_puzzle(prompt):
    """Extract cipher->plain pairs and the query."""
    lines = prompt.strip().split('\n')
    pairs = []
    query = None
    for line in lines:
        match = re.match(r'\s*(.*?)\s*->\s*(.*?)\s*$', line)
        if match:
            cipher = match.group(1).strip()
            plain = match.group(2).strip()
            pairs.append((cipher, plain))
        elif 'decrypt' in line.lower() or 'text:' in line.lower():
            query = re.search(r'(?:decrypt|text)[:\s]+(.*?)(?:\s*$)', line, re.IGNORECASE)
            if query:
                query = query.group(1).strip()
    return pairs, query

# Analyze 15 examples
add("### Sample Analysis of 15 Encryption Puzzles\n")

all_mappings = defaultdict(list)  # cipher_char -> [plain_chars]
shift_values = []

enc_samples = cats['encryption'].sample(min(15, len(cats['encryption'])), random_state=42)

for idx, row in enc_samples.iterrows():
    pairs, query = parse_encryption_puzzle(row['prompt'])
    if not pairs:
        continue
    
    add(f"#### Puzzle {row['id']}\n")
    add(f"  Query: '{query}' -> '{row['answer']}'\n")
    
    # Build mapping for this puzzle
    puzzle_map = {}
    for cipher, plain in pairs:
        # Map letter by letter preserving position
        if len(cipher) == len(plain):
            for c, p in zip(cipher, plain):
                if c != ' ':
                    if c in puzzle_map and puzzle_map[c] != p:
                        add(f"  WARNING: Inconsistent mapping '{c}' -> '{puzzle_map[c]}' vs '{p}'")
                    puzzle_map[c] = p
    
    # Check if it's a shift cipher
    shifts = []
    for c, p in puzzle_map.items():
        if c.isalpha() and p.isalpha():
            shift = (ord(p.lower()) - ord(c.lower())) % 26
            shifts.append(shift)
    
    if shifts and len(set(shifts)) == 1:
        shift_val = shifts[0]
        shift_values.append(shift_val)
        add(f"  **SHIFT CIPHER with shift = {shift_val}**")
        add(f"  Mapping: {dict(list(puzzle_map.items())[:10])}")
    else:
        add(f"  **SUBSTITUTION CIPHER** (not a simple shift)")
        add(f"  Shifts found: {set(shifts) if shifts else 'none'}")
        add(f"  Mapping: {dict(list(puzzle_map.items())[:15])}")
    
    # Verify answer
    if query:
        decrypted = ''.join(puzzle_map.get(c, '?') for c in query if c != ' ')
        add(f"  Decrypted query (using map): '{decrypted}'")
        add(f"  Expected answer: '{row['answer']}'")
        if decrypted.replace(' ', '') == row['answer'].replace(' ', ''):
            add(f"  ✓ Match!")
        else:
            add(f"  ✗ Mismatch")
    add("")

# Overall statistics
add(f"### Overall Encryption Statistics\n")
add(f"Analyzed shift ciphers: {len(shift_values)}")
shift_dist = Counter(shift_values)
add(f"Shift value distribution: {dict(shift_dist.most_common())}")

# Count shift vs substitution across all puzzles
shift_count = 0
sub_count = 0
for idx, row in cats['encryption'].iterrows():
    pairs, query = parse_encryption_puzzle(row['prompt'])
    if not pairs:
        continue
    puzzle_map = {}
    for cipher, plain in pairs:
        if len(cipher) == len(plain):
            for c, p in zip(cipher, plain):
                if c != ' ':
                    puzzle_map[c] = p
    shifts = []
    for c, p in puzzle_map.items():
        if c.isalpha() and p.isalpha():
            shift = (ord(p.lower()) - ord(c.lower())) % 26
            shifts.append(shift)
    if shifts and len(set(shifts)) == 1:
        shift_count += 1
    else:
        sub_count += 1

add(f"\nShift ciphers: {shift_count} ({100*shift_count/len(cats['encryption']):.1f}%)")
add(f"Substitution ciphers: {sub_count} ({100*sub_count/len(cats['encryption']):.1f}%)")

# Detailed word analysis
add("\n### Word Frequency Analysis (Plaintext)\n")
word_freq = Counter()
for idx, row in cats['encryption'].iterrows():
    pairs, query = parse_encryption_puzzle(row['prompt'])
    if pairs:
        for cipher, plain in pairs:
            for word in plain.split():
                word_freq[word.lower()] += 1

add(f"Top 30 plaintext words: {word_freq.most_common(30)}")

# ==============================================================================
# 3. BASE CONVERSION ANALYSIS
# ==============================================================================
section("3. BASE CONVERSION ANALYSIS")

add(f"Total examples: {len(cats['base_conversion'])}\n")

def parse_base_puzzle(prompt):
    """Extract number -> numeral pairs and query."""
    lines = prompt.strip().split('\n')
    pairs = []
    query = None
    for line in lines:
        match = re.match(r'\s*(\d+)\s*->\s*(.*?)\s*$', line)
        if match:
            pairs.append((int(match.group(1)), match.group(2).strip()))
        elif 'write' in line.lower() or 'number' in line.lower():
            q_match = re.search(r'number\s+(\d+)', line, re.IGNORECASE)
            if q_match:
                query = int(q_match.group(1))
    return pairs, query

def int_to_roman(n):
    """Convert integer to Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    sym = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    roman = ''
    for i in range(len(val)):
        while n >= val[i]:
            roman += sym[i]
            n -= val[i]
    return roman

# Verify roman numeral mapping
add("### Verifying Roman Numeral Mapping\n")
roman_correct = 0
roman_wrong = 0
for idx, row in cats['base_conversion'].head(50).iterrows():
    pairs, query = parse_base_puzzle(row['prompt'])
    if not pairs:
        continue
    for num, numeral in pairs:
        expected_roman = int_to_roman(num)
        if numeral == expected_roman:
            roman_correct += 1
        else:
            roman_wrong += 1

add(f"Roman numeral check (first 50 puzzles): {roman_correct} correct, {roman_wrong} wrong\n")

# Sample analysis
add("### Sample Analysis of 15 Base Conversion Puzzles\n")
base_samples = cats['base_conversion'].sample(min(15, len(cats['base_conversion'])), random_state=42)

for idx, row in base_samples.iterrows():
    pairs, query = parse_base_puzzle(row['prompt'])
    if not pairs:
        continue
    
    add(f"#### Puzzle {row['id']}\n")
    add(f"  Pairs: {pairs[:6]}")
    add(f"  Query: {query} -> {row['answer']}")
    
    # Check if all are Roman numerals
    all_roman = all(p[1] == int_to_roman(p[0]) for p in pairs)
    if all_roman:
        add(f"  **All standard Roman numerals** ✓")
    else:
        add(f"  **Non-standard numerals detected**")
        for num, numeral in pairs:
            expected = int_to_roman(num)
            if numeral != expected:
                add(f"    {num}: got '{numeral}', expected '{expected}'")
    
    # Check query answer
    if query:
        expected = int_to_roman(query)
        add(f"  Expected Roman: {expected}, Answer: {row['answer']}, Match: {expected == row['answer']}")
    add("")

# Full verification across all puzzles
add(f"### Full Base Conversion Verification\n")
full_correct = 0
full_wrong = 0
wrong_examples = []
for idx, row in cats['base_conversion'].iterrows():
    pairs, query = parse_base_puzzle(row['prompt'])
    if not pairs:
        continue
    all_match = all(p[1] == int_to_roman(p[0]) for p in pairs)
    if all_match:
        full_correct += 1
    else:
        full_wrong += 1
        if len(wrong_examples) < 5:
            wrong_examples.append((row['id'], pairs))

add(f"All standard Roman: {full_correct} ({100*full_correct/len(cats['base_conversion']):.1f}%)")
add(f"Non-standard: {full_wrong} ({100*full_wrong/len(cats['base_conversion']):.1f}%)")
if wrong_examples:
    add("\nNon-standard examples:")
    for eid, pairs in wrong_examples:
        add(f"  {eid}: {pairs[:5]}")

# Range analysis
add("\n### Number Range Analysis\n")
all_nums = []
for idx, row in cats['base_conversion'].iterrows():
    pairs, query = parse_base_puzzle(row['prompt'])
    if pairs:
        for num, numeral in pairs:
            all_nums.append(num)
        if query:
            all_nums.append(query)

if all_nums:
    add(f"Min: {min(all_nums)}, Max: {max(all_nums)}, Mean: {sum(all_nums)/len(all_nums):.1f}")

# ==============================================================================
# 4. UNIT CONVERSION ANALYSIS
# ==============================================================================
section("4. UNIT CONVERSION ANALYSIS")

add(f"Total examples: {len(cats['unit_conversion'])}\n")

def parse_unit_puzzle(prompt):
    """Extract input_value -> output_value pairs and query."""
    lines = prompt.strip().split('\n')
    pairs = []
    query = None
    for line in lines:
        match = re.match(r'\s*([\d.]+)\s*m\s+becomes\s+([\d.]+)\s*$', line)
        if match:
            pairs.append((float(match.group(1)), float(match.group(2))))
        elif 'convert' in line.lower():
            q_match = re.search(r'([\d.]+)\s*m', line)
            if q_match:
                query = float(q_match.group(1))
    return pairs, query

add("### Sample Analysis of 20 Unit Conversion Puzzles\n")

conversion_factors = []
unit_samples = cats['unit_conversion'].sample(min(20, len(cats['unit_conversion'])), random_state=42)

for idx, row in unit_samples.iterrows():
    pairs, query = parse_unit_puzzle(row['prompt'])
    if not pairs:
        continue
    
    add(f"#### Puzzle {row['id']}\n")
    add(f"  Pairs: {pairs}")
    add(f"  Query: {query} m -> {row['answer']}")
    
    # Calculate ratios
    ratios = [out/inp for inp, out in pairs if inp != 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    ratio_std = (sum((r - avg_ratio)**2 for r in ratios) / len(ratios))**0.5 if ratios else 0
    
    conversion_factors.append(avg_ratio)
    add(f"  Ratios (out/in): {[f'{r:.6f}' for r in ratios]}")
    add(f"  Average ratio: {avg_ratio:.6f} ± {ratio_std:.6f}")
    
    # Verify answer
    if query:
        predicted = query * avg_ratio
        expected = float(row['answer'])
        add(f"  Predicted: {predicted:.2f}, Expected: {expected}, Error: {abs(predicted - expected):.4f}")
    add("")

# Overall statistics
add(f"### Overall Unit Conversion Statistics\n")
add(f"Total puzzles with conversion factors: {len(conversion_factors)}")
add(f"Factor range: {min(conversion_factors):.6f} to {max(conversion_factors):.6f}")
add(f"Mean factor: {sum(conversion_factors)/len(conversion_factors):.6f}")

# Distribution of factors
factor_buckets = defaultdict(int)
for f in conversion_factors:
    bucket = round(f, 2)
    factor_buckets[bucket] += 1

add(f"\nFactor distribution (rounded to 0.01):")
for bucket, count in sorted(factor_buckets.items(), key=lambda x: -x[1])[:20]:
    add(f"  {bucket}: {count}")

# Check if factors cluster around known conversions
add(f"\nKnown conversion factors for reference:")
add(f"  meters to feet: 3.28084")
add(f"  meters to yards: 1.09361")
add(f"  km to miles: 0.62137")
add(f"  kg to pounds: 2.20462")
add(f"  meters to feet (reversed): {1/3.28084:.6f}")
add(f"  feet to meters: 0.30480")

# Check how many are close to common factors
add(f"\nProximity to common factors:")
for target_name, target_val in [("m->ft", 3.28084), ("m->yd", 1.09361), ("ft->m", 0.3048), ("in->cm", 2.54)]:
    close = sum(1 for f in conversion_factors if abs(f - target_val) < 0.01)
    add(f"  Near {target_name} ({target_val}): {close}")

# ==============================================================================
# 5. GRAVITATIONAL ANALYSIS
# ==============================================================================
section("5. GRAVITATIONAL ANALYSIS")

add(f"Total examples: {len(cats['gravitational'])}\n")

def parse_gravity_puzzle(prompt):
    """Extract t -> d pairs and query t."""
    pairs = []
    query_t = None
    for line in prompt.split('\n'):
        match = re.match(r'\s*For t\s*=\s*([\d.]+)s,\s*distance\s*=\s*([\d.]+)\s*m', line)
        if match:
            pairs.append((float(match.group(1)), float(match.group(2))))
        elif 'determine' in line.lower():
            q_match = re.search(r't\s*=\s*([\d.]+)s', line)
            if q_match:
                query_t = float(q_match.group(1))
    return pairs, query_t

add("### Sample Analysis of 20 Gravitational Puzzles\n")

g_values = []
g_consistency = []

grav_samples = cats['gravitational'].sample(min(20, len(cats['gravitational'])), random_state=42)

for idx, row in grav_samples.iterrows():
    pairs, query_t = parse_gravity_puzzle(row['prompt'])
    if not pairs:
        continue
    
    add(f"#### Puzzle {row['id']}\n")
    add(f"  t->d pairs: {pairs}")
    add(f"  Query t: {query_t}s -> answer: {row['answer']}")
    
    # Calculate g for each pair: d = 0.5 * g * t^2 => g = 2*d / t^2
    g_vals = []
    for t, d in pairs:
        if t > 0:
            g = 2 * d / (t ** 2)
            g_vals.append(g)
    
    if g_vals:
        avg_g = sum(g_vals) / len(g_vals)
        g_std = (sum((g - avg_g)**2 for g in g_vals) / len(g_vals))**0.5
        g_values.append(avg_g)
        g_consistency.append(g_std / avg_g if avg_g > 0 else float('inf'))
        
        add(f"  g values: {[f'{g:.4f}' for g in g_vals]}")
        add(f"  Average g: {avg_g:.4f} ± {g_std:.4f} (CV: {100*g_std/avg_g:.2f}%)")
        
        # Verify answer
        if query_t:
            predicted_d = 0.5 * avg_g * (query_t ** 2)
            expected_d = float(row['answer'])
            add(f"  Predicted d: {predicted_d:.2f}, Expected: {expected_d}, Error: {abs(predicted_d - expected_d):.4f}")
    add("")

# Overall statistics
add(f"### Overall Gravitational Statistics\n")
add(f"Total puzzles analyzed: {len(g_values)}")
add(f"g value range: {min(g_values):.4f} to {max(g_values):.4f}")
add(f"Mean g: {sum(g_values)/len(g_values):.4f}")
add(f"Earth's g = 9.80665 m/s²")
add(f"Mean CV (coefficient of variation): {sum(g_consistency)/len(g_consistency)*100:.2f}%")

# Distribution
g_buckets = defaultdict(int)
for g in g_values:
    bucket = round(g, 1)
    g_buckets[bucket] += 1

add(f"\ng value distribution (rounded to 0.1):")
for bucket, count in sorted(g_buckets.items(), key=lambda x: -x[1])[:20]:
    add(f"  {bucket}: {count}")

# ==============================================================================
# 6. CRYPTARITHM ANALYSIS
# ==============================================================================
section("6. CRYPTARITHM ANALYSIS")

add(f"Total examples: {len(cats['cryptarithm'])}\n")

def parse_cryptarithm_puzzle(prompt):
    """Extract equation pairs and query."""
    lines = prompt.strip().split('\n')
    pairs = []
    query = None
    for line in lines:
        match = re.match(r'\s*(.*?)\s*=\s*(.*?)\s*$', line)
        if match:
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            if lhs and rhs and 'Alice' not in lhs and ' Wonderland' not in lhs and 'examples' not in lhs:
                pairs.append((lhs, rhs))
        elif 'determine' in line.lower() or 'result' in line.lower():
            q_match = re.search(r'(?:for|:)\s*(.+?)(?:\s*$)', line)
            if q_match:
                query = q_match.group(1).strip()
    return pairs, query

# Analyze different subtypes
add("### Subtype Analysis\n")

subtype_counts = defaultdict(int)
subtype_examples = defaultdict(list)

for idx, row in cats['cryptarithm'].head(200).iterrows():
    pairs, query = parse_cryptarithm_puzzle(row['prompt'])
    if not pairs:
        continue
    
    # Determine subtype
    sample_lhs = pairs[0][0]
    
    # Check if it uses only printable ASCII symbols
    has_letters = any(c.isalpha() for c in sample_lhs)
    has_digits = any(c.isdigit() for c in sample_lhs)
    has_symbols = any(not c.isalnum() and c not in ' +-*/' for c in sample_lhs)
    has_operators = any(c in '+-*/|&^' for c in sample_lhs)
    
    if has_digits and has_operators:
        subtype = 'digit_operators'
    elif has_letters:
        subtype = 'letter_substitution'
    elif has_symbols and not has_digits and not has_letters:
        subtype = 'symbol_substitution'
    elif has_digits and not has_operators:
        subtype = 'digit_equation'
    else:
        subtype = 'other'
    
    subtype_counts[subtype] += 1
    if len(subtype_examples[subtype]) < 5:
        subtype_examples[subtype].append((row['id'], pairs, query, row['answer']))

add("Subtypes found in first 200:")
for st, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
    add(f"  {st}: {count}")

# Show examples of each subtype
for st, examples in subtype_examples.items():
    add(f"\n### {st.upper()} Examples\n")
    for eid, pairs, query, answer in examples:
        add(f"  Puzzle {eid}:")
        add(f"    Pairs: {pairs[:5]}")
        add(f"    Query: '{query}' -> '{answer}'")
        add("")

# Deeper analysis of digit_operators subtype
add("\n### Deep Analysis: Digit+Operators Subtype\n")

digit_op_count = 0
for idx, row in cats['cryptarithm'].iterrows():
    pairs, query = parse_cryptarithm_puzzle(row['prompt'])
    if not pairs:
        continue
    sample_lhs = pairs[0][0]
    has_digits = any(c.isdigit() for c in sample_lhs)
    has_operators = any(c in '+-*/|&^' for c in sample_lhs)
    
    if has_digits and has_operators:
        digit_op_count += 1
        if digit_op_count <= 5:
            add(f"\n  Puzzle {row['id']}:")
            add(f"    Pairs: {pairs}")
            add(f"    Query: '{query}' -> '{row['answer']}'")
            
            # Try to figure out the operator mapping
            # Parse LHS as expression with digits and a single operator
            for lhs, rhs in pairs[:3]:
                # Find the operator
                op_match = re.match(r'(\d+)([+\-*/|&^\\])(\d+)', lhs)
                if op_match:
                    a, op, b = int(op_match.group(1)), op_match.group(2), int(op_match.group(3))
                    rhs_val = rhs
                    try:
                        rhs_int = int(rhs)
                    except:
                        try:
                            rhs_int = float(rhs)
                        except:
                            rhs_int = None
                    
                    if rhs_int is not None:
                        add(f"      {a} {op} {b} = {rhs_int}")
                        # Check known operations
                        results = []
                        results.append(f"+:{a+b}", )
                        results.append(f"-:{a-b}")
                        results.append(f"*:{a*b}")
                        if b != 0:
                            results.append(f"/:{a/b}")
                        results_str = ", ".join(results[:4])
                        add(f"      Known: {results_str}")

add(f"\n  Total digit+operator puzzles: {digit_op_count}")

# Analyze symbol substitution subtype more deeply
add("\n### Deep Analysis: Symbol Substitution Subtype\n")

sym_count = 0
for idx, row in cats['cryptarithm'].iterrows():
    pairs, query = parse_cryptarithm_puzzle(row['prompt'])
    if not pairs:
        continue
    sample_lhs = pairs[0][0]
    has_letters = any(c.isalpha() for c in sample_lhs)
    has_digits = any(c.isdigit() for c in sample_lhs)
    has_symbols = any(not c.isalnum() and c not in ' +-*/' for c in sample_lhs)
    
    if has_symbols and not has_digits and not has_letters:
        sym_count += 1
        if sym_count <= 5:
            add(f"\n  Puzzle {row['id']}:")
            add(f"    Pairs: {pairs[:5]}")
            add(f"    Query: '{query}' -> '{row['answer']}")
            
            # Check if it's a simple character-to-character mapping
            # Try to build a mapping from input chars to output chars
            char_map = {}
            consistent = True
            for lhs, rhs in pairs:
                if len(lhs) == len(rhs):
                    for c1, c2 in zip(lhs, rhs):
                        if c1 in char_map:
                            if char_map[c1] != c2:
                                consistent = False
                        else:
                            char_map[c1] = c2
            
            add(f"    Mapping (consistent={consistent}): {dict(list(char_map.items())[:10])}")
            
            # Try applying mapping to query
            if query and consistent:
                result = ''.join(char_map.get(c, '?') for c in query)
                add(f"    Applied mapping to query: '{result}' vs answer '{row['answer']}'")

add(f"\n  Total symbol substitution puzzles: {sym_count}")

# Full count of consistent symbol mappings
add("\n### Symbol Substitution: Consistency Check\n")
consistent_count = 0
inconsistent_count = 0
for idx, row in cats['cryptarithm'].iterrows():
    pairs, query = parse_cryptarithm_puzzle(row['prompt'])
    if not pairs:
        continue
    sample_lhs = pairs[0][0]
    has_letters = any(c.isalpha() for c in sample_lhs)
    has_digits = any(c.isdigit() for c in sample_lhs)
    has_symbols = any(not c.isalnum() and c not in ' +-*/' for c in sample_lhs)
    
    if has_symbols and not has_digits and not has_letters:
        char_map = {}
        consistent = True
        for lhs, rhs in pairs:
            if len(lhs) == len(rhs):
                for c1, c2 in zip(lhs, rhs):
                    if c1 in char_map:
                        if char_map[c1] != c2:
                            consistent = False
                            break
                    else:
                        char_map[c1] = c2
                if not consistent:
                    break
        
        if consistent:
            consistent_count += 1
        else:
            inconsistent_count += 1

add(f"Consistent mappings: {consistent_count}")
add(f"Inconsistent mappings: {inconsistent_count}")

# ==============================================================================
# SUMMARY
# ==============================================================================
section("SUMMARY AND SOLVER RECOMMENDATIONS")

add("""
## Category Summary

### 1. Bit Manipulation (1602 puzzles)
- Most puzzles use a CONSISTENT XOR mask across all example pairs
- The XOR mask is the same for every input->output pair within a puzzle
- Solver strategy: Extract any one pair, compute XOR mask = input XOR output, apply to query
- Some puzzles may use other operations (NOT, rotations, reversals) but XOR is dominant
- Output can be 7 or 8 bits (leading zero may be dropped)

### 2. Encryption (1576 puzzles)
- Two subtypes: Simple shift ciphers and general substitution ciphers
- For shift ciphers: identify the shift from any letter pair, apply uniformly
- For substitution ciphers: build character mapping from all examples, apply to query
- Space characters are preserved (not encrypted)
- Solver strategy: Build mapping from examples, check if shift, apply to query

### 3. Base Conversion (3180 puzzles, but some may overlap with other categories)
- Standard Roman numeral conversion from Arabic integers
- All examples verified: input is integer, output is Roman numeral
- Solver strategy: Use standard int_to_roman conversion
- Range typically 1-100+

### 4. Unit Conversion (1594 puzzles)
- Each puzzle has a CONSISTENT conversion factor (ratio = output/input)
- Factor varies between puzzles (no universal factor)
- Solver strategy: Calculate ratio from any example pair, multiply query by ratio
- Factor precision matters - use average across all pairs for accuracy

### 5. Gravitational (1597 puzzles)
- Uses d = 0.5 * g * t^2 with a non-standard g value per puzzle
- g is consistent within each puzzle (low CV)
- Solver strategy: Calculate g = 2*d/t^2 from examples, average them, apply to query t
- g values vary widely across puzzles

### 6. Cryptarithm (1555 puzzles)
- Multiple subtypes:
  a) Symbol substitution: character-to-character mapping (consistent)
  b) Digit + operator: modified arithmetic (operator substitution)
  c) Letter substitution: letter-to-digit or letter-to-letter mapping
- Solver strategy depends on subtype:
  a) Build char map from examples, apply to query
  b) Figure out operator semantics from examples, evaluate query
  c) Similar to symbol substitution but with letters

## Key Insights for Building Solvers

1. **Bit manipulation**: XOR mask extraction is the primary pattern. Handle 7/8-bit variations.
2. **Encryption**: Check shift first (faster), fall back to substitution mapping.
3. **Base conversion**: Standard Roman numeral converter.
4. **Unit conversion**: Simple ratio multiplication.
5. **Gravitational**: Linear regression or average g estimation.
6. **Cryptarithm**: Most complex - need to identify subtype and apply appropriate logic.
""")

# Write output
with open('analysis/deep_analysis.md', 'w') as f:
    f.write('\n'.join(output_parts))

print("\n\nAnalysis complete! Results saved to analysis/deep_analysis.md")
print(f"Total output length: {len('\\n'.join(output_parts))} characters")
