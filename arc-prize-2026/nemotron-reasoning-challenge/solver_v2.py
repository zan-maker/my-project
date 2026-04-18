"""
Nemotron Reasoning Challenge - Complete Solver v2
Fixed: bit manipulation string/int bug, rounding, cryptarithm symbol handling, encryption vocabulary
"""

import csv
import re
import os
import sys
import math
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ============================================================================
# CATEGORY 1: BASE CONVERSION (Roman Numerals) - 100%
# ============================================================================

def int_to_roman(num: int) -> str:
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    roman = ''
    for i, v in enumerate(val):
        while num >= v:
            roman += syms[i]
            num -= v
    return roman

def solve_base_conversion(prompt: str) -> str:
    # Find the query number
    match = re.search(r'(?:write the number|convert the number|number)\s+(\d+)', prompt, re.IGNORECASE)
    if not match:
        match = re.search(r'Now.*?(\d+)\s*(?:in the|in this|using)', prompt, re.IGNORECASE)
    if not match:
        numbers = re.findall(r'\b(\d+)\b', prompt)
        if numbers:
            return int_to_roman(int(numbers[-1]))
    if match:
        return int_to_roman(int(match.group(1)))
    return ""

# ============================================================================
# CATEGORY 2: UNIT CONVERSION - Fixed rounding
# ============================================================================

def smart_round(val: float, decimals: int = 2) -> str:
    """Round to specified decimals, matching the answer format exactly."""
    rounded = round(val, decimals)
    # Format without trailing zeros for cases like 45.0 -> "45.0" not "45.00"
    if decimals == 2:
        formatted = f"{rounded:.2f}"
    else:
        formatted = f"{rounded:.{decimals}f}"
    return formatted

def solve_unit_conversion(prompt: str) -> str:
    lines = prompt.strip().split('\n')
    pairs = []
    query_val = None
    
    for line in lines:
        m1 = re.search(r'([\d.]+)\s*m\s*(?:becomes|->|=)\s*([\d.]+)', line)
        if m1:
            pairs.append((float(m1.group(1)), float(m1.group(2))))
        m2 = re.search(r'convert.*?:\s*([\d.]+)\s*m', line, re.IGNORECASE)
        if m2:
            query_val = float(m2.group(1))
    
    if not pairs or query_val is None:
        return ""
    
    # Calculate median ratio for robustness
    ratios = sorted([out_val / in_val for in_val, out_val in pairs])
    if len(ratios) % 2 == 1:
        avg_ratio = ratios[len(ratios) // 2]
    else:
        avg_ratio = (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2
    
    result = query_val * avg_ratio
    
    # Try multiple rounding approaches and pick the one that best matches example precision
    # First check: do examples use standard .XX format?
    has_zero = any(str(p[1]).endswith('.0') for p in pairs)
    
    if has_zero:
        # Check if result is close to an integer
        if abs(result - round(result)) < 0.05:
            return f"{round(result):.1f}"
    
    return smart_round(result)

# ============================================================================
# CATEGORY 3: GRAVITATIONAL - Fixed rounding and parsing
# ============================================================================

def solve_gravitational(prompt: str) -> str:
    lines = prompt.strip().split('\n')
    g_values = []
    query_t = None
    
    for line in lines:
        m1 = re.search(r'For\s+t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)', line, re.IGNORECASE)
        if m1:
            t = float(m1.group(1))
            d = float(m1.group(2))
            if t > 0:
                g = 2 * d / (t ** 2)
                g_values.append(g)
        m2 = re.search(r'(?:Now|determine|for)\s+.*?t\s*=\s*([\d.]+)', line, re.IGNORECASE)
        if m2:
            query_t = float(m2.group(1))
    
    if not g_values or query_t is None:
        return ""
    
    # Use median g for robustness
    g_values.sort()
    if len(g_values) % 2 == 1:
        avg_g = g_values[len(g_values) // 2]
    else:
        avg_g = (g_values[len(g_values) // 2 - 1] + g_values[len(g_values) // 2]) / 2
    
    d = 0.5 * avg_g * (query_t ** 2)
    return smart_round(d)

# ============================================================================
# CATEGORY 4: ENCRYPTION - Expanded vocabulary and multi-word matching
# ============================================================================

ENCRYPTION_VOCAB = set([
    # Articles/determiners
    'the', 'a', 'an',
    # Nouns (subjects)
    'knight', 'hatter', 'student', 'wizard', 'queen', 'alice', 'rabbit',
    'turtle', 'dragon', 'mouse', 'cat', 'king', 'princess', 'bird', 'teacher',
    # Verbs
    'dreams', 'creates', 'found', 'studies', 'draws', 'sees', 'reads',
    'writes', 'imagines', 'chases', 'follows', 'watches', 'discovers',
    'explores', 'finds', 'catches', 'hides', 'paints', 'sings', 'dances',
    'opens', 'closes', 'builds', 'breaks', 'climbs', 'flies', 'runs',
    'jumps', 'swims', 'sleeps', 'eats', 'drinks', 'gives', 'takes',
    'makes', 'loves', 'hates', 'helps', 'fights', 'wins', 'loses',
    'knows', 'thinks', 'feels', 'says', 'tells', 'asks', 'answers',
    'calls', 'names', 'shows', 'teaches', 'learns', 'plays', 'works',
    'wakes', 'smiles', 'laughs', 'cries', 'listens', 'speaks', 'looks',
    'waits', 'searches', 'collects', 'grows', 'changes', 'moves', 'touches',
    'holds', 'carries', 'brings', 'sends', 'keeps', 'leaves', 'starts',
    'stops', 'falls', 'rises', 'shines', 'burns', 'breaks', 'fixes',
    # Adjectives
    'strange', 'hidden', 'wise', 'silver', 'dark', 'colorful', 'curious',
    'bright', 'ancient', 'magical', 'mysterious', 'golden', 'tiny', 'huge',
    'fast', 'slow', 'old', 'new', 'big', 'small', 'long', 'short',
    'happy', 'sad', 'brave', 'kind', 'cold', 'hot', 'red', 'blue',
    'green', 'white', 'black', 'quiet', 'loud', 'gentle', 'wild',
    'sweet', 'bitter', 'rich', 'poor', 'strong', 'weak', 'young',
    # Locations
    'forest', 'garden', 'castle', 'river', 'mountain', 'cave', 'tower',
    'village', 'lake', 'bridge', 'door', 'room', 'house', 'path', 'road',
    'sky', 'ground', 'tree', 'flower', 'stone', 'wall', 'island', 'sea',
    'ocean', 'desert', 'field', 'meadow', 'hill', 'valley', 'pond',
    # Other words
    'secret', 'key', 'crystal', 'puzzle', 'map', 'book', 'treasure',
    'in', 'near', 'behind', 'above', 'below', 'beside', 'inside', 'outside',
    'under', 'over', 'through', 'with', 'from', 'to', 'at', 'by',
    'and', 'but', 'or', 'not', 'no', 'yes', 'all', 'every', 'each',
    'some', 'any', 'one', 'two', 'three', 'four', 'five', 'six',
    'seven', 'eight', 'nine', 'ten',
    'first', 'last', 'next', 'other', 'another',
    'day', 'night', 'morning', 'evening', 'time',
    'man', 'woman', 'child', 'people', 'world',
    'story', 'song', 'letter', 'word', 'question', 'answer',
    'life', 'death', 'dream', 'magic', 'wonder', 'power',
    'light', 'shadow', 'fire', 'water', 'wind', 'earth',
    'star', 'moon', 'sun', 'rain', 'snow', 'cloud',
    'sword', 'shield', 'crown', 'ring', 'cloak',
    'food', 'drink', 'bread', 'fruit',
    'wonderland', 'mirror', 'clock', 'teapot', 'cake',
    'smile', 'tears', 'fear', 'hope', 'joy', 'peace',
    'truth', 'lie', 'song', 'tale', 'wish', 'spell',
    'lost', 'safe', 'free', 'alone', 'together',
    'friend', 'enemy', 'hero', 'villain',
    'left', 'right', 'up', 'down', 'back', 'away',
    'never', 'always', 'often', 'sometimes',
    'very', 'much', 'more', 'less', 'too',
    'here', 'there', 'where', 'when', 'how', 'why', 'what',
    'still', 'again', 'once', 'twice',
])

def match_word_pattern(pattern: str, vocab: set) -> Optional[str]:
    """Match a pattern like '?oo?' against vocabulary words of the same length."""
    if not pattern:
        return None
    candidates = [w for w in vocab if len(w) == len(pattern)]
    matches = []
    for word in candidates:
        ok = True
        for p_char, w_char in zip(pattern, word):
            if p_char != '?' and p_char != w_char:
                ok = False
                break
        if ok:
            matches.append(word)
    return matches[0] if len(matches) == 1 else (matches[0] if matches else None)

def solve_encryption(prompt: str) -> str:
    lines = prompt.strip().split('\n')
    
    cipher_to_plain = {}
    
    for line in lines:
        m = re.search(r'([a-z ]+?)\s*->\s*([a-z ]+)', line)
        if m:
            cipher_text = m.group(1).strip()
            plain_text = m.group(2).strip()
            
            c_words = cipher_text.split()
            p_words = plain_text.split()
            
            if len(c_words) == 1 and len(p_words) == 1:
                # Single word - map char by char
                if len(c_words[0]) == len(p_words[0]):
                    for c, p in zip(c_words[0], p_words[0]):
                        if c not in cipher_to_plain:
                            cipher_to_plain[c] = p
            elif len(c_words) == len(p_words):
                for cw, pw in zip(c_words, p_words):
                    if len(cw) == len(pw):
                        for c, p in zip(cw, pw):
                            if c not in cipher_to_plain:
                                cipher_to_plain[c] = p
    
    # Extract query
    query = None
    for line in lines:
        m = re.search(r'(?:decrypt|Now)[^:]*:\s*([a-z ]+)', line, re.IGNORECASE)
        if m:
            query = m.group(1).strip()
    if query is None:
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[a-z ]+$', line) and len(line) > 2 and '->' not in line:
                query = line
                break
    
    if query is None:
        return ""
    
    decrypted_words = []
    for cipher_word in query.split():
        partial = ''
        unknown_chars = set()
        for c in cipher_word:
            if c in cipher_to_plain:
                partial += cipher_to_plain[c]
            else:
                partial += '?'
                unknown_chars.add(c)
        
        if '?' not in partial:
            decrypted_words.append(partial)
        else:
            matched = match_word_pattern(partial, ENCRYPTION_VOCAB)
            if matched:
                # Update cipher mapping for newly discovered chars
                for c, m_char in zip(cipher_word, matched):
                    if c not in cipher_to_plain:
                        cipher_to_plain[c] = m_char
                decrypted_words.append(matched)
            else:
                # Try to use character frequency analysis
                decrypted_words.append(partial)
    
    return ' '.join(decrypted_words)

# ============================================================================
# CATEGORY 5: CRYPTARITHM - Improved operator identification
# ============================================================================

def solve_cryptarithm(prompt: str) -> str:
    lines = prompt.strip().split('\n')
    
    examples = []
    query_expr = None
    is_symbol_sub = False
    
    for line in lines:
        line = line.strip()
        m = re.search(r'(.+?)\s*=\s*(.+)', line)
        if not m:
            continue
        
        left = m.group(1).strip()
        right = m.group(2).strip()
        
        # Check for symbol substitution (non-digit characters on both sides)
        if not re.match(r'^[\d]+$', left) and not re.match(r'^[\d]+$', right):
            # Symbol substitution puzzle
            op_match = re.search(r'([\d]+)\s*([^\d\s\w]+)\s*([\d]+)', left)
            if not op_match:
                is_symbol_sub = True
                # Store character-level mapping
                cipher = left
                plain = right
                examples.append(('sym', cipher, plain))
                if 'Now' in line or 'determine' in line:
                    query_expr = ('sym_query', cipher)
                continue
        
        op_match = re.search(r'([\d]+)\s*([^\d\s\w]+)\s*([\d]+)', left)
        if op_match:
            a = int(op_match.group(1))
            op = op_match.group(2)
            b = int(op_match.group(3))
            
            if re.match(r'^[\d]+$', right):
                result = int(right)
                examples.append(('op', a, op, b, result))
                if 'Now' in line or 'determine' in line:
                    query_expr = ('op_query', a, op, b)
            else:
                # Non-numeric result - could be symbol substitution
                examples.append(('sym', left, right))
                if 'Now' in line or 'determine' in line:
                    query_expr = ('sym_query', left)
    
    if not examples:
        return ""
    
    # Handle symbol substitution
    sym_examples = [(c, p) for e in examples if e[0] == 'sym' for c, p in [(e[1], e[2])]]
    op_examples = [e for e in examples if e[0] == 'op']
    
    # Check if query is symbol substitution
    if query_expr and query_expr[0] == 'sym_query':
        char_map = {}
        for cipher, plain in sym_examples:
            if len(cipher) == len(plain):
                for c, p in zip(cipher, plain):
                    if c not in char_map:
                        char_map[c] = p
        
        query_cipher = query_expr[1]
        result = ''
        for c in query_cipher:
            result += char_map.get(c, '?')
        return result
    
    # Handle operator puzzles
    if op_examples and query_expr and query_expr[0] == 'op_query':
        operators = set(e[2] for e in op_examples)
        
        ALL_OPS = [
            ('add', lambda a, b: a + b),
            ('sub', lambda a, b: a - b),
            ('abs_sub', lambda a, b: abs(a - b)),
            ('mul', lambda a, b: a * b),
            ('mul_minus_1', lambda a, b: a * b - 1),
            ('add_minus_1', lambda a, b: a + b - 1),
            ('add_plus_1', lambda a, b: a + b + 1),
            ('mul_plus_1', lambda a, b: a * b + 1),
            ('floordiv', lambda a, b: a // b if b != 0 else 999999),
            ('mod', lambda a, b: a % b if b != 0 else 999999),
            ('concat', lambda a, b: int(str(a) + str(b))),
            ('reverse_concat', lambda a, b: int(str(b) + str(a))),
            ('max', lambda a, b: max(a, b)),
            ('min', lambda a, b: min(a, b)),
            ('bit_and', lambda a, b: a & b),
            ('bit_or', lambda a, b: a | b),
            ('bit_xor', lambda a, b: a ^ b),
            ('lshift', lambda a, b: a << b if b < 10 else 999999),
            ('mul_sub_a', lambda a, b: a * b - a),
            ('mul_sub_b', lambda a, b: a * b - b),
            ('a_sq_plus_b', lambda a, b: a * a + b),
            ('a_plus_b_sq', lambda a, b: a + b * b),
            ('sub_then_mul', lambda a, b: (a - b) * b if a >= b else (b - a) * a),
            ('double', lambda a, b: 2 * a),
        ]
        
        op_meanings = {}
        for op in operators:
            op_exs = [(a, b, r) for _, a, o, b, r in op_examples if o == op]
            if not op_exs:
                continue
            
            best_ops = []
            for op_name, op_func in ALL_OPS:
                consistent = True
                for a, b, expected in op_exs:
                    try:
                        result = op_func(a, b)
                        if result != expected:
                            consistent = False
                            break
                    except:
                        consistent = False
                        break
                if consistent:
                    best_ops.append((op_name, op_func))
            
            if best_ops:
                priority = ['add', 'sub', 'mul', 'floordiv', 'mod', 'concat',
                           'abs_sub', 'add_minus_1', 'mul_minus_1', 'pow',
                           'reverse_concat', 'bit_and', 'bit_or', 'bit_xor',
                           'max', 'min', 'lshift']
                op_meanings[op] = None
                for p in priority:
                    for name, func in best_ops:
                        if name == p:
                            op_meanings[op] = func
                            break
                    if op_meanings[op] is not None:
                        break
                if op_meanings[op] is None:
                    op_meanings[op] = best_ops[0][1]
        
        _, a, op, b = query_expr
        if op in op_meanings:
            try:
                return str(op_meanings[op](a, b))
            except:
                pass
    
    return ""

# ============================================================================
# CATEGORY 6: BIT MANIPULATION - Fixed int conversion
# ============================================================================

def solve_bit_manipulation(prompt: str) -> str:
    lines = prompt.strip().split('\n')
    
    examples = []
    query = None
    
    for line in lines:
        m = re.search(r'([01]{8})\s*->\s*([01]{8})', line)
        if m:
            examples.append((m.group(1), m.group(2)))
        m2 = re.search(r'(?:for|determine|output for):\s*([01]{8})', line, re.IGNORECASE)
        if m2:
            query = m2.group(1)
    
    if not examples or query is None:
        return ""
    
    query_bits = [int(b) for b in query]
    num_bits = 8
    result_bits = []
    
    for out_pos in range(num_bits):
        # Build truth table: list of (input_bits_tuple, expected_output_bit)
        tt = []
        for inp, out in examples:
            input_bits = tuple(int(b) for b in inp)
            output_bit = int(out[out_pos])
            tt.append((input_bits, output_bit))
        
        found = False
        
        # Try constant
        all_same = all(tt[i][1] == tt[0][1] for i in range(len(tt)))
        if all_same:
            result_bits.append(str(tt[0][1]))
            found = True
        
        if found:
            continue
        
        # Try single-bit identity
        for in_pos in range(num_bits):
            if all(tt[i][0][in_pos] == tt[i][1] for i in range(len(tt))):
                result_bits.append(str(query_bits[in_pos]))
                found = True
                break
        
        if found:
            continue
        
        # Try single-bit NOT
        for in_pos in range(num_bits):
            if all((1 - tt[i][0][in_pos]) == tt[i][1] for i in range(len(tt))):
                result_bits.append(str(1 - query_bits[in_pos]))
                found = True
                break
        
        if found:
            continue
        
        # Try 2-input functions
        for p1 in range(num_bits):
            for p2 in range(p1 + 1, num_bits):
                ops_2 = [
                    lambda a, b: a ^ b,           # XOR
                    lambda a, b: a & b,           # AND
                    lambda a, b: a | b,           # OR
                    lambda a, b: 1 - (a ^ b),     # XNOR
                    lambda a, b: 1 - (a & b),     # NAND
                    lambda a, b: 1 - (a | b),     # NOR
                    lambda a, b: a & (1 - b),     # AND NOT
                    lambda a, b: (1 - a) & b,     # NOT AND
                ]
                for op_func in ops_2:
                    if all(op_func(tt[i][0][p1], tt[i][0][p2]) == tt[i][1] for i in range(len(tt))):
                        result_bits.append(str(op_func(query_bits[p1], query_bits[p2])))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
        
        # Try 3-input majority and other 3-bit functions
        for p1 in range(num_bits):
            for p2 in range(p1 + 1, num_bits):
                for p3 in range(p2 + 1, num_bits):
                    # Majority
                    if all(sum([tt[i][0][p1], tt[i][0][p2], tt[i][0][p3]]) >= 2 == bool(tt[i][1]) for i in range(len(tt))):
                        s = query_bits[p1] + query_bits[p2] + query_bits[p3]
                        result_bits.append('1' if s >= 2 else '0')
                        found = True
                        break
                    # Any 2
                    if all((tt[i][0][p1] + tt[i][0][p2] + tt[i][0][p3]) >= 2 == bool(tt[i][1]) for i in range(len(tt))):
                        s = query_bits[p1] + query_bits[p2] + query_bits[p3]
                        result_bits.append('1' if s >= 2 else '0')
                        found = True
                        break
                    # All 3
                    if all(tt[i][0][p1] and tt[i][0][p2] and tt[i][0][p3] == bool(tt[i][1]) for i in range(len(tt))):
                        result_bits.append('1' if query_bits[p1] and query_bits[p2] and query_bits[p3] else '0')
                        found = True
                        break
                    # None
                    if all(not (tt[i][0][p1] or tt[i][0][p2] or tt[i][0][p3]) == bool(tt[i][1]) for i in range(len(tt))):
                        result_bits.append('1' if not (query_bits[p1] or query_bits[p2] or query_bits[p3]) else '0')
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            # Fallback: majority vote from examples
            ones = sum(1 for _, out in examples if out[out_pos] == '1')
            result_bits.append('1' if ones > len(examples) / 2 else '0')
    
    return ''.join(result_bits)

# ============================================================================
# CATEGORY DETECTION
# ============================================================================

def classify_puzzle(prompt: str) -> str:
    first_line = prompt.split('\n')[0].strip().lower()
    
    if 'bit manipulation' in first_line or 'binary' in first_line:
        return 'bit_manipulation'
    if 'cryptarithm' in first_line or ('transformation rules' in first_line and 'equation' in first_line.lower()):
        return 'cryptarithm'
    if 'encryption' in first_line:
        return 'encryption'
    if 'numeral system' in first_line or 'converted into a' in first_line:
        return 'base_conversion'
    if 'unit conversion' in first_line:
        return 'unit_conversion'
    if 'gravitational' in first_line:
        return 'gravitational'
    
    # Broader patterns
    if re.search(r'[01]{8}\s*->\s*[01]{8}', prompt):
        return 'bit_manipulation'
    if '->' in prompt and re.search(r'[a-z]{4,}\s+[a-z]+\s*->\s*[a-z]+', prompt):
        return 'encryption'
    if 'numeral' in prompt[:300].lower():
        return 'base_conversion'
    if 'becomes' in prompt[:300] and 'm' in prompt[:500]:
        return 'unit_conversion'
    if 'd = 0.5' in prompt or 'gravitational' in prompt[:500].lower():
        return 'gravitational'
    if re.search(r'\d+[^\d\w]\d+\s*=', prompt):
        return 'cryptarithm'
    
    return 'unknown'

# ============================================================================
# MAIN SOLVER
# ============================================================================

SOLVER_MAP = {
    'base_conversion': solve_base_conversion,
    'unit_conversion': solve_unit_conversion,
    'gravitational': solve_gravitational,
    'encryption': solve_encryption,
    'cryptarithm': solve_cryptarithm,
    'bit_manipulation': solve_bit_manipulation,
}

def solve_puzzle(prompt: str, verbose: bool = False) -> str:
    category = classify_puzzle(prompt)
    if verbose:
        print(f"  Category: {category}")
    
    if category in SOLVER_MAP:
        try:
            return SOLVER_MAP[category](prompt)
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return ""
    return ""

def validate_on_training(data_path: str, max_samples: int = 100):
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    correct = 0
    total = 0
    errors_by_cat = defaultdict(int)
    totals_by_cat = defaultdict(int)
    mismatches = defaultdict(list)
    
    for row in rows[:max_samples]:
        prompt = row['prompt']
        expected = row['answer']
        category = classify_puzzle(prompt)
        totals_by_cat[category] += 1
        
        predicted = solve_puzzle(prompt)
        
        if predicted.strip() == expected.strip():
            correct += 1
        else:
            errors_by_cat[category] += 1
            if len(mismatches[category]) < 5:
                mismatches[category].append((predicted, expected))
        
        total += 1
    
    print(f"\n=== VALIDATION RESULTS (first {total} examples) ===")
    print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")
    for cat in sorted(totals_by_cat.keys()):
        cc = totals_by_cat[cat] - errors_by_cat[cat]
        ct = totals_by_cat[cat]
        pct = cc / ct * 100 if ct > 0 else 0
        print(f"  {cat:20s}: {cc:3d}/{ct:3d} = {pct:5.1f}%")
    
    print("\nSample mismatches:")
    for cat, ms in mismatches.items():
        for pred, exp in ms[:2]:
            print(f"  [{cat}] pred='{pred}' exp='{exp}'")

def generate_submission(test_path: str, output_path: str):
    with open(test_path, 'r') as f:
        reader = csv.DictReader(f)
        test_rows = list(reader)
    
    results = []
    for row in test_rows:
        answer = solve_puzzle(row['prompt'], verbose=True)
        results.append({'id': row['id'], 'answer': answer})
        print(f"  => {row['id']}: {answer}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSubmission saved: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        validate_on_training(os.path.join(DATA_DIR, 'train.csv'), max_n)
    elif len(sys.argv) > 1 and sys.argv[1] == 'submit':
        generate_submission(
            os.path.join(DATA_DIR, 'test.csv'),
            os.path.join(os.path.dirname(DATA_DIR), 'submissions', 'submission.csv')
        )
    else:
        print("=== VALIDATING ===")
        validate_on_training(os.path.join(DATA_DIR, 'train.csv'), 500)
        print("\n\n=== SUBMITTING ===")
        generate_submission(
            os.path.join(DATA_DIR, 'test.csv'),
            os.path.join(os.path.dirname(DATA_DIR), 'submissions', 'submission.csv')
        )
