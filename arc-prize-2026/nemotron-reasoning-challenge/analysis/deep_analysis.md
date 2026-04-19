# Nemotron Reasoning Challenge - Deep Pattern Analysis

## Dataset Overview

- **Total puzzles**: 9,500
- **Categories**: 6 (perfectly balanced ~1,575-1,602 each)

| Category | Count | Difficulty |
|----------|-------|-----------|
| bit_manipulation | 1,602 | **Hard** |
| cryptarithm | 1,555 | **Hard** |
| encryption | 1,576 | Medium |
| base_conversion | 1,576 | **Easy** |
| unit_conversion | 1,594 | **Easy** |
| gravitational | 1,597 | **Easy** |

---

## 1. BIT MANIPULATION (1,602 puzzles)

### Format
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,
and possibly majority or choice functions.

Here are some examples of input -> output:
01010001 -> 11011101
00001001 -> 01101101
...
Now, determine the output for: 00110100
Answer: 10010111
```

- **Input**: 8-bit binary numbers
- **Output**: 8-bit binary numbers  
- **Examples per puzzle**: 7-10 input→output pairs
- **Query**: One new input (not in examples), need to predict output

### Key Findings

**Simple single-operation transforms are RARE (only ~14% of puzzles).**
- Left/right rotation by k: ~10% (these are equivalent: left_rot_k = right_rot_{8-k})
- NOT, AND, OR with constant: ~4%
- XOR with constant: **0%** (XOR is NEVER consistent across examples)

**The transform is an arbitrary per-bit boolean function.** Each output bit position is computed as a boolean function of 1-3 input bit positions:
- ~20% of output bits are identity copies of a single input bit
- ~15% are NOT of a single input bit  
- ~30% are 2-input boolean functions (XOR, AND, OR, XNOR, etc.)
- ~25% are 3-input boolean functions (majority, etc.)
- ~10% are constants (always 0 or always 1)

**There is massive ambiguity.** With only 7-10 examples, each output bit typically has 10-50+ consistent boolean function candidates. The "correct" answer depends on choosing the function a human would identify as the natural pattern.

### Solver Strategy
1. For each output bit, enumerate all 1-bit, 2-bit, and 3-bit input subsets
2. Build truth tables from examples
3. Prefer simpler functions (fewer input bits)  
4. When ambiguous, the puzzle is designed for LLM reasoning (hard to solve programmatically)
5. **Expected accuracy with simple approach**: ~50-55%
6. **Expected accuracy with LLM**: ~85-95%

### Example Analysis
```
Puzzle 0031df9c (right rotation by 2):
  out[0]=in[6], out[1]=in[7], out[2]=in[0], out[3]=in[1],
  out[4]=in[2], out[5]=in[3], out[6]=in[4], out[7]=in[5]

Puzzle 00066667 (mixed operations):
  out[0] = NOT(in[1] XOR in[7])  -- 2-input function
  out[1] = NOT in[2]             -- NOT
  out[2] = NOT in[3]             -- NOT
  out[3] = NOT in[4]             -- NOT
  out[4] = NOT in[5]             -- NOT
  out[5] = NOT in[6]             -- NOT
  out[6] = NOT in[7]             -- NOT
  out[7] = 1 (constant)
```

---

## 2. ENCRYPTION (1,576 puzzles)

### Format
```
In Alice's Wonderland, secret encryption rules are used on text.
Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
...
Now, decrypt the following text: trb wzrswvog hffk
Answer: cat imagines book
```

- **Cipher**: lowercase English text (words separated by spaces)
- **Plaintext**: lowercase English text
- **Encryption type**: **100% substitution cipher** (bijective letter-to-letter mapping)
- **0% shift ciphers** - all are general substitution ciphers
- **Space is preserved** (not encrypted)

### Key Findings

**The cipher is a bijective (one-to-one) substitution cipher within each puzzle.**
Each puzzle has its own random letter-to-letter mapping.

**Limited vocabulary**: Only **77 unique words** across all 1,576 puzzles!

**Top 30 plaintext words:**
```
the (3141), dreams (481), creates (477), found (473), studies (467),
draws (458), sees (458), reads (452), teacher (449), writes (449),
imagines (445), chases (440), follows (438), knight (437), hatter (436),
watches (436), student (434), wizard (434), discovers (427), queen (426),
alice (426), secret (425), rabbit (424), turtle (418), dragon (415),
mouse (414), explores (403), cat (402), king (399), princess (398)
```

**Full noun list**: knight, hatter, student, wizard, queen, alice, rabbit, turtle, dragon, mouse, cat, king, princess, bird, teacher  
**Full verb list**: dreams, creates, found, studies, draws, sees, reads, writes, imagines, chases, follows, watches, discovers, explores  
**Full adjective list**: strange, hidden, wise, silver, dark, colorful, curious, bright, ancient  
**Full location list**: forest, garden, castle  
**Full other words**: the, secret, key, crystal, puzzle, map, book, treasure, in

### Missing Character Coverage
- **38%** of puzzles have all query characters covered by examples → direct decryption works
- **38%** have 1 missing character  
- **17%** have 2 missing characters
- **7%** have 3+ missing characters

### Solver Strategy
1. Build cipher→plain mapping from all example pairs (word-by-word, position-by-position)
2. Decrypt query using the mapping (unknown chars → '?')
3. Match partial decryption against the 77-word vocabulary
4. Use word length and known letters to uniquely identify words
5. **Expected accuracy**: 95-100% (vocabulary is tiny, so dictionary matching is very effective)

### Example
```
Cipher:  trb wzrswvog hffk
Mapping: t→c, r→a, b→t, w→i, z→m, s→g, v→n, o→e, g→s, f→o, k→b
Decrypt:  cat imagines ?oo?  → "book" matches pattern "?oo?" with known vocab
Answer:   cat imagines book ✓
```

---

## 3. BASE CONVERSION (1,576 puzzles)

### Format
```
In Alice's Wonderland, numbers are secretly converted into a different numeral system.
Some examples are given below:
11 -> XI
15 -> XV
94 -> XCIV
19 -> XIX
Now, write the number 38 in the Wonderland numeral system.
Answer: XXXVIII
```

### Key Findings

**100% standard Roman numeral conversion.** Every single puzzle is just converting Arabic integers to Roman numerals.

- **Number range**: 1 to 100 (mean: 50.2)
- **Verification**: All 1,576 puzzles confirmed as standard Roman numerals
- **0 non-standard mappings**

### Solver Strategy
1. Extract the query integer from the prompt
2. Convert to standard Roman numeral
3. **Expected accuracy**: 100%

### Standard Roman Numeral Reference
```
I=1, V=5, X=10, L=50, C=100
IV=4, IX=9, XL=40, XC=90
```

---

## 4. UNIT CONVERSION (1,594 puzzles)

### Format
```
In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
10.08 m becomes 6.69
17.83 m becomes 11.83
35.85 m becomes 23.79
Now, convert the following measurement: 25.09 m
Answer: 16.65
```

### Key Findings

**Each puzzle has a unique, consistent conversion factor** (ratio = output/input).
The factor is constant across all examples within a puzzle but varies between puzzles.

- **Factor range**: 0.502 to 2.000
- **Mean factor**: 1.255
- **Factors do NOT correspond to real-world unit conversions** (not meters→feet, etc.)
- The factor is essentially a random multiplier in [0.5, 2.0] range

### Consistency
- **100% of puzzles have error < 0.05** using average ratio
- **99% of puzzles have error < 0.01**
- Mean prediction error: 0.0031
- Max prediction error: 0.0183

### Solver Strategy
1. Calculate ratio = output/input for each example pair
2. Average the ratios
3. Multiply query input by average ratio
4. Round to 2 decimal places (matching answer format)
5. **Expected accuracy**: ~100% (errors are all below 0.02)

### Example
```
Pairs: [(29.13, 35.47), (47.55, 57.89), (49.37, 60.11), (24.09, 29.33)]
Ratios: [1.2176, 1.2175, 1.2175, 1.2175]
Average: 1.2175
Query: 44.2 × 1.2175 = 53.82 ✓
```

---

## 5. GRAVITATIONAL (1,597 puzzles)

### Format
```
In Alice's Wonderland, the gravitational constant has been secretly changed.
Here are some example observations:
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
...
Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2.
Answer: 154.62
```

### Key Findings

**Each puzzle uses d = 0.5 * g * t² with a unique g value.** 
The g value is extremely consistent within each puzzle.

- **g range**: 4.91 to 19.58 m/s²
- **Mean g**: 12.13 m/s² (Earth's g = 9.81)
- **g is NOT Earth's gravity** - it's a random value per puzzle

### Consistency (within each puzzle)
- **Mean coefficient of variation**: 0.011% (extremely precise!)
- **Max CV**: 0.086%
- **100% of puzzles have prediction error < 0.05**
- **91% of puzzles have prediction error < 0.01**
- Mean prediction error: 0.0044

### Solver Strategy
1. Calculate g = 2*d/t² for each example
2. Average all g values (they're nearly identical)
3. Compute d = 0.5 * g_avg * t_query²
4. Round to 2 decimal places
5. **Expected accuracy**: ~100% (g is incredibly consistent)

### Example
```
t->d pairs: [(1.04, 3.93), (2.21, 17.76), (1.75, 11.14)]
g values: [7.2670, 7.2726, 7.2751]
Average g: 7.2716
Query: d = 0.5 * 7.2716 * 4.19² = 63.83 (answer: 63.85, error: 0.02)
```

---

## 6. CRYPTARITHM (1,555 puzzles)

### Format
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations.
Below are a few examples:
96$54 = 5184
50$41 = 2050
51$95 = 4845
Now, determine the result for: 59$49
Answer: 2891
```

### Subtypes (from first 200 puzzles analyzed)

| Subtype | Count | Description |
|---------|-------|-------------|
| symbol_substitution | 113 | Symbol-to-symbol character mapping |
| digit_equation | 46 | Digit + operator, operator = unknown operation |
| digit_operators | 41 | Multiple operators, each = different operation |

### Key Findings

**Three distinct puzzle types:**

#### Type 1: Symbol Substitution (~55-60%)
Symbol strings map to other symbol strings through a character-by-character substitution (like the encryption puzzles but with non-alphabetic characters).

#### Type 2: Digit + Operator Equations (~30-35%)
Two-digit numbers are combined with a non-standard operator (symbol) that represents a specific arithmetic operation. The solver must identify what each operator means.

**Common operations include:**
- Addition (a + b)
- Subtraction (a - b)
- Absolute subtraction (|a - b|)
- Multiplication (a * b)
- Multiplication minus 1 (a * b - 1)
- Integer division (a // b)
- Modulo (a % b)
- Concatenation (str(a) + str(b))
- Reverse concatenation (str(b) + str(a))
- Addition minus 1 (a + b - 1)

**Important**: Each operator symbol is **consistent within a puzzle** but has **different meanings across puzzles**. Standard operators like +, -, * may also be remapped!

#### Type 3: Mixed Equations (~5-10%)
Some puzzles mix standard operators (with their normal meaning) with non-standard operators.

### Solver Strategy

**For symbol substitution:**
1. Build character mapping from example pairs (position-by-position)
2. Apply mapping to query
3. Same approach as encryption puzzles

**For digit + operator equations:**
1. Parse each equation to extract (left_operand, operator, right_operand, result)
2. For each operator, try all candidate operations
3. Verify consistency across all examples with that operator
4. Apply identified operation to the query
5. Handle edge cases: negative results, leading zeros, concatenated results

**Expected accuracy**: 
- Symbol substitution: 85-95%
- Digit equations: 75-85% (some operations are unusual/ambiguous)

### Example (digit equation)
```
Puzzle 04322d27:
  96$54 = 5184  → 96*54 = 5184, so $ = multiplication
  50$41 = 2050  → 50*41 = 2050 ✓
  51$95 = 4845  → 51*95 = 4845 ✓
  89$47 = 4183  → 89*47 = 4183 ✓
  Query: 59$49 = 59*49 = 2891 ✓
```

### Example (mixed operators)
```
Puzzle 03f07b43:
  55`39 = 16    → 55-39 = 16, so ` = subtraction
  61\65 = 126   → 61+65 = 126, so \ = addition  
  42>23 = 4223  → "42" + "23" = "4223", so > = concatenation
  Query: 81`20 = 81-20 = 61 ✓
```

---

## SOLVER PRIORITY & DIFFICULTY RANKING

### Tier 1: Trivial (100% accuracy expected)
| Category | Method | Lines of Code |
|----------|--------|---------------|
| base_conversion | Standard Roman numeral converter | ~15 |
| unit_conversion | Calculate ratio, multiply | ~20 |
| gravitational | Calculate g, apply formula | ~20 |

### Tier 2: High accuracy (90-100%)
| Category | Method | Lines of Code |
|----------|--------|---------------|
| encryption | Substitution cipher + dictionary matching | ~40 |

### Tier 3: Moderate accuracy (75-85%)
| Category | Method | Lines of Code |
|----------|--------|---------------|
| cryptarithm | Operator identification + symbol substitution | ~80 |

### Tier 4: Hard (50-60% without LLM)
| Category | Method | Lines of Code |
|----------|--------|---------------|
| bit_manipulation | Boolean function enumeration | ~60 |

### Tier 4 with LLM boost (85-95%)
For bit manipulation, pass the examples + query to an LLM and ask it to identify the pattern and predict the output.

---

## IMPLEMENTATION NOTES

1. **Parsing**: Each category has a distinct prompt format - use regex patterns shown above
2. **Rounding**: Unit conversion and gravitational answers are rounded to 2 decimal places
3. **Leading zeros**: Bit manipulation may have 7-bit or 8-bit outputs; gravitational answers don't have leading zeros
4. **Error tolerance**: For unit conversion and gravitational, average across all examples to minimize rounding errors
5. **Vocabulary**: The encryption vocabulary is only 77 words - hardcode this list for perfect dictionary matching
