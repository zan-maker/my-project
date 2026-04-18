---
Task ID: REMINDER
Agent: Main Agent
Task: Set reminder to submit ARC-AGI-2 v2 tomorrow

Reminder Details:
- DATE: 2026-04-19 (tomorrow)
- TIMEZONE: America/New_York (ET)
- REASON: ARC-AGI-2 has 1 submission/day limit. v1 was submitted 2026-04-18.
- V2 kernel already pushed and completed successfully.

---

## EXACT COMMANDS TO RUN TOMORROW:

### 1. Submit ARC-AGI-2 v2:
```bash
export PATH="/home/z/.local/bin:$PATH"
export KAGGLE_API_TOKEN=KGAT_19e16eaf2bc84622c9bebb9c73c86e1d

kaggle competitions submit \
  -c arc-prize-2026-arc-agi-2 \
  -f submission.json \
  -k shyamdesigan/annotatex-arc-agi-2-v2 \
  -v 1 \
  -m "AnnotateX v2: 26 heuristics, object analysis, connected components"
```

### 2. Check submission status:
```bash
kaggle competitions submissions -c arc-prize-2026-arc-agi-2
```

### 3. Check ARC-AGI-3 score (submitted 2026-04-18):
```bash
kaggle competitions submissions -c arc-prize-2026-arc-agi-3
```

---

## CURRENT SUBMISSION STATUS (as of 2026-04-18):

| Competition | Kernel | Status | Notes |
|---|---|---|---|
| ARC-AGI-2 | shyamdesigan/annotatex-arc-agi-2-v1 | COMPLETE | v1 baseline, 25 tasks solved |
| ARC-AGI-2 | shyamdesigan/annotatex-arc-agi-2-v2 | READY | v2 improved, 34 tasks solved, push tomorrow |
| ARC-AGI-3 | shyamdesigan/annotatex-arc-agi-3-agent | PENDING | Agent kernel submitted |
| Paper Track | (not yet built) | PENDING | Research paper needed |

---

## PENDING WORK:

### Tomorrow (Apr 19):
- [ ] Submit ARC-AGI-2 v2
- [ ] Build & submit Paper Track research paper
- [ ] Check ARC-AGI-3 score

### Ongoing:
- [ ] Build ARC-AGI-2 v3 with LLM-based ICL (Qwen3 on Kaggle GPU)
- [ ] Improve ARC-AGI-3 agent (top score is 0.68, ours is baseline)
- [ ] Paper Track paper must link to code submission

### Deadlines:
- ARC-AGI-3: Oct 26, 2026
- ARC-AGI-2: Nov 2, 2026
- Paper Track: Nov 9, 2026
