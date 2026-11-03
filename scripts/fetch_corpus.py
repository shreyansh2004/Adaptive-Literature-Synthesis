import time
from tools.paper_fetch import fetch_and_save

# Pin the papers that matter for your demo/eval — don't leave this to live search drift.
CORE_PAPER_IDS = [
    "2106.09685",   # LoRA (original) — non-negotiable, keep this pinned
    "2005.11401",   # RAG
    "2411.14961",   # LoRA-FAIR
    "2501.19389",   # Federated Sketching LoRA
    "2602.04998",   # Learning Rate Matters
    "2602.05988",   # Layer-wise LoRA
    "2410.16954",   # LoRA-C
    "2510.27364",   # Video LoRA fine-tuning
    "2403.00946",   # Very Large Dropout
    "2102.03983",   # Partial fine-tuning
]

for arxiv_id in CORE_PAPER_IDS:
    fetch_and_save(arxiv_id, include_full_text=True)
    time.sleep(3)