"""One-off script to estimate total tokens and cost for remaining corpus."""
import json
import random
from pathlib import Path

from src.config import Config
from src.zotero.database import ZoteroDatabase
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.analysis.semantic_prompts import build_pass_user_prompt, SEMANTIC_SYSTEM_PROMPT

config = Config.load()
db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
papers = [p for p in db.get_all_papers() if p.pdf_path and p.pdf_path.exists()]

sa_data = json.loads(Path("data/index/semantic_analyses.json").read_text(encoding="utf-8"))
existing_ids = set(sa_data.get("extractions", {}).keys())
remaining = [p for p in papers if p.paper_id not in existing_ids]

print(f"Total with PDFs: {len(papers)}")
print(f"Already extracted: {len(existing_ids)}")
print(f"Remaining: {len(remaining)}")

pdf = PDFExtractor(enable_ocr=True)
tc = TextCleaner()

random.seed(42)
sample = random.sample(remaining, min(25, len(remaining)))

tokens_per_paper = []
for p in sample:
    try:
        text = pdf.extract_text(p.pdf_path)
        if text:
            text = tc.clean(text)
            text = tc.truncate_for_llm(text)
            prompt = build_pass_user_prompt(1, p.title, "", None, p.item_type, text)
            full = SEMANTIC_SYSTEM_PROMPT + prompt
            tokens_est = len(full) // 4
            tokens_per_paper.append(tokens_est * 6)
    except Exception:
        continue

avg_tokens = sum(tokens_per_paper) / len(tokens_per_paper)
median_tokens = sorted(tokens_per_paper)[len(tokens_per_paper) // 2]

n = len(remaining)
total_input = int(avg_tokens * n)
output_per_paper = 12000
total_output = output_per_paper * n

std_in = total_input / 1_000_000 * 2.50
std_out = total_output / 1_000_000 * 15.00
batch_in = total_input / 1_000_000 * 1.25
batch_out = total_output / 1_000_000 * 7.50

print(f"\n=== Token Estimation (sampled {len(tokens_per_paper)} papers) ===")
print(f"Avg input tokens/paper (6 passes): {avg_tokens:,.0f}")
print(f"Median: {median_tokens:,.0f}")
print(f"Range: {min(tokens_per_paper):,.0f} - {max(tokens_per_paper):,.0f}")
print(f"Output tokens/paper (est): {output_per_paper:,}")

print(f"\n=== {n} Remaining Papers ===")
print(f"Total input tokens: {total_input:,} ({total_input/1_000_000:.1f}M)")
print(f"Total output tokens: {total_output:,} ({total_output/1_000_000:.1f}M)")
print(f"Total tokens: {(total_input + total_output):,} ({(total_input + total_output)/1_000_000:.1f}M)")

print(f"\n=== Cost ===")
print(f"Standard API:       ${std_in + std_out:>8.2f}  (in ${std_in:.2f} + out ${std_out:.2f})")
print(f"Batch API (50% off): ${batch_in + batch_out:>8.2f}  (in ${batch_in:.2f} + out ${batch_out:.2f})")

print(f"\n=== Tier Planning (batch API) ===")
print(f"Tier 1 (~900K enqueued):  ~{n // 8} batches of ~8 papers")
print(f"Tier 2 (~2M enqueued):    ~{n // 20} batches of ~20 papers")
print(f"Tier 3 (~5M enqueued):    ~{n // 50} batches of ~50 papers")
print(f"Tier 4 (~20M enqueued):   ~{n // 200} batches of ~200 papers")
print(f"Tier 5 (~100M enqueued):  1-2 batches")

print(f"\n=== Spend to Unlock Tiers ===")
print(f"Tier 2: $50 cumulative API spend + 7 days")
print(f"Tier 3: $100 cumulative API spend + 7 days")
print(f"Tier 4: $250 cumulative API spend + 14 days")
print(f"Tier 5: $1,000 cumulative API spend + 30 days")
print(f"\nTotal batch cost for this job: ${batch_in + batch_out:.2f}")
print(f"  -> This spend alone would unlock Tier 4 (if first API usage)")
