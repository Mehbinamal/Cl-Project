from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFC
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
import itertools
import re

# ---------------------------------------------------------------------------
# Chillu normalization: collapses the legacy <consonant, VIRAMA, ZWJ>
# chillu encoding to the atomic chillu codepoint. NFC alone does NOT do
# this (both forms are independently valid Unicode, so NFC treats them
# as canonically distinct) -- without this step, the same word can end
# up as two different BPE vocab entries depending on which encoding the
# source text happened to use, silently splitting its true frequency
# and corrupting the fertility comparison exactly on chillu-final words
# (which include some of the most common suffixes in Malayalam).
# ---------------------------------------------------------------------------
ZWJ = "\u200D"
VIRAMA = "\u0D4D"
CHILLU_MAP = {
    "\u0D23": "\u0D7A",  # ണ -> chillu NN
    "\u0D28": "\u0D7B",  # ന -> chillu N
    "\u0D30": "\u0D7C",  # ര -> chillu RR
    "\u0D32": "\u0D7D",  # ല -> chillu L
    "\u0D33": "\u0D7E",  # ള -> chillu LL
    "\u0D15": "\u0D7F",  # ക -> chillu K
}
_CHILLU_PATTERN = re.compile(
    r"[" + "".join(CHILLU_MAP.keys()) + r"]" + VIRAMA + ZWJ
)

def normalize_chillu(text: str) -> str:
    return _CHILLU_PATTERN.sub(lambda m: CHILLU_MAP[m.group(0)[0]], text)

def normalize_text(text: str) -> str:
    """Full normalization pass: NFC first, then chillu collapse."""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    return normalize_chillu(text)


def load_local_corpus(corpus_dir: Path):
    if not corpus_dir.exists():
        return []

    texts = []
    for file_path in sorted(corpus_dir.rglob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if content:
            texts.append(content)
    return texts


def load_smc_corpus(max_examples=100000):
    ds = load_dataset(
        "VishnuPJ/Malayalam_CultureX_IndicCorp_SMC",
        split="train",
        streaming=True,
    )
    for text in itertools.islice((example["text"] for example in ds if example.get("text")), max_examples):
        yield text


def get_corpus(max_examples=100000):
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    texts = load_local_corpus(data_dir)
    if texts:
        return texts

    corpus_file = project_root / "malayalam_corpus.txt"
    if corpus_file.exists():
        content = corpus_file.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            return [content]

    return list(load_smc_corpus(max_examples=max_examples))


def train_bpe(corpus_texts, vocab_size=16000, min_frequency=2):
    # Normalize the corpus BEFORE training -- this is what fixes the
    # chillu dual-encoding vocab-dilution bug. Do it once here rather
    # than relying only on a HF Normalizer, so it's visible/debuggable.
    normalized_corpus = (normalize_text(t) for t in corpus_texts)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFC()])  # belt-and-suspenders; chillu
                                                # collapse already happened above
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=min_frequency,
    )

    tokenizer.train_from_iterator(normalized_corpus, trainer=trainer)
    return tokenizer


def evaluate_fertility(tokenizer, words):
    """Fertility per word, WITH explicit UNK tracking. A word that hits
    [UNK] is a failure case, not a cheap 'fertility: 1' win -- if we
    don't flag it, the average silently rewards the tokenizer for
    failing on the hardest words."""
    results = []
    for word in words:
        norm_word = normalize_text(word)
        encoded = tokenizer.encode(norm_word)
        has_unk = "[UNK]" in encoded.tokens
        results.append({
            "word": word,
            "tokens": encoded.tokens,
            "fertility": len(encoded.tokens),
            "has_unk": has_unk,
        })
    return results


def summarize_fertility(eval_results):
    clean = [r for r in eval_results if not r["has_unk"]]
    unk_count = len(eval_results) - len(clean)
    unk_rate = unk_count / len(eval_results) if eval_results else 0.0
    avg_fertility_clean = (
        sum(r["fertility"] for r in clean) / len(clean) if clean else float("nan")
    )
    avg_fertility_all = sum(r["fertility"] for r in eval_results) / len(eval_results)
    return {
        "avg_fertility_excluding_unk": avg_fertility_clean,
        "avg_fertility_including_unk": avg_fertility_all,
        "unk_rate": unk_rate,
        "unk_count": unk_count,
        "total_words": len(eval_results),
    }


def main():
    train_size = 100000
    test_words = [
        "മലയാളം", "ഭാഷ", "അറിവ്", "വിദ്യാഭ്യാസം", "സാഹിത്യം",
        "കേരളം", "വ്യവസായം", "വൈദ്യുതി", "സാങ്കേതികവിദ്യ", "പ്രവർത്തനം",
        "ബന്ധം", "അഭിപ്രായം", "പ്രദേശീയ", "പ്രതീക്ഷ", "അവസരം",
        "പതിപ്പ്", "വികസനം", "സ്പർദ്ദ", "സമീപം", "ഭാവി",
    ]

    print(f"Loading up to {train_size} Malayalam text examples from the SMC corpus...")
    corpus_texts = get_corpus(max_examples=train_size)
    if not corpus_texts:
        raise SystemExit(
            "No Malayalam corpus found. Add data to 'data/' or 'malayalam_corpus.txt'."
        )

    print(f"Training BPE on {len(corpus_texts)} examples (normalized)...")
    tokenizer = train_bpe(corpus_texts, vocab_size=16000, min_frequency=2)

    output_path = Path(__file__).resolve().parent / "malayalam_bpe_16k_normalized.json"
    tokenizer.save(str(output_path))
    print(f"Saved Malayalam BPE tokenizer to {output_path}")

    eval_results = evaluate_fertility(tokenizer, test_words)
    summary = summarize_fertility(eval_results)

    print("\nMalayalam fertility evaluation (tokens/word):")
    for r in eval_results:
        flag = "  [UNK present]" if r["has_unk"] else ""
        print(f"{r['word']}: {r['fertility']} -> {r['tokens']}{flag}")

    print(f"\nAverage fertility (excluding UNK words): {summary['avg_fertility_excluding_unk']:.3f}")
    print(f"Average fertility (including UNK words):  {summary['avg_fertility_including_unk']:.3f}")
    print(f"UNK rate: {summary['unk_count']}/{summary['total_words']} "
          f"({summary['unk_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()