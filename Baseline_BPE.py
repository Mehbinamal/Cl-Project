import itertools
import re
from collections import defaultdict

from datasets import load_dataset


# ============================================================
# 1. Corpus
# ============================================================
def load_smc_corpus(max_examples=100000):
    ds = load_dataset(
        "VishnuPJ/Malayalam_CultureX_IndicCorp_SMC",
        split="train",
        streaming=True,
    )
    for text in itertools.islice((example["text"] for example in ds if example.get("text")), max_examples):
        yield text


# ============================================================
# 2. Word splitter: keep Malayalam base + combining marks together
# ============================================================
WORD_RE = re.compile(
    r'[\u0D00-\u0D7F\u200c\u200d]+'        # Malayalam block (incl. ZWJ/ZWNJ for chillu)
    r'|[A-Za-z0-9_]+'                       # latin words / digits
    r'|[^\s\w\u0D00-\u0D7F\u200c\u200d]'   # everything else, one char at a time
)


# ============================================================
# 3. List-based merge (no string.replace -> no cross-boundary matching)
# ============================================================
def _merge(symbols, pair, new_token):
    out, i, n = [], 0, len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_token)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BaselineBPE:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = {}          # ordered: (a, b) -> merged

    @staticmethod
    def _tokenize_words(text):
        return WORD_RE.findall(text)

    @staticmethod
    def _init_symbols(word_with_marker):
        # keep "</w>" atomic
        if word_with_marker.endswith("</w>"):
            return list(word_with_marker[:-4]) + ["</w>"]
        return list(word_with_marker)

    # ---------------------------------------------------- training
    def train(self, corpus):
        print("Step 1: counting word/punct frequencies ...")
        word_freqs = defaultdict(int)
        for text in corpus:
            for w in self._tokenize_words(text):
                word_freqs[w + "</w>"] += 1

        avg_len = sum(len(w) - 4 for w in word_freqs) / max(len(word_freqs), 1)
        print(f"  unique types: {len(word_freqs)} | avg type length: {avg_len:.2f} chars")

        print("Step 2: initialising character vocabulary ...")
        chars = set()
        for w in word_freqs:
            chars.update(self._init_symbols(w))
        self.vocab = sorted(chars)
        num_merges = max(self.vocab_size - len(self.vocab), 0)
        print(f"  initial vocab: {len(self.vocab)} | merges to learn: {num_merges}")

        splits = {w: self._init_symbols(w) for w in word_freqs}   # lists, not strings

        print("Step 3: learning merges ...")
        for step in range(num_merges):
            pair_freqs = defaultdict(int)
            for w, freq in word_freqs.items():
                sym = splits[w]
                for i in range(len(sym) - 1):
                    pair_freqs[(sym[i], sym[i + 1])] += freq

            if not pair_freqs:
                print(f"  stopped early at {step} merges (no pairs left)")
                break

            best = max(pair_freqs, key=pair_freqs.get)
            new_token = best[0] + best[1]
            self.vocab.append(new_token)
            self.merges[best] = new_token

            for w in splits:
                if best[0] in splits[w]:                      # cheap guard, avoids list churn
                    splits[w] = _merge(splits[w], best, new_token)

            if step % 500 == 0:
                print(f"  {step}/{num_merges} | vocab {len(self.vocab)} | {best} -> {new_token}")

        print("Training complete!\n")

    # ---------------------------------------------------- encoding
    def encode(self, text):
        out = []
        for w in self._tokenize_words(text):
            symbols = list(w) + ["</w>"]
            for pair, merged in self.merges.items():
                if pair[0] in symbols:
                    symbols = _merge(symbols, pair, merged)
            for s in symbols:
                clean = s.replace("</w>", "")     # strip marker from *every* token
                if clean:
                    out.append(clean)
        return out

    # ---------------------------------------------------- fertility
    def check_fertility(self, texts):
        """Fertility measured against REAL whitespace words, not regex fragments."""
        total_words = total_chars = total_tokens = 0
        for text in texts:
            words = text.split()                   # <-- real words
            total_words += len(words)
            total_chars += sum(len(w) for w in words)
            total_tokens += len(self.encode(text))

        return {
            "whitespace_words":        total_words,
            "chars_in_words":          total_chars,
            "bpe_tokens":              total_tokens,
            "avg_word_len_chars":      round(total_chars / total_words, 2) if total_words else 0,
            "fertility_tok_per_word":  round(total_tokens / total_words, 3) if total_words else 0,
            "fertility_tok_per_char":  round(total_tokens / total_chars, 3) if total_chars else 0,
            "compression_chars_per_tok": round(total_chars / total_tokens, 3) if total_tokens else 0,
        }


# ============================================================
# 4. Run
# ============================================================
if __name__ == "__main__":
    print("Loading corpus ...")
    corpus = list(load_smc_corpus(max_examples=5000))
    print(f"Loaded {len(corpus)} texts.\n")

    bpe = BaselineBPE(vocab_size=8000)
    bpe.train(corpus)

    test = "കേരളം, ഇന്ത്യയുടെ തെക്കേ അറ്റത്തുള്ള ഒരു സംസ്ഥാനമാണ്!"
    print("--- Encoding ---")
    print("text:  ", test)
    print("tokens:", bpe.encode(test), "\n")

    print("--- Fertility (first 500 texts) ---")
    for k, v in bpe.check_fertility(corpus[:500]).items():
        print(f"  {k}: {v}")

    print("\n--- Sample of learned subwords (last 20 merges) ---")
    print(list(bpe.merges.values())[-20:])