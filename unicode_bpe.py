import itertools
from collections import defaultdict

# CRITICAL: Use the `regex` module, NOT standard `re`
import regex 
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
# 2. Unicode-Aware Pre-tokenizer
# ============================================================
# \p{Malayalam} matches the entire Malayalam Unicode block.
# \p{L} and \p{N} match any Unicode letter/number.
WORD_RE = regex.compile(
    r'\p{Malayalam}+'             # Malayalam words
    r'|[A-Za-z0-9_]+'             # Latin words / digits
    r'|[^\s\p{L}\p{N}]'           # Punctuation (anything not a letter/number/space)
)

# \X matches an Extended Grapheme Cluster (the Unicode standard for "what a user sees as one character")
GRAPHENE_RE = regex.compile(r'\X')

# ============================================================
# 3. List-based merge (Unchanged, still the most robust method)
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

class UnicodeBPE:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = {}

    @staticmethod
    def _tokenize_words(text):
        return WORD_RE.findall(text)

    @staticmethod
    def _init_symbols(word_with_marker):
        # Extract grapheme clusters instead of raw code points!
        if word_with_marker.endswith("</w>"):
            base_word = word_with_marker[:-4]
            return GRAPHENE_RE.findall(base_word) + ["</w>"]
        return GRAPHENE_RE.findall(word_with_marker)

    # ---------------------------------------------------- training
    def train(self, corpus):
        print("Step 1: counting word/punct frequencies ...")
        word_freqs = defaultdict(int)
        for text in corpus:
            for w in self._tokenize_words(text):
                word_freqs[w + "</w>"] += 1

        # Calculate average grapheme length (not code-point length)
        avg_len = sum(len(self._init_symbols(w)) - 1 for w in word_freqs) / max(len(word_freqs), 1)
        print(f"  unique types: {len(word_freqs)} | avg type length: {avg_len:.2f} graphemes")

        print("Step 2: initialising grapheme vocabulary ...")
        chars = set()
        for w in word_freqs:
            chars.update(self._init_symbols(w))
        self.vocab = sorted(chars)
        
        num_merges = max(self.vocab_size - len(self.vocab), 0)
        print(f"  initial vocab (graphemes): {len(self.vocab)} | merges to learn: {num_merges}")

        splits = {w: self._init_symbols(w) for w in word_freqs}

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
                if best[0] in splits[w]:
                    splits[w] = _merge(splits[w], best, new_token)

            if step % 500 == 0:
                print(f"  {step}/{num_merges} | vocab {len(self.vocab)} | {best} -> {new_token}")

        print("Training complete!\n")

    # ---------------------------------------------------- encoding
    def encode(self, text):
        out = []
        for w in self._tokenize_words(text):
            # Initialize with graphemes
            symbols = GRAPHENE_RE.findall(w) + ["</w>"]
            
            for pair, merged in self.merges.items():
                if pair[0] in symbols:
                    symbols = _merge(symbols, pair, merged)
                    
            for s in symbols:
                clean = s.replace("</w>", "")
                if clean:
                    out.append(clean)
        return out

    # ---------------------------------------------------- fertility
    def check_fertility(self, texts):
        total_words = total_chars = total_tokens = 0
        for text in texts:
            words = text.split()
            total_words += len(words)
            # Note: For Unicode fertility, we count graphemes, not code points!
            total_chars += sum(len(GRAPHENE_RE.findall(w)) for w in words)
            total_tokens += len(self.encode(text))

        return {
            "whitespace_words":        total_words,
            "graphemes_in_words":      total_chars,
            "bpe_tokens":              total_tokens,
            "avg_word_len_graphemes":  round(total_chars / total_words, 2) if total_words else 0,
            "fertility_tok_per_word":  round(total_tokens / total_words, 3) if total_words else 0,
            "fertility_tok_per_graph": round(total_tokens / total_chars, 3) if total_chars else 0,
            "compression_graph_per_tok": round(total_chars / total_tokens, 3) if total_tokens else 0,
        }

# ============================================================
# 4. Run
# ============================================================
if __name__ == "__main__":
    print("Loading corpus ...")
    corpus = list(load_smc_corpus(max_examples=5000))
    print(f"Loaded {len(corpus)} texts.\n")

    # 8000 is plenty when starting with graphemes instead of code points
    bpe = UnicodeBPE(vocab_size=8000)
    bpe.train(corpus)

    test = "കേരളം, ഇന്ത്യയുടെ തെക്കേ അറ്റത്തുള്ള ഒരു സംസ്ഥാനമാണ്!"
    print("--- Encoding ---")
    print("text:  ", test)
    print("tokens:", bpe.encode(test), "\n")

    print("--- Fertility (first 500 texts) ---")
    for k, v in bpe.check_fertility(corpus[:500]).items():
        print(f"  {k}: {v}")

    print("\n--- Sample of learned subwords (last 15 merges) ---")
    print(list(bpe.merges.values())[-15:])