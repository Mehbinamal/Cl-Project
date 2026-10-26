import itertools
from collections import defaultdict
import regex
from datasets import load_dataset

# ============================================================
# 1. Corpus
# ============================================================
def load_smc_corpus(max_examples=100000):
    ds = load_dataset("VishnuPJ/Malayalam_CultureX_IndicCorp_SMC", split="train", streaming=True)
    for text in itertools.islice((example["text"] for example in ds if example.get("text")), max_examples):
        yield text

# ============================================================
# 2. Rule Engine for Pre-Tokenization
# ============================================================
class TokenizationRule:
    """Base class for custom tokenization rules."""
    def __init__(self, name, condition_func):
        self.name = name
        self.condition_func = condition_func

    def applies_to(self, word):
        return self.condition_func(word)

# --- Define Specific Rules ---

# Rule 1: Words ending in Malayalam Chillus
# Independent chillus: \u0D7A to \u0D7F (ൺ, ം, റ, ഴ, ൽ, ൿ)
# Classic chillus: Consonant + Virama (\u0D4D) + ZWNJ (\u200C)
def ends_with_chillu(word):
    return bool(regex.search(r'[\u0D7A-\u0D7F]$', word) or regex.search(r'\u0D4D\u200C$', word))

# You can easily add more rules later! For example:
# def is_english_loanword(word):
#     return bool(regex.fullmatch(r'[A-Za-z0-9_]+', word))

ACTIVE_RULES = [
    TokenizationRule("Chillu_Ending_Word", ends_with_chillu),
    # TokenizationRule("English_Loanword", is_english_loanword), 
]

# ============================================================
# 3. Base Regex Splitters
# ============================================================
WORD_RE = regex.compile(
    r'\p{Malayalam}+'
    r'|[A-Za-z0-9_]+'
    r'|[^\s\p{L}\p{N}]'
)
GRAPHENE_RE = regex.compile(r'\X')

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

# ============================================================
# 4. Rule-Based Unicode BPE
# ============================================================
class RuleBasedUnicodeBPE:
    def __init__(self, vocab_size=8000, rules=None):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = {}
        self.rules = rules if rules is not None else ACTIVE_RULES

    @staticmethod
    def _tokenize_words(text):
        return WORD_RE.findall(text)

    def _init_symbols(self, word_with_marker):
        """Initializes symbols, applying rules BEFORE grapheme splitting."""
        has_marker = word_with_marker.endswith("</w>")
        base_word = word_with_marker[:-4] if has_marker else word_with_marker
        
        # CHECK RULES: If any rule applies, treat the ENTIRE word as 1 atomic symbol
        for rule in self.rules:
            if rule.applies_to(base_word):
                symbols = [base_word]  # Protected! No grapheme splitting.
                if has_marker:
                    symbols.append("</w>")
                return symbols
                
        # DEFAULT: Split into Unicode Grapheme Clusters
        symbols = GRAPHENE_RE.findall(base_word)
        if has_marker:
            symbols.append("</w>")
        return symbols

    # ---------------------------------------------------- training
    def train(self, corpus):
        print("Step 1: counting frequencies ...")
        word_freqs = defaultdict(int)
        for text in corpus:
            for w in self._tokenize_words(text):
                word_freqs[w + "</w>"] += 1

        print("Step 2: initialising vocabulary with rules ...")
        chars = set()
        protected_count = 0
        for w in word_freqs:
            syms = self._init_symbols(w)
            chars.update(syms)
            # Count how many words were protected by rules (length 1 means it wasn't split)
            if len(syms) == 1 or (len(syms) == 2 and syms[-1] == "</w>"):
                protected_count += 1

        self.vocab = sorted(chars)
        num_merges = max(self.vocab_size - len(self.vocab), 0)
        
        print(f"  initial vocab: {len(self.vocab)} | merges to learn: {num_merges}")
        print(f"  [Rule Engine] Protected {protected_count} unique word types from grapheme splitting.")

        splits = {w: self._init_symbols(w) for w in word_freqs}

        print("Step 3: learning merges ...")
        for step in range(num_merges):
            pair_freqs = defaultdict(int)
            for w, freq in word_freqs.items():
                sym = splits[w]
                for i in range(len(sym) - 1):
                    pair_freqs[(sym[i], sym[i + 1])] += freq

            if not pair_freqs:
                print(f"  stopped early at {step} merges.")
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
            # Apply rules during encoding too!
            symbols = self._init_symbols(w + "</w>")
            
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
            total_chars += sum(len(GRAPHENE_RE.findall(w)) for w in words)
            total_tokens += len(self.encode(text))

        return {
            "whitespace_words":        total_words,
            "graphemes_in_words":      total_chars,
            "bpe_tokens":              total_tokens,
            "fertility_tok_per_word":  round(total_tokens / total_words, 3) if total_words else 0,
            "compression_graph_per_tok": round(total_chars / total_tokens, 3) if total_tokens else 0,
        }

# ============================================================
# 5. Run & Test
# ============================================================
if __name__ == "__main__":
    print("Loading corpus ...")
    corpus = list(load_smc_corpus(max_examples=5000))
    
    bpe = RuleBasedUnicodeBPE(vocab_size=8000)
    bpe.train(corpus)

    # Test sentence specifically designed to trigger the Chillu rule
    # അവൻ (avan - he), ഞാൻ (njaan - I), വാൾ (vaal - sword), സംസ്ഥാനം (state - ends in am, not chillu)
    test_text = "അവൻ ഞാൻ വാൾ സംസ്ഥാനം!"
    
    print("--- Encoding with Rules ---")
    print(f"Text:   {test_text}")
    tokens = bpe.encode(test_text)
    print(f"Tokens: {tokens}\n")
    
    print("--- Fertility ---")
    for k, v in bpe.check_fertility(corpus[:500]).items():
        print(f"  {k}: {v}")