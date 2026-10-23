"""check_tokenizer.py TOKENIZER_JSON MLT ML_TOK: ml_tok against the Hugging Face
tokenizers library on a corpus of tricky strings (and any extra text files
given with --text): the same ids when encoding, and the same text when
decoding the ids after the leading BOS (which is how ml_run prints). Texts
holding added tokens such as "<s>" are checked for encoding only: ml_tok, like
ml_run, treats every BOS as the start of the text. Exits 77 (skip) if the
tokenizers package is missing."""
import argparse
import random
import subprocess
import sys
import unicodedata

CORPUS = [
    "Hello world", "Once upon a time, there was a little girl named Lily.",
    "I'm sure they'll say it's fine, but we've DON'T and You'RE and I'd.",
    "'sup 'Sup ''s 's'", "12345678901 3.14159 1,000,000 0x1F 2^10",
    "naïve café résumé Ångström", "日本語のテキストを分割する。", "안녕하세요, 세계! 반갑습니다.",
    "Привет, мир! Ελληνικά", "emoji 🙂👍🏽 and flags 🇰🇷", "मानक हिन्दी", "العربية النص",
    "  leading spaces", "trailing spaces   ", "a  b   c    d", "tab\tseparated\tvalues",
    "line one\nline two\n\nline four", "windows\r\nline\r\n", "\n\n\nstarts with newlines",
    "ends with newlines\n\n", " \n \n mixed  \n\n  whitespace \t\n", "   ", "\t", "\n",
    "def f(x):\n    return x ** 2  # square\n", "if (a && b) { c = d[i]; } else { e(); }",
    "...!!!??? --- ___ *** ### @@@", "email@example.com https://example.com/a?b=c&d=e",
    "x", "A", "ab", "The quick brown fox jumps over the lazy dog. 1234 567 89",
    "Mixed123Numbers456and789Letters", "ʼapostrophe ‘quotes’ “double” «guillemets»",
    "Zero​width nbsp　ideographic sep",
    "<s>special</s> in <|begin_of_text|> text <|eot_id|> and <|reserved_special_token_7|>",
    "a" * 300, "ab " * 120, "🙂" * 50,
    # NFC (Qwen2): decomposed accents, out-of-order marks, conjoining jamo,
    # singletons, exclusions and blocked compositions
    "cafe\u0301 re\u0301sume\u0301 A\u030a \u212b \u2126 \u0958 \u0344 \u0f73",
    "a\u0301\u0323 a\u0323\u0301 q\u0307\u0323 o\u031b\u0323\u0302 e\u0302\u0301\u0301",
    "\u1100\u1161\u11a8 \u1112\u1161\u11ab\u1100\u1173\u11af \uac00\u11a8 \u1100\u1161 \u1100\u11a8",
    "\u0915\u093c \u09c7\u09be \u0b47\u0b56 \u1025\u102e \u3046\u3099 \u30cf\u309a",
]
# every composite NFC builds, decomposed (letters split by spaces in chunks;
# tools/gen_unicode.py follows the tokenizers library in leaving U+11938 out)
_COMP = [c for c in map(chr, range(0x110000)) if unicodedata.normalize("NFD", c) != c
         and unicodedata.normalize("NFC", unicodedata.normalize("NFD", c)) == c and c != "\U00011938"]
CORPUS += [" ".join(unicodedata.normalize("NFD", c) for c in _COMP[i:i + 200])
           for i in range(0, len(_COMP), 200)]


POOL = list("aZqé漢字한글Жж123٣٤ \n\r\t'’.,!?-_()[]{}<>/\\@#$%^&*\"`~:;|+=") + [
    "\u0301", "\u0323", "\u0302", "\u031b", "\u1100", "\u1161", "\u11a8", "\u3099", "e\u0301", "\u0915", "\u093f", "\u00a0", "\u3000", "\u2028", "🙂", "👍🏽", "\U0001F1F0", "s", "t", "re", "ll", "  ", "\n\n"]


def fuzz(n, seed):
    """n random strings over a pool of letters, numbers, marks, spaces and symbols."""
    rng = random.Random(seed)
    return ["".join(rng.choice(POOL) for _ in range(rng.randint(1, 40))) for _ in range(n)]


def ml_tok(binary, mlt, args, data):
    r = subprocess.run([binary, mlt] + args, input=data, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"ml_tok {args} failed: {r.stderr.decode(errors='replace')}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_json")
    ap.add_argument("mlt")
    ap.add_argument("ml_tok")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--fuzz", type=int, default=200, help="random strings to add (default 200)")
    a = ap.parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("skip: the tokenizers package is not installed")
        return 77
    hf = Tokenizer.from_file(a.tokenizer_json)
    texts = CORPUS + [open(p, encoding="utf-8").read() for p in a.text] + fuzz(a.fuzz, 1)
    bos = hf.encode("").ids[0] if hf.encode("").ids else -1
    bad = 0
    for s in texts:
        want = hf.encode(s).ids
        got = [int(x) for x in ml_tok(a.ml_tok, a.mlt, [], s.encode("utf-8")).split()]
        if got != want:
            bad += 1
            i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
            print(f"MISMATCH {s[:60]!r}: first difference at {i}: ml_tok {got[i:i + 6]} hf {want[i:i + 6]}")
            continue
        if any(t.content in s for t in hf.get_added_tokens_decoder().values()):
            continue
        body = got[1:] if got and got[0] == bos else got
        back = ml_tok(a.ml_tok, a.mlt, ["--decode"], " ".join(map(str, body)).encode()).decode("utf-8", errors="replace")
        ref = hf.decode(body, skip_special_tokens=False)
        if back != ref:
            bad += 1
            print(f"DECODE {s[:60]!r}: ml_tok {back[:60]!r} hf {ref[:60]!r}")
    print(f"{a.mlt}: {len(texts) - bad}/{len(texts)} texts match the tokenizers library")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
