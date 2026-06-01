from pathlib import Path


def load_hotwords(path: str = "data/hotwords.txt") -> str:
    """Load FunASR hotwords from a newline-separated text file."""
    hotword_path = Path(path)
    if not hotword_path.exists():
        return ""

    words = []
    for line in hotword_path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            words.append(word)
    return " ".join(words)
