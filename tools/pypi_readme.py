"""Rewrite README.md's relative links and images to absolute GitHub URLs at one ref, for PyPI."""
# Usage (release workflow, before `python -m build`): python tools/pypi_readme.py v0.1.0
# PyPI renders the long description without the repository, so `docs/X.md` and `code_paper_figs/x.png`
# must become github.com/.../blob/<ref>/docs/X.md and raw.githubusercontent.com/.../<ref>/code_paper_figs/x.png.
import re
import sys
from pathlib import Path

REPO = "Ulli1991/LUNA"
_ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.I)


def absolute(target, ref, image):
    """`target` as an absolute URL at `ref` (unchanged if already absolute or an anchor)."""
    if _ABSOLUTE.match(target):
        return target
    if image:
        return f"https://raw.githubusercontent.com/{REPO}/{ref}/{target}"
    kind = "tree" if target.endswith("/") else "blob"
    return f"https://github.com/{REPO}/{kind}/{ref}/{target}"


def rewrite(text, ref):
    """README text with every relative markdown link, markdown image and <img src> made absolute."""
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)",
                  lambda m: f"{m[1]}[{m[2]}]({absolute(m[3], ref, bool(m[1]))})", text)
    return re.sub(r'(<img\s[^>]*src=")([^"]+)(")', lambda m: m[1] + absolute(m[2], ref, True) + m[3], text)


if __name__ == "__main__":
    ref = sys.argv[1]
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("README.md")
    path.write_text(rewrite(path.read_text(), ref))
