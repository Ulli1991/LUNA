"""Check that edits between two git revisions changed only docstrings and comments.

Usage: python tools/check_docstring_only.py BASE [HEAD] [--stats]
"""

import argparse
import ast
import io
import subprocess
import sys
import tokenize


def _show(rev, path):
    """Return the file text at a revision ('' if absent); HEAD=None reads the worktree."""
    if rev is None:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    out = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


def _strip_docstrings(tree):
    """Remove the leading docstring Expr of the module, every class and every function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_dump(src):
    """AST dump of the source with docstrings removed and no position attributes."""
    return ast.dump(_strip_docstrings(ast.parse(src)), include_attributes=False)


def doc_lines(src):
    """Count docstring lines and comment lines in the source."""
    tree = ast.parse(src)
    n_doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                n_doc += body[0].end_lineno - body[0].lineno + 1
    comment_rows = {tok.start[0] for tok in tokenize.generate_tokens(io.StringIO(src).readline)
                    if tok.type == tokenize.COMMENT}
    return n_doc, len(comment_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("head", nargs="?", default=None, help="revision; default is the worktree")
    ap.add_argument("--stats", action="store_true", help="print doc/comment/total line counts")
    args = ap.parse_args()
    cmd = ["git", "diff", "--name-only", "--diff-filter=M", args.base]
    if args.head:
        cmd.append(args.head)
    files = [f for f in subprocess.run(cmd, capture_output=True, text=True, check=True)
             .stdout.split() if f.endswith(".py")]
    bad = []
    for f in files:
        old, new = _show(args.base, f), _show(args.head, f)
        same = code_dump(old) == code_dump(new)
        if not same:
            bad.append(f)
        if args.stats:
            (d0, c0), (d1, c1) = doc_lines(old), doc_lines(new)
            print(f"{'OK  ' if same else 'DIFF'} {f}: doc {d0}->{d1} comment {c0}->{c1} "
                  f"total {old.count(chr(10))}->{new.count(chr(10))}")
    print(f"{len(files) - len(bad)}/{len(files)} changed .py files have identical code ASTs")
    for f in bad:
        print("CODE CHANGED:", f)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
