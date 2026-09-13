"""语义检查（AST，纯标准库，不依赖 torch/networkx/yaml）。

做两件事：

1. **跨模块符号检查**：把 `from src.x.y import a, b` 解析到真实文件，确认 a、b
   真的定义在那里（能抓出改名残留、漏 export、函数被覆盖）。
2. **文件内未定义名字检查**：把 import 进来的名字、定义的名字、builtins、
   lambda 参数、赋值/推导式/with/except 绑定的名字都收进作用域，找出被读取但
   从未定义的名字（NameError 风险）。

用法::

    python tools/semantic_check.py            # 检查 src/ 与 scripts/
    python tools/semantic_check.py --strict   # 有问题时返回码 1

设计说明：这个脚本是纯静态的，所以在没有 torch 的机器上也能跑 —— 项目里
`AGENTS.md` 明确禁止用 `python -c` / heredoc 跑长脚本，这类检查应当落成文件。
"""

from __future__ import annotations

import argparse
import ast
import builtins
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}


def module_name_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect_definitions(tree: ast.Module) -> set[str]:
    """模块顶层定义的符号 + 顶层 import 进来的名字。"""
    names: set[str] = set()

    def add_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)

    def add_imports(node: ast.AST) -> None:
        for alias in node.names:
            # `from pkg import sub.module` 绑定的是 sub
            names.add(alias.asname or alias.name.split(".")[0])

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                add_target(target)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            add_target(node.target)
        elif isinstance(node, ast.AugAssign):
            add_target(node.target)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            add_imports(node)
    return names


def imported_from(tree: ast.Module) -> list[tuple[str, str, int]]:
    """返回 [(module, name, lineno)]，只看 `from X import a, b`。"""
    out: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                out.append((node.module, alias.name, node.lineno))
    return out


def scope_names(tree: ast.Module) -> set[str]:
    """近似地收集整个文件里所有"被绑定"的名字。"""
    names: set[str] = set()

    def add_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)
        elif isinstance(target, ast.Starred):
            add_target(target.value)

    for child in ast.walk(tree):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = child.args
                for arg in (
                    list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
                ):
                    names.add(arg.arg)
                if args.vararg:
                    names.add(args.vararg.arg)
                if args.kwarg:
                    names.add(args.kwarg.arg)
        elif isinstance(child, ast.Lambda):
            args = child.args
            for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                names.add(arg.arg)
            if args.vararg:
                names.add(args.vararg.arg)
            if args.kwarg:
                names.add(args.kwarg.arg)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                add_target(target)
        elif isinstance(child, ast.AnnAssign) and child.target is not None:
            add_target(child.target)
        elif isinstance(child, (ast.AugAssign, ast.NamedExpr)):
            add_target(child.target)
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            add_target(child.target)
        elif isinstance(child, ast.comprehension):
            add_target(child.target)
        elif isinstance(child, ast.withitem) and child.optional_vars is not None:
            add_target(child.optional_vars)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.MatchAs) and child.name:  # pragma: no cover
            names.add(child.name)
        elif isinstance(child, ast.MatchStar) and child.name:  # pragma: no cover
            names.add(child.name)
        elif isinstance(child, ast.Global):
            names.update(child.names)
        elif isinstance(child, ast.Nonlocal):
            names.update(child.names)
    return names


def undefined_names(tree: ast.Module) -> list[tuple[str, int]]:
    loaded: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.append((node.id, node.lineno))
    known = collect_definitions(tree) | BUILTINS | scope_names(tree)
    return sorted({(name, line) for name, line in loaded if name not in known})


def main() -> int:
    parser = argparse.ArgumentParser(description="AST-based semantic checks")
    parser.add_argument("--strict", action="store_true", help="有问题时返回码 1")
    args = parser.parse_args()

    files = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted(
        (REPO_ROOT / "scripts").glob("*.py")
    ) + sorted((REPO_ROOT / "tools").glob("*.py"))
    tables: dict[str, set[str]] = {}
    trees: dict[Path, ast.Module] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[path] = tree
        tables[module_name_for(path, REPO_ROOT)] = collect_definitions(tree)

    tests_conftest = REPO_ROOT / "tests" / "conftest.py"
    if tests_conftest.exists():
        tables["tests.conftest"] = collect_definitions(
            ast.parse(tests_conftest.read_text(encoding="utf-8"))
        )

    problems = 0

    # 1. 跨模块符号是否存在
    for path, tree in trees.items():
        for module, name, lineno in imported_from(tree):
            if not (module.startswith("src") or module in ("tests.conftest",)):
                continue
            if module not in tables:
                print(f"MISSING MODULE {path.relative_to(REPO_ROOT)}:{lineno} -> {module}")
                problems += 1
                continue
            if name.split(".")[0] in tables[module]:
                continue
            # `from src.data import branch_segments as bs` 导入的是子模块
            if f"{module}.{name.split('.')[0]}" in tables:
                continue
            print(
                f"MISSING SYMBOL {path.relative_to(REPO_ROOT)}:{lineno} -> "
                f"{module}.{name}"
            )
            problems += 1

    # 2. 文件内未定义就被读取的名字
    for path, tree in trees.items():
        for name, lineno in undefined_names(tree):
            print(f"POSSIBLE NameError {path.relative_to(REPO_ROOT)}:{lineno} -> {name}")
            problems += 1

    print(f"checked {len(trees)} modules, {problems} problems")
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
