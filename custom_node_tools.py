"""Read-only inspection of installed ComfyUI custom node packages.

Exposed to the workflow chat as function tools. Every path is resolved inside
one custom node package directory; hidden files and directories (any path
component starting with ".") are never listed, searched or read.
"""

import json
import os
import re

import folder_paths
import nodes


MAX_LIST_ENTRIES = 400
MAX_READ_CHARS = 40_000
MAX_READ_FILE_BYTES = 5_000_000
MAX_SEARCH_RESULTS = 60
MAX_SEARCH_FILES = 3_000
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_NODE_TYPES_PER_PACKAGE = 200
SKIPPED_DIR_NAMES = {"__pycache__", "node_modules"}
BLOCKED_FILE_PATTERN = re.compile(
    r"(^id_(rsa|dsa|ecdsa|ed25519)|\.(pem|key|p12|pfx|crt|keystore)$|secret|credential|password)",
    re.IGNORECASE,
)
SEARCHABLE_EXTENSIONS = {
    ".py", ".js", ".mjs", ".ts", ".vue", ".css", ".html", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".md", ".txt", ".rst", ".sh", ".glsl",
    ".cpp", ".c", ".h", ".cu",
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "list_custom_nodes",
        "description": (
            "List installed ComfyUI custom node packages with their version, "
            "whether they are disabled and which node types each one registers."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_custom_node_files",
        "description": (
            "List non-hidden files of one custom node package, recursively from "
            "an optional subdirectory. Returns relative paths and sizes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package directory name from list_custom_nodes.",
                },
                "subdir": {
                    "type": ["string", "null"],
                    "description": "Relative subdirectory, or null for the package root.",
                },
            },
            "required": ["package", "subdir"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_custom_node_file",
        "description": (
            "Read a range of lines from one non-hidden text file inside a custom "
            "node package. Long files are returned in chunks; use start_line to "
            "continue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package directory name from list_custom_nodes.",
                },
                "path": {
                    "type": "string",
                    "description": "File path relative to the package directory.",
                },
                "start_line": {
                    "type": ["integer", "null"],
                    "description": "1-based first line to return, or null for 1.",
                },
            },
            "required": ["package", "path", "start_line"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_custom_node_files",
        "description": (
            "Case-insensitive literal text search over non-hidden source files of "
            "custom node packages. Use it to find where a class, input name or "
            "error message is defined."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Literal text to find (at least 3 characters).",
                },
                "package": {
                    "type": ["string", "null"],
                    "description": "Limit to one package directory name, or null for all.",
                },
            },
            "required": ["query", "package"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOL_DEFINITIONS}


class ToolError(Exception):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False)


def _custom_node_roots():
    return [
        root
        for root in folder_paths.get_folder_paths("custom_nodes")
        if os.path.isdir(root)
    ]


def _is_hidden_part(part):
    return part.startswith(".") or part in ("__pycache__",)


def _package_dirs():
    packages = {}
    for root in _custom_node_roots():
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if _is_hidden_part(name) or not os.path.isdir(path):
                continue
            packages.setdefault(name, path)
    return packages


def _resolve_package(package):
    if not isinstance(package, str) or not package or "/" in package or "\\" in package:
        raise ToolError("package must be a directory name from list_custom_nodes.")
    path = _package_dirs().get(package)
    if path is None:
        raise ToolError(f"Custom node package not found: {package}")
    return os.path.realpath(path)


def _resolve_inside(package_root, relative_path):
    relative_path = (relative_path or "").replace("\\", "/").strip("/")
    parts = [part for part in relative_path.split("/") if part not in ("", ".")]
    if any(part == ".." or _is_hidden_part(part) for part in parts):
        raise ToolError("Hidden files and parent-directory paths are not allowed.")
    target = os.path.realpath(os.path.join(package_root, *parts))
    if target != package_root and not target.startswith(package_root + os.sep):
        raise ToolError("Path is outside the custom node package.")
    return target, "/".join(parts)


def _walk_visible(start):
    for current, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(
            name for name in dirnames
            if not _is_hidden_part(name) and name not in SKIPPED_DIR_NAMES
        )
        for name in sorted(filenames):
            if not _is_hidden_part(name):
                yield os.path.join(current, name)


def _package_metadata(path):
    metadata = {}
    pyproject = os.path.join(path, "pyproject.toml")
    if os.path.isfile(pyproject):
        try:
            with open(pyproject, encoding="utf-8", errors="replace") as handle:
                text = handle.read(20_000)
            for key in ("name", "version", "description"):
                match = re.search(rf'^{key}\s*=\s*"([^"\n]*)"', text, re.MULTILINE)
                if match:
                    metadata[key] = match.group(1)
        except OSError:
            pass
    return metadata


def _node_types_by_package():
    by_package = {}
    for node_type, node_class in nodes.NODE_CLASS_MAPPINGS.items():
        module = getattr(node_class, "RELATIVE_PYTHON_MODULE", None)
        if not isinstance(module, str) or not module.startswith("custom_nodes."):
            continue
        package = module.split(".", 1)[1]
        by_package.setdefault(package, []).append(node_type)
    return by_package


def list_custom_nodes():
    node_types = _node_types_by_package()
    packages = []
    for name, path in _package_dirs().items():
        node_list = sorted(node_types.get(name, []))
        entry = {
            "package": name,
            "disabled": name.endswith(".disabled"),
            "symlink": os.path.islink(path),
            "node_type_count": len(node_list),
            "node_types": node_list[:MAX_NODE_TYPES_PER_PACKAGE],
            **_package_metadata(path),
        }
        if len(node_list) > MAX_NODE_TYPES_PER_PACKAGE:
            entry["node_types_truncated"] = True
        packages.append(entry)
    return {"packages": packages}


def list_custom_node_files(package, subdir):
    package_root = _resolve_package(package)
    start, relative_start = _resolve_inside(package_root, subdir)
    if not os.path.isdir(start):
        raise ToolError("subdir is not a directory.")

    files = []
    truncated = False
    for path in _walk_visible(start):
        if len(files) >= MAX_LIST_ENTRIES:
            truncated = True
            break
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        files.append({
            "path": os.path.relpath(path, package_root).replace(os.sep, "/"),
            "size": size,
        })
    result = {"package": package, "subdir": relative_start, "files": files}
    if truncated:
        result["truncated"] = True
        result["note"] = "Narrow the listing with subdir."
    return result


def _read_text(path, limit):
    with open(path, "rb") as handle:
        data = handle.read(limit)
    if b"\x00" in data[:8192]:
        raise ToolError("Binary files cannot be read.")
    return data.decode("utf-8", errors="replace")


def read_custom_node_file(package, path, start_line, redact):
    package_root = _resolve_package(package)
    target, relative_path = _resolve_inside(package_root, path)
    if not os.path.isfile(target):
        raise ToolError("File not found.")
    if BLOCKED_FILE_PATTERN.search(os.path.basename(target)):
        raise ToolError("This file may contain credentials and cannot be read.")
    if os.path.getsize(target) > MAX_READ_FILE_BYTES:
        raise ToolError("File is too large to read.")

    lines = _read_text(target, MAX_READ_FILE_BYTES).splitlines()
    start_line = max(1, start_line or 1)
    if start_line > max(1, len(lines)):
        raise ToolError(f"start_line is past the end of the file ({len(lines)} lines).")

    chunk = []
    size = 0
    end_line = start_line - 1
    for line in lines[start_line - 1:]:
        size += len(line) + 1
        if chunk and size > MAX_READ_CHARS:
            break
        chunk.append(line[:MAX_READ_CHARS])
        end_line += 1

    result = {
        "package": package,
        "path": relative_path,
        "total_lines": len(lines),
        "start_line": start_line,
        "end_line": end_line,
        "content": redact("\n".join(chunk)),
    }
    if end_line < len(lines):
        result["next_start_line"] = end_line + 1
    return result


def search_custom_node_files(query, package, redact):
    if not isinstance(query, str) or len(query.strip()) < 3:
        raise ToolError("query must be at least 3 characters.")
    needle = query.strip().lower()
    if package:
        roots = [(package, _resolve_package(package))]
    else:
        roots = [(name, os.path.realpath(path)) for name, path in _package_dirs().items()]

    matches = []
    scanned = 0
    for name, root in roots:
        for path in _walk_visible(root):
            if os.path.splitext(path)[1].lower() not in SEARCHABLE_EXTENSIONS:
                continue
            if BLOCKED_FILE_PATTERN.search(os.path.basename(path)):
                continue
            if not os.path.realpath(path).startswith(root + os.sep):
                continue
            scanned += 1
            if scanned > MAX_SEARCH_FILES:
                return {"matches": matches, "truncated": True,
                        "note": "File scan limit reached; pass package to narrow."}
            try:
                if os.path.getsize(path) > MAX_SEARCH_FILE_BYTES:
                    continue
                text = _read_text(path, MAX_SEARCH_FILE_BYTES)
            except (OSError, ToolError):
                continue
            if needle not in text.lower():
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if needle in line.lower():
                    matches.append({
                        "package": name,
                        "path": os.path.relpath(path, root).replace(os.sep, "/"),
                        "line": number,
                        "text": redact(line.strip()[:300]),
                    })
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        return {"matches": matches, "truncated": True}
    return {"matches": matches}


def call_tool(name, arguments, redact):
    """Run one custom-node tool and return its JSON string output."""
    try:
        if name == "list_custom_nodes":
            result = list_custom_nodes()
        elif name == "list_custom_node_files":
            result = list_custom_node_files(arguments.get("package"), arguments.get("subdir"))
        elif name == "read_custom_node_file":
            start_line = arguments.get("start_line")
            if start_line is not None and not isinstance(start_line, int):
                raise ToolError("start_line must be an integer or null.")
            result = read_custom_node_file(
                arguments.get("package"), arguments.get("path"), start_line, redact
            )
        elif name == "search_custom_node_files":
            result = search_custom_node_files(
                arguments.get("query"), arguments.get("package"), redact
            )
        else:
            raise ToolError(f"Unknown tool: {name}")
    except ToolError as error:
        return _json({"error": str(error)})
    except OSError as error:
        return _json({"error": f"Filesystem error: {error.strerror or error.__class__.__name__}"})
    return _json(result)
