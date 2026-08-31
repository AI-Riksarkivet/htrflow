#!/usr/bin/env python3
"""Sync docs/features/**.md to Azure DevOps work items (AI-labbet project).

  azure_sync.py story   <story.md> [--dry-run]
      create (no id) or update (id set) a Product Backlog Item
  azure_sync.py feature <feature.md> [--dry-run]
      update a Feature's description (everything before '## Stories')

Front matter is the mapping: `type`, `id`, `parent`, `title`. On create the new
id is written back into the file. Work items get the tags
`htrflow; story-<ID>` (ID = first three chars of the file name), so the
mapping is two-way: Azure tag -> file, file `id:` -> Azure item.
State and assignee are Azure's; this script never touches them.

Auth: a PAT with Work Items read+write in ~/.azdo_pat (mode 600).
"""

import html
import json
import os
import re
import subprocess
import sys

BASE = "https://devops.ra.se/DataLab/AI-labbet/_apis/wit"
TAG_PREFIX = "htrflow"


def pat():
    return open(os.path.expanduser("~/.azdo_pat")).read().strip()


def md_inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    s = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s
    )  # links dropped: Azure has no repo context
    return s


def md_block(md):
    """Small Markdown subset -> HTML: headings, paragraphs, bullet lists, tables."""
    out, para, lst, table = [], [], [], []

    def flush():
        nonlocal para, lst, table
        if para:
            out.append("<p>" + md_inline(" ".join(para)) + "</p>")
            para = []
        if lst:
            out.append(
                "<ul>" + "".join("<li>" + md_inline(i) + "</li>" for i in lst) + "</ul>"
            )
            lst = []
        if table:
            rows = [r for r in table if not re.match(r"^\|[-| ]+\|$", r)]
            cells = [
                [md_inline(c.strip()) for c in r.strip("|").split("|")] for r in rows
            ]
            h = "<tr>" + "".join(f"<th>{c}</th>" for c in cells[0]) + "</tr>"
            b = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                for r in cells[1:]
            )
            out.append('<table border="1" cellpadding="4">' + h + b + "</table>")
            table = []

    for line in md.splitlines():
        if line.startswith("#"):
            flush()
            lvl = min(len(line) - len(line.lstrip("#")) + 1, 4)
            out.append(f"<h{lvl}>{md_inline(line.lstrip('# ').strip())}</h{lvl}>")
        elif line.startswith("|"):
            if para or lst:
                flush()
            table.append(line)
        elif re.match(r"^\s*- ", line):
            if para or table:
                flush()
            lst.append(re.sub(r"^\s*- (\[ \] )?", "", line))
        elif line.strip() == "":
            flush()
        elif lst and line.startswith("  "):
            lst[-1] += " " + line.strip()
        else:
            if lst or table:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def read(path):
    t = open(path).read()
    parts = t.split("---")
    fm, body = parts[1], "---".join(parts[2:])
    meta = dict(re.findall(r"^(\w+):[ \t]*(.*)$", fm, re.M))
    body = re.sub(
        r"^# .*\n", "", body.strip() + "\n", count=1
    )  # H1 duplicates the title field
    return t, meta, body


def call(method, url, ops):
    r = subprocess.run(
        [
            "curl",
            "-sS",
            "-m",
            "30",
            "-u",
            f":{pat()}",
            "-X",
            method,
            "-H",
            "Content-Type: application/json-patch+json",
            "-w",
            "\n%{http_code}",
            "-d",
            json.dumps(ops),
            url,
        ],
        capture_output=True,
        text=True,
    )
    out, code = r.stdout.rsplit("\n", 1)
    return code, (json.loads(out) if out.strip().startswith("{") else out)


def story(path, dry):
    t, meta, body = read(path)
    sid = os.path.basename(path)[:3]
    parts = re.split(r"^## (?:Done when|Klart när)\s*$", body, flags=re.M)
    desc_md, ac_md = parts[0], (parts[1] if len(parts) > 1 else "")
    ops = [
        {
            "op": "add",
            "path": "/fields/System.Title",
            "value": re.sub(r"[`*]", "", meta["title"]),
        },
        {"op": "add", "path": "/fields/System.Description", "value": md_block(desc_md)},
        {
            "op": "add",
            "path": "/fields/System.Tags",
            "value": f"{TAG_PREFIX}; story-{sid}",
        },
    ]
    if ac_md.strip():
        ops.append(
            {
                "op": "add",
                "path": "/fields/Microsoft.VSTS.Common.AcceptanceCriteria",
                "value": md_block(ac_md),
            }
        )
    existing = meta.get("id", "").strip()
    if existing:
        for o in ops:
            if o["path"] == "/fields/System.Tags":
                o["op"] = "replace"  # 'add' merges tags; the file is authoritative
        method, url = "PATCH", f"{BASE}/workitems/{existing}?api-version=7.0"
    else:
        ops.append(
            {
                "op": "add",
                "path": "/relations/-",
                "value": {
                    "rel": "System.LinkTypes.Hierarchy-Reverse",
                    "url": f"{BASE}/workItems/{meta['parent']}",
                },
            }
        )
        method, url = (
            "POST",
            f"{BASE}/workitems/${meta['type'].replace(' ', '%20')}?api-version=7.0",
        )
    if dry:
        print(method, url)
        print(json.dumps(ops, indent=1, ensure_ascii=False)[:3000])
        return
    code, d = call(method, url, ops)
    if not code.startswith("2"):
        print(sid, "http", code, str(d)[:400])
        sys.exit(1)
    print(sid, "id", d["id"], "| state", d["fields"]["System.State"])
    if not existing:
        open(path, "w").write(
            re.sub(r"^id:.*$", f"id: {d['id']}", t, count=1, flags=re.M)
        )


def feature(path, dry):
    t, meta, body = read(path)
    body = re.split(r"^## (?:Stories|Berättelser)\s*$", body, flags=re.M)[0]
    body = re.sub(r'!!! \w+ "[^"]*"\n(\n    .*)+', "", body)  # docs-site admonitions
    ops = [{"op": "add", "path": "/fields/System.Description", "value": md_block(body)}]
    if dry:
        print(meta["id"])
        print(ops[0]["value"][:2000])
        return
    code, d = call("PATCH", f"{BASE}/workitems/{meta['id']}?api-version=7.0", ops)
    print(
        "feature",
        meta["id"],
        "http",
        code,
        "" if code.startswith("2") else str(d)[:300],
    )


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("story", "feature"):
        print(__doc__)
        sys.exit(2)
    (story if sys.argv[1] == "story" else feature)(sys.argv[2], "--dry-run" in sys.argv)
