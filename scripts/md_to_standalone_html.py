"""把教程 Markdown 转成单文件离线 HTML。

用途
    W3 及后续每周的教程都要有一份可离线阅读的 HTML。本脚本负责：
    (1) 把 Markdown 正文转成 HTML 片段（表格、代码块、标题锚点）
    (2) 生成左侧目录，并在滚动时高亮当前章节
    (3) 内联全部 CSS 与 JS，产出单文件，双击即可打开，不依赖网络
    (4) 打印时隐藏目录，正文独占一栏

输入
    CLI 第一个参数：源 Markdown 路径；第二个参数：输出 HTML 路径。

输出
    一个 UTF-8 的 .html 文件，并在控制台打印正文汉字数与校验结果。

跑完应该看到的自检数字
    控制台最后一行是「内容保真校验：通过」
    输出的 HTML 里不含 http 外链（除文末参考资料里读者自己点开的链接以外，
    样式与脚本全部内联）
"""

import html as htmllib
import re
import sys
from pathlib import Path

import markdown


# ==== 1. 样式与脚本（内联进产物，保证离线可用） ====

CSS = """
:root {
  --bg: #fdfdfb; --fg: #1f2328; --muted: #6b7280; --line: #e5e7eb;
  --accent: #1f6feb; --code-bg: #f6f8fa; --sidebar: #f7f7f4;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  font-size: 16px; line-height: 1.85;
}
#layout { display: flex; max-width: 1180px; margin: 0 auto; }
#toc {
  width: 268px; flex: 0 0 268px; position: sticky; top: 0; align-self: flex-start;
  height: 100vh; overflow-y: auto; padding: 32px 18px 48px 24px;
  background: var(--sidebar); border-right: 1px solid var(--line);
  font-size: 14px; line-height: 1.6;
}
#toc .toc-title { font-size: 12px; letter-spacing: .12em; color: var(--muted);
  text-transform: uppercase; margin-bottom: 14px; }
#toc ul { list-style: none; margin: 0; padding-left: 0; }
#toc ul ul { padding-left: 14px; }
#toc li { margin: 2px 0; }
#toc a { color: var(--muted); text-decoration: none; display: block;
  padding: 3px 8px; border-left: 2px solid transparent; border-radius: 3px; }
#toc a:hover { color: var(--fg); background: #ecece6; }
#toc a.active { color: var(--accent); border-left-color: var(--accent);
  background: #eef4ff; font-weight: 600; }
#content { flex: 1 1 auto; min-width: 0; padding: 40px 56px 120px; }
h1 { font-size: 30px; line-height: 1.4; margin: 0 0 28px; letter-spacing: -.01em; }
h2 { font-size: 22px; margin: 46px 0 14px; padding-top: 12px;
  border-top: 1px solid var(--line); }
h3 { font-size: 17px; margin: 28px 0 10px; color: #33383f; }
p { margin: 12px 0; }
a { color: var(--accent); }
strong { font-weight: 650; }
code { font-family: "Cascadia Code", Consolas, "Courier New", monospace;
  font-size: .88em; background: var(--code-bg); padding: 1px 5px;
  border-radius: 4px; border: 1px solid #eceef1; }
pre { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 8px; padding: 14px 16px; overflow-x: auto;
  line-height: 1.55; margin: 16px 0; }
pre code { background: none; border: 0; padding: 0; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14.5px; }
th, td { border: 1px solid var(--line); padding: 8px 11px; text-align: left;
  vertical-align: top; }
th { background: #f3f4f6; font-weight: 650; }
tbody tr:nth-child(even) { background: #fafafa; }
blockquote { margin: 16px 0; padding: 10px 16px; border-left: 3px solid #c9ced6;
  background: #f8f8f5; color: #444; }
ol, ul { padding-left: 26px; }
li { margin: 6px 0; }
ol > li::marker { color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--line); margin: 32px 0; }
@media (max-width: 900px) {
  #layout { display: block; }
  #toc { position: static; width: auto; height: auto; border-right: 0;
    border-bottom: 1px solid var(--line); }
  #content { padding: 24px 20px 80px; }
}
@media print {
  #toc { display: none; }
  #layout { display: block; max-width: none; }
  #content { padding: 0; }
  body { background: #fff; font-size: 11pt; line-height: 1.6; }
  h2 { page-break-after: avoid; }
  pre, table, blockquote { page-break-inside: avoid; }
  a { color: #000; text-decoration: none; }
}
"""

JS = """
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('#toc a'));
  var heads = links.map(function (a) { return document.getElementById(a.hash.slice(1)); });
  function activate(idx) {
    links.forEach(function (a, i) { a.classList.toggle('active', i === idx); });
  }
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        var i = heads.indexOf(e.target);
        if (i >= 0) { activate(i); }
      }
    });
  }, { rootMargin: '0px 0px -75% 0px', threshold: 0 });
  heads.forEach(function (h) { if (h) { obs.observe(h); } });
  activate(0);
})();
"""


# ==== 2. 目录生成 ====

def render_toc(tokens, depth=0):
    """把 markdown 的 toc_tokens 递归渲染成嵌套 ul。只取前两级的标题。"""
    if not tokens:
        return ""
    out = ["<ul>"]
    for tk in tokens:
        out.append(f'<li><a href="#{tk["id"]}">{tk["name"]}</a>')
        if tk.get("children") and depth < 1:
            out.append(render_toc(tk["children"], depth + 1))
        out.append("</li>")
    out.append("</ul>")
    return "".join(out)


# ==== 3. 主流程 ====

def convert(src, dst):
    """读 src，写 dst。返回 (正文汉字数, 标题字符串)。"""
    text = src.read_text(encoding="utf-8")

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        # toc_depth 从 2 起：H1 已作为页头单独渲染，放进目录会指向一个不存在的锚点
        extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
    )
    body = md.convert(text)
    toc = render_toc(md.toc_tokens)

    # 标题取第一个 H1，同时把它从正文里摘掉（页头单独渲染一次）
    m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    title = m.group(1) if m else src.stem
    body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)

    hanzi = len(re.findall(r"[\u4e00-\u9fff]", re.sub(r"```.*?```", "", text, flags=re.S)))

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div id="layout">
<nav id="toc">
  <div class="toc-title">目录</div>
  {toc}
</nav>
<main id="content">
<h1>{title}</h1>
{body}
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page, encoding="utf-8")
    return hanzi, title


def normalize(s):
    """两侧共用的归一化：反转义 HTML 实体，再只保留字母数字和汉字。

    去掉标点的原因是 Markdown 的 `[[0,-1],[1,0]]` 渲染成 HTML 后括号还在，
    如果只在源侧删符号，比对会永远失败。两侧用同一个函数才不会假阳性。
    """
    s = htmllib.unescape(s)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", s)


def verify(src_text, html_text):
    """内容保真校验：Markdown 的每行正文都应能在 HTML 的纯文本里找到。

    只比对「有实际内容」的行，跳过围栏、空行、表格分隔线这类结构行。
    返回缺失行的列表，空列表表示通过。
    """
    plain = re.sub(r"<script.*?</script>|<style.*?</style>", "", html_text, flags=re.S)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = normalize(plain)

    missing = []
    for raw in src_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```") or re.fullmatch(r"\|[-: |]+\|", line):
            continue
        # 有序/无序列表的编号由 HTML 的 <ol>/<ul> 生成，源侧的「1. 」不在正文里
        line = re.sub(r"^(\d+[.、]|[-*+])\s+", "", line)
        probe = normalize(line)
        if len(probe) < 6:
            continue
        if probe not in plain:
            missing.append(line[:60])
    return missing


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 3:
        print("用法：python md_to_standalone_html.py <教程.md> <输出.html>")
        return 2

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    src_text = src.read_text(encoding="utf-8")
    hanzi, title = convert(src, dst)
    missing = verify(src_text, dst.read_text(encoding="utf-8"))

    print(f"标题：{title}")
    print(f"正文 {hanzi} 字")
    print(f"输出：{dst}")
    if missing:
        print(f"内容保真校验：{len(missing)} 行缺失")
        for line in missing[:10]:
            print("  -", line)
        return 1
    print("内容保真校验：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
