"""Build the 27 November progress report from whatever `analyses/` holds. No torch.

The deck presents the plan of context.md §8 — map where the answer depends, vary the
alignment weight, realign where the dependence is largest — one slide a step, and each
step's result on the slides after it once that result is on file, between an opening
slide that says what was done since the last report (and how many of the realigned runs
have trained, counted from git's index) and a closing one that gathers the readings. So
the same command writes the plan before anything has run and the report after: the job
that runs the experiment (`scripts/run_plan.sh`) calls it after every stage, and pulling
its commits is what updates the deck.

What a result slide says is not composed here. Where the research document fixed an
outcome's meaning before the runs existed, the analyses name the outcome they fell in and
this prints the sentence that outcome was given (presentations/principles.md §1). Where
the document only describes what was measured (the map, the differences at the six
positions, the layer runs), the sentence is chosen by the same test the document states,
and a description the numbers no longer bear stops the build rather than being printed.
The document still has to be brought up to the same results by hand — a slide written
here is ahead of it until then.

The package is written at the level of its parts, from the parts of an earlier report
(principles.md §5): the lab's template, 游明朝, 24pt body lines, the template's table
style, and red for the one conclusion a slide draws. Standard library only, so it runs on
the machine that trained the runs and on the one that opens the deck alike.

    python scripts/update_progress.py [OUT.pptx [ANALYSES_DIR]]

writes the deck to `presentations/20261127_進捗報告.pptx` from `analyses/` unless told
otherwise, and leaves the file alone when nothing but its modification time would change.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
import zipfile
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
ANALYSES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "analyses"
BASE = ROOT / "presentations" / "20261106_進捗報告.pptx"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "presentations" / "20261127_進捗報告.pptx"

EMU = 914400
FONT = "游明朝"
MATH = "Cambria Math"
INK, RED, WHITE = "000000", "FF0000", "FFFFFF"
TABLE_STYLE = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"  # 雛形の表の様式
BODY, TABLE = 2400, 1800  # 本文と表の字の大きさ（1/100 pt）
DIAG = TABLE  # 模式図の中の語（principles.md §3：表と図は 14pt 以上）
TOKEN, PITCH = 0.3, 0.42  # トークンの正方形の一辺と，並べる間隔（インチ）
X0, W = 0.92, 11.5  # 本文の左端と幅（雛形の題と同じ）
Y0, GAP = 1.8, 0.25  # 本文の始まりと，塊どうしの間隔（principles.md §4）

# ③で学習する run の数：proposal と ablation を乱数種3つで（研究文書 §8.5 第2歩）
RELOCATED_RUNS = 6

NS = (
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)


def emu(v: float) -> int:
    return round(v * EMU)


def text_width(text: str, sz: int) -> float:
    """字幅の見積もり（インチ）．和文は全角，欧文と数字は半角の 0.55 とみる．"""
    pt = sz / 100
    return sum(pt if ord(c) > 0x2E7F else 0.55 * pt for c in text) / 72


def line_h(sz: int) -> float:
    return sz / 100 * 1.3 / 72


def fits(text: str, sz: int, width: float) -> None:
    """折り返しを描く環境に任せない：幅の 94% に収まらない行を見つけたら止める．"""
    text = typeset(text)  # run() と同じ字で測る
    w = text_width(text, sz)
    if w > 0.94 * width:
        raise SystemExit(f"収まらない（{w:.2f} > {0.94 * width:.2f}in）：{text}")


def faces(text: str) -> list[tuple[str, str]]:
    """文字を書体ごとに切る．𝒉 と λ は数式の書体で組む（游明朝では λ が全角になり，前後が空きすぎる）．"""
    return [(part, MATH if part in ("𝒉", "λ") else FONT) for part in re.split("(𝒉|λ)", text) if part]


# _TOUCHING はラテン文字の語（と 𝒉）が仮名・漢字に接する所．数の後の pt は単位なので除く．
# 数字と約物はどちらの字にも数えないので，その隣には空白を入れない．_CIRCLED は丸数字の後ろの空白．
_KANA_KANJI = "ぁ-ゖァ-ヺ一-鿿"
_TOUCHING = re.compile(rf"(?<=[A-Za-z𝒉])(?<!\dpt)(?=[{_KANA_KANJI}])|(?<=[{_KANA_KANJI}])(?=[A-Za-z𝒉])")
_CIRCLED = re.compile("(?<=[①-⑳]) ")


def typeset(text: str) -> str:
    """principles.md §3 の空白：ラテン文字の語と和文の間に一つ入れ，丸数字の後ろの空白は除く．

    位置の名を文に差し込むと語の境が詰まりうるので，書いた文でなく，組むときに揃える．
    """
    return _CIRCLED.sub("", _TOUCHING.sub(" ", text))


def run(text: str, sz: int = BODY, *, color: str = INK, bold: bool = False) -> str:
    """一続きの文字．空白を typeset で整え，書体ごとに run を分ける．"""
    text = typeset(text)
    return "".join(
        f'<a:r><a:rPr lang="ja-JP" sz="{sz}" b="{int(bold)}" i="0">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f'<a:latin typeface="{face}"/><a:ea typeface="{face}"/><a:cs typeface="{face}"/>'
        f"</a:rPr><a:t>{escape(part)}</a:t></a:r>"
        for part, face in faces(text)
    )


def para(runs: str, *, algn: str = "l", sz: int = BODY, before: int = 0) -> str:
    """一つの段落．before は前の段落との間隔（1/100 pt）．"""
    return (
        f'<a:p><a:pPr algn="{algn}" marL="0" indent="0"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
        f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef><a:buNone/></a:pPr>{runs}'
        f'<a:endParaRPr lang="ja-JP" sz="{sz}"/></a:p>'
    )


class Slide:
    """本文の塊を本文の始まり Y0 から上に詰めて置き，塊どうしを GAP 空ける．"""

    def __init__(self, title: str):
        self.title = title
        self.shapes: list[str] = []
        self.media: list[bytes] = []
        self._last_id = 3  # 図形の番号は，この次から振る（1 は図形の木，2 は題）
        self.y = Y0

    def _id(self) -> int:
        self._last_id += 1
        return self._last_id

    def take(self, h: float) -> float:
        y = self.y
        self.y = y + h + GAP
        return y

    def _sp(self, x: float, y: float, w: float, h: float, *, geom: str = "rect", fill: str | None = None,
            line: str | None = None, body: str = "", txbox: bool = False, pad: float = 0.0) -> None:
        i = self._id()
        f = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
        ln = (f'<a:ln w="15875"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
              if line else "<a:ln><a:noFill/></a:ln>")
        tx = (f'<p:txBody><a:bodyPr wrap="square" lIns="{emu(pad)}" tIns="0" rIns="{emu(pad)}" bIns="0" anchor="ctr">'
              f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{body or "<a:p><a:endParaRPr lang=\"ja-JP\"/></a:p>"}</p:txBody>')
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Shape {i}"/><p:cNvSpPr{" txBox=\"1\"" if txbox else ""}/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{f}{ln}</p:spPr>{tx}</p:sp>'
        )

    def text(self, lines: list[tuple[str, int, str]], *, x: float = X0, w: float = W, y: float | None = None,
             algn: str = "l", gap_after: float = GAP, pad: float = 0.1) -> float:
        """行の並び（文字，大きさ，色）．行どうしは 9pt 空ける（principles.md §4 の一段目の間隔）．"""
        for t, sz, _ in lines:
            fits(t, sz, w - 2 * pad)
        ps, h = [], 0.0
        for k, (t, sz, color) in enumerate(lines):
            before = 0 if k == 0 else 900
            ps.append(para(run(t, sz, color=color), algn=algn, sz=sz, before=before))
            h += line_h(sz) + before / 100 / 72
        if y is None:
            y = self.y
            self.y = y + h + gap_after
        # 文字の箱は，これまでの進捗報告と同じく左右に 0.1 インチの内側の余白を持つ（字の端が題の字の端と揃う）
        self._sp(x, y, w, h, body="".join(ps), txbox=True, pad=pad)
        return y

    def box(self, x: float, y: float, w: float, h: float, lines: list[str] | None = None, *,
            fill: str | None = None, line: str | None = INK, algn: str = "ctr", sz: int = BODY, pad: float = 0.15,
            geom: str = "rect") -> None:
        for t in lines or []:
            fits(t, sz, w - 2 * pad)
        body = "".join(para(run(t, sz), algn=algn, sz=sz) for t in lines) if lines else ""
        self._sp(x, y, w, h, fill=fill, line=line, body=body, pad=pad, geom=geom)

    def label(self, x: float, y: float, w: float, t: str, *, algn: str = "ctr", sz: int = DIAG) -> None:
        """図の中の語．本文の流れを進めず，指定の位置に一行で置く．"""
        self.text([(t, sz, INK)], x=x, y=y, w=w, algn=algn, pad=0.0)

    def line(self, x1: float, y1: float, x2: float, y2: float, *, arrow: bool = True, dash: bool = False) -> None:
        """水平か垂直の線．arrow なら終点に三角の頭，dash なら破線．影の付く既定の書式（p:style）は持たせない．"""
        assert x1 == x2 or y1 == y2
        i = self._id()
        head = '<a:tailEnd type="triangle" w="med" len="med"/>' if arrow else ""
        flip = ' flipV="1"' if y2 < y1 else (' flipH="1"' if x2 < x1 else "")
        self.shapes.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{i}" name="Line {i}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm{flip}><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/>'
            f'<a:ext cx="{emu(abs(x2 - x1))}" cy="{emu(abs(y2 - y1))}"/></a:xfrm>'
            '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:ln w="15875"><a:solidFill><a:srgbClr val="{INK}"/></a:solidFill>'
            f'{"<a:prstDash val=\"dash\"/>" if dash else ""}{head}</a:ln></p:spPr></p:cxnSp>'
        )

    def picture(self, png: bytes, x: float, y: float, w: float, h: float) -> None:
        """PNG を置き，墨の線で縁取る．部品は build が ppt/media/ に書き，rId は置いた順に 2 から振る．"""
        i = self._id()
        self.media.append(png)
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="Picture {i}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="rId{len(self.media) + 1}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'<a:ln w="15875"><a:solidFill><a:srgbClr val="{INK}"/></a:solidFill></a:ln></p:spPr></p:pic>'
        )

    def table(self, cols: list[float], rows: list[list[str]], *, row_h: float = 0.45, center: tuple[int, ...] = (),
              bold: tuple[tuple[int, int], ...] = ()) -> None:
        """雛形の表の様式．見出しの行は accent1 の 40%，ほかの行は 20%（これまでの進捗報告と同じ）．

        bold は太字にするセル（行，列）．区間が 0 を含まない差に使う（principles.md §2）．
        """
        i = self._id()
        y = self.take(row_h * len(rows))
        trs = []
        for r, row in enumerate(rows):
            shade = ('<a:lumMod val="40000"/><a:lumOff val="60000"/>' if r == 0
                     else '<a:lumMod val="20000"/><a:lumOff val="80000"/>')
            tcs = []
            for c, cell in enumerate(row):
                fits(cell, TABLE, cols[c] - 0.16)
                algn = "ctr" if r == 0 or c in center else "l"
                tcs.append(
                    f"<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{para(run(cell, TABLE, bold=(r, c) in bold), algn=algn, sz=TABLE)}</a:txBody>"
                    f'<a:tcPr anchor="ctr" marL="73152" marR="73152"><a:solidFill><a:schemeClr val="accent1">{shade}'
                    "</a:schemeClr></a:solidFill></a:tcPr></a:tc>"
                )
            trs.append(f'<a:tr h="{emu(row_h)}">{"".join(tcs)}</a:tr>')
        grid = "".join(f'<a:gridCol w="{emu(c)}"/>' for c in cols)
        self.shapes.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{i}" name="Table {i}"/>'
            '<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{emu(X0 + (W - sum(cols)) / 2)}" y="{emu(y)}"/><a:ext cx="{emu(sum(cols))}" cy="{emu(row_h * len(rows))}"/></p:xfrm>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr firstRow="1" bandRow="0"><a:tableStyleId>{TABLE_STYLE}</a:tableStyleId></a:tblPr>'
            f'<a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>'
        )

    def xml(self) -> str:
        set_title = typeset(self.title)
        assert text_width(set_title, 4400) < 11.0, self.title  # 題は一行（44pt で約18字まで）
        title = (
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            '<p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>'
            "<a:p>"
            + "".join(f'<a:r><a:rPr lang="ja-JP"><a:latin typeface="{face}"/><a:ea typeface="{face}"/></a:rPr>'
                      f"<a:t>{escape(part)}</a:t></a:r>" for part, face in faces(set_title))
            + "</a:p></p:txBody></p:sp>"
        )
        return (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:sld {NS}><p:cSld><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>'
            + title + "".join(self.shapes)
            + "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        )


# ---------------------------------------------------------------- スライド


def cover() -> str:
    def ph(kind: str, idx: str, i: int, lines: list[str]) -> str:
        ps = "".join(
            f'<a:p><a:r><a:rPr lang="ja-JP"><a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/></a:rPr>'
            f"<a:t>{escape(t)}</a:t></a:r></a:p>"
            for t in lines
        )
        return (
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{kind} {i}"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            f'<p:nvPr><p:ph type="{kind}"{idx}/></p:nvPr></p:nvSpPr><p:spPr/>'
            f"<p:txBody><a:bodyPr/><a:lstStyle/>{ps}</p:txBody></p:sp>"
        )

    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:sld {NS}><p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>'
        + ph("ctrTitle", "", 2, ["進捗報告"])
        + ph("subTitle", ' idx="1"', 3, ["杉村研究室", "26860646　中野 友晴"])
        + "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )


def tokens(s: Slide, x: float, y: float, n: int, *, last_h: bool = False) -> list[float]:
    """横に並ぶトークンの正方形．白地に墨の線で，last_h なら最後の一つを 𝒉 として墨で塗る．各々の左端の x を返す．"""
    xs = [x + k * PITCH for k in range(n)]
    for k, xk in enumerate(xs):
        h = last_h and k == n - 1
        s.box(xk, y, TOKEN, TOKEN, fill=INK if h else WHITE)
    return xs


def axes(s: Slide, x: float, y: float, title: str, groups: list[tuple[float, float, str, str]], *,
         h: float = 2.05) -> None:
    """𝒉 の空間の模式．縦は答えを動かす方向，横は動かさない方向．groups は（横，縦，塗り，名）の手順の塊．

    h は題の行の上端から横軸までの高さ．
    """
    s.label(x, y, 5.4, title, algn="l")
    ox, oy = x + 0.3, y + h
    s.line(ox, oy, ox, y + 0.45)
    s.line(ox, oy, x + 5.2, oy)
    s.label(ox + 0.12, y + 0.42, 2.6, "答えを動かす方向", algn="l")
    s.label(x + 2.6, oy + 0.06, 2.6, "動かさない方向", algn="r")
    for gx, gy, fill, name in groups:
        cx, cy = ox + gx, oy - gy
        for dx, dy in ((0.0, 0.0), (0.3, 0.12), (0.12, -0.2)):
            s.box(cx + dx, cy + dy, 0.2, 0.2, fill=fill, geom="ellipse")
        s.label(cx + 0.6, cy - 0.04, 1.2, name, algn="l")


def s_done(measured: bool, swept: bool, trained: int) -> Slide:
    """前回からの差分（principles.md §2）．手順の番号はまだ出ていないので，手順の名で言う．"""
    s = Slide("今回したこと")
    lines = ["学会での質問と，使われない理由の候補から，4つの手順を立てた"]
    if measured or swept:
        lines += [t for t, d in (("測定：答えが依存する層と位置のマップ", measured),
                                 ("学習と測定：整合の強さを変えた3本", swept)) if d]
    if 0 < trained < RELOCATED_RUNS:
        lines.append(f"依存の大きい位置への掛け直し：{RELOCATED_RUNS}本中{trained}本を学習")
    s.text([(t, BODY, INK) for t in lines])
    return s


def s_questions() -> Slide:
    s = Slide("学会での質問と応え方")
    s.text([("要旨は発表者の聞き取り", BODY, INK)])
    s.table(
        [0.7, 5.4, 5.4],
        [
            ["番号", "質問の要旨", "応え方"],
            ["1", "causal attention が悪さをしたのでは", "答えが依存する層と位置を測る"],
            ["2", "整合の強さ λ と層を増やせば変わるのでは", "λ を変え，層も1つずつ変えて測る"],
            ["3", "発火を調べ，整合を掛ける位置を決めては", "依存の大きい位置に整合を掛け直す"],
            ["4", "𝒉 に Attention の Q が出るか（曖昧）", "𝒉 は Q を作る元．要旨が不確かで測らず"],
        ],
        center=(0,),
    )
    s.text([("→ 1–3は，層・位置・強さを変えれば手順の情報が使われるかを問う", BODY, RED)])
    return s


def s_candidates() -> Slide:
    s = Slide("手順の情報が十分には使われない理由")
    y = s.take(1.75 + 0.38 + line_h(DIAG))  # 位置の図の下端は，トークンの下の名の行
    # 位置：手順の情報は 𝒉 に載るが，後段の層は他の位置を主に読む（𝒉 からの矢印を破線にする）
    left = X0 + 0.2
    s.box(left, y + 0.4, 5.0, 0.5, ["後段の層（第17–32層）"], sz=DIAG)
    ty = y + 1.75
    xs = tokens(s, left + 0.35, ty, 4) + tokens(s, left + 2.35, ty, 4) + tokens(s, left + 4.35, ty, 1, last_h=True)
    for xk in xs:
        s.line(xk + TOKEN / 2, ty, xk + TOKEN / 2, y + 0.9, dash=xk == xs[-1])
    s.label(left + 0.35, ty + 0.38, 1.56, "画像")
    s.label(left + 2.35, ty + 0.38, 1.56, "質問文")
    s.label(left + 4.0, ty + 0.38, 1.0, "𝒉")
    s.label(left, y, 5.0, "位置", algn="l")
    # 方向：手順の情報が，答えを動かさない方向に載る
    axes(s, X0 + 6.0, y, "方向", [(0.6, 0.45, INK, "手順 A"), (2.9, 0.45, WHITE, "手順 B")])
    s.text([
        ("候補：答えを良くしないなら，手順の情報は後段にあまり読まれない", BODY, INK),
        ("位置：手順の情報は 𝒉 に載るが，後段の層は他の位置を主に読む", BODY, INK),
        ("方向：手順の情報が，𝒉 の中で答えを動かさない方向に載る", BODY, INK),
    ])
    s.text([("→ 掛ける位置と方向を順に変えて試す", BODY, RED)])
    return s


def s_plan(done: tuple[bool, ...], trained: int) -> Slide:
    """done は手順ごとに，結果が analyses/ にあるか．ある手順には「済」を，学習の途中の③には本数を添える．"""
    s = Slide("方針と進み具合")
    steps = [
        "① 答えが依存する層と位置のマップを作る",
        "② 整合の強さ λ を変える",
        "③ 答えの依存が大きい位置に整合を掛け直す",
        "④ 答えを動かす方向に限って整合を掛ける",
    ]
    steps = [f"{t}：済" if d else t for t, d in zip(steps, done, strict=True)]
    if not done[2] and trained:
        steps[2] += f"：{RELOCATED_RUNS}本中{trained}本を学習"
    bh, ah = 0.55, 0.15  # 手順は計画の順に並べるだけで，矢印で結ばない（依存を見せない）
    y = s.take(len(steps) * bh + (len(steps) - 1) * ah)
    bw = max(text_width(typeset(t), BODY) for t in steps) / 0.94 + 0.5
    bx = X0 + (W - bw) / 2  # 図と表は左右の中央に置く（principles.md §4）
    for t in steps:
        s.box(bx, y, bw, bh, [t], algn="l", sz=BODY, pad=0.2)
        y += bh + ah
    s.text([
        ("並行して：学習にほとんど出ない署名でも，読み取り精度を測る", BODY, INK),
        ("その後：別のモデルとデータ．署名に代えて yes/no 問か否かで整合", BODY, INK),
    ])
    return s


def s_map() -> Slide:
    s = Slide("① 層と位置ごとの相対感度のマップ")
    cols = [("書式", 3), ("画像", 8), ("質問文", 6), ("書式", 5)]
    rows = ["第32層", "第16層", "第1層"]  # 全32層を測る．間の層は縦の点で略す
    n = sum(k for _, k in cols)
    cw, rh, gap = 0.3, 0.42, 0.36
    gw = n * cw
    gx = X0 + (W - gw) / 2
    y = s.take(len(rows) * rh + (len(rows) - 1) * gap + 0.08 + line_h(DIAG))
    for r, name in enumerate(rows):
        ry = y + r * (rh + gap)
        s.label(gx - 1.15, ry + 0.06, 1.0, name, algn="r")
        for k in range(n):
            s.box(gx + k * cw, ry, cw, rh, fill=INK if r == 1 and k == n - 1 else None)
        if r:
            s.label(gx - 1.15, ry - gap + 0.02, 1.0, "⋮", algn="r")
    s.label(gx + gw + 0.15, y + rh + gap + 0.06, 1.05, "今の 𝒉", algn="l")
    x = gx
    for name, k in cols:
        s.label(x - 0.3, y + len(rows) * rh + (len(rows) - 1) * gap + 0.08, k * cw + 0.6, name)
        x += k * cw
    s.text([
        ("ねらい：答えがどの層・位置に依存するかを，3群 × 乱数種3つで見る", BODY, INK),
        ("書式：会話の枠の語．User: や Assistant: など", BODY, INK),
        ("注意：マスの和は一定でない．依存が他の位置へ移ったかは言えない", BODY, INK),
        ("ほかに：第8・24層に整合を掛けた試走2本でも測る", BODY, INK),
    ])
    return s


def s_lambda() -> Slide:
    s = Slide("② 整合の強さ λ の変更")
    bw, bh = 1.8, 0.6
    gap = (W - 5 * bw) / 4
    y = s.take(bh + 0.15 + 2 * 0.3)
    points = [("λ = 0", ["baseline", "測定済み"]), ("λ = 0.03", ["新たに学習"]), ("λ = 0.1", ["proposal", "測定済み"]),
              ("λ = 0.3", ["新たに学習"]), ("λ = 1.0", ["新たに学習"])]
    for k, (t, notes) in enumerate(points):
        x = X0 + k * (bw + gap)
        s.box(x, y, bw, bh, [t], sz=DIAG)
        if k < len(points) - 1:
            s.line(x + bw, y + bh / 2, x + bw + gap, y + bh / 2, arrow=False)
        for j, note in enumerate(notes):
            s.label(x, y + bh + 0.12 + j * 0.3, bw, note)
    s.text([
        ("ねらい：読み取り精度の利得に，正解率の代償が伴う λ を見る", BODY, INK),
        ("測る：各強さで読み取り精度・正解率・相対感度．乱数種は1つ", BODY, INK),
    ])
    s.text([("判定：相対感度が λ に伴い単調に下がれば，今の低下は強さに応じる", BODY, RED)])
    return s


def s_relocate(choice: dict | None) -> Slide:
    """choice は①が選んだ場所．選んであれば，選び方の次の行で，規則でなく選んだ結果を言う．"""
    s = Slide("③ 依存の大きい位置への掛け直し")
    groups = [("書式", 3), ("画像", 5), ("質問文", 4), ("書式", 5)]
    sep = 0.3  # 区分どうしの間に足す空き
    span = sum(n for _, n in groups) * PITCH - (PITCH - TOKEN) + (len(groups) - 1) * sep
    y = s.take(0.8 + 0.08 + line_h(DIAG))
    x, xs = X0 + (W - span) / 2, []
    for k, (name, n) in enumerate(groups):
        row = tokens(s, x, y, n, last_h=k == len(groups) - 1)
        s.label(row[0] - 0.3, y + 0.4, row[-1] + TOKEN - row[0] + 0.6, name)
        xs += row
        x = row[-1] + PITCH + sep
    s.label(xs[-1] + TOKEN + 0.1, y - 0.01, 1.1, "今の 𝒉", algn="l")
    cand = xs[-6:]
    by = y + 0.8
    s.line(xs[0], by, xs[7] + TOKEN, by, arrow=False)
    s.label(xs[0] - 0.3, by + 0.08, xs[7] + TOKEN - xs[0] + 0.6, "質問を見ない70位置")
    s.line(cand[0], by, cand[-1] + TOKEN, by, arrow=False)
    s.label(cand[0] - 0.7, by + 0.08, cand[-1] + TOKEN - cand[0] + 1.4, "選ぶ先：質問を読み終えた6位置")
    lines = [
        "選び方：第16層の6位置のうち，baseline の相対感度が最大の位置",
        (f"掛ける先：最大は{place(choice['largest'])}で今と同じなので，{chosen(choice)}" if choice
         else "最大が今の 𝒉 なら，同じ一点なので6位置すべてに掛ける"),
        f"学習：proposal・ablation × 乱数種3つの{RELOCATED_RUNS}本",
        "測る：相対感度と正解率を，baseline・ablation と比べる",
    ]
    s.text([(t, BODY, INK) for t in lines])
    s.text([
        ("判定：相対感度が両方を下回れば，十分には使われない", BODY, RED),
        ("相対感度が保たれ，正解率が両方を超えれば，問いに肯定の材料", BODY, INK),
    ])
    return s


def s_direction() -> Slide:
    s = Slide("④ 答えを動かす方向に限った整合")
    y = s.take(1.75 + 0.06 + line_h(DIAG))  # 軸の図の下端は，横軸の名の行
    axes(s, X0 + 0.1, y, "今の整合：動かさない方向で分かれうる",
         [(0.6, 0.4, INK, "手順 A"), (2.9, 0.4, WHITE, "手順 B")], h=1.75)
    axes(s, X0 + 6.0, y, "④：答えを動かす方向で分ける",
         [(2.9, 0.95, INK, "手順 A"), (2.9, 0.4, WHITE, "手順 B")], h=1.75)
    s.text([
        ("方向：第16層で答えを動かす方向．baseline で求め，学習中は固定", BODY, INK),
        ("測る：相対感度と正解率を，baseline・ablation と比べる", BODY, INK),
    ])
    s.text([
        ("判定：相対感度が両方を下回れば，十分には使われない", BODY, RED),
        ("相対感度が保たれれば，方向の候補と両立する", BODY, INK),
    ])
    return s


# ---------------------------------------------------------------- 結果
#
# 結果のスライドは，analyses/ にその結果があるときだけ作る．値は analyses/ から取り，表示の桁で丸める．
# 読みの文の選び方は冒頭の docstring にある．読みを返す関数（*_reading と RELOCATION）は文だけを返し，
# 結論の印の「→ 」はスライドの側で付ける——まとめのスライドは同じ文を印なしで並べ直す．


def load(name: str) -> dict | None:
    """analyses/ の結果．まだ無ければ None．"""
    path = ANALYSES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _study_statistics():
    """研究文書 §4.3 の区間を作る定義（taskalign/analysis/statistics.py）．

    区間の作り方は一か所にしか置かない．パッケージの __init__ は torch を読むので，標準ライブラリだけで
    書かれたこのファイルを，パッケージを通さずに読む．
    """
    spec = importlib.util.spec_from_file_location(
        "taskalign_statistics", ROOT / "taskalign" / "analysis" / "statistics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired_difference = _study_statistics().paired_difference


def trained_relocated() -> int:
    """③で学習し終えた run の数．終えた run は snapshots/ に評価を残す．

    git の索引から数える：ノートPCの作業ツリーは snapshots/ を持たず（sparse checkout），研究室機の
    ジョブは記録の前に済んだ run を索引に載せるので，どちらの機でも同じ数になる．
    """
    listed = subprocess.run(["git", "ls-files", "snapshots"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    return sum(bool(re.fullmatch(r"snapshots/[^/]*_relocated_[^/]*/metrics/eval\.json", p)) for p in listed)


def num(v: float, digits: int, *, signed: bool = False) -> str:
    """表示の桁で丸めた値．負号は U+2212（principles.md §3）．"""
    return f"{v:{'+' if signed else ''}.{digits}f}".replace("-", "−")


def interval(d: dict, digits: int) -> str:
    """差と，その95%信頼区間．"""
    low, high = (num(d[k], digits, signed=True) for k in ("ci_low", "ci_high"))
    return f"{num(d['mean'], digits, signed=True)} [{low}, {high}]"


def png(rows: list[list[float]], cell: int = 8) -> bytes:
    """0–1 の値の格子を，値が大きいほど濃い灰色の PNG にする．一マスは cell 画素の正方形．"""
    raw = b"".join(
        b"\x00" + bytes(g for v in row for g in [round(255 * (1 - v))] * cell)
        for row in rows for _ in range(cell)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", len(rows[0]) * cell, len(rows) * cell, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# 質問を読み終えた6位置の名．プロンプトの末尾から数えた位置で引く（研究文書 §8.5 の並び）．
ENDING = {-6: "質問文の末尾", -5: "<end_of_utterance>", -4: "改行", -3: "Ass", -2: "istant", -1: ":（今の 𝒉）"}
LEAD = 3  # プロンプトの先頭で，画像の前に来る書式の語の数
ARMS = ("baseline", "proposal", "ablation")
# 本筋で見せる群の比較（principles.md §2）
PAIRS = (("proposal", "baseline"), ("proposal", "ablation"))


def seed_mean(runs: dict, cells: Callable[[dict], list[list[float]]]) -> list[list[float]]:
    """乱数種の平均．cells は一つの run から（層，位置）の格子を取り出す．"""
    grids = [cells(run) for run in runs.values()]
    return [[sum(v) / len(v) for v in zip(*rows)] for rows in zip(*grids)]


def place(offset: int) -> str:
    """文の中で位置を指す名．表の見出しでは ENDING のトークンの名を使う．"""
    return "今の 𝒉" if offset == -1 else ENDING[offset]


def chosen(choice: dict) -> str:
    """③で整合を掛ける場所を，図の名で言う．"""
    if len(choice["positions"]) > 1:
        return f"第{choice['layer']}層の6位置すべて"
    return f"第{choice['layer']}層の「{place(choice['positions'][0])}」"


def baseline_map(mapped: dict) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """baseline の，乱数種で平均したマップ．（先頭，質問文，末尾の6位置）の三つで，どれも層ごとの行．

    先頭は書式の LEAD 語と画像．質問文は，半数以上の問が届く位置までに切る．
    """
    assert mapped["layout"] == {"head": 70, "tail": 5}, mapped["layout"]  # LEAD・図の区分・ENDING の前提
    runs, prompts = mapped["sets"]["reported"]["baseline"], mapped["jacobian_prompts"]
    reach = sum(2 * n >= prompts for n in next(iter(runs.values()))["map"]["question_prompts"])
    head = seed_mean(runs, lambda run: run["map"]["head"])
    question = [row[:reach] for row in seed_mean(runs, lambda run: run["map"]["question"])]
    ending = seed_mean(runs, lambda run: run["ending"][str(mapped["primary_dim"])])
    return head, question, ending


def s_map_result(mapped: dict) -> Slide:
    s = Slide("① の結果：baseline のマップ")
    head, question, ending = baseline_map(mapped)
    grid = [h + q + e[1:] for h, q, e in zip(head, question, ending, strict=True)]
    # 濃さの上限は，第1–31層の，先頭の位置を除いた値から取る：最後の層は定義から末尾だけが 1 になり，
    # 先頭の書式の語は値が特に大きい（左端の黒）．濃さは平方根に比例させ，小さい値も見えるようにする
    top = max(v for row in grid[:-1] for v in row[1:])
    shade = [[min(v / top, 1.0) ** 0.5 for v in row] for row in reversed(grid)]

    cols = [("書式", LEAD), ("画像", len(head[0]) - LEAD), ("質問文", len(question[0])), ("書式", len(ending[0]) - 1)]
    n, layers = len(shade[0]), len(shade)
    cw, rh = 9.3 / n, 0.085
    gx = X0 + (W - n * cw) / 2
    # 層の名は行の中心に置くので，最上段の名が本文の始まりより上に出ない分だけ図を下げる
    lift = line_h(DIAG) / 2 - rh / 2
    y = s.take(lift + layers * rh + 0.1 + line_h(DIAG)) + lift
    s.picture(png(shade), gx, y, n * cw, layers * rh)
    for layer in (layers, mapped["site_layer"], 1):
        s.label(gx - 1.15, y + (layers - layer + 0.5) * rh - line_h(DIAG) / 2, 1.0, f"第{layer}層", algn="r")
    s.label(gx + n * cw + 0.15, y + (layers - mapped["site_layer"] + 0.5) * rh - line_h(DIAG) / 2, 1.05,
            "今の 𝒉", algn="l")
    x, by = gx, y + layers * rh + 0.06
    for name, k in cols:
        s.line(x + 0.03, by, x + k * cw - 0.03, by, arrow=False)
        s.label(x - 0.3, by + 0.04, k * cw + 0.6, name)
        x += k * cw
    s.text([
        ("濃いほど相対感度が大．質問文は問の半数が届く位置まで", BODY, INK),
        ("左端の黒：先頭の書式の語．値が特に大きい．理由は未調査", BODY, INK),
    ])
    s.text([(f"→ {map_reading(mapped)}", BODY, RED)])
    return s


def map_reading(mapped: dict) -> str:
    """baseline のマップの記述（研究文書 §8.5 第1歩の結果）．整合を掛けた層で，画像のどの位置より
    質問文（問の半数が届く位置まで）と今の 𝒉 が大きいことを確かめてから言う．
    """
    layer = mapped["site_layer"]
    head, question, ending = (grid[layer - 1] for grid in baseline_map(mapped))
    if not max(head[LEAD:]) < min(question) <= max(question) < ending[-1]:
        raise SystemExit("マップの記述が値に合わない：研究文書 §8.5 第1歩の結果を読み直し，この文を書き直す")
    return f"第{layer}層では，画像より質問文と今の 𝒉 の相対感度が大きい"


def s_site_result(mapped: dict) -> Slide:
    choice, dim, layer = mapped["choice"], str(mapped["primary_dim"]), mapped["site_layer"]
    s = Slide(f"① の結果：第{layer}層の6位置")
    s.text([
        ("6位置：質問を読み終えた位置．質問文の末尾と後ろの書式", BODY, INK),
        ("Assistant: は Ass・istant・: の3トークンに分かれる", BODY, INK),
    ])
    rows = [["群", *(ENDING[o] for o in choice["offsets"])]]
    for arm in ARMS:
        row = seed_mean(mapped["sets"]["reported"][arm], lambda run: run["ending"][dim])[layer - 1]
        rows.append([arm, *(num(v, 3) for v in row)])
    s.table([1.5, 1.8, 2.8, 0.9, 0.9, 1.1, 1.9], rows, center=tuple(range(1, 7)),
            bold=((1, 1 + choice["offsets"].index(choice["largest"])),))
    s.text([
        ("値：相対感度（方向32本）．乱数種3つの平均", BODY, INK),
        ("太字：baseline の最大", BODY, INK),
    ])
    s.text([(f"→ baseline の相対感度は，{place(choice['largest'])}で最大", BODY, RED)])
    return s


def site_differences(mapped: dict) -> list[list[dict]]:
    """整合を掛けた層の6位置ごとの，PAIRS の差と 95% 区間（研究文書 §4.3）．位置の順に並ぶ．"""
    reported, dim, layer = mapped["sets"]["reported"], str(mapped["primary_dim"]), mapped["site_layer"]

    def at(arm: str, k: int) -> dict[int, float]:
        return {int(seed): run["ending"][dim][layer - 1][k] for seed, run in reported[arm].items()}

    return [[paired_difference(at(a, k), at(b, k)) for a, b in PAIRS] for k in range(len(ENDING))]


def site_reading(mapped: dict) -> str:
    """6位置での proposal の動き．両方の比較で区間が 0 を含まない位置だけを，向きごとに名指す．"""
    offsets, moved = mapped["choice"]["offsets"], {"低下": [], "上昇": []}
    for offset, pair in zip(offsets, site_differences(mapped), strict=True):
        if all(d["excludes_zero"] for d in pair) and len({d["mean"] > 0 for d in pair}) == 1:
            moved["上昇" if pair[0]["mean"] > 0 else "低下"].append(place(offset))
    said = "，".join(f"{'・'.join(where)}で{way}" for way, where in moved.items() if where)
    return f"proposal は{said}" if said else "proposal が両方と違う位置は無い"


def s_site_differences(mapped: dict) -> Slide:
    s = Slide(f"① の結果：第{mapped['site_layer']}層の6位置での差")
    rows = [["位置", *(f"{a} − {b}" for a, b in PAIRS)]]
    bold = []
    for r, (offset, pair) in enumerate(zip(mapped["choice"]["offsets"], site_differences(mapped), strict=True), 1):
        rows.append([ENDING[offset], *(interval(d, 3) for d in pair)])
        bold += [(r, c) for c, d in enumerate(pair, 1) if d["excludes_zero"]]
    s.table([3.0, 3.8, 3.8], rows, center=(1, 2), bold=tuple(bold))
    s.text([
        ("値：相対感度の差と95%信頼区間．太字は区間が 0 を含まない差", BODY, INK),
        ("注意：位置どうしの和は一定でなく，依存が移ったとは言えない", BODY, INK),
    ])
    s.text([(f"→ {site_reading(mapped)}", BODY, RED)])
    return s


def layer_tails(mapped: dict, seed: str = "42") -> list[tuple[int, float, float]]:
    """整合を掛けた層ごとの（層，baseline の末尾，proposal の末尾）．どれも同じ乱数種で，層の順．"""
    dim, sets = str(mapped["primary_dim"]), mapped["sets"]

    def tail(run: dict, layer: int) -> float:
        return run["ending"][dim][layer - 1][-1]

    return [(layer, tail(sets["reported"]["baseline"][seed], layer), tail(runs[seed], layer))
            for layer, runs in sorted([(mapped["site_layer"], sets["reported"]["proposal"]),
                                       *((r[seed]["layer"], r) for r in sets["layer"].values())])]


def layer_reading(mapped: dict) -> str:
    """掛けた層の末尾の向き（研究文書 §8.5 第1歩）．どの層でも下がるときだけ「層によらない」と言う．"""
    tails = layer_tails(mapped)
    fell = [f"第{layer}層" for layer, before, after in tails if after < before]
    rose = [f"第{layer}層" for layer, before, after in tails if after >= before]
    if not rose:
        return "どの層でも低下．依存の低下は層によらない見込み"
    if fell == [f"第{mapped['site_layer']}層"]:
        return f"低下は第{mapped['site_layer']}層だけ．第{mapped['site_layer']}層に固有の見込み"
    return f"{'・'.join(fell)}で低下，{'・'.join(rose)}で上昇．層によらないとは言えない"


def s_layers_result(mapped: dict) -> Slide:
    s = Slide("① の結果：整合を掛ける層の変更")
    rows = [["整合を掛けた層", "baseline", "proposal", "変化"]]
    for layer, before, after in layer_tails(mapped):
        rows.append([f"第{layer}層", num(before, 3), num(after, 3), num(100 * (after / before - 1), 1, signed=True) + "%"])
    s.table([2.6, 2.0, 2.0, 2.0], rows, center=(0, 1, 2, 3))
    s.text([
        ("値：掛けた層の，プロンプトの最後の位置の相対感度", BODY, INK),
        ("変化：(proposal − baseline) ÷ baseline", BODY, INK),
        ("乱数種42の1本．区間は作れず，向きだけを読む", BODY, INK),
    ])
    s.text([(f"→ {layer_reading(mapped)}", BODY, RED)])
    return s


def weight(w: float) -> str:
    """λ の表記．②の計画のスライドと同じく 0 と 1.0 と書く．"""
    return "0" if w == 0 else str(w)


def lambda_cost(curve: dict) -> float:
    """λ > 0 の各点の正解率の，λ = 0 からの差の絶対値の最大（pt）を，小数第1位へ切り上げたもの．

    「以内」と書くので，丸めで上限を下げない．差は百分率の引き算なので，先に第2位で丸めて浮動小数の端数を落とす．
    """
    zero = curve["accuracy"][curve["align_weight"].index(0.0)]
    largest = max(abs(a - zero) for w, a in zip(curve["align_weight"], curve["accuracy"], strict=True) if w)
    return math.ceil(round(largest, 2) * 10) / 10


def lambda_below(curve: dict, dim: str) -> bool:
    """λ > 0 のどの点でも，相対感度が λ = 0 より低いか（研究文書 §8.4 の結果の記述）．"""
    leaning, zero = curve["sensitivity"][dim], curve["align_weight"].index(0.0)
    return all(v < leaning[zero] for k, v in enumerate(leaning) if k != zero)


def lambda_shape(curve: dict, dim: str) -> str:
    """まとめの一行：λ > 0 の点が λ = 0 より低いかと，単調か．"""
    monotone = curve["sensitivity_falls_at_every_step"]
    if lambda_below(curve, dim) and not monotone:
        return "相対感度は λ > 0 のどの点も λ = 0 より低いが，単調でない"
    return "相対感度は λ に伴い単調に低下" if monotone else "相対感度は λ = 0 より高い点がある"


def lambda_reading(curve: dict) -> str:
    """研究文書 §8.4 が測る前に決めた判定：相対感度がどの段でも下がるか．"""
    return ("相対感度は λ に伴い単調に低下．今の低下は強さに応じる"
            if curve["sensitivity_falls_at_every_step"]
            else "相対感度は単調に下がらず，今の低下は強さに応じるとは言えない")


def s_lambda_result(curve: dict, dim: str, spread: float) -> Slide:
    """spread は baseline を乱数種だけ替えたときの正解率の幅（pt．研究文書 §5.2）．"""
    s = Slide("② の結果：整合の強さ λ の変更")
    leaning = curve.get("sensitivity", {}).get(dim)
    rows = [["λ", "読み取り精度（%）", "正解率（%）", *(["相対感度"] if leaning else [])]]
    for k, w in enumerate(curve["align_weight"]):
        rows.append([weight(w), num(100 * curve["probe_accuracy"][k], 1), num(curve["accuracy"][k], 2),
                     *([num(leaning[k], 3)] if leaning else [])])
    s.table([1.6] + [2.6] * (len(rows[0]) - 1), rows, center=tuple(range(len(rows[0]))))
    s.text([
        ("乱数種42の1本．区間は作れず，向きと形を読む", BODY, INK),
        (f"正解率：λ = 0 との差は{num(lambda_cost(curve), 1)}pt以内．乱数種だけでも{num(spread, 2)}pt動く", BODY, INK),
        *([("相対感度：λ > 0 のどの点も λ = 0 より低い", BODY, INK)] if leaning and lambda_below(curve, dim) else []),
    ])
    if leaning:
        s.text([(f"→ {lambda_reading(curve)}", BODY, RED)])
    return s


# 研究文書 §8.5 第2歩の表の行ごとの読み．行は summarize_relocation.py が名指す．
RELOCATION = {
    "falls_by_labels": "この場所でも答えの依存は下がる．低下はラベルの意味による",
    "falls_not_by_labels": "依存は下がるが，ラベルの意味には帰せない",
    "kept_accuracy_up": "正解率が上がる．問いに初めて肯定の材料",
    "kept_accuracy_down": "手順の情報は答えの生成に関与し，正解率を損なう",
    "kept_accuracy_undecided": "答えの生成に関与するが，正解率への効果の向きは決められない",
}


def s_relocate_result(relocation: dict) -> Slide:
    s = Slide("③ の結果：整合を掛ける位置の変更")
    pairs = tuple(f"{a}_minus_{b}" for a, b in PAIRS)  # summarize_relocation.py の鍵
    leaning = relocation["sensitivity"][str(relocation["primary_dim"])]
    accuracy = relocation["accuracy"]
    s.text([(f"整合を掛けた場所：{chosen(relocation)}", BODY, INK)])
    s.table([2.6, 2.4, 2.4, 2.4], [
        ["乱数種3つの平均", *ARMS],
        ["相対感度", *(num(leaning[arm]["mean"], 3) for arm in ARMS)],
        ["正解率（%）", *(num(accuracy[arm]["mean"], 2) for arm in ARMS)],
    ], center=(1, 2, 3))
    s.table([2.6, 3.6, 3.6], [
        ["差と95%信頼区間", *(f"{a} − {b}" for a, b in PAIRS)],
        ["相対感度", *(interval(leaning[pair], 3) for pair in pairs)],
        ["正解率（pt）", *(interval(accuracy[pair], 2) for pair in pairs)],
    ], center=(1, 2), bold=tuple(
        (r, c) for r, read in enumerate((leaning, accuracy), start=1)
        for c, pair in enumerate(pairs, start=1) if read[pair]["excludes_zero"]
    ))
    s.text([
        ("太字：区間が 0 を含まない差", BODY, INK),
        (f"→ {RELOCATION[relocation['outcome']]}", BODY, RED),
    ])
    return s


def s_summary(mapped: dict | None, curve: dict | None, relocation: dict | None, trained: int) -> Slide:
    """結果のスライドの読みを，同じ関数から一行ずつ集め直し，次の一手を添える．"""
    s = Slide("まとめと次の一手")
    dim = str(mapped["primary_dim"]) if mapped else ""
    lines = []
    if mapped:
        lines += [f"①：{t}" for t in (map_reading(mapped), site_reading(mapped))]
        if "layer" in mapped["sets"]:
            lines.append(f"①：掛けた層では{layer_reading(mapped).split('．')[0]}（乱数種1つ）")
    if curve:
        lines.append(f"②：乱数種1つ．λ = 1.0 まで正解率の差は{num(lambda_cost(curve), 1)}pt以内")
        lines.append(f"②：{lambda_shape(curve, dim)}" if dim in curve.get("sensitivity", {})
                     else "②：相対感度は測っていない")
    if relocation:
        lines.append(f"③：{RELOCATION[relocation['outcome']]}")
    s.text([(t, BODY, INK) for t in lines])
    if relocation:
        nxt = "→ 次：④の実装と学習" if relocation["outcome"] == "falls_by_labels" else "→ 次：④は行わず，次の問いへ"
    else:
        nxt = f"→ 次：③の残り{RELOCATED_RUNS - trained}本を学習し，相対感度と正解率で判定"
    s.text([(nxt, BODY, RED)])
    return s


def slides() -> list[Slide]:
    """方針のスライドに，結果のあるものだけ結果のスライドを続け，最後にまとめる．"""
    mapped = load("probe-dependence-map.json")
    sweep = load("summarize-sweep.json")
    relocation = load("summarize-relocation.json")
    measured = mapped is not None and "reported" in mapped["sets"]
    curve = (sweep or {}).get("strength_curve")
    trained = trained_relocated()

    deck = [s_done(measured, bool(curve), trained), s_questions(), s_candidates(),
            s_plan((measured, bool(curve), bool(relocation), False), trained), s_map()]
    if measured:
        deck += [s_map_result(mapped), s_site_result(mapped), s_site_differences(mapped)]
        if "layer" in mapped["sets"]:
            deck.append(s_layers_result(mapped))
    deck.append(s_lambda())
    if curve:
        spread = next(p for p in sweep["points"] if p["arm"] == "baseline")["accuracy"]["range"]
        deck.append(s_lambda_result(curve, str(mapped["primary_dim"]) if mapped else "", spread))
    deck.append(s_relocate(mapped["choice"] if measured else None))
    if relocation:
        deck.append(s_relocate_result(relocation))
    deck.append(s_direction())
    if measured or curve:
        deck.append(s_summary(mapped if measured else None, curve, relocation, trained))
    return deck


# ---------------------------------------------------------------- 包み


def build() -> None:
    """BASE の部品から，スライドと画像と縮小図を外し，組んだスライドを入れて OUT に書く．"""
    with zipfile.ZipFile(BASE) as z:
        parts = {name: z.read(name) for name in z.namelist()
                 if not name.startswith(("ppt/slides/", "ppt/media/")) and name != "docProps/thumbnail.jpeg"}
    root_rels = parts["_rels/.rels"].decode("utf-8")
    parts["_rels/.rels"] = re.sub(r"<Relationship [^>]*thumbnail[^>]*/>", "", root_rels).encode("utf-8")

    # スライドの関係：雛形のレイアウトの第1（タイトル スライド）を表紙に，第6（タイトルのみ）を本文に使う
    slide_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
        'Target="../slideLayouts/slideLayout{n}.xml"/></Relationships>'
    )
    titles = ["進捗報告"]
    parts["ppt/slides/slide1.xml"] = cover().encode("utf-8")
    parts["ppt/slides/_rels/slide1.xml.rels"] = slide_rels.format(n=1).encode("utf-8")
    image_rel = (
        '<Relationship Id="rId{j}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="../media/{name}"/></Relationships>'
    )
    for k, slide in enumerate(slides(), start=2):
        titles.append(slide.title)
        rels = slide_rels.format(n=6)
        for j, png_bytes in enumerate(slide.media, start=2):
            name = f"progress{k}_{j - 1}.png"
            parts[f"ppt/media/{name}"] = png_bytes
            rels = rels.replace("</Relationships>", image_rel.format(j=j, name=name))
        parts[f"ppt/slides/slide{k}.xml"] = slide.xml().encode("utf-8")
        parts[f"ppt/slides/_rels/slide{k}.xml.rels"] = rels.encode("utf-8")

    count = len(titles)
    pres = parts["ppt/presentation.xml"].decode("utf-8")
    ids = "".join(f'<p:sldId id="{256 + k}" r:id="rId{201 + k}"/>' for k in range(count))
    parts["ppt/presentation.xml"] = re.sub(
        r"<p:sldIdLst>.*?</p:sldIdLst>", f"<p:sldIdLst>{ids}</p:sldIdLst>", pres, flags=re.DOTALL).encode("utf-8")

    prels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    prels = re.sub(r"\s*<Relationship [^>]*relationships/slide\"[^>]*/>", "", prels)
    prels = prels.replace("</Relationships>", "".join(
        f'<Relationship Id="rId{201 + k}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{k + 1}.xml"/>' for k in range(count)) + "</Relationships>")
    parts["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")

    ct = parts["[Content_Types].xml"].decode("utf-8")
    ct = re.sub(r"\s*<Override PartName=\"/ppt/slides/[^>]*/>", "", ct)
    ct = ct.replace("</Types>", "".join(
        f'<Override PartName="/ppt/slides/slide{k + 1}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for k in range(count)) + "</Types>")
    parts["[Content_Types].xml"] = ct.encode("utf-8")

    # app.xml：枚数・題の数・TitlesOfParts の size と要素数の四つを揃える
    app = parts["docProps/app.xml"].decode("utf-8")
    fonts = [FONT, MATH, "游ゴシック Light", "游ゴシック"]  # 本文・𝒉・テーマの題と本文
    items = [f"<vt:lpstr>{escape(t)}</vt:lpstr>" for t in [*fonts, "Office テーマ", *titles]]
    heading = (
        '<HeadingPairs><vt:vector size="6" baseType="variant">'
        f"<vt:variant><vt:lpstr>使用されているフォント</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(fonts)}</vt:i4></vt:variant>"
        "<vt:variant><vt:lpstr>テーマ</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant>"
        f"<vt:variant><vt:lpstr>スライド タイトル</vt:lpstr></vt:variant><vt:variant><vt:i4>{count}</vt:i4></vt:variant>"
        "</vt:vector></HeadingPairs>"
    )
    app = re.sub(r"<HeadingPairs>.*?</HeadingPairs>", heading, app, flags=re.DOTALL)
    app = re.sub(r"<TitlesOfParts>.*?</TitlesOfParts>",
                 f'<TitlesOfParts><vt:vector size="{len(items)}" baseType="lpstr">{"".join(items)}</vt:vector></TitlesOfParts>',
                 app, flags=re.DOTALL)
    app = re.sub(r"<Slides>\d+</Slides>", f"<Slides>{count}</Slides>", app)
    app = re.sub(r"<TotalTime>\d+</TotalTime>", "<TotalTime>0</TotalTime>", app)
    parts["docProps/app.xml"] = app.encode("utf-8")

    core = parts["docProps/core.xml"].decode("utf-8")
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts["docProps/core.xml"] = re.sub(r"(<dcterms:modified[^>]*>)[^<]*", rf"\g<1>{now}", core).encode("utf-8")

    # 中身が変わらなければ書かない：更新時刻だけが違う版を作ると，組み直すたびに記録が要る
    if OUT.exists():
        with zipfile.ZipFile(OUT) as z:
            filed = {name: z.read(name) for name in z.namelist() if name != "docProps/core.xml"}
        if filed == {name: b for name, b in parts.items() if name != "docProps/core.xml"}:
            print(f"nothing to change: {OUT}")
            return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    order = ["[Content_Types].xml"] + sorted(name for name in parts if name != "[Content_Types].xml")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in order:
            z.writestr(name, parts[name])
    print(OUT)


if __name__ == "__main__":
    build()
