"""Build the 27 November progress report from whatever `analyses/` holds. No torch.

The deck runs in one order: what was done since the last report, the questions taken at
the conference, what the language model is given and which positions the experiments
name, why the alignment may go unused, and the five experiments of context.md §8 with
where each stands. Then every experiment with a result on file gets its result slides,
each opening on why it is looked at and saying how it was measured, and the experiments
without one share a single "今後" slide that gives their methods alone, so no slide
promises a result that is not there; a closing slide gathers the readings. The same
command writes the deck before anything has run and after: the job that runs the
experiments (`scripts/run_plan.sh`) calls it after every stage, and pulling its commits
is what updates the deck.

What a result slide says is not composed here. Where the research document fixed an
outcome's meaning before the runs existed, the analyses name the outcome they fell in and
this prints the sentence that outcome was given (presentations/principles.md §1). Where
the document only describes what was measured (the map, the differences at the six
positions, the layer runs), the sentence is chosen by the same test the document states,
and a description the numbers no longer bear stops the build rather than being printed.
The document still has to be brought up to the same results by hand — a slide written
here is ahead of it until then.

The package is written at the level of its parts, from the parts of an earlier report
(principles.md §5): the lab's template, 游明朝, 24pt body lines, and the template's table
style. Colour comes from the poster's greens alone — the tables' fill, the map, and the
bold deep green of the one conclusion a slide draws (principles.md §4). Standard library
only, so it runs on the machine that trained the runs and on the one that opens the deck
alike.

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
INK, WHITE = "000000", "FFFFFF"
# 色はポスターの緑の濃淡だけで組む（principles.md §4）．PALE は表の地，MID はマップの中間と表の見出しの地，
# DEEP はマップの最も濃い色．結論の字（ACCENT）はポスターの基調の緑で，DEEP では暗くて目立たない．
PALE, MID, DEEP = "E2EFDA", "8FBF78", "2E6B34"
ACCENT = "4A8C45"
TABLE_STYLE = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"  # 雛形の表の様式．地の色だけを上の緑で塗り替える
BODY, SUB, TABLE = 2400, 2000, 1800  # 本文，二段目の箇条，表の字の大きさ（1/100 pt）
TABLE_HANG = 0.25  # 表の中の箇条の記号を左に出す幅（インチ）：18pt の一字分
DIAG = TABLE  # 模式図の中の語（principles.md §3：表と図は 14pt 以上）
SMALL = 1400  # 図の中のトークンの文字列（表と図の下限の 14pt）
TOKEN, PITCH = 0.3, 0.42  # トークンの正方形の一辺と，並べる間隔（インチ）
X0, W = 0.92, 11.5  # 本文の左端と幅（雛形の題と同じ）
Y0, GAP = 1.8, 0.25  # 本文の始まりと，塊どうしの間隔（principles.md §4）
BOTTOM = 7.0  # 本文の下端：スライドの高さ 7.5 インチから，端の 0.5 インチを除く
INDENT = {0: 0.0, 1: 0.35, 2: 0.8}  # 箇条の段ごとの左の余白（インチ）．記号は 0.35 だけ左に出す

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
    """折り返しを描く環境に任せない：幅の 94% に収まらない行を見つけたら止める．改行（\\n）で割った行ごとに測る．"""
    for line in typeset(text).split("\n"):  # run() と同じ字で測る
        w = text_width(line, sz)
        if w > 0.94 * width:
            raise SystemExit(f"収まらない（{w:.2f} > {0.94 * width:.2f}in）：{line}")


def faces(text: str) -> list[tuple[str, str]]:
    """文字を書体ごとに切る．𝒉 と λ は数式の書体で組む（游明朝では λ が全角になり，前後が空きすぎる）．"""
    return [(part, MATH if part in ("𝒉", "λ") else FONT) for part in re.split("(𝒉|λ)", text) if part]


# _TOUCHING はラテン文字の語（と 𝒉）が仮名・漢字に接する所．数の後の pt は単位なので除く．
# 数字と約物はどちらの字にも数えないので，その隣には空白を入れない．_CIRCLED は丸数字の後ろで，
# 次が空白・約物・範囲の – でない所（「③–⑤」「①・②」には入れない）．
_KANA_KANJI = "ぁ-ゖァ-ヺ一-鿿"
_TOUCHING = re.compile(rf"(?<=[A-Za-z𝒉])(?<!\dpt)(?=[{_KANA_KANJI}])|(?<=[{_KANA_KANJI}])(?=[A-Za-z𝒉])")
_CIRCLED = re.compile("(?<=[①-⑳])(?=[^\\s・，．（）「」–])")


def typeset(text: str) -> str:
    """principles.md §3 の空白：ラテン文字の語と和文の間と，丸数字の後ろに一つ入れる．

    位置の名や手順の番号を文に差し込むと語の境が詰まりうるので，書いた文でなく，組むときに揃える．
    """
    return _CIRCLED.sub(" ", _TOUCHING.sub(" ", text))


def run(text: str, sz: int = BODY, *, color: str = INK, bold: bool = False) -> str:
    """一続きの文字．空白を typeset で整え，書体ごとに run を分ける．

    改行（\\n）は段落の中の改行にする：続きの行は段落の字下げを保つので，箇条の続きが新しい項目に見えない．
    """
    def piece(line: str) -> str:
        return "".join(
            f'<a:r><a:rPr lang="ja-JP" sz="{sz}" b="{int(bold)}" i="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{face}"/><a:ea typeface="{face}"/><a:cs typeface="{face}"/>'
            f"</a:rPr><a:t>{escape(part)}</a:t></a:r>"
            for part, face in faces(line)
        )

    return f'<a:br><a:rPr lang="ja-JP" sz="{sz}"/></a:br>'.join(piece(line) for line in typeset(text).split("\n"))


def para(runs: str, *, algn: str = "l", sz: int = BODY, before: int = 0, left: float = 0.0,
         hang: float = 0.0) -> str:
    """一つの段落．before は前の段落との間隔（1/100 pt），left は左の余白（インチ）．

    hang が正なら「・」の箇条にし，記号をその幅だけ左に出す．
    """
    bullet = f'<a:buFont typeface="{FONT}"/><a:buChar char="・"/>' if hang else "<a:buNone/>"
    return (
        f'<a:p><a:pPr algn="{algn}" marL="{emu(left)}" indent="{-emu(hang)}">'
        f'<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
        f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef>{bullet}</a:pPr>{runs}'
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

    def text(self, lines: list[tuple[str, int, str] | tuple[str, int, str, int]], *, x: float = X0, w: float = W, y: float | None = None,
             algn: str = "l", gap_after: float = GAP, pad: float = 0.1) -> float:
        """行の並び（文字，大きさ，色[，段]）．段 1 は箇条，段 2 はその下の箇条．

        行どうしは，二段目の箇条の前を 3pt，ほかを 9pt 空ける（principles.md §4）．結論の色 ACCENT の行は太字にする
        （principles.md §3 の強調の項）．文字の中の改行（\\n）は，同じ段落の続きの行にする．
        """
        ps, h = [], 0.0
        for k, line in enumerate(lines):
            t, sz, color, level = (*line, 0)[:4]
            fits(t, sz, w - 2 * pad - INDENT[level])
            before = 0 if k == 0 else 300 if level == 2 else 900
            ps.append(para(run(t, sz, color=color, bold=color == ACCENT), algn=algn, sz=sz, before=before,
                           left=INDENT[level], hang=INDENT[1] if level else 0.0))
            h += (t.count("\n") + 1) * line_h(sz) + before / 100 / 72
        if y is None:
            y = self.y
            self.y = y + h + gap_after
        # 文字の箱は，これまでの進捗報告と同じく左右に 0.1 インチの内側の余白を持つ（字の端が題の字の端と揃う）
        self._sp(x, y, w, h, body="".join(ps), txbox=True, pad=pad)
        return y

    def box(self, x: float, y: float, w: float, h: float, lines: list[str] | None = None, *,
            fill: str | None = None, line: str | None = INK, algn: str = "ctr", sz: int = BODY, pad: float = 0.15,
            geom: str = "rect", color: str = INK) -> None:
        for t in lines or []:
            fits(t, sz, w - 2 * pad)
        body = "".join(para(run(t, sz, color=color), algn=algn, sz=sz) for t in lines) if lines else ""
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

    def table(self, cols: list[float], rows: list[list[str | list[str] | None]], *,
              center: tuple[int, ...] = (), bold: tuple[tuple[int, int], ...] = ()) -> None:
        """雛形の表の様式で，地をポスターの緑で塗る：見出しの行は MID，ほかの行は PALE（principles.md §4）．

        セルは文字か，文字の並びか，None．並びは一つずつ「・」の箇条にする——一つのセルに項目を改行で並べると，
        文の折り返しと見分けられない（principles.md §3）．文字の中の改行（\\n）は，同じ文の続きの行にする．
        None は上のセルと縦に結合する（同じ群の行を一つのセルで括る．principles.md §3 の表の項）．
        行の高さは，行の中で最も行数の多いセルに合わせる（結合したセルは一行と数える）．bold は太字にするセル
        （行，列）．区間が 0 を含まない差に使う（principles.md §2）．
        """
        items = [[[] if cell is None else [cell] if isinstance(cell, str) else cell for cell in row] for row in rows]
        heights = [max(sum(t.count("\n") + 1 for t in cell) for cell in row) * line_h(TABLE) + 0.095 for row in items]

        def span(r: int, c: int) -> int:
            """r 行 c 列のセルが，下の None をいくつ括るか（自身を含む行数）．"""
            n = 1
            while r + n < len(rows) and rows[r + n][c] is None:
                n += 1
            return n

        i = self._id()
        y = self.take(sum(heights))
        trs = []
        for r, row in enumerate(items):
            tcs = []
            for c, cell in enumerate(row):
                algn = "ctr" if r == 0 or c in center else "l"
                hang = TABLE_HANG if isinstance(rows[r][c], list) else 0.0
                ps = []
                for t in cell:
                    fits(t, TABLE, cols[c] - 0.16 - hang)
                    ps.append(para(run(t, TABLE, bold=(r, c) in bold), algn=algn, sz=TABLE, left=hang, hang=hang))
                merge = ' vMerge="1"' if rows[r][c] is None else f' rowSpan="{span(r, c)}"' if span(r, c) > 1 else ""
                tcs.append(
                    f"<a:tc{merge}><a:txBody><a:bodyPr/><a:lstStyle/>"
                    f"{''.join(ps) or '<a:p><a:endParaRPr lang=\"ja-JP\"/></a:p>'}</a:txBody>"
                    f'<a:tcPr anchor="ctr" marL="73152" marR="73152" marT="27432" marB="27432">'
                    f'<a:solidFill><a:srgbClr val="{MID if r == 0 else PALE}"/></a:solidFill></a:tcPr></a:tc>'
                )
            trs.append(f'<a:tr h="{emu(heights[r])}">{"".join(tcs)}</a:tr>')
        grid = "".join(f'<a:gridCol w="{emu(c)}"/>' for c in cols)
        self.shapes.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{i}" name="Table {i}"/>'
            '<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{emu(X0 + (W - sum(cols)) / 2)}" y="{emu(y)}"/><a:ext cx="{emu(sum(cols))}" cy="{emu(sum(heights))}"/></p:xfrm>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr firstRow="1" bandRow="0"><a:tableStyleId>{TABLE_STYLE}</a:tableStyleId></a:tblPr>'
            f'<a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>'
        )

    def xml(self) -> str:
        set_title = typeset(self.title)
        assert text_width(set_title, 4400) < 11.0, self.title  # 題は一行（44pt で約18字まで）
        assert self.y - GAP <= BOTTOM + 1e-6, f"本文が下端を越える（{self.y - GAP:.2f}in）：{self.title}"
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


# ---------------------------------------------------------------- 図の部品


def tokens(s: Slide, x: float, y: float, n: int, *, last_h: bool = False, dots: bool = False) -> list[float]:
    """横に並ぶ n 個のトークンの正方形．白地に墨の線で，last_h なら最後の一つを 𝒉 として墨で塗る．

    dots なら，最後の一つの手前に縦の点の代わりの「…」を一つ分置き，略したトークンを示す．各正方形の左端の x を返す．
    """
    slots = [x + k * PITCH for k in range(n + dots)]
    xs = slots[:n - 1] + slots[-1:] if dots else slots
    if dots:
        s.label(slots[n - 1] - 0.1, y - 0.04, TOKEN + 0.2, "…")
    for k, xk in enumerate(xs):
        s.box(xk, y, TOKEN, TOKEN, fill=INK if last_h and k == n - 1 else WHITE)
    return xs


# 質問を読み終えた6位置の，図での名．プロンプトの末尾から数えた位置で引く（研究文書 §8.5 の並び）．
# 文と表では，place が最後の位置を今の 𝒉 と呼ぶ．
ENDING = {-6: "質問文の末尾", -5: "<end_of_utterance>", -4: "改行", -3: "Ass", -2: "istant", -1: ":"}


def ending_widths() -> list[float]:
    """ending_row の箱の幅．字が幅の 94% に収まる幅に左右の余白を足し，0.6 インチを下限とする．"""
    return [max(text_width(typeset(t), DIAG) / 0.94 + 0.3, 0.6) for t in ENDING.values()]


def ending_span() -> tuple[float, float]:
    """ending_row が並べる箱の左端と右端の x．"""
    span = sum(ending_widths()) + 0.1 * (len(ENDING) - 1)
    return X0 + (W - span) / 2, X0 + (W + span) / 2


def ending_row(s: Slide, y: float) -> None:
    """質問を読み終えた6位置を，トークンの文字を入れた箱で横に並べる．最後の一つ（今の 𝒉）は墨で塗る．"""
    x = ending_span()[0]
    for k, (t, w) in enumerate(zip(ENDING.values(), ending_widths(), strict=True)):
        last = k == len(ENDING) - 1
        s.box(x, y, w, 0.45, [t], sz=DIAG, fill=INK if last else WHITE, color=WHITE if last else INK, pad=0.1)
        x += w + 0.1
    s.label(x + 0.02, y + 0.04, 1.1, "今の 𝒉", algn="l")


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


# ---------------------------------------------------------------- 前置きと計画


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


# 研究文書 §8 の順の5つの手順と，行ったときに「今回したこと」で言う文
STEPS = (
    "① 答えが依存する層と位置を測る",
    "② 整合の強さ λ を変える",
    "③ 掛ける層を変え，乱数種5つで学習し直す",
    "④ 依存の大きい位置に整合を掛け直す",
    "⑤ 答えを動かす方向に限って整合を掛ける",
)
DID = (
    "答えが，どの層・位置の表現に依存するかを測った",
    "整合の強さ λ を変えて学習し，効果を比べた",
    "掛ける層を変えて学習し直し，第16層と比べた",
    "整合を掛ける位置を変えて学習し，効果を比べた",
    "答えを動かす方向に限って整合を掛け，効果を比べた",
)


def s_done(done: tuple[bool, ...]) -> Slide:
    """前回からの差分（principles.md §2）．したことだけを言い，途中まで学習した本数は言わない．

    学会の質問から立てたのは①②④⑤の4つで，③は，学会で示した結果が第16層の外でも成り立つかを
    確かめるために本人が足した（研究文書 §8 の冒頭）．
    """
    s = Slide("今回したこと")
    lines = [("学会で受けた質問をもとに，4つの実験を計画した", BODY, INK, 1)]
    did = [t for t, d in zip(DID, done, strict=True) if d]
    if did:
        lines.append((f"そのうち{len(did)}つを行った", BODY, INK, 1))
        lines += [(t, SUB, INK, 2) for t in did]
    lines.append(("学会の結果が層によらないかを確かめる実験を，計画に加えた", BODY, INK, 1))
    s.text(lines)
    return s


def s_questions() -> Slide:
    """質疑の記録（presentations/20261116_PCSJ-IMPS2026_質疑.md）の聞き取りを，意味を保って言い直す（principles.md §2）．

    質問は一つの文の続きとして句読点の後で折り，応え方は項目ごとの箇条にする（principles.md §3 の表の項）．
    応え方には，聴き手の判断に関わらない実装の事情を書かない（principles.md §1）．
    """
    s = Slide("学会で受けた質問と応え方")
    s.text([("今回の実験は，学会で受けた次の質問から立てた", BODY, INK)])
    s.table(
        [6.3, 5.2],
        [
            ["受けた質問", "応え方"],
            ["なぜ causal attention の言語モデルなのか．\ncausal でなくてよく，むしろ結果を損ねたのでは",
             ["使ったモデルの性質で，選んでいない", "causal でないモデルと比べていない", "答えが依存する層・位置は測る"]],
            ["整合の制約が弱いのでは．λ を大きくするか，\n掛ける層を増やせば，何か変わるのでは",
             ["λ を変えて学習する", "層は増やさず，一つずつ変えて試す"]],
            ["入力ごとに LLM のどこが強く働くかは，\n多くの研究が調べている．それを参考にすれば，\n整合を掛けるべき場所が分かるのでは",
             ["依存の大きい位置に整合を掛け直す"]],
            ["𝒉 には Attention の Q の情報が強く出て，\nほかの情報は潰れているのでは．だから，\n𝒉 から読み取れる情報も限られるのでは（曖昧）",
             ["𝒉 は Q そのものではなく，\n次の層が Q・K・V を作る元", "聞き取りが曖昧で，測らない"]],
        ],
    )
    return s


def s_input() -> Slide:
    """プロンプトの並び（研究文書 §8.5 第1歩）と，以降のスライドが名指す位置．

    書式の文字列は，SmolVLM-500M-Instruct のプロセッサで組んだトークンの列と一字ずつ合わせてある（2026-11-27）．
    """
    s = Slide("言語モデル部への入力")
    s.text([("causal attention のもとで，各位置が何を読むかを示す", BODY, INK)])
    groups = [("書式", 3, False, "<|im_start|>User:"), ("画像", 5, True, "67トークン"),
              ("質問文", 5, True, "What's the girl holding?"), ("書式", 5, False, "")]
    sep = 0.45
    span = sum((n + dots) * PITCH for _, n, dots, _ in groups) - (PITCH - TOKEN) + sep * (len(groups) - 1)
    y = s.take(0.38 + 2 * line_h(DIAG) + 0.05 + 0.75 + 0.45)  # トークン，名の2行，括り線，広げた6位置
    x, xs = X0 + (W - span) / 2, []
    for k, (name, n, dots, note) in enumerate(groups):
        row = tokens(s, x, y, n, last_h=k == len(groups) - 1, dots=dots)
        mid, wide = (row[0] + row[-1] + TOKEN) / 2, row[-1] + TOKEN - row[0] + sep
        s.label(mid - wide / 2 - 0.3, y + 0.38, wide + 0.6, name)
        if note:
            s.label(mid - wide / 2 - 0.4, y + 0.38 + line_h(DIAG), wide + 0.8, note, sz=SMALL)
        xs += row
        x = row[-1] + PITCH + sep
    # 質問を読み終えた6位置（質問文の最後のトークンと，後ろの書式）を下に広げて見せる
    # 括り線の中央から下ろし，広げた並びの中央へ折って入れる（線は水平と垂直だけ）
    start, by = xs[-6], y + 0.38 + 2 * line_h(DIAG) + 0.05
    s.line(start, by, xs[-1] + TOKEN, by, arrow=False)
    s.label(start - 3.1, by + 0.04, 3.0, "質問を読み終えた6位置", algn="r")
    left, right = ending_span()
    down, turn = (start + xs[-1] + TOKEN) / 2, by + 0.4
    s.line(down, by, down, turn, arrow=False)
    s.line(down, turn, (left + right) / 2, turn, arrow=False)
    s.line((left + right) / 2, turn, (left + right) / 2, by + 0.75)
    ending_row(s, by + 0.75)
    s.text([
        ("causal attention なので，各位置は前のトークンだけを読む", BODY, INK, 1),
        ("画像の位置は，質問文をまだ読んでいない", BODY, INK, 1),
        ("今の 𝒉 は第16層の最後の位置で，ここに整合を掛けている", BODY, INK, 1),
    ])
    return s


def s_candidates() -> Slide:
    s = Slide("手順の情報が十分には使われない理由")
    s.text([("答えを良くしないなら，手順の情報は後段の層にあまり読まれない", BODY, INK),
            ("読まれにくい所は，次の二通りが考えられる", BODY, INK)])
    y = s.take(line_h(DIAG) + 1.75 + 0.38 + line_h(DIAG))
    # 位置：後段の層は 𝒉 より他の位置を主に読む（𝒉 からの矢印を破線にする）
    left = X0 + 0.2
    s.label(left, y, 5.4, "後段の層は，𝒉 より他の位置を主に読む", algn="l")
    gy = y + line_h(DIAG)
    s.box(left, gy + 0.4, 5.0, 0.5, ["後段の第17–32層"], sz=DIAG)
    ty = gy + 1.75
    xs = tokens(s, left + 0.35, ty, 4) + tokens(s, left + 2.35, ty, 4) + tokens(s, left + 4.35, ty, 1, last_h=True)
    for xk in xs:
        s.line(xk + TOKEN / 2, ty, xk + TOKEN / 2, gy + 0.9, dash=xk == xs[-1])
    s.label(left + 0.35, ty + 0.38, 1.56, "画像")
    s.label(left + 2.35, ty + 0.38, 1.56, "質問文")
    s.label(left + 4.0, ty + 0.38, 1.0, "𝒉")
    # 方向：手順の情報が，答えを動かさない方向に載る
    axes(s, X0 + 6.0, y, "手順の情報が，答えを動かさない方向に載る",
         [(0.6, 0.45, INK, "手順 A"), (2.9, 0.45, WHITE, "手順 B")], h=line_h(DIAG) + 1.75)
    s.text([("→ 整合を掛ける位置と方向を変えて試す", BODY, ACCENT)])
    return s


def s_plan(done: tuple[bool, ...]) -> Slide:
    """手順を計画の順に縦に並べ，右に状況を添える．矢印で結ばない（依存を見せない．principles.md §2）．

    状況は，結果が analyses/ にあれば「今回報告」，無ければ「未着手」．走っているかは git から読めないので，
    「学習中」とは書かない（principles.md §2 の状況の項）．
    """
    s = Slide("実験の計画")
    s.text([("質問と，十分には使われない理由をもとに，次の順で実験する", BODY, INK)])
    sw = max(text_width(typeset(t), BODY) for t in STEPS) / 0.94 + 0.2 + 0.6
    y = s.text([(t, BODY, INK) for t in STEPS], w=sw)
    s.text([("今回報告" if d else "未着手", BODY, INK) for d in done], x=X0 + sw, w=2.0, y=y)
    reported = "・".join(t[0] for t, d in zip(STEPS, done, strict=True) if d)
    if reported:
        s.text([(f"→ 今回は{reported}の結果を報告", BODY, ACCENT)])
    return s


# ---------------------------------------------------------------- 結果
#
# 結果のスライドは，analyses/ にその結果があるときだけ作る．値は analyses/ から取り，表示の桁で丸める．
# 読みの文の選び方は冒頭の docstring にある．読みを返す関数（*_reading と RELOCATION）は文だけを返し，
# 結論の印の「→ 」はスライドの側で付ける——まとめのスライドは同じ文を印なしで並べ直す．二行の読みは
# 一行目だけを並べ，マップの読みは二行とも並べる．λ の読みは，一行目だけでは二行目の「単調には下がらない」が
# 落ちる場合があるので，まとめの一行を別に返す．


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


def num(v: float, digits: int, *, signed: bool = False) -> str:
    """表示の桁で丸めた値．負号は U+2212（principles.md §3）．"""
    return f"{v:{'+' if signed else ''}.{digits}f}".replace("-", "−")


def interval(d: dict, digits: int) -> str:
    """差と，その95%信頼区間．"""
    low, high = (num(d[k], digits, signed=True) for k in ("ci_low", "ci_high"))
    return f"{num(d['mean'], digits, signed=True)} [{low}, {high}]"


def png(rows: list[list[float]], cell: int = 8) -> bytes:
    """0–1 の値の格子を，0 が白，0.5 が MID，1 が DEEP の緑になる PNG にする．一マスは cell 画素の正方形．"""
    def rgb(v: float) -> bytes:
        low, high, k = (WHITE, MID, 2 * v) if v < 0.5 else (MID, DEEP, 2 * v - 1)
        return bytes(round(a + (b - a) * k) for a, b in zip(bytes.fromhex(low), bytes.fromhex(high), strict=True))

    raw = b"".join(
        b"\x00" + b"".join(rgb(v) * cell for v in row)
        for row in rows for _ in range(cell)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", len(rows[0]) * cell, len(rows) * cell, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


LEAD = 3  # プロンプトの先頭で，画像の前に来る書式の語の数
ARMS = ("baseline", "proposal", "ablation")
# 本筋で見せる群の比較（principles.md §2）
PAIRS = (("proposal", "baseline"), ("proposal", "ablation"))
TRIAL = "42"  # 掛ける層を変えた試走の乱数種（研究文書 §A.5）
PROPOSAL_WEIGHT = 0.1  # proposal の整合の強さ λ（研究文書 §3.2）


def seed_mean(runs: dict, cells: Callable[[dict], list[list[float]]]) -> list[list[float]]:
    """乱数種の平均．cells は一つの run から（層，位置）の格子を取り出す．"""
    grids = [cells(run) for run in runs.values()]
    return [[sum(v) / len(v) for v in zip(*rows)] for rows in zip(*grids)]


def place(offset: int) -> str:
    """文と表で位置を指す名．最後の位置だけは，トークンの文字でなく今の 𝒉 と呼ぶ．"""
    return "今の 𝒉" if offset == -1 else ENDING[offset]


def chosen(choice: dict) -> str:
    """③で整合を掛ける場所を，図の名で言う．"""
    if len(choice["positions"]) > 1:
        return f"第{choice['layer']}層の6位置すべて"
    return f"第{choice['layer']}層の「{place(choice['positions'][0])}」"


# ---------------------------------------------------------------- ① の結果


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


def seeds(mapped: dict) -> int:
    """①の各群の乱数種の数．"""
    return len(mapped["sets"]["reported"]["baseline"])


def s_map_result(mapped: dict) -> Slide:
    s = Slide("① の結果：baseline のマップ")
    head, question, ending = baseline_map(mapped)
    grid = [h + q + e[1:] for h, q, e in zip(head, question, ending, strict=True)]
    # 濃さの上限は，第1–31層の，先頭の位置を除いた値から取る：最後の層は定義から末尾だけが 1 になり，
    # 先頭の書式の語は値が特に大きい（左端の濃い列）．濃さは平方根に比例させ，小さい値も見えるようにする
    top = max(v for row in grid[:-1] for v in row[1:])
    shade = [[min(v / top, 1.0) ** 0.5 for v in row] for row in reversed(grid)]

    s.text([("掛ける場所を選ぶため，baseline の相対感度を全層・全位置で見る", BODY, INK),
            (f"濃いほど大きい．乱数種{seeds(mapped)}つの平均", BODY, INK)])
    cols = [("書式", LEAD), ("画像", len(head[0]) - LEAD), ("質問文", len(question[0])), ("書式", len(ending[0]) - 1)]
    n, layers = len(shade[0]), len(shade)
    cw, rh = 9.3 / n, 0.045
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
    first, second = map_reading(mapped)
    s.text([("左端の濃い列は先頭の <|im_start|> で，理由は未調査", BODY, INK),
            (f"→ {first}", BODY, ACCENT), (second, BODY, ACCENT)])
    return s


def map_reading(mapped: dict) -> tuple[str, str]:
    """baseline のマップの記述（研究文書 §8.5 第1歩の結果）．整合を掛けた層で，画像のどの位置より
    質問文（問の半数が届く位置まで）と今の 𝒉 が大きいことを確かめてから言う．
    二行で返し，マップのスライドとまとめが同じ二行を使う．「画像までしか読んでいない位置」は先頭の書式
    （<|im_start|> は値が特に大きい）も含むので，比べる相手は画像のトークンの位置と書く．
    """
    layer = mapped["site_layer"]
    head, question, ending = (grid[layer - 1] for grid in baseline_map(mapped))
    if not max(head[LEAD:]) < min(question) <= max(question) < ending[-1]:
        raise SystemExit("マップの記述が値に合わない：研究文書 §8.5 第1歩の結果を読み直し，この文を書き直す")
    return (f"第{layer}層の baseline では，質問文の位置と今の 𝒉 のほうが，",
            "画像のトークンの位置より相対感度が大きい")


def site_differences(mapped: dict) -> list[list[dict]]:
    """整合を掛けた層の6位置ごとの，PAIRS の差と 95% 区間（研究文書 §4.3）．位置の順に並ぶ．"""
    reported, dim, layer = mapped["sets"]["reported"], str(mapped["primary_dim"]), mapped["site_layer"]

    def at(arm: str, k: int) -> dict[int, float]:
        return {int(seed): run["ending"][dim][layer - 1][k] for seed, run in reported[arm].items()}

    return [[paired_difference(at(a, k), at(b, k)) for a, b in PAIRS] for k in range(len(ENDING))]


def site_reading(mapped: dict) -> str:
    """6位置での proposal の動き．両方の比較で区間が 0 を含まない位置だけを，向きごとに名指す．"""
    offsets, moved = mapped["choice"]["offsets"], {"低く": [], "高く": []}
    for offset, pair in zip(offsets, site_differences(mapped), strict=True):
        if all(d["excludes_zero"] for d in pair) and len({d["mean"] > 0 for d in pair}) == 1:
            moved["高く" if pair[0]["mean"] > 0 else "低く"].append(place(offset))
    said = "，".join(f"{'・'.join(where)}で{way}" for way, where in moved.items() if where)
    # 連用形でつないだ最後の「く」を「い」に替えて，文を閉じる
    return f"proposal は両群より，{said[:-1]}い" if said else "proposal が両群と違う位置は無い"


def s_sites(mapped: dict) -> Slide:
    """第16層の6位置（④の候補．研究文書 §8.5 第2歩が測る前に決めた）での baseline の値と，群の差．

    baseline の列が④の場所を決め（最大の位置），差の列が群の比較を見せる．
    """
    s = Slide(f"① の結果：第{mapped['site_layer']}層の6位置")
    s.text([("質問を読み終えた6位置は，測る前に決めた④の候補", BODY, INK)])
    diffs = site_differences(mapped)
    rows = [["位置", "baseline", *(f"{a} − {b}" for a, b in PAIRS)]]
    for offset, value, pair in zip(mapped["choice"]["offsets"], mapped["choice"]["baseline"], diffs, strict=True):
        rows.append([place(offset), num(value, 3), *(interval(d, 3) for d in pair)])
    s.table([2.8, 1.4, 3.65, 3.65], rows, center=(1, 2, 3), bold=tuple(
        (r, c) for r, pair in enumerate(diffs, start=1) for c, d in enumerate(pair, start=2) if d["excludes_zero"]))
    s.text([(f"値は相対感度，乱数種{seeds(mapped)}つの平均．太字は区間が 0 を含まない差", BODY, INK)])
    s.text([(f"→ {site_reading(mapped)}", BODY, ACCENT)])
    return s


def layer_tails(mapped: dict) -> list[tuple[int, float, float]]:
    """整合を掛けた層ごとの（層，baseline の末尾，proposal の末尾）．どれも乱数種 TRIAL で，層の順．"""
    dim, sets = str(mapped["primary_dim"]), mapped["sets"]

    def tail(run: dict, layer: int) -> float:
        return run["ending"][dim][layer - 1][-1]

    return [(layer, tail(sets["reported"]["baseline"][TRIAL], layer), tail(runs[TRIAL], layer))
            for layer, runs in sorted([(mapped["site_layer"], sets["reported"]["proposal"]),
                                       *((r[TRIAL]["layer"], r) for r in sets["layer"].values())])]


def layer_reading(mapped: dict) -> tuple[str, str]:
    """掛けた層の末尾の向き（研究文書 §8.5 第1歩）．どの層でも下がるときだけ「層によらない」と言う．二行で返す．"""
    tails = layer_tails(mapped)
    fell = "・".join(str(layer) for layer, before, after in tails if after < before)
    rose = "・".join(str(layer) for layer, before, after in tails if after >= before)
    if not rose:
        return "相対感度はどの層でも低下", "依存の低下は層によらない見込み"
    if fell == str(mapped["site_layer"]):
        return f"相対感度の低下は第{fell}層だけ", f"依存の低下は第{fell}層に固有の見込み"
    return f"相対感度は第{fell}層で低下，第{rose}層で上昇", "依存の低下は，層によらないとは言えない"


def s_layers(mapped: dict, sweep: dict) -> Slide:
    """掛ける層を変えた試走（研究文書 §A.5）と，第16層に掛けた同じ乱数種の proposal．

    表の行はどれも proposal で，読み取り精度と正解率は summarize-sweep.json の points から，相対感度は掛けた層の
    末尾の値を取る．baseline は行に置かず，正解率は冒頭の行で，相対感度は差の列で比べる（principles.md §2 の
    表の項）．読み取り精度は，baseline では第16層でしか測っていないので比べない．
    """
    s = Slide("① の結果：掛ける層を変えた試走")
    points = {p["layer"]: p for p in sweep["points"] if p["arm"] == "proposal" and p["align_weight"] == PROPOSAL_WEIGHT}
    base = next(p for p in sweep["points"] if p["arm"] == "baseline")
    tails = layer_tails(mapped)
    trials = "・".join(str(layer) for layer, _, _ in tails if layer != mapped["site_layer"])
    s.text([(f"学会の結果が掛ける層によるかを，第{trials}層に掛けた試走で見る", BODY, INK),
            (f"表は proposal．乱数種はどれも{TRIAL}で，baseline の正解率は{num(base['accuracy']['per_seed'][TRIAL], 2)}%",
             BODY, INK)])
    rows = [["掛けた層", "読み取り精度（%）", "正解率（%）", "相対感度", "相対感度の差\nproposal − baseline"]]
    for layer, before, after in tails:
        point = points[layer]
        rows.append([f"第{layer}層", num(100 * point["probe_accuracy"]["per_seed"][TRIAL], 1),
                     num(point["accuracy"]["per_seed"][TRIAL], 2), num(after, 3), num(after - before, 3, signed=True)])
    s.table([1.4, 2.5, 1.7, 2.0, 3.0], rows, center=(0, 1, 2, 3, 4))
    s.text([(f"加工による 𝒉 の動きの比は，第{trials}層では未測定", BODY, INK)])
    first, second = layer_reading(mapped)
    s.text([(f"→ {first}", BODY, ACCENT), (second, BODY, ACCENT)])
    return s


# ---------------------------------------------------------------- ② の結果


def weight(w: float) -> str:
    """λ の表記．0 と 1.0 と書く．"""
    return "0" if w == 0 else str(w)


def lambda_cost(curve: dict) -> float:
    """λ > 0 の各点の正解率の，λ = 0 からの差の絶対値の最大（pt）を，小数第1位へ切り上げたもの．

    「以内」と書くので，丸めで上限を下げない．差は百分率の引き算なので，先に第2位で丸めて浮動小数の端数を落とす．
    """
    zero = curve["accuracy"][curve["align_weight"].index(0.0)]
    largest = max(abs(a - zero) for w, a in zip(curve["align_weight"], curve["accuracy"], strict=True) if w)
    return math.ceil(round(largest, 2) * 10) / 10


def lambda_reading(curve: dict, dim: str) -> tuple[str, str, str]:
    """研究文書 §8.4 の結果の記述と，測る前に決めた判定（相対感度がどの段でも下がるか）．

    スライドの二行と，まとめの一行を返す．下がったと言うときは，何より下がったか（baseline）を書く．
    """
    if curve["sensitivity_falls_at_every_step"]:
        falls = "相対感度は λ に伴って単調に下がる"
        return falls, "学会で示した依存の低下は，強さに応じる", falls
    leaning, zero = curve["sensitivity"][dim], curve["align_weight"].index(0.0)
    if all(v < leaning[zero] for k, v in enumerate(leaning) if k != zero):
        return ("相対感度は，λ > 0 のどれでも baseline より低い",
                "ただし単調には下がらず，依存の低下が強さに応じるとは言えない",
                "相対感度は λ > 0 のどれでも baseline より低いが，単調には下がらない")
    unsteady = "相対感度は λ に伴って単調には下がらない"
    return unsteady, "学会で示した依存の低下が，強さに応じるとは言えない", unsteady


def cost_reading(curve: dict) -> str:
    """正解率の動いた幅の読み．②の結果のスライドとまとめが同じ文を使う．"""
    return f"正解率は baseline から{num(lambda_cost(curve), 1)}pt以内"


def s_lambda_result(curve: dict, dim: str, spread: float) -> Slide:
    """spread は baseline を乱数種だけ替えたときの正解率の幅（pt．研究文書 §5.2）．

    群を左の列に置き，同じ群の行（λ > 0 はどれも proposal）を一つのセルに結合し，λ の列には値だけを置く
    （principles.md §3 の表の項）．
    """
    s = Slide("② の結果：整合の強さ λ の変更")
    s.text([(f"依存の低下が強さに応じるかを，λ を変えて見る．乱数種{curve['seed']}の1本", BODY, INK)])
    leaning = curve.get("sensitivity", {}).get(dim)
    rows: list[list[str | None]] = [["群", "λ", "読み取り精度（%）", "正解率（%）", *(["相対感度"] if leaning else [])]]
    before = None  # 前の行の群．同じなら None を置いて，上のセルに結合する
    for k, w in enumerate(curve["align_weight"]):
        arm = "proposal" if w else "baseline"
        rows.append([None if arm == before else arm, weight(w),
                     num(100 * curve["probe_accuracy"][k], 1), num(curve["accuracy"][k], 2),
                     *([num(leaning[k], 3)] if leaning else [])])
        before = arm
    s.table([1.8, 1.4, 2.6, 2.2, 2.0][:len(rows[0])], rows, center=tuple(range(len(rows[0]))))
    s.text([(f"{cost_reading(curve)}．乱数種だけで{num(spread, 2)}pt動く", BODY, INK)])
    if leaning:
        first, second, _ = lambda_reading(curve, dim)
        s.text([(f"→ {first}", BODY, ACCENT), (second, BODY, ACCENT)])
    return s


# ---------------------------------------------------------------- 今後と ④ の結果


AIMS = {2: "結果が層によらないか", 3: "使われるようになるか", 4: "使われるようになるか"}


def s_future(done: tuple[bool, ...], choice: dict | None) -> Slide:
    """結果の無い手順を一枚に集め，方法だけを言う（principles.md §2：まだ結果の無い実験は「今後」と添えて方法だけ）．

    冒頭の行は，各手順が整合の何を変えるか（③は層，④は位置，⑤は方向）．
    """
    if choice is None:
        site = "第16層の6位置のうち，baseline の最大の位置に掛ける"
    elif len(choice["positions"]) > 1:
        site = f"baseline の最大が{place(choice['largest'])}と同じなので，6位置すべてに掛ける"
    else:
        site = f"baseline の最大だった{place(choice['largest'])}に掛ける"
    plans = {
        2: ("層", ["baseline と，第8・16・24層に掛けた proposal を学習",
                        "正解率・読み取り精度・相対感度と，加工による 𝒉 の動きの比を比べる"]),
        3: ("位置", [site, "proposal と ablation を乱数種3つずつ学習し，baseline と比べる"]),
        4: ("方向", ["答えを動かす方向へ射影した表現で，整合損失を計算する"]),
    }
    left = [k for k in plans if not done[k]]
    marks = [STEPS[k][0] for k in left]
    # 続く番号は範囲で，飛ぶ番号は並べて言う（「③–⑤」が④を含むと読まれないように）
    contiguous = left == list(range(left[0], left[-1] + 1))
    span = f"{marks[0]}–{marks[-1]}" if len(left) > 1 and contiguous else "・".join(marks)
    s = Slide(f"今後：{span} の方法")
    # 冒頭の行は，ねらいの同じ手順をまとめて言う（③は学会の結果が層によらないか，④⑤は使われるようになるか）
    aims: dict[str, list[str]] = {}
    for k in left:
        aims.setdefault(AIMS[k], []).append(STEPS[k][0])
    s.text([("，".join(f"{'・'.join(m)} で{aim}" for aim, m in aims.items()) + "を見る", BODY, INK)])
    lines = []
    for k in left:
        lines.append((STEPS[k], BODY, INK))
        lines += [(t, SUB, INK, 2) for t in plans[k][1]]
    s.text(lines)
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
    s = Slide("④ の結果：掛ける位置の変更")
    pairs = tuple(f"{a}_minus_{b}" for a, b in PAIRS)  # summarize_relocation.py の鍵
    leaning = relocation["sensitivity"][str(relocation["primary_dim"])]
    accuracy = relocation["accuracy"]
    s.text([(f"{chosen(relocation)}に掛けた．乱数種3つ", BODY, INK)])
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
        ("太字は区間が 0 を含まない差", BODY, INK),
        (f"→ {RELOCATION[relocation['outcome']]}", BODY, ACCENT),
    ])
    return s


# ---------------------------------------------------------------- まとめ


def s_summary(mapped: dict | None, curve: dict | None, dim: str, relocation: dict | None,
              done: tuple[bool, ...]) -> Slide:
    """結果のスライドの読みを，同じ関数から集め直し，次の一手を添える．

    手順ごとに名を親の行に置き，読みをその下の箇条に並べる．乱数種1つの但し書きは，②では親の行に一度だけ，
    ①の層の試走では「試走」の語で言う（principles.md §3 の但し書きの項）．
    dim は相対感度を読む幅で，②の結果のスライドと同じものを受け取る．
    次の一手は，研究文書 §8 の順で最初の未着手の手順．⑤は④が表の1行目に当たったときにだけ行う（§8.5 第3歩）．
    """
    s = Slide("まとめと次の一手")
    lines = []
    if mapped:
        lines.append((STEPS[0], BODY, INK))
        lines += [("\n".join(map_reading(mapped)), SUB, INK, 2), (site_reading(mapped), SUB, INK, 2)]
        if "layer" in mapped["sets"]:
            lines.append((f"掛ける層を変えた試走では，{layer_reading(mapped)[0]}", SUB, INK, 2))
    if curve:
        lines.append((f"{STEPS[1]}．乱数種はどれも1つ", BODY, INK))
        lines.append((cost_reading(curve), SUB, INK, 2))
        if curve.get("sensitivity", {}).get(dim):
            lines.append((lambda_reading(curve, dim)[2], SUB, INK, 2))
    if relocation:
        lines += [(STEPS[3], BODY, INK), (RELOCATION[relocation["outcome"]], SUB, INK, 2)]
    s.text(lines)
    left = [k for k, d in enumerate(done) if not d]
    if relocation and relocation["outcome"] != "falls_by_labels":
        left.remove(4)
    s.text([(f"→ 次は，{STEPS[left[0]]}" if left else "→ 次は，次の問いへ進む", BODY, ACCENT)])
    return s


def slides() -> list[Slide]:
    """前置きと計画の後に，結果のある実験は結果を，結果の無い実験は「今後」の一枚に方法を続け，最後にまとめる．

    ③（層を変えた学習に乱数種を揃える）を読む分析はまだ無いので，③は常に未着手として扱う．
    """
    mapped = load("probe-dependence-map.json")
    sweep = load("summarize-sweep.json")
    relocation = load("summarize-relocation.json")
    measured = mapped is not None and "reported" in mapped["sets"]
    curve = (sweep or {}).get("strength_curve")
    dim = str(mapped["primary_dim"]) if mapped else ""
    done = (measured, bool(curve), False, bool(relocation), False)

    deck = [s_done(done), s_questions(), s_input(), s_candidates(), s_plan(done)]
    if measured:
        deck += [s_map_result(mapped), s_sites(mapped)]
        if "layer" in mapped["sets"] and sweep:  # 表の正解率と読み取り精度は summarize-sweep.json にある
            deck.append(s_layers(mapped, sweep))
    if curve:
        spread = next(p for p in sweep["points"] if p["arm"] == "baseline")["accuracy"]["range"]
        deck.append(s_lambda_result(curve, dim, spread))
    if relocation:
        deck.append(s_relocate_result(relocation))
    if not all(done[2:]):
        deck.append(s_future(done, mapped["choice"] if measured else None))
    if measured or curve:
        deck.append(s_summary(mapped if measured else None, curve, dim, relocation, done))
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
