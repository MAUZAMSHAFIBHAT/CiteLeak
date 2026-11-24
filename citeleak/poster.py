"""Auto-generated 36" x 24" poster (PDF) built from results/metrics.json.

Every number and every claim on the poster is computed from the metrics, and the
wording adapts to the direction of the result (it will not claim a leak that the
data does not show). Edit poster/config.json for your name, course and repository
link, then run:  python -m citeleak.poster
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Flowable, Image, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
EPS_LO = 1.0
W, H = 36 * inch, 24 * inch

# ---------------------------------------------------------------- palette
NAVY = HexColor("#012169")       # Duke navy
NAVY2 = HexColor("#0b3a8c")
INK, INK2, MUTED = HexColor("#16181d"), HexColor("#4a4d55"), HexColor("#8a8d94")
BG, CARD, LINE = HexColor("#eef0f4"), white, HexColor("#dfe3ea")
BLUE, ORANGE, AQUA, VIOLET = HexColor("#2a78d6"), HexColor("#eb6834"), HexColor("#1baf7a"), HexColor("#6250d6")
TINT = {"blue": HexColor("#e7f0fb"), "orange": HexColor("#fdeee7"), "aqua": HexColor("#e5f6ef"),
        "violet": HexColor("#eeebfb"), "amber": HexColor("#fff4dc")}
GOOD, PART, BAD = HexColor("#1a8f5a"), HexColor("#4f6a8f"), HexColor("#c4362f")

# ---------------------------------------------------------------- fonts
_fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_fd / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_fd / "DejaVuSans-Bold.ttf")))
_lato = ROOT / "poster" / "fonts"
if (_lato / "Lato-Regular.ttf").exists():
    for name, f in [("R", "Regular"), ("B", "Bold"), ("K", "Black"), ("I", "Italic"),
                    ("L", "Light"), ("BI", "BoldItalic")]:
        pdfmetrics.registerFont(TTFont(f"Lato-{name}", str(_lato / f"Lato-{f}.ttf")))
    F = {"r": "Lato-R", "b": "Lato-B", "k": "Lato-K", "i": "Lato-I", "l": "Lato-L", "bi": "Lato-BI"}
    pdfmetrics.registerFontFamily("Lato-R", normal="Lato-R", bold="Lato-B", italic="Lato-I",
                                  boldItalic="Lato-BI")
else:
    pdfmetrics.registerFont(TTFont("DVI", str(_fd / "DejaVuSans-Oblique.ttf")))
    F = {"r": "DV", "b": "DVB", "k": "DVB", "i": "DVI", "l": "DV", "bi": "DVB"}
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")
CHECK, CROSS = '<font name="DVB">✓</font>', '<font name="DVB">✗</font>'


def styles(s: float):
    P = ParagraphStyle
    return {
        "b": P("b", fontName=F["r"], fontSize=21 * s, leading=29.5 * s, textColor=INK, spaceAfter=7 * s),
        "bl": P("bl", fontName=F["r"], fontSize=21 * s, leading=29 * s, textColor=INK,
                leftIndent=24 * s, bulletIndent=2 * s, spaceAfter=6 * s, bulletFontName=F["b"],
                bulletColor=NAVY2),
        "small": P("sm", fontName=F["r"], fontSize=14.5 * s, leading=19.5 * s, textColor=INK2),
        "cap": P("cap", fontName=F["i"], fontSize=16.5 * s, leading=22.5 * s, textColor=INK2,
                 spaceBefore=4 * s, spaceAfter=4 * s),
        "big": P("big", fontName=F["k"], fontSize=62 * s, leading=66 * s, alignment=TA_CENTER),
        "bigl": P("bigl", fontName=F["r"], fontSize=17 * s, leading=21 * s, textColor=INK2,
                  alignment=TA_CENTER),
        "mono": P("mono", fontName=F["r"], fontSize=16 * s, leading=21 * s, textColor=INK),
        "monob": P("monob", fontName=F["b"], fontSize=16.5 * s, leading=22 * s, textColor=NAVY),
        "callout": P("co", fontName=F["r"], fontSize=22 * s, leading=30 * s, textColor=NAVY),
        "sub": P("sub", fontName=F["b"], fontSize=22 * s, leading=28 * s, textColor=NAVY2,
                 spaceBefore=8 * s, spaceAfter=4 * s),
    }


# ---------------------------------------------------------------- building blocks
class Card(Flowable):
    """White rounded card with a numbered badge + title, drawn around child flowables."""

    def __init__(self, num, title, flows, s, accent=NAVY2, gap=6):
        super().__init__()
        self.num, self.title, self.flows, self.s, self.accent = num, title, flows, s, accent
        self.pad = 24 * s
        self.head = 58 * s
        self.gap = gap * s
        self.extra = 0.0          # vertical slack added when columns are justified

    def wrap(self, aw, ah):
        self.w = aw
        iw = aw - 2 * self.pad
        self.sizes = [f.wrap(iw, 10 ** 6) for f in self.flows]
        body = sum(h for _, h in self.sizes) + self.gap * max(0, len(self.flows) - 1)
        self.h = self.head + body + 2 * self.pad - 6 * self.s + self.extra
        return aw, self.h

    def draw(self):
        c, s, w, h = self.canv, self.s, self.w, self.h
        r = 16 * s
        c.setFillColor(HexColor("#d5d9e1"))
        c.roundRect(3 * s, -5 * s, w, h, r, fill=1, stroke=0)              # soft shadow
        c.setFillColor(CARD)
        c.setStrokeColor(LINE)
        c.setLineWidth(1.2 * s)
        c.roundRect(0, 0, w, h, r, fill=1, stroke=1)
        c.setFillColor(self.accent)                                        # top accent bar
        c.roundRect(0, h - 8 * s, w, 8 * s, 4 * s, fill=1, stroke=0)
        c.rect(0, h - 8 * s, w, 4 * s, fill=1, stroke=0)
        # badge + title
        bx, by, br = self.pad + 19 * s, h - self.pad - 22 * s, 19 * s
        if self.num:
            c.setFillColor(self.accent)
            c.circle(bx, by, br, fill=1, stroke=0)
            c.setFillColor(white)
            c.setFont(F["k"], 22 * s)
            c.drawCentredString(bx, by - 8 * s, str(self.num))
            tx = bx + br + 14 * s
        else:
            tx = self.pad
        c.setFillColor(NAVY)
        c.setFont(F["k"], 34 * s)
        c.drawString(tx, by - 12 * s, self.title)
        # children
        y = h - self.head - self.pad
        for f, (fw, fh) in zip(self.flows, self.sizes):
            y -= fh
            f.drawOn(c, self.pad, y)
            y -= self.gap


class Pipeline(Flowable):
    def __init__(self, width, s):
        super().__init__()
        self.w, self.s = width, s
        self.h = 2.55 * inch * s

    def wrap(self, *_):
        return self.w, self.h

    def draw(self):
        c, s = self.canv, self.s
        boxes = [("Attacker", "knows name,\nage, city", NAVY2, True),
                 ("Retriever", "BM25, top-4\nprivate store", BLUE, False),
                 ("Guarded LLM", "prompt rule +\noutput filter", BLUE, False),
                 ("Attribution", "2⁴ ablations\n(ContextCite)", BLUE, False),
                 ("UI", "answer +\nsources panel", NAVY2, True)]
        n, gap = len(boxes), 0.36 * inch * s
        bw = (self.w - gap * (n - 1)) / n
        bh = 1.9 * inch * s
        y = self.h - bh
        for i, (t, sub, col, solid) in enumerate(boxes):
            x = i * (bw + gap)
            c.setFillColor(col if solid else TINT["blue"])
            c.setStrokeColor(col)
            c.setLineWidth(2.4 * s)
            c.roundRect(x, y, bw, bh, 14 * s, fill=1, stroke=1)
            c.setFillColor(white if solid else NAVY)
            c.setFont(F["k"], 19 * s)
            c.drawCentredString(x + bw / 2, y + bh - 0.5 * inch * s, t)
            c.setFont(F["r"], 14.5 * s)
            c.setFillColor(white if solid else INK2)
            for j, ln in enumerate(sub.split("\n")):
                c.drawCentredString(x + bw / 2, y + bh - (0.92 + 0.27 * j) * inch * s, ln)
            if i < n - 1:
                ax, ay = x + bw + 5 * s, y + bh / 2
                c.setStrokeColor(MUTED)
                c.setLineWidth(3 * s)
                c.line(ax, ay, ax + gap - 16 * s, ay)
                c.setFillColor(MUTED)
                p = c.beginPath()
                tip = ax + gap - 9 * s
                p.moveTo(tip, ay)
                p.lineTo(tip - 12 * s, ay + 8 * s)
                p.lineTo(tip - 12 * s, ay - 8 * s)
                p.close()
                c.drawPath(p, fill=1, stroke=0)
        c.setFont(F["i"], 15 * s)
        c.setFillColor(INK2)
        c.drawString(0, y - 0.36 * inch * s,
                     "Datastore: synthetic patient records + care protocols. No real patient data.")


def img(path: Path, width: float):
    from PIL import Image as PImage
    w, h = PImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def box(flows, width, s, bg, pad=16, border=None):
    t = Table([[flows]], colWidths=[width])
    st = [("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), pad * s),
          ("RIGHTPADDING", (0, 0), (-1, -1), pad * s), ("TOPPADDING", (0, 0), (-1, -1), 12 * s),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 12 * s), ("ROUNDEDCORNERS", [12 * s] * 4)]
    if border is not None:
        st.append(("LINEBEFORE", (0, 0), (0, -1), 6 * s, border))
    t.setStyle(TableStyle(st))
    return t


def stat_tiles(items, width, S, s):
    """items: [(value, label, color, tint)]"""
    cells, gap = [], 14 * s
    tw = (width - gap * (len(items) - 1)) / len(items)
    for v, lab, col, tint in items:
        big = ParagraphStyle("x", parent=S["big"], textColor=col)
        inner = Table([[Paragraph(v, big)], [Paragraph(lab, S["bigl"])]], colWidths=[tw - 4])
        inner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tint),
                                   ("ROUNDEDCORNERS", [14 * s] * 4),
                                   ("TOPPADDING", (0, 0), (-1, 0), 14 * s),
                                   ("BOTTOMPADDING", (0, -1), (-1, -1), 16 * s),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10 * s),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 10 * s)]))
        cells.append(inner)
    row = []
    for i, cl in enumerate(cells):
        row.append(cl)
        if i < len(cells) - 1:
            row.append("")
    t = Table([row], colWidths=sum([[tw, gap] for _ in cells], [])[:-1])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def chip(text, col, S, s):
    st = ParagraphStyle("chip", fontName=F["b"], fontSize=19 * s, leading=24 * s, textColor=white,
                        alignment=TA_CENTER)
    t = Table([[Paragraph(text, st)]])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col), ("ROUNDEDCORNERS", [16 * s] * 4),
                           ("TOPPADDING", (0, 0), (-1, -1), 6 * s), ("BOTTOMPADDING", (0, 0), (-1, -1), 9 * s),
                           ("LEFTPADDING", (0, 0), (-1, -1), 16 * s), ("RIGHTPADDING", (0, 0), (-1, -1), 16 * s)]))
    return t


def view_table(v, S, s, title, width, accent):
    rows = [[Paragraph(title, S["monob"]), "", ""]]
    for src in v["sources"]:
        share = "—" if src["share"] is None else f"{src['share']:.2f}"
        kind = src["kind"].replace("_", " ") if src["kind"] else "type hidden"
        rows.append([Paragraph(src["label"], S["mono"]), Paragraph(kind, S["mono"]),
                     Paragraph(f"<b>{share}</b>", S["mono"])])
    t = Table(rows, colWidths=[width * 0.44, width * 0.38, width * 0.18])
    t.setStyle(TableStyle([("SPAN", (0, 0), (-1, 0)), ("LINEBELOW", (0, 0), (-1, 0), 2 * s, accent),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return t


# ---------------------------------------------------------------- content
def content(M, cfg, fig: Path, colw: float, s: float, k=(1.0, 1.0, 1.0)):
    S = styles(s)
    iw = colw - 48 * s                       # inner card width
    mi, fa, at, g = M["membership"], M["faithfulness"], M["attribute"], M["guardrail"]
    a0, a1, a2 = mi["L0"]["auc"], mi["L1"]["auc"], mi["L2"]["auc"]
    tpr2, tpr0, tpr1 = mi["L2"]["tpr@5%fpr"], mi["L0"]["tpr@5%fpr"], mi["L1"]["tpr@5%fpr"]
    ung = M.get("membership_unguarded", {}).get("L0", {}).get("auc")
    leaked = a2 > a0 + 0.02

    near = [p for p in mi if p.startswith("shield(") and mi[p]["auc"] <= a0 + 0.03]
    rec = (max(near, key=lambda p: fa[p]["top1_agreement"]) if near else
           min((p for p in mi if p.startswith("shield(")), key=lambda p: mi[p]["auc"]))
    eps_rec = rec[rec.index("(") + 1:-1]
    f2, fr = fa["L2"]["top1_agreement"], fa[rec]["top1_agreement"]
    fu, au = fa["unlink"]["top1_agreement"], mi["unlink"]["auc"]
    gap = max(a2 - a0, 1e-9)
    dp_mid = mi["dp(2.0)"]["auc"]
    best_shield = min(mi[p]["auc"] for p in mi if p.startswith("shield("))
    one_fix_fails = au > a2 - 0.25 * gap and dp_mid > a2 - 0.25 * gap and best_shield <= a0 + 0.05
    link_frac = (a2 - au) / gap
    shield_f_lo = fa[f"shield({EPS_LO})"]["top1_agreement"]

    ci0, ci2 = mi["L0"]["auc_ci"], mi["L2"]["auc_ci"]
    h1, h2 = ci2[0] > ci0[1], a0 < a1 < a2
    shields_ok = [p for p in mi if p.startswith("shield(") and mi[p]["auc"] <= a0 + 0.05]
    h3_priv = bool(shields_ok)
    h3_full = h3_priv and max(fa[p]["top1_agreement"] for p in shields_ok) >= 0.5 * f2

    # ---------------- column 1
    c1 = [
        Card(1, "The problem", [
            Paragraph("AI assistants show <b>sources</b>, often with attribution scores, so users "
                      "can check answers: good explainability. But when RAG searches a <b>private</b> "
                      "collection, the sources panel is also a <b>side channel</b>: it reveals what "
                      "the system found and how much the model relied on it.", S["b"]),
            box([Paragraph("<b>Can the sources panel leak private information even when the "
                           "chatbot's text answer is guarded? Can we fix it without destroying "
                           "the explanation?</b>", S["callout"])], iw, s, TINT["blue"], border=BLUE),
        ], s),
        Card(2, "System: MediAssist", [
            Pipeline(iw, s * min(k[0], 1.25)),
            Paragraph("A fictional hospital assistant. The LLM is told never to reveal or confirm "
                      "patients, and an output filter replaces any answer about a named person with "
                      "a refusal. Attribution follows ContextCite [1]: the answer is re-scored under "
                      "all 16 subsets of the 4 retrieved sources, and a linear fit gives each "
                      "source's share of influence.", S["cap"]),
        ], s),
        Card(3, "Hypotheses & setup", [
            Paragraph("<b>H1</b>  Showing attribution increases membership-inference success over "
                      "answer-only access.", S["bl"], bulletText="●"),
            Paragraph("<b>H2</b>  Leakage grows with granularity: nothing &lt; citations &lt; scores.",
                      S["bl"], bulletText="●"),
            Paragraph("<b>H3</b>  A privacy-aware panel can return leakage to the answer-only floor "
                      "while keeping much of its faithfulness.", S["bl"], bulletText="●"),
            Paragraph("Setup", S["sub"]),
            Paragraph(f"{M.get('n_docs', 1666):,} synthetic records + care protocols · "
                      f"{M['n_targets']} target people, {M['n_members']} truly in the store · "
                      f"{M['model'].split('/')[-1]} running locally · BM25 top-{M['k']}.", S["b"]),
            Paragraph("Attacker knows name, age, sex, city; asks 3 questions + 1 control; logistic "
                      "regression, out-of-fold, <b>retrained per defense</b>. Metrics: ROC-AUC (95% "
                      "CI), TPR at 5% FPR, faithfulness vs. exact leave-one-out effects.", S["b"]),
        ], s),
        Card(4, "What is new", [
            Paragraph("RAG stores are known to leak through <i>answers</i> [2,3], and model "
                      "explanations are known to leak <i>training</i> data [4,5,6]. We measure the "
                      "<b>extra</b> leakage from the <b>sources panel</b> of RAG over a private store, "
                      "separate <b>linkable ids</b> from <b>scores</b>, and map the privacy–faithfulness "
                      "trade-off against an adaptive attacker.", S["b"]),
            Paragraph("[1] Cohen-Wang et al., ContextCite, NeurIPS 2024 · [2] Anderson et al., "
                      "arXiv:2405.20446 · [3] Naseh et al., Riddle Me This!, CCS 2025 · [4] Shokri "
                      "et al., AIES 2021 · [5] Liu et al., Please Tell Me More, IEEE S&amp;P 2024 · "
                      "[6] DeepLeak, arXiv:2601.03429", S["small"]),
        ], s),
    ]

    # ---------------- column 2
    tiles = stat_tiles([
        (f"{a0:.2f}", "attack AUC with the<br/>answer text only", BLUE, TINT["blue"]),
        (f"{a2:.2f}", "attack AUC with<br/>sources + scores", AQUA, TINT["aqua"]),
        (f"{tpr2:.0%}", "of patients exposed at<br/>5% false alarms", NAVY2, TINT["blue"]),
    ], iw, S, s)
    guard_txt = (f"<b>Prompt-only guardrails failed:</b> with just the system prompt, the model "
                 f"repeated the person's full name in {g['full_name_in_answer']:.0%} of answers and a "
                 f"patient's diagnosis in {g['member_condition_in_answer']:.0%}"
                 + (f" (answer-only AUC {ung:.2f})" if ung is not None else "")
                 + ". All results therefore use the output filter.")
    ex = M.get("example")
    c2_res = [
        tiles,
        img(fig / "fig_leak.png", min(iw, iw * 0.70 * s * k[1])),
        Paragraph(f"Adding attribution scores {'raised' if leaked else 'did not clearly raise'} "
                  f"attack AUC from {a0:.2f} to {a2:.2f}; citations alone gave {a1:.2f}. Patients "
                  f"caught at a 5% false-alarm rate: {tpr0:.0%} → {tpr1:.0%} → {tpr2:.0%}.", S["cap"]),
        box([Paragraph(guard_txt, S["b"])], iw, s, TINT["blue"], border=NAVY2),
    ]
    c2 = [Card(5, "Result: the sources panel leaks" if leaked else "Result: leakage by disclosure",
               c2_res, s, accent=AQUA)]
    if ex:
        half = (iw - 20 * s) / 2
        t = Table([[view_table(ex["L2"], S, s, "Status quo (ids + scores)", half, NAVY2), "",
                    view_table(ex["shield"], S, s, "Privacy Shield ε=1", half, BLUE)]],
                  colWidths=[half, 20 * s, half])
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        c2.append(Card(6, "What the attacker sees", [
            Paragraph(f"<i>“{ex['query']}”</i>", S["cap"]), t,
        ], s))
    lo2 = at["L2"]["ci"][0]
    attr_note = ("<b>Not better than chance.</b> Likely reason: the name pulls the person's record "
                 "into the top sources whichever condition is asked about, so the panel looks the "
                 "same for every candidate. <i>Whether</i> someone is a patient leaked; <i>what</i> "
                 "they have did not, with this simple attack." if lo2 <= 0.2 else
                 "Scores let the attacker beat chance at guessing the condition.")
    attr_tbl = Table([[img(fig / "fig_attr.png", min(iw * 0.52, iw * 0.46 * s * k[1])),
                       Paragraph(f"Five candidate conditions, chance 20%. Correct guesses: "
                                 f"{at['L0']['accuracy']:.0%} from answers, {at['L2']['accuracy']:.0%} "
                                 f"with scores (n={at['L2']['n']}).<br/><br/>{attr_note}", S["cap"])]],
                     colWidths=[iw * 0.52, iw * 0.48])
    attr_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    c2.append(Card(7, "Can it guess the diagnosis?", [attr_tbl], s, accent=VIOLET))

    # ---------------- column 3
    if one_fix_fails:
        defense_cap = (f"Each fix alone fails: hiding only the ids gives AUC {au:.2f}; noising only "
                       f"the scores (ε=2) gives {dp_mid:.2f}. Combining both (Shield ε={eps_rec}) "
                       f"reaches AUC {mi[rec]['auc']:.2f}, but faithfulness falls from {f2:.0%} to "
                       f"{fr:.0%}.")
        take2 = (f"Ids and scores <b>each</b> leak on their own. Removing only one barely helps "
                 f"(AUC {au:.2f} / {dp_mid:.2f}); only removing both reaches the answer-only floor "
                 f"(AUC {best_shield:.2f}).")
    else:
        defense_cap = (f"Unlinkable labels alone: AUC {au:.2f}, faithfulness {fu:.0%}. Shield "
                       f"ε={eps_rec}: AUC {mi[rec]['auc']:.2f}, faithfulness {fr:.0%} (status quo "
                       f"{a2:.2f}, {f2:.0%}).")
        take2 = (f"Unlinkable labels remove about {max(min(link_frac, 1), 0):.0%} of the extra "
                 "leakage" + ("; stable ids are a major part of the problem." if link_frac >= 0.4
                              else "; the scores themselves carry most of the signal."))
    take3 = (f"Formal per-query DP is costly: at ε={EPS_LO:g} the shown top source matches the true "
             f"one only {shield_f_lo:.0%} of the time (status quo {f2:.0%}). No free lunch between "
             f"transparency and privacy.")
    chips = Table([[chip(f"H1 {CHECK if h1 else CROSS}", GOOD if h1 else BAD, S, s), "",
                    chip(f"H2 {CHECK if h2 else CROSS}", GOOD if h2 else BAD, S, s), "",
                    chip("H3 " + (CHECK if h3_full else "partly: private " + CHECK + ", faithful " + CROSS
                                  if h3_priv else CROSS),
                         GOOD if h3_full else PART if h3_priv else BAD, S, s)]],
                  colWidths=[1.45 * inch * s, 12 * s, 1.45 * inch * s, 12 * s, None])
    chips.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("ALIGN", (0, 0), (-1, -1), "LEFT")]))
    c3 = [
        Card(8, "Defense: Privacy Shield", [
            Paragraph("<b>Unlinkable labels.</b> Fresh random labels per answer and hidden source "
                      "type, so the same record can't be recognized across questions.", S["bl"],
                      bulletText="●"),
            Paragraph("<b>Differentially private scores.</b> Laplace(2/ε) noise on the share vector: "
                      "ε-DP per release (L1 sensitivity ≤ 2 on the simplex).",
                      S["bl"], bulletText="●"),
            img(fig / "fig_frontier.png", min(iw, iw * 0.80 * s * k[2])),
            Paragraph(defense_cap, S["cap"]),
        ], s, accent=BLUE),
        Card(9, "Takeaways", [
            chips, Spacer(1, 4 * s),
            Paragraph(("Explanations are an attack surface. " if leaked else "")
                      + "Transparency features need a threat model like any other output.",
                      S["bl"], bulletText="●"),
            Paragraph(take2, S["bl"], bulletText="●"),
            Paragraph(take3, S["bl"], bulletText="●"),
            Paragraph("A system prompt is not a privacy control: the model ignored it most of the time.",
                      S["bl"], bulletText="●") if g["full_name_in_answer"] > 0.5 else Spacer(1, 0),
        ], s, accent=NAVY2),
        Card(10, "Limitations, ethics & next steps", [
            Paragraph("<b>Limits.</b> Templated synthetic notes are easier to match than real ones; "
                      "one small LLM and a BM25 retriever; the DP guarantee covers the score channel "
                      "per query (q queries cost qε), not the answer text.", S["b"]),
            Paragraph("<b>Ethics.</b> All data is fake; we attack only our own system and release "
                      "the defense with the attack. <b>Next:</b> dense retrievers, larger LLMs, real "
                      "de-identified corpora, and a user study on noisy sources.", S["b"]),
        ], s, accent=MUTED),
    ]
    headline = (f"With the text answer filtered, attackers were near coin-flip (AUC {a0:.2f}), yet the "
                f"sources panel exposed <b>{tpr2:.0%} of patients</b> at a 5% false-alarm rate "
                f"(AUC {a2:.2f})." if leaked else
                f"Answer-only attack AUC {a0:.2f}; with the sources panel {a2:.2f}.")
    return c1, c2, c3, headline


# ---------------------------------------------------------------- page
def draw_header(c, cfg, hh):
    c.setFillColor(NAVY)
    c.rect(0, H - hh, W, hh, fill=1, stroke=0)
    # accent stripe
    seg = W / 4
    for i, col in enumerate([HexColor("#6aaaf0"), BLUE, HexColor("#1f4fa8"), HexColor("#4a90e2")]):
        c.setFillColor(col)
        c.rect(i * seg, H - hh - 0.12 * inch, seg, 0.12 * inch, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(F["k"], 84)
    c.drawString(0.9 * inch, H - 1.45 * inch, "CiteLeak")
    tw = pdfmetrics.stringWidth("CiteLeak", F["k"], 84)
    c.setFont(F["l"], 68)
    c.drawString(0.9 * inch + tw + 0.3 * inch, H - 1.45 * inch, "when the “Sources” button leaks secrets")
    c.setFont(F["r"], 33)
    c.setFillColor(HexColor("#c9d6f0"))
    c.drawString(0.9 * inch, H - 2.15 * inch,
                 "Privacy side channels in explainable retrieval-augmented generation, and a defense")
    c.setFont(F["b"], 27)
    c.setFillColor(white)
    c.drawString(0.9 * inch, H - 2.78 * inch, cfg["author"])
    aw = pdfmetrics.stringWidth(cfg["author"], F["b"], 27)
    c.setFont(F["r"], 27)
    c.setFillColor(HexColor("#c9d6f0"))
    c.drawString(0.9 * inch + aw, H - 2.78 * inch, f"   ·   {cfg['affiliation']}   ·   {cfg['course']}")
    # QR card
    size = 2.25 * inch
    qx, qy = W - 3.35 * inch, H - hh + 0.42 * inch
    c.setFillColor(white)
    c.roundRect(qx - 0.12 * inch, qy - 0.12 * inch, size + 0.24 * inch, size + 0.24 * inch, 12, fill=1, stroke=0)
    q = QrCodeWidget(cfg["repo_url"])
    b = q.getBounds()
    d = Drawing(size, size, transform=[size / (b[2] - b[0]), 0, 0, size / (b[3] - b[1]), 0, 0])
    d.add(q)
    renderPDF.draw(d, c, qx, qy)
    c.setFillColor(HexColor("#c9d6f0"))
    c.setFont(F["b"], 19)
    c.drawRightString(qx - 0.3 * inch, qy + 1.25 * inch, "Code, data &")
    c.drawRightString(qx - 0.3 * inch, qy + 0.95 * inch, "live demo")
    c.setFont(F["r"], 30)
    c.drawRightString(qx - 0.3 * inch, qy + 0.45 * inch, "→")


def draw_banner(c, text, top, s):
    st = ParagraphStyle("ban", fontName=F["r"], fontSize=29, leading=37, textColor=NAVY, alignment=TA_LEFT)
    x, w = 0.7 * inch, W - 1.4 * inch
    p = Paragraph(f'<font name="{F["k"]}" color="#2a78d6">KEY FINDING  </font>{text}', st)
    pw, ph = p.wrap(w - 1.2 * inch, 400)
    bh = ph + 0.55 * inch
    y = top - bh
    c.setFillColor(HexColor("#d5d9e1"))
    c.roundRect(x + 3, y - 5, w, bh, 18, fill=1, stroke=0)
    c.setFillColor(white)
    c.roundRect(x, y, w, bh, 18, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.roundRect(x, y, 0.22 * inch, bh, 6, fill=1, stroke=0)
    p.drawOn(c, x + 0.6 * inch, y + (bh - ph) / 2)
    return y


def layout_column(cards, colw, avail, gap_min):
    hs = [cd.wrap(colw, avail)[1] for cd in cards]
    total = sum(hs) + gap_min * (len(cards) - 1)
    return total, hs


def build_poster(metrics_path, fig_dir, out_pdf, config_path=None):
    M = json.loads(Path(metrics_path).read_text())
    cfg = json.loads((Path(config_path) if config_path else ROOT / "poster" / "config.json").read_text())
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    margin, gutter, hh = 0.7 * inch, 0.45 * inch, 3.05 * inch
    colw = (W - 2 * margin - 2 * gutter) / 3
    def column_total(cards, avail):
        return sum(cd.wrap(colw, avail)[1] for cd in cards) + gap_min * (len(cards) - 1)

    gap_min = 0.3 * inch
    for s in [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70]:
        c = canvas.Canvas(str(out_pdf), pagesize=(W, H))
        c.setTitle("CiteLeak poster")
        c.setAuthor(cfg["author"])
        c.setFillColor(BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        draw_header(c, cfg, hh)
        headline = content(M, cfg, Path(fig_dir), colw, s)[3]
        ban_bottom = draw_banner(c, headline, H - hh - 0.42 * inch, s)
        top = ban_bottom - 0.38 * inch
        avail = top - margin
        base = content(M, cfg, Path(fig_dir), colw, s)
        if any(column_total(base[i], avail) > avail for i in range(3)):
            continue
        # grow each column's figures to use its spare height
        ks = [1.0, 1.0, 1.0]
        for i in range(3):
            for kk in [1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05]:
                trial = list(ks)
                trial[i] = kk
                if column_total(content(M, cfg, Path(fig_dir), colw, s, tuple(trial))[i], avail) <= avail:
                    ks[i] = kk
                    break
        cols = content(M, cfg, Path(fig_dir), colw, s, tuple(ks))[:3]
        for i, cards in enumerate(cols):
            total = column_total(cards, avail)
            per = (avail - total) / len(cards)
            x = margin + i * (colw + gutter)
            y = top
            for cd in cards:
                cd.extra = per
                _, h = cd.wrap(colw, avail)
                y -= h
                cd.drawOn(c, x, y)
                y -= gap_min
        if M.get("_preview"):
            c.setFillColor(HexColor("#c4362f"))
            c.setFont(F["b"], 26)
            c.drawCentredString(W / 2, 0.3 * inch,
                                "PREVIEW – smoke-test numbers from a random-weight model, NOT results")
        c.save()
        print(f"[poster] wrote {out_pdf} (text scale {s}, figure scale {ks})")
        return out_pdf
    raise RuntimeError("poster content does not fit even at the smallest scale")


if __name__ == "__main__":
    build_poster(ROOT / "results" / "metrics.json", ROOT / "results" / "figures",
                 ROOT / "poster" / "CiteLeak_poster.pdf")
