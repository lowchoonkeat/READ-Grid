"""Build the official IEM Journal Word manuscript from 13_author_manuscript_draft_iem.md
using iem_template_original_backup.docx as pristine base.

Implements all IEM Journal requirements and /ccf-humanization principles:
- UK English spelling
- Abstract strictly < 250 words (237 words)
- APA style referencing (Author, Year) and alphabetized reference list (58 references)
- Humanized academic tone: no defensive posturing, no arbitrary boldings in body text, reduced colon usage
- First page 100% identical to template:
    P0: Title
    P1: Authors with superscripts
    P2: Divider line
    Table 0: Left = Abstract, Right = Nested Table (grey box with dates & affiliations) + divider + Keywords
    P4: Divider line
    P5: Continuous section break to 2 columns
- Author CRediT table 100% identical to template:
    Inside 2-column layout under DECLARATIONS
    Single-column width (247.4 pt: 75 pt Author + 172.4 pt Roles)
    Borderless (val="nil")
    Header fill #E6E6E6
    Alternating row fills (white, #E6E6E6, white, #E6E6E6, white)
    Author names bold Arial 8.5 pt, roles Arial 8.5 pt regular
    No "Table X" caption; preceded by "Author contributions. Roles are described using the CRediT taxonomy."
    Followed by "All authors reviewed and approved the final manuscript." and divider rule
- Main tables (Tables 1-6) span full width (516.1 pt) with light grey headers (#D9D9D9)
- Figures (Figures 1-7) span full width with centered alignment and captions below
- All raw <img ...> HTML artifacts stripped
- Author Profiles at end: 1-column layout with grey banner "PROFILES" and all 5 authors with photos
"""

import os
import re
import sys
from pathlib import Path

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

ROOT = Path(r"C:\Users\Low\Downloads\P359")
PRISTINE_TEMPLATE = ROOT / "paper_package" / "template" / "iem_template_original_backup.docx"
MD_PATH = ROOT / "paper_package" / "13_author_manuscript_draft_iem.md"
FIG_DIR = ROOT / "paper_package" / "figures"
TEMPLATE_DIR = ROOT / "paper_package" / "template"
OUT_DOCX = ROOT / "paper_package" / "IEM_Journal_READ_Grid_Manuscript.docx"
TEMPLATE_DOCX = ROOT / "paper_package" / "template" / "iem_template.docx"

FIG_MAP = {
    1: FIG_DIR / "fig_v2_training_ladder.png",
    2: FIG_DIR / "fig_v2_val_arms.png",
    3: FIG_DIR / "fig_v2_gap_closure.png",
    4: FIG_DIR / "fig_v2_baselines.png",
    5: FIG_DIR / "fig_v2_primary_forest.png",
    6: FIG_DIR / "fig_v2_outage_coverage.png",
    7: FIG_DIR / "fig_v2_transfer.png",
}

TABLE_CAPTIONS = {
    1: "Table 1: Policy arm definitions and training configurations across the two evaluation runs.",
    2: "Table 2: Run 1 descriptive means, averaged over the 21 episodes with each episode weighted equally.",
    3: "Table 3: Validation-split results for the corrected policy set across 21 episodes and four fault conditions.",
    4: "Table 4: Held-out arm means, one value per episode averaged over the 21 test episodes (frozen selection objective; learned arms averaged over five seeds).",
    5: "Table 5: Frozen composite selection score pooled over 105 runs per condition across shifted operating points.",
    6: "Table 6: Measured wall-clock inference time on the evaluation host per control step for the full feeder.",
}

# Notes printed directly beneath a table, still inside the full-width flow.
TABLE_NOTES = {
    1: ("A deterministic arm is run once, so the seed columns count trained checkpoints per arm. "
        "The row totals give 7 instances in Run 1 and 30 in Run 2."),
    2: ("Achievement is averaged over the 21 episodes with each episode weighted equally. The switches "
        "column counts transitions across all 50 groups, and Table 3 reports the same quantity per group. "
        "Episode-mean and pooled figures differ, and each table states which it reports."),
    4: ("Each column is built from one value per episode, and the 21 episodes are averaged. The selection "
        "score column is Equation (5) applied to the three columns to its left. The pre-registered primary "
        "contrast weights each fault id equally within an episode and is therefore not the difference of two "
        "rows here; it is −0.0040 with a 95% interval of [−0.0239, +0.0174]. The eligible-step weighting used "
        "in this table gives −0.0017 [−0.0218, +0.0200], and both are inconclusive at δ = 0.02."),
    5: ("Pooled means are shown for level comparison. The paired differences reported in the text and in "
        "Figure 7 are computed per episode and then averaged, so a pooled difference of two table entries is "
        "not the reported contrast. At δ_DR = 8% the pooled difference is +0.0047 while the paired difference "
        "is +0.0212."),
    6: ("Times are wall-clock per control step for the whole feeder, obtained as the per-episode inference "
        "time divided by 1,440 steps. The ratio column divides unrounded means, so the rounded means imply "
        "33.3× for C4v2 where the table reports 32.8×."),
}

# Column widths in points for full-page tables (total = 516.1 pt)
TABLE_COL_WIDTHS = {
    1: [42.0, 112.0, 104.0, 116.0, 68.0, 74.0],
    2: [116.1, 80.0, 80.0, 80.0, 80.0, 80.0],
    3: [176.1, 80.0, 80.0, 90.0, 90.0],
    4: [110.0, 78.0, 82.0, 82.0, 76.0, 88.1],
    5: [116.1, 100.0, 100.0, 100.0, 100.0],
    6: [126.1, 130.0, 130.0, 130.0],
}


# --------------------------------------------------------------------------- #
# XML & Styling Helpers
# --------------------------------------------------------------------------- #

def set_cell_borders(cell, color="808080", sz="4"):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        b = OxmlElement(f'w:{edge}')
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), sz)
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        tcBorders.append(b)
    tcPr.append(tcBorders)

def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for edge, val in (('top', top), ('bottom', bottom), ('left', left), ('right', right)):
        node = OxmlElement(f'w:{edge}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color="D9D9D9"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>')
    tcPr.append(shd)

def add_divider_line_paragraph(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="8" w:space="1" w:color="000000"/></w:pBdr>')
    pPr.append(pBdr)
    return p

def add_formatted_runs(p, text, font_name="Arial", font_size_pt=9.0, default_bold=False, default_italic=False):
    # Parse inline formatting: `code`, *italic*, **bold**
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**') and len(token) >= 4:
            r = p.add_run(token[2:-2])
            r.font.name = font_name
            r.font.size = Pt(font_size_pt)
            r.bold = True
            if default_italic:
                r.italic = True
        elif token.startswith('*') and token.endswith('*') and len(token) >= 2:
            r = p.add_run(token[1:-1])
            r.font.name = font_name
            r.font.size = Pt(font_size_pt)
            r.italic = True
            if default_bold:
                r.bold = True
        elif token.startswith('`') and token.endswith('`') and len(token) >= 2:
            r = p.add_run(token[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(font_size_pt * 0.95)
        else:
            r = p.add_run(token)
            r.font.name = font_name
            r.font.size = Pt(font_size_pt)
            if default_bold:
                r.bold = True
            if default_italic:
                r.italic = True

def switch_to_1col(doc):
    sec = doc.add_section(WD_SECTION_START.CONTINUOUS)
    cols = sec._sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '1')
    return sec

def switch_to_2col(doc):
    sec = doc.add_section(WD_SECTION_START.CONTINUOUS)
    cols = sec._sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2')
        cols[0].set(qn('w:space'), '425')
    return sec


# --------------------------------------------------------------------------- #
# Document Assembly
# --------------------------------------------------------------------------- #

def main():
    print(f"Loading pristine base template: {PRISTINE_TEMPLATE}")
    doc = docx.Document(PRISTINE_TEMPLATE)
    body = doc._body._element

    # 1. Update running headers across all header parts
    RUNNING_TITLE = "GUARDED EDGE FALLBACK UNDER COMMUNICATION OUTAGES"
    for rId, part in doc.part.related_parts.items():
        if 'header' in part.partname:
            hdr_elem = part._element
            for child in list(hdr_elem):
                hdr_elem.remove(child)
            p_elem = OxmlElement('w:p')
            pPr = OxmlElement('w:pPr')
            pStyle = OxmlElement('w:pStyle')
            pStyle.set(qn('w:val'), 'Header')
            pPr.append(pStyle)
            jc = OxmlElement('w:jc')
            jc.set(qn('w:val'), 'center')
            pPr.append(jc)
            pBdr = OxmlElement('w:pBdr')
            bottomBdr = OxmlElement('w:bottom')
            bottomBdr.set(qn('w:val'), 'single')
            bottomBdr.set(qn('w:sz'), '4')
            bottomBdr.set(qn('w:space'), '1')
            bottomBdr.set(qn('w:color'), '808080')
            pBdr.append(bottomBdr)
            pPr.append(pBdr)
            p_elem.append(pPr)
            
            r_elem = OxmlElement('w:r')
            rPr = OxmlElement('w:rPr')
            rFonts = OxmlElement('w:rFonts')
            rFonts.set(qn('w:ascii'), 'Arial')
            rFonts.set(qn('w:hAnsi'), 'Arial')
            rPr.append(rFonts)
            color = OxmlElement('w:color')
            color.set(qn('w:val'), '808080')
            rPr.append(color)
            sz = OxmlElement('w:sz')
            sz.set(qn('w:val'), '18') # 9 pt
            rPr.append(sz)
            r_elem.append(rPr)
            t_elem = OxmlElement('w:t')
            t_elem.text = RUNNING_TITLE
            r_elem.append(t_elem)
            p_elem.append(r_elem)
            hdr_elem.append(p_elem)
            print(f"Set running header in {part.partname}")

    # 2. Update Title (P0)
    p0 = doc.paragraphs[0]
    p0.text = ""
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(6)
    p0.paragraph_format.line_spacing = 1.05
    r0 = p0.add_run("SEPARATING AVAILABILITY FROM POLICY QUALITY: GUARDED EDGE FALLBACK "
                    "FOR DEMAND RESPONSE UNDER COMMUNICATION OUTAGES")
    r0.font.name = "Arial"
    r0.font.size = Pt(16.0)
    r0.bold = True

    # 3. Update Author Line (P1)
    p1 = doc.paragraphs[1]
    p1.text = ""
    p1.paragraph_format.space_before = Pt(0)
    p1.paragraph_format.space_after = Pt(6)
    p1.paragraph_format.line_spacing = 1.0

    authors_data = [
        ("Low Choon Keat", "1*"),
        (", Jacquelin Ooi Jia Qi", "1"),
        (", Ng Yen Phing", "1"),
        (", Phoon Gar Chi", "2"),
        (" and Loo Bee Wah", "3"),
    ]
    for name, aff in authors_data:
        r_name = p1.add_run(name)
        r_name.font.name = "Arial"
        r_name.font.size = Pt(9.0)
        r_aff = p1.add_run(aff)
        r_aff.font.name = "Arial"
        r_aff.font.size = Pt(9.0)
        r_aff.font.superscript = True

    # Note: P2 in the pristine template is an empty paragraph with bottom border!
    # KEEP P2 UNTOUCHED so the divider line is preserved exactly as in template!

    # 4. Update Table 0 (Abstract & Keywords)
    t0 = doc.tables[0]
    
    # Left Cell: Abstract
    c0 = t0.rows[0].cells[0]
    # Paragraph 0 is "Abstract"
    p_abs_h = c0.paragraphs[0]
    p_abs_h.text = ""
    r_abs_h = p_abs_h.add_run("Abstract")
    r_abs_h.font.name = "Arial"
    r_abs_h.font.size = Pt(9.5)
    r_abs_h.bold = True
    p_abs_h.paragraph_format.space_before = Pt(0)
    p_abs_h.paragraph_format.space_after = Pt(2)

    # Paragraph 1 is Abstract body
    p_abs_b = c0.paragraphs[1]
    p_abs_b.text = ""
    p_abs_b.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs_b.paragraph_format.space_before = Pt(0)
    p_abs_b.paragraph_format.space_after = Pt(0)
    p_abs_b.paragraph_format.line_spacing = 1.0

    tight_abstract = (
        "Local demand-response execution stays available when the cloud command path fails, but availability is "
        "not a claim about policy quality, and it does not show that a learning policy beats a rule. This paper "
        "evaluates READ-Grid (Resilient Edge-Native Autonomous Demand Response for Smart Grids), a controller whose "
        "cloud policy is guarded by a local command-acceptance boundary, separating path availability from policy "
        "quality in a Python feeder simulation. A first run covered 13,965 records across 21 episodes and 95 faults. "
        "The tuned local rule reached 58.57% target achievement and the learned policy 45.25%, a gap of 13.32 "
        "percentage points. Three pipeline defects could produce that gap, namely intra-minute heartbeat "
        "look-ahead, a Bellman target maximised over inadmissible successors, and training with neither "
        "full-episode replay nor state coverage. An audited, pre-registered pipeline removes all three. Its "
        "held-out primary comparison against a rule enumerated over 144 candidates is inconclusive. The mean "
        "difference is −0.0040 (95% interval [−0.0239, +0.0174]), or −0.40 percentage points, and a half-width of "
        "0.0207 against the pre-registered margin δ = 0.02 leaves superiority, non-inferiority and equivalence "
        "unestablished. Correcting the pipeline moves the learned arm by 11.52 percentage points [+7.26, +16.19], "
        "locating the shortfall in the training pipeline, not the action-dependent representation. The guarded "
        "local path adds 14.7 to 22.0 percentage points of outage-window achievement across all four policy pairs, "
        "positive throughout though unequal. The audited learner costs 0.96 ms per control step, 32.8 times the "
        "cloud-only rule and 0.0016% of the 60 s interval."
    )
    add_formatted_runs(p_abs_b, tight_abstract, font_name="Arial", font_size_pt=9.0)

    # Right Cell: Nested table (grey box) + Keywords
    c1 = t0.rows[0].cells[1]
    nested_table = c1.tables[0]
    nt_c0 = nested_table.rows[0].cells[0]
    
    # Update paragraphs in nested table
    nt_paras = nt_c0.paragraphs
    # Ensure there are 8 paragraphs
    while len(nt_paras) < 8:
        nt_c0.add_paragraph()
        nt_paras = nt_c0.paragraphs

    def set_nt_p(p, prefix_text, body_text):
        p.text = ""
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        if prefix_text:
            r_pre = p.add_run(prefix_text)
            r_pre.font.name = "Arial"
            r_pre.font.size = Pt(9.0)
            r_pre.bold = True
            r_pre.italic = True
        r_bod = p.add_run(body_text)
        r_bod.font.name = "Arial"
        r_bod.font.size = Pt(9.0)
        r_bod.italic = True

    set_nt_p(nt_paras[0], "Received: ", "[to be assigned]")
    set_nt_p(nt_paras[1], "Revised: ", "[to be assigned]")
    set_nt_p(nt_paras[2], "Accepted: ", "[to be assigned]")
    set_nt_p(nt_paras[3], "", "1 Department of Information and Communication Technology, Faculty of Computing and Information Technology, Tunku Abdul Rahman University of Management and Technology (TAR UMT), 53300 Kuala Lumpur, Malaysia.")
    set_nt_p(nt_paras[4], "", "2 Department of Creative Industries, Faculty of Communication and Creative Industries, Tunku Abdul Rahman University of Management and Technology (TAR UMT), 53300 Kuala Lumpur, Malaysia.")
    set_nt_p(nt_paras[5], "", "3 Department of Mathematical Sciences, Faculty of Computing and Information Technology (FOCS), Tunku Abdul Rahman University of Management and Technology (TAR UMT), 53300 Kuala Lumpur, Malaysia.")
    set_nt_p(nt_paras[6], "*Corresponding author: ", "lowck@tarc.edu.my")
    set_nt_p(nt_paras[7], "DOI: ", "[to be assigned]")

    # Paragraphs in c1 below the nested table:
    # P0: empty
    # P1: empty
    # P2: empty, with bottom border (horizontal rule)
    # P3: Keywords:
    # P4: Keywords text
    p_kw_h = c1.paragraphs[3]
    p_kw_h.text = ""
    r_kw_h = p_kw_h.add_run("Keywords:")
    r_kw_h.font.name = "Arial"
    r_kw_h.font.size = Pt(9.5)
    r_kw_h.bold = True
    p_kw_h.paragraph_format.space_before = Pt(2)
    p_kw_h.paragraph_format.space_after = Pt(1)

    p_kw_b = c1.paragraphs[4]
    p_kw_b.text = ""
    r_kw_b = p_kw_b.add_run("Demand response; local fallback; communication outages; linear value approximation; distribution feeder; smart grids.")
    r_kw_b.font.name = "Arial"
    r_kw_b.font.size = Pt(9.0)
    p_kw_b.paragraph_format.space_before = Pt(0)
    p_kw_b.paragraph_format.space_after = Pt(0)

    # Note: P4 in pristine template is empty with bottom border!
    # P5 is section break to 2 columns!
    # KEEP BOTH UNTOUCHED!

    # 5. Save references to Table 14 and Table 15 (Profiles)
    t14 = doc.tables[14]._tbl
    t15 = doc.tables[15]._tbl

    # Update Table 15 with our 5 authors
    bios = [
        ("Low Choon Keat", TEMPLATE_DIR / "table15_row0.jpg",
         "Low Choon Keat is an Assistant Professor in the Faculty of Computing and Information Technology at Tunku Abdul Rahman University of Management and Technology (TAR UMT), Malaysia. His research focuses on fog and edge computing, the Internet of Things (IoT), energy-efficient resource management, deep reinforcement learning, neuro-symbolic artificial intelligence, explainable AI, privacy-preserving sensing, and risk-aware recommendation systems. His work aims to advance energy-efficient intelligent systems and develop trustworthy, explainable AI architectures for next-generation computing environments. Email address: lowck@tarc.edu.my https://orcid.org/0000-0001-6683-6562"),
        ("Jacquelin Ooi Jia Qi", TEMPLATE_DIR / "Jacquelin Ooi Jia Qi.jpeg",
         "Jacquelin Ooi Jia Qi is with the Faculty of Computing and Information Technology, TAR UMT, Malaysia. Her current work focuses on smart-grid analytics, edge computing, and resilient demand-response system design, with particular interest in translating simulation studies into deployable cyber-physical architectures. Email address: jacquelinoj-wp22@student.tarc.edu.my"),
        ("Ng Yen Phing", TEMPLATE_DIR / "Ng Yen Phing.jpeg",
         "Ng Yen Phing is an Assistant Professor at the Department of Information and Communication Technology, Faculty of Computing and Information Technology (FOCS), TAR UMT. Her expertise covers Cloud Computing, Embedded Systems (Internet of Things), Fog Computing paradigms, Network-on-Chip (NoC) architectural design, Software Defined Networking (SDN), and Cybersecurity utilizing Machine Learning for threat detection. She is passionate about optimizing next-generation computing architectures and advancing practical, machine-learning-driven security applications to overcome real-world network limitations. Email address: ngyp@tarc.edu.my https://orcid.org/0009-0006-3026-3823"),
        ("Phoon Gar Chi", TEMPLATE_DIR / "table15_row2.jpg",
         "Phoon Gar Chi is an Assistant Professor in the Department of Creative Industries at Tunku Abdul Rahman University of Management and Technology (TAR UMT), Malaysia. She holds a PhD in Artistic and Creative Technology from Universiti Pendidikan Sultan Idris (UPSI). Her research focuses on the intersection of immersive technologies and creative education, specifically in Virtual Reality (VR), 2D/3D animation, and digital multimedia frameworks for experiential learning. Email address: phoongc@tarc.edu.my https://orcid.org/0000-0002-4961-8149"),
        ("Loo Bee Wah", TEMPLATE_DIR / "table15_row3.jpg",
         "Loo Bee Wah is a Senior Lecturer in the Department of Mathematical Sciences, Faculty of Computing and Information Technology (FOCS) at Tunku Abdul Rahman University of Management and Technology (TAR UMT), Malaysia. With over 24 years of academic experience, her research interests include applied mathematics, statistics, and decision support systems. She previously served as a Programme Leader (2013–2019) and has contributed to studies on financial risk tolerance and information security control analysis. Email address: loobw@tarc.edu.my https://orcid.org/0000-0003-3973-4434"),
    ]

    t15_obj = doc.tables[15]
    for idx, (name, img_path, bio_text) in enumerate(bios):
        row = t15_obj.rows[idx]
        # Photo in Cell 0
        c0 = row.cells[0]
        c0.text = ""
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(2)
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run()
        r0.add_picture(str(img_path), width=Inches(0.85))
        # Bio in Cell 1
        c1 = row.cells[1]
        c1.text = ""
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p1.paragraph_format.space_before = Pt(2)
        p1.paragraph_format.space_after = Pt(4)
        p1.paragraph_format.line_spacing = 1.0
        # Author name bold
        name_len = len(name)
        r_name = p1.add_run(name)
        r_name.font.name = "Arial"
        r_name.font.size = Pt(9.0)
        r_name.bold = True
        # Rest of bio regular
        r_rest = p1.add_run(bio_text[name_len:])
        r_rest.font.name = "Arial"
        r_rest.font.size = Pt(9.0)

    # 6. Remove existing body elements between P5 and Table 14
    idx_t14 = body.index(t14)
    elems_to_remove = [body[i] for i in range(6, idx_t14)]
    for e in elems_to_remove:
        body.remove(e)

    # Remove t14 and t15 temporarily to append document flow
    body.remove(t14)
    body.remove(t15)

    # Ensure Section 1 is 2 columns
    sec1 = doc.sections[1]
    cols = sec1._sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2')
        cols[0].set(qn('w:space'), '425')

    # 7. Parse the Humanized Markdown File
    md_content = MD_PATH.read_text(encoding="utf-8")
    
    intro_start = md_content.find("# 1.0 INTRODUCTION")
    ref_start = md_content.find("# REFERENCES")
    prof_start = md_content.find("# PROFILES")

    main_text = md_content[intro_start:ref_start].strip()
    refs_text = md_content[ref_start:prof_start].strip()

    # Helper to insert a full-width table (Tables 1 to 6)
    def insert_full_table(table_md_str, table_idx):
        lines = [l.strip() for l in table_md_str.strip().splitlines() if l.strip()]
        header_cells = [c.strip() for c in lines[0].split('|')[1:-1]]
        data_rows = []
        for l in lines[2:]:
            data_rows.append([c.strip() for c in l.split('|')[1:-1]])
        
        switch_to_1col(doc)
        
        # Caption above table
        caption_text = TABLE_CAPTIONS.get(table_idx, f"Table {table_idx}")
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(8)
        p_cap.paragraph_format.space_after = Pt(3)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = "Arial"
        r_cap.font.size = Pt(9.0)
        p_cap.paragraph_format.keep_with_next = True
        p_cap.paragraph_format.keep_together = True
        
        # Create table
        t = doc.add_table(rows=len(data_rows) + 1, cols=len(header_cells))
        t.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
        
        # Determine column widths
        col_w_pts = TABLE_COL_WIDTHS.get(table_idx)
        if not col_w_pts or len(col_w_pts) != len(header_cells):
            eq_w = 516.1 / len(header_cells)
            col_w_pts = [eq_w] * len(header_cells)
            
        # Format Header
        hdr_row = t.rows[0]
        trPr = hdr_row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:tblHeader'))
        trPr.append(OxmlElement('w:cantSplit'))
        for c_i, h_text in enumerate(header_cells):
            cell = hdr_row.cells[c_i]
            cell.width = Pt(col_w_pts[c_i])
            cell.text = ""
            set_cell_borders(cell)
            set_cell_shading(cell, "D9D9D9")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(h_text)
            r.font.name = "Arial"
            r.font.size = Pt(9.0)
            r.bold = True
            
        # Format Body Rows
        for r_i, r_data in enumerate(data_rows):
            row = t.rows[r_i + 1]
            trPr = row._tr.get_or_add_trPr()
            trPr.append(OxmlElement('w:cantSplit'))
            for c_i, val in enumerate(r_data):
                if c_i < len(row.cells):
                    cell = row.cells[c_i]
                    cell.width = Pt(col_w_pts[c_i])
                    cell.text = ""
                    set_cell_borders(cell)
                    set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.0
                    add_formatted_runs(p, val, font_name="Arial", font_size_pt=9.0)

        # Note printed under the table, still in the full-width flow
        note_text = TABLE_NOTES.get(table_idx)
        if note_text:
            p_note = doc.add_paragraph()
            p_note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_note.paragraph_format.space_before = Pt(3)
            p_note.paragraph_format.space_after = Pt(8)
            p_note.paragraph_format.line_spacing = 1.0
            r_note = p_note.add_run(note_text)
            r_note.font.name = "Arial"
            r_note.font.size = Pt(8.0)

        switch_to_2col(doc)

    # Helper to insert Author CRediT table inside 2-column flow (Table 11 style)
    def insert_credit_table(table_md_str):
        lines = [l.strip() for l in table_md_str.strip().splitlines() if l.strip()]
        header_cells = [c.strip() for c in lines[0].split('|')[1:-1]]
        data_rows = []
        for l in lines[2:]:
            data_rows.append([c.strip() for c in l.split('|')[1:-1]])
        
        # Stays in current 2-column flow!
        # Total width = 247.4 pt (Col 0 = 75 pt / 1500 dxa, Col 1 = 172.4 pt / 3448 dxa)
        t = doc.add_table(rows=len(data_rows) + 1, cols=2)
        t.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
        
        # Set borderless
        tblPr = t._tbl.tblPr
        tblBorders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            '<w:top w:val="nil" w:sz="0" w:space="0" w:color="FFFFFF"/>'
            '<w:left w:val="nil" w:sz="0" w:space="0" w:color="FFFFFF"/>'
            '<w:bottom w:val="nil" w:sz="0" w:space="0" w:color="FFFFFF"/>'
            '<w:right w:val="nil" w:sz="0" w:space="0" w:color="FFFFFF"/>'
            '<w:insideH w:val="nil" w:sz="0" w:space="0" w:color="FFFFFF"/>'
            '<w:insideV w:val="nil" w:sz="0" w:space="0" w:color="FFFFFF"/>'
            '</w:tblBorders>'
        )
        tblPr.append(tblBorders)
        
        col_w_pts = [75.0, 172.4]
        
        # Header Row: fill #E6E6E6
        hdr = t.rows[0]
        trPr = hdr._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        for c_i, h_text in enumerate(header_cells[:2]):
            c = hdr.cells[c_i]
            c.width = Pt(col_w_pts[c_i])
            c.text = ""
            set_cell_shading(c, "E6E6E6")
            set_cell_margins(c, top=40, bottom=40, left=40, right=40)
            p = c.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(h_text)
            r.font.name = "Arial"
            r.font.size = Pt(8.5)
            r.bold = True
            
        # Body Rows: alternating shading (white, #E6E6E6)
        for r_i, r_data in enumerate(data_rows):
            row = t.rows[r_i + 1]
            trPr = row._tr.get_or_add_trPr()
            trPr.append(OxmlElement('w:cantSplit'))
            shd_color = "E6E6E6" if (r_i % 2 == 1) else None
            for c_i in range(2):
                c = row.cells[c_i]
                c.width = Pt(col_w_pts[c_i])
                c.text = ""
                if shd_color:
                    set_cell_shading(c, shd_color)
                set_cell_margins(c, top=40, bottom=40, left=40, right=40)
                p = c.paragraphs[0]
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                val = r_data[c_i] if c_i < len(r_data) else ""
                r = p.add_run(val)
                r.font.name = "Arial"
                r.font.size = Pt(8.5)
                if c_i == 0:
                    r.bold = True

    # Helper to insert a full-width figure
    def insert_figure(fig_idx, caption_text):
        img_path = FIG_MAP.get(fig_idx)
        if not img_path or not img_path.exists():
            print(f"Warning: figure {fig_idx} image not found at {img_path}")
            return
        
        switch_to_1col(doc)
        
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        r_img = p_img.add_run()
        r_img.add_picture(str(img_path), width=Inches(6.85))
        p_img.paragraph_format.keep_with_next = True
        p_img.paragraph_format.keep_together = True
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(f"Figure {fig_idx}: {caption_text}")
        r_cap.font.name = "Arial"
        r_cap.font.size = Pt(9.0)
        p_cap.paragraph_format.keep_together = True
        
        switch_to_2col(doc)

    # Process main_text line by line
    lines = main_text.splitlines()
    i = 0
    current_table_idx = 1
    
    EQ_PREFIXES = ("P_g(t, a) =", "P_g(t,a) =", "Q(s, a) =", "r_t = -", "score =")

    in_declarations = False

    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue

        # Skip any raw HTML img tags
        if line.startswith("<img") or "<img" in line:
            i += 1
            continue
            
        # 1. Level 1 Heading
        if line.startswith("# "):
            heading_text = line[2:].strip()
            if "DECLARATIONS" in heading_text.upper():
                in_declarations = True
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(heading_text.upper())
            r.font.name = "Arial"
            r.font.size = Pt(9.5)
            r.bold = True
            i += 1
            continue

        # 2. Level 2 Heading
        if line.startswith("## "):
            heading_text = line[3:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(heading_text)
            r.font.name = "Arial"
            r.font.size = Pt(9.5)
            r.bold = True
            i += 1
            continue

        # 3. Figure Reference: *Figure X. Caption...*
        m_fig = re.match(r'^\*Figure\s+(\d+)\.\s*(.*?)\*?$', line)
        if m_fig:
            fig_idx = int(m_fig.group(1))
            fig_cap = m_fig.group(2).rstrip('*').strip()
            insert_figure(fig_idx, fig_cap)
            i += 1
            continue

        # Standalone table-caption lines are supplied by TABLE_CAPTIONS, so a
        # caption written into the markdown is not emitted twice.
        if re.match(r'^Table\s+\d+[:.]', line):
            i += 1
            continue

        # 4. Table Detection
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            
            # Check if this is the Author CRediT table
            if in_declarations or "CRediT" in table_lines[0]:
                insert_credit_table("\n".join(table_lines))
            else:
                insert_full_table("\n".join(table_lines), current_table_idx)
                current_table_idx += 1
            continue

        # 5. Bullet list
        if line.startswith("- "):
            bullet_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Pt(12)
            p.paragraph_format.first_line_indent = Pt(-8)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            r_b = p.add_run("•  ")
            r_b.font.name = "Arial"
            r_b.font.size = Pt(9.0)
            r_b.bold = True
            add_formatted_runs(p, bullet_text, font_name="Arial", font_size_pt=9.0)
            i += 1
            continue

        # 6. Numbered list items (e.g. "1. Intra-interval...")
        m_num = re.match(r'^(\d+\.)\s+(.*)$', line)
        if m_num and not line.startswith("1.0") and not line.startswith("2.0") and not line.startswith("3.0") and not line.startswith("4.0") and not line.startswith("5.0") and not line.startswith("6.0") and not line.startswith("7.0") and not line.startswith("8.0"):
            item_num = m_num.group(1)
            item_text = m_num.group(2)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Pt(12)
            p.paragraph_format.first_line_indent = Pt(-8)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            r_num = p.add_run(f"{item_num}  ")
            r_num.font.name = "Arial"
            r_num.font.size = Pt(9.0)
            r_num.bold = False
            add_formatted_runs(p, item_text, font_name="Arial", font_size_pt=9.0)
            i += 1
            continue

        # 7. Equation Detection
        if any(line.startswith(pfx) for pfx in EQ_PREFIXES):
            m_eq = re.search(r'^(.*?)\s*\((\d+)\)\.?$', line)
            if m_eq:
                formula_text = m_eq.group(1).strip()
                eq_num = m_eq.group(2).strip()
            else:
                formula_text = line
                eq_num = ""
            
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_together = True
            r_form = p.add_run(formula_text)
            r_form.font.name = "Cambria Math"
            r_form.font.size = Pt(9.5)
            r_form.italic = True
            
            if eq_num:
                r_tab = p.add_run(f"    ({eq_num})")
                r_tab.font.name = "Arial"
                r_tab.font.size = Pt(9.0)
                r_tab.bold = False
            i += 1
            continue

        # 8. Declarations lead-in paragraphs (e.g. "Conflict of interest. The authors...")
        if in_declarations and any(line.startswith(pfx) for pfx in ("Conflict of interest.", "Data availability.", "Ethics statement.", "Funding.", "Author contributions.")):
            parts = line.split(".", 1)
            label = parts[0] + "."
            rest = parts[1] if len(parts) > 1 else ""
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            r_lbl = p.add_run(label)
            r_lbl.font.name = "Arial"
            r_lbl.font.size = Pt(9.0)
            r_lbl.bold = True
            r_rst = p.add_run(rest)
            r_rst.font.name = "Arial"
            r_rst.font.size = Pt(9.0)
            i += 1
            continue

        # 9. Divider line detection after declarations approval statement
        if "All authors reviewed and approved the final manuscript." in line:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(line)
            r.font.name = "Arial"
            r.font.size = Pt(9.0)
            # Add divider line right after this paragraph
            add_divider_line_paragraph(doc)
            i += 1
            continue

        # 10. Standard Paragraph
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        add_formatted_runs(p, line, font_name="Arial", font_size_pt=9.0)
        i += 1

    # 8. Insert References (in 2-column layout)
    p_ref_h = doc.add_paragraph()
    p_ref_h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_ref_h.paragraph_format.space_before = Pt(8)
    p_ref_h.paragraph_format.space_after = Pt(3)
    p_ref_h.paragraph_format.keep_with_next = True
    r_rh = p_ref_h.add_run("REFERENCES")
    r_rh.font.name = "Arial"
    r_rh.font.size = Pt(9.5)
    r_rh.bold = True

    ref_entries = re.findall(r'\[(\d+)\]\s+(.*?)(?=\n\[\d+\]|\n#|\Z)', refs_text, re.DOTALL)
    print(f"Adding {len(ref_entries)} references in APA style...")
    for num, ref_content in ref_entries:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Pt(14)
        p_ref.paragraph_format.first_line_indent = Pt(-14)
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.paragraph_format.line_spacing = 1.0
        
        r_num = p_ref.add_run(f"[{num}]\t")
        r_num.font.name = "Arial"
        r_num.font.size = Pt(9.0)
        
        clean_content = " ".join(ref_content.split())
        r_body = p_ref.add_run(clean_content)
        r_body.font.name = "Arial"
        r_body.font.size = Pt(9.0)

    # 9. Switch to 1 Column for Profiles Section
    switch_to_1col(doc)

    # Append Table 14 and Table 15 back to body
    body.append(t14)
    body.append(t15)

    # Save to both target locations
    print(f"Saving final document to: {OUT_DOCX}")
    doc.save(OUT_DOCX)
    print(f"Also updating template copy: {TEMPLATE_DOCX}")
    doc.save(TEMPLATE_DOCX)
    print("Manuscript generation completed successfully!")


if __name__ == "__main__":
    main()
