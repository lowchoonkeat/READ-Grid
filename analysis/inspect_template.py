import docx
from docx.oxml import parse_xml

doc = docx.Document('paper_package/template/iem_template_original_backup.docx')

print(f"Number of sections: {len(doc.sections)}")
for i, s in enumerate(doc.sections):
    sectPr = s._sectPr
    cols = sectPr.xpath('./w:cols')
    num_cols = cols[0].get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num') if cols else '1'
    print(f"Sect {i}: start={s.start_type}, cols={num_cols}, w={s.page_width.pt}pt, top={s.top_margin.pt}pt, bot={s.bottom_margin.pt}pt, left={s.left_margin.pt}pt, right={s.right_margin.pt}pt")

print("\n--- Body elements order (first 20) ---")
body_children = list(doc.element.body)
for idx, child in enumerate(body_children[:25]):
    tag = child.tag.split('}')[-1]
    if tag == 'p':
        p = docx.text.paragraph.Paragraph(child, doc)
        text_preview = p.text[:80].strip()
        print(f"{idx}: [P] {text_preview!r} (align={p.alignment})")
    elif tag == 'tbl':
        tbl = docx.table.Table(child, doc)
        print(f"{idx}: [TBL] rows={len(tbl.rows)}, cols={len(tbl.columns)}, text00={tbl.rows[0].cells[0].text[:40]!r}")
    elif tag == 'sectPr':
        print(f"{idx}: [sectPr]")

print("\n--- Table 11 (CRediT) inspection ---")
for t_idx, tbl in enumerate(doc.tables):
    txt = "".join(c.text for row in tbl.rows for c in row.cells)
    if 'CRediT' in txt or t_idx == 11:
        print(f"Found table index {t_idx}:")
        for r_i, r in enumerate(tbl.rows):
            print(f"  Row {r_i}: {[c.text.strip() for c in r.cells]}")
            print(f"    widths: {[c.width.pt if c.width else 'None' for c in r.cells]}")
        tblPr = tbl._tbl.tblPr
        borders = tblPr.xpath('./w:tblBorders')
        if borders:
            import xml.etree.ElementTree as ET
            print("  Borders raw:", ET.tostring(borders[0], encoding='utf-8').decode('utf-8'))
        # Let's inspect cell styles, font names, sizes in tbl
        for c in tbl.rows[0].cells:
            for p in c.paragraphs:
                for run in p.runs:
                    print(f"  Header run: font={run.font.name}, size={run.font.size.pt if run.font.size else None}, bold={run.bold}")
        for c in tbl.rows[1].cells:
            for p in c.paragraphs:
                for run in p.runs:
                    print(f"  Body run: font={run.font.name}, size={run.font.size.pt if run.font.size else None}, bold={run.bold}")
        break

print("\n--- Table 0 (Abstract) inspection ---")
t0 = doc.tables[0]
for r_i, r in enumerate(t0.rows):
    for c_i, c in enumerate(r.cells):
        print(f"Cell ({r_i},{c_i}) width={c.width.pt if c.width else 'None'} pt")
        print(f"  XML tcPr: {c._tc.tcPr.xml if c._tc.tcPr is not None else 'None'}")
        for p_i, p in enumerate(c.paragraphs):
            print(f"  P{p_i}: align={p.alignment} text={p.text[:60]!r}")
            for run in p.runs:
                print(f"    Run: text={run.text[:30]!r}, font={run.font.name}, size={run.font.size.pt if run.font.size else None}, bold={run.bold}")

print("\n--- First 5 paragraphs inspection ---")
for idx, child in enumerate(body_children[:10]):
    tag = child.tag.split('}')[-1]
    if tag == 'p':
        p = docx.text.paragraph.Paragraph(child, doc)
        print(f"P{idx}: XML pPr={p._p.pPr.xml if p._p.pPr is not None else 'None'}")
        print(f"  Text: {p.text!r}")
        for run in p.runs:
            print(f"    Run: font={run.font.name}, size={run.font.size.pt if run.font.size else None}, bold={run.bold}, italic={run.italic}, text={run.text!r}")
