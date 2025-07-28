def classify_heading_levels(headings, y_threshold=5):
    """
    headings: list of tuples (text, x0, font_size, bold, page_num, distance_from_top)
    """
    from collections import defaultdict

    # Group headings by page number
    pages = defaultdict(list)
    for h in headings:
        pages[h[4]].append(h)  # page_num = h[4]

    outline = []
    title_assigned = False  # 🟡 Flag to mark first heading as TITLE

    for page in sorted(pages.keys()):
        lines = pages[page]

        # Sort by vertical position (top to bottom)
        lines.sort(key=lambda x: x[5])  # distance_from_top

        merged = []
        current_group = []

        for line in lines:
            if not current_group:
                current_group.append(line)
                continue

            # Group lines close in vertical distance
            if abs(line[5] - current_group[-1][5]) <= y_threshold:
                current_group.append(line)
            else:
                merged.append(current_group)
                current_group = [line]

        if current_group:
            merged.append(current_group)

        for group in merged:
            rep = group[0]
            text = " ".join([g[0] for g in group])
            x0 = rep[1]
            font_size = rep[2]
            bold = rep[3]
            distance_from_top = rep[5]

            # Determine heading level using X-position
            x_values = sorted(set([l[1] for l in lines]))
            if abs(x0 - x_values[0]) < 0.5:
                level = "H1"
            elif len(x_values) > 1 and abs(x0 - x_values[1]) < 0.5:
                level = "H2"
            else:
                level = "H3"

            # Adjust level based on font size and boldness
            if len(x_values) == 1 or abs(x0 - x_values[0]) < 0.5:
                if bold and font_size >= 13:
                    level = "H1"
                elif bold or font_size >= 11:
                    level = "H2"
                else:
                    level = "H3"

            # ✅ Assign TITLE only once to the first heading encountered
            if not title_assigned:
                level = "TITLE"
                title_assigned = True

            outline.append({
                "level": level,
                "text": text,
                "page": page
            })

    return outline
