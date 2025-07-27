import re
import json
import os
import copy

def parse_regulation(md_content):
    """
    Parses the EU Regulation markdown text into a hierarchical JSON structure.
    Improved Annex parsing logic.
    """
    lines = md_content.splitlines()
    regulation_data = {
        "metadata": {},
        "preamble": {"citations": [], "recitals": []},
        "enacting_terms": {},
        "annexes": [],
        "final_provisions": {}
    }

    current_part_key = None
    current_title_key = None
    current_chapter_key = None
    current_section_key = None
    current_sub_section_key = None

    current_article_lines = []
    current_article_number = None
    current_article_title = None

    current_annex_lines = []
    current_annex_id = None
    current_annex_title = None

    state = "start"

    article_start_pattern = re.compile(r"^\s*(?:#\s*)?Article\s+(\d+)", re.IGNORECASE)

    def get_current_level():
        level = regulation_data["enacting_terms"]
        if current_part_key:
            if current_part_key not in level: level[current_part_key] = {}
            level = level[current_part_key]
        if current_title_key:
            if current_title_key not in level: level[current_title_key] = {}
            level = level[current_title_key]
        if current_chapter_key:
            if current_chapter_key not in level: level[current_chapter_key] = {}
            level = level[current_chapter_key]
        if current_section_key:
            if current_section_key not in level: level[current_section_key] = {}
            level = level[current_section_key]
        if current_sub_section_key:
            if current_sub_section_key not in level: level[current_sub_section_key] = {}
            level = level[current_sub_section_key]
        return level

    def finalize_article():
        nonlocal current_article_lines, current_article_number, current_article_title
        if current_article_number is not None:
            if not current_article_lines and not current_article_title:
                 current_article_number = None
                 return
            try:
                current_level = get_current_level()
            except KeyError as e:
                print(f"ERROR (finalize_article): KeyError accessing structure for Article {current_article_number}. Error: {e}")
                current_article_lines = []
                current_article_number = None
                current_article_title = None
                return

            article_text = "\n".join(current_article_lines).strip()
            article_text = re.sub(r"^\n+", "", article_text)
            article_text = re.sub(r"\n+$", "", article_text)

            article_entry = {
                "number": current_article_number,
                "text": article_text
            }
            if current_article_title and current_article_title.strip():
                 article_entry["title"] = current_article_title.strip()

            if "articles" not in current_level:
                 current_level["articles"] = []

            # Check for duplicates before appending
            is_duplicate = False
            if current_level["articles"]:
                last_article = current_level["articles"][-1]
                if last_article.get("number") == current_article_number and last_article.get("text") == article_text:
                    is_duplicate = True # Avoid appending exact same content again

            if not is_duplicate:
                 current_level["articles"].append(article_entry)

            current_article_lines = []
            current_article_number = None
            current_article_title = None

    def finalize_annex():
        nonlocal current_annex_lines, current_annex_id, current_annex_title
        if current_annex_id is not None:
             # Don't create empty annexes
            if not current_annex_lines and not current_annex_title:
                current_annex_id = None
                return

            annex_text = "\n".join(current_annex_lines).strip()
            annex_entry = {
                "id": current_annex_id,
                "title": current_annex_title.strip() if current_annex_title else None,
                "text": annex_text
            }
            regulation_data["annexes"].append(annex_entry)
            current_annex_lines = []
            current_annex_id = None
            current_annex_title = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        original_line = lines[i]
        next_line = lines[i+1].strip() if i + 1 < len(lines) else ""

        # --- State Machine Logic ---

        if state in ["start", "metadata", "preamble_citations", "preamble_recitals"]:
             # (Preamble/metadata logic remains the same)
            if state == "start":
                if line.startswith("(Legislative acts)"):
                    regulation_data["metadata"]["act_type"] = "REGULATIONS"
                    state = "metadata"
                elif line:
                     if "metadata_misc" not in regulation_data["metadata"]: regulation_data["metadata"]["metadata_misc"] = []
                     regulation_data["metadata"]["metadata_misc"].append(line)

            elif state == "metadata":
                if re.match(r"^REGULATION \(EU\) No \d+/\d+ OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL", line):
                    regulation_data["metadata"]["title"] = line
                    if i + 1 < len(lines) and re.match(r"^of \d{1,2} \w+ \d{4}", lines[i+1].strip()):
                        regulation_data["metadata"]["date"] = lines[i+1].strip().replace("of ", "")
                        i += 1
                    if i + 1 < len(lines) and re.match(r"^\(Text with EEA relevance\)", lines[i+1].strip()):
                         regulation_data["metadata"]["eea_relevance"] = True
                         i += 1
                elif line.startswith("Having regard to"):
                    state = "preamble_citations"
                    regulation_data["preamble"]["citations"].append(original_line)
                elif line:
                     if "metadata_misc" not in regulation_data["metadata"]: regulation_data["metadata"]["metadata_misc"] = []
                     regulation_data["metadata"]["metadata_misc"].append(line)

            elif state == "preamble_citations":
                if line.startswith("Having regard to") or \
                   line.startswith("After transmission of") or \
                   line.startswith("Acting in accordance"):
                     regulation_data["preamble"]["citations"].append(original_line)
                elif line.startswith("Whereas:"):
                    state = "preamble_recitals"
                elif line.startswith("HAVE ADOPTED THIS REGULATION:"):
                    state = "enacting_terms"
                elif regulation_data["preamble"]["citations"] and line:
                    regulation_data["preamble"]["citations"][-1] += "\n" + original_line

            elif state == "preamble_recitals":
                recital_match = re.match(r"^\((\d+)\)\s*(.*)", original_line, re.DOTALL)
                if recital_match:
                    recital_num = int(recital_match.group(1))
                    recital_text = recital_match.group(2).strip()
                    k = i + 1
                    while k < len(lines):
                        next_recital_line = lines[k].strip()
                        if re.match(r"^\(\d+\)", next_recital_line) or next_recital_line.startswith("HAVE ADOPTED THIS REGULATION:"):
                            break
                        recital_text += "\n" + lines[k]
                        k += 1
                    i = k - 1

                    regulation_data["preamble"]["recitals"].append({
                        "number": recital_num,
                        "text": recital_text.strip()
                    })
                elif line.startswith("HAVE ADOPTED THIS REGULATION:"):
                    state = "enacting_terms"

        elif state in ["enacting_terms", "annex", "final_provisions"]:
            part_match = re.match(r"^PART (\w+)", line, re.IGNORECASE)
            title_match = re.match(r"^TITLE ([IVXLCDM]+)", line, re.IGNORECASE)
            chapter_match = re.match(r"^CHAPTER (\d+)", line, re.IGNORECASE)
            section_match = re.match(r"^S\s*e\s*c\s*t\s*i\s*o\s*n\s*(\d+)", line, re.IGNORECASE)
            sub_section_match = re.match(r"^S\s*u\s*b\s*-\s*S\s*e\s*c\s*t\s*i\s*o\s*n\s*(\d+)", line, re.IGNORECASE)
            article_match = article_start_pattern.match(line)
            annex_match = re.match(r"^ANNEX ([IVXLCDM]+)", line, re.IGNORECASE)
            final_provisions_match = re.match(r"^This Regulation shall be binding", line)
            done_at_match = re.match(r"^Done at", line)

            structural_element_found = False

            # --- Process Structure or Content ---

            # 1. Check for Annex Start (High Priority after Preamble)
            if annex_match:
                finalize_article()
                finalize_annex() # Finalize previous annex
                state = "annex"
                current_annex_id = annex_match.group(1)
                current_annex_title = ""
                k = i + 1
                title_lines = []
                # Capture potential multi-line titles
                while k < len(lines):
                    potential_title = lines[k].strip()
                    # Heuristic: Stop title capture if it looks like content (e.g., starts with number, lower case, table syntax)
                    # Or if another ANNEX starts
                    if not potential_title \
                       or re.match(r"^(ANNEX|\d+\.|\(?[a-z]\)|\(?[ivxlcdm]+\)|Table|TABLE|---+)", potential_title, re.IGNORECASE) \
                       or potential_title.startswith("Column") or potential_title.startswith("Points"):
                        break
                    title_lines.append(potential_title)
                    k += 1
                if title_lines:
                    current_annex_title = " ".join(title_lines)
                    i = k - 1 # Adjust main loop counter past the title lines
                structural_element_found = True # Processed this line

            # 2. Check for Article Start
            elif article_match and state != "final_provisions":
                # If we were in an annex, finalize it first
                if state == "annex":
                    finalize_annex()
                finalize_article() # Finalize previous article
                state = "enacting_terms" # Switch back if needed
                current_article_number = article_match.group(1)
                current_article_title = ""
                title_on_same_line = re.match(r"^\s*(?:#\s*)?Article\s+\d+\s+(.+)", line)
                if title_on_same_line:
                    current_article_title = title_on_same_line.group(1).strip()
                elif next_line.startswith("#"):
                     current_article_title = next_line.replace("#", "").strip()
                     i += 1
                structural_element_found = True

            # 3. Check for other Enacting Terms Structure
            elif part_match and state != "annex": # Don't allow PART inside ANNEX
                finalize_article()
                state = "enacting_terms"
                current_part_key = f"PART_{part_match.group(1)}"
                current_title_key, current_chapter_key, current_section_key, current_sub_section_key = None, None, None, None
                part_title = next_line if next_line and not re.match(r"^(TITLE|Article)", next_line, re.IGNORECASE) else None
                if current_part_key not in regulation_data["enacting_terms"]:
                    regulation_data["enacting_terms"][current_part_key] = {}
                if part_title:
                    regulation_data["enacting_terms"][current_part_key]["title"] = part_title.strip()
                    i += 1
                structural_element_found = True
            elif title_match and current_part_key and state != "annex":
                finalize_article()
                state = "enacting_terms"
                current_title_key = f"TITLE_{title_match.group(1)}"
                current_chapter_key, current_section_key, current_sub_section_key = None, None, None
                title_title = next_line if next_line and not re.match(r"^(CHAPTER|Article)", next_line, re.IGNORECASE) else None
                parent_level = regulation_data["enacting_terms"][current_part_key]
                if current_title_key not in parent_level: parent_level[current_title_key] = {}
                if title_title:
                    parent_level[current_title_key]["title"] = title_title.strip()
                    i += 1
                structural_element_found = True
            elif chapter_match and current_title_key and state != "annex":
                finalize_article()
                state = "enacting_terms"
                current_chapter_key = f"CHAPTER_{chapter_match.group(1)}"
                current_section_key, current_sub_section_key = None, None
                chapter_title = ""
                if next_line.startswith("#"):
                    chapter_title = next_line.replace("#", "").strip()
                    i += 1
                elif next_line and not re.match(r"^(Section|S\s*e\s*c\s*t\s*i\s*o\s*n|Article|^\(?[a-zivxlcdm]+\)|^\d+\.)", next_line, re.IGNORECASE):
                    chapter_title = next_line.strip()
                    i += 1
                parent_level = regulation_data["enacting_terms"][current_part_key][current_title_key]
                if current_chapter_key not in parent_level: parent_level[current_chapter_key] = {}
                if chapter_title: parent_level[current_chapter_key]["title"] = chapter_title
                structural_element_found = True
            elif section_match and current_chapter_key and state != "annex":
                finalize_article()
                state = "enacting_terms"
                current_section_key = f"Section_{section_match.group(1)}"
                current_sub_section_key = None
                section_title = ""
                k = i + 1
                title_lines = []
                while k < len(lines):
                    potential_title = lines[k].strip()
                    if not potential_title or re.match(r"^(Sub-Section|S\s*u\s*b\s*-\s*S\s*e\s*c\s*t\s*i\s*o\s*n|Article|^\(?[a-zivxlcdm]+\)|^\d+\.)", potential_title, re.IGNORECASE):
                        break
                    title_lines.append(potential_title)
                    k += 1
                if title_lines:
                    section_title = " ".join(title_lines)
                    i = k - 1
                parent_level = regulation_data["enacting_terms"][current_part_key][current_title_key][current_chapter_key]
                if current_section_key not in parent_level: parent_level[current_section_key] = {}
                if section_title: parent_level[current_section_key]["title"] = section_title
                structural_element_found = True
            elif sub_section_match and current_section_key and state != "annex":
                finalize_article()
                state = "enacting_terms"
                current_sub_section_key = f"Sub-Section_{sub_section_match.group(1)}"
                sub_section_title = ""
                k = i + 1
                title_lines = []
                while k < len(lines):
                    potential_title = lines[k].strip()
                    if not potential_title or re.match(r"^(Article|^\(?[a-zivxlcdm]+\)|^\d+\.)", potential_title, re.IGNORECASE):
                         break
                    title_lines.append(potential_title)
                    k += 1
                if title_lines:
                    sub_section_title = " ".join(title_lines)
                    i = k - 1
                parent_level = regulation_data["enacting_terms"][current_part_key][current_title_key][current_chapter_key][current_section_key]
                if current_sub_section_key not in parent_level: parent_level[current_sub_section_key] = {}
                if sub_section_title: parent_level[current_sub_section_key]["title"] = sub_section_title
                structural_element_found = True

            # 4. Check for Final Provisions Start
            elif final_provisions_match or done_at_match:
                 finalize_article()
                 finalize_annex()
                 state = "final_provisions"
                 if final_provisions_match and "concluding_formula" not in regulation_data["final_provisions"]:
                      regulation_data["final_provisions"]["concluding_formula"] = original_line
                 structural_element_found = True

            # --- Append content ---
            if not structural_element_found and line:
                 if state == "enacting_terms" and current_article_number is not None:
                      # Avoid appending the article marker/title line itself as content
                      is_marker_line = (line == f"Article {current_article_number}")
                      is_title_line = (current_article_title and line.replace("#", "").strip() == current_article_title)
                      if not is_marker_line and not is_title_line:
                          current_article_lines.append(original_line)
                 elif state == "annex" and current_annex_id is not None:
                      # Avoid appending the annex marker/title line itself as content
                      is_marker_line = (line == f"ANNEX {current_annex_id}")
                      is_title_line = (current_annex_title and line == current_annex_title) # Simple check for single-line title
                      if not is_marker_line and not is_title_line:
                          current_annex_lines.append(original_line)

        elif state == "final_provisions":
             if "final_text_block" not in regulation_data["final_provisions"]:
                 regulation_data["final_provisions"]["final_text_block"] = []
             if line:
                 regulation_data["final_provisions"]["final_text_block"].append(original_line)

        i += 1

    # Finalize the last captured article or annex
    finalize_article()
    finalize_annex()

    # Consolidate final provisions text (keep as before)
    if "final_text_block" in regulation_data["final_provisions"]:
        full_final_text = "\n".join(regulation_data["final_provisions"]["final_text_block"]).strip()
        del regulation_data["final_provisions"]["final_text_block"]
        if "concluding_formula" not in regulation_data["final_provisions"]:
             concluding_match = re.search(r"This Regulation shall be binding.*?applicable in all Member States\.", full_final_text, re.DOTALL | re.IGNORECASE)
             if concluding_match:
                 regulation_data["final_provisions"]["concluding_formula"] = concluding_match.group(0).strip()
        if "place_date" not in regulation_data["final_provisions"]:
             done_at_match = re.search(r"Done at (.*?), (\d{1,2} \w+ \d{4})\.", full_final_text)
             if done_at_match:
                 regulation_data["final_provisions"]["place_date"] = f"Done at {done_at_match.group(1)}, {done_at_match.group(2)}."
        if "signatories" not in regulation_data["final_provisions"]:
            signatories = re.findall(r"(For the European Parliament|For the Council)\s+The President\s+([A-Z][A-Z.\s]+)", full_final_text, re.IGNORECASE)
            if signatories:
                 regulation_data["final_provisions"]["signatories"] = [f"{s[0]}\nThe President\n{s[1].strip()}" for s in signatories]

    return regulation_data

def convert_to_array_structure(input_dict):
    """
    Recursively converts a nested dictionary using structural keys
    (PART_X, TITLE_Y, CHAPTER_Z, Section_A, Sub-Section_B)
    into a structure using lists of objects with 'type', 'id', 'title',
    'children', and 'articles' keys.

    Args:
        input_dict: The dictionary level to transform.

    Returns:
        A list of transformed objects for the current level.
    """
    output_list = []

    # Define the order of processing for better structure preservation
    # (Adjust if specific ordering beyond key insertion is needed)
    key_order_prefixes = ["PART_", "TITLE_", "CHAPTER_", "Section_", "Sub-Section_"]

    # Sort keys roughly by structure type - this relies on Python 3.7+ dict order preservation
    # A more robust sort could parse the numbers/roman numerals if needed.
    def sort_key(item):
        key = item[0]
        for i, prefix in enumerate(key_order_prefixes):
            if key.startswith(prefix):
                # Try to extract number for sorting within type
                num_part = key.split('_')[-1]
                try:
                    # Handle Roman numerals for Titles
                    if prefix == "TITLE_":
                        roman_map = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
                        num = 0
                        last_value = 0
                        for numeral in reversed(num_part):
                            value = roman_map[numeral]
                            if value < last_value:
                                num -= value
                            else:
                                num += value
                            last_value = value
                        return (i, num)
                    else:
                         return (i, int(num_part)) # Sort by type index, then numeric value
                except (ValueError, KeyError):
                    return (i, 0) # Fallback for non-standard keys
        if key == "articles":
            return (len(key_order_prefixes), 0) # Articles come last
        if key == "title":
            return (-1, 0) # Title comes first (though usually handled separately)
        return (len(key_order_prefixes) + 1, 0) # Other keys after articles


    # Iterate through sorted keys to maintain document order as much as possible
    # Note: This relies on Python 3.7+ insertion order preservation in the *original* dict
    # If older Python or unstable order, a more complex pre-sort might be needed.
    for key, value in input_dict.items():
        # Regex to identify and parse structural keys
        match = re.match(r"^(PART|TITLE|CHAPTER|Section|Sub-Section)_([^_]+)$", key, re.IGNORECASE)

        if match:
            type_str = match.group(1).upper()
             # Adjust display name for sections
            if type_str == "SECTION": type_display = "Section"
            elif type_str == "SUB-SECTION": type_display = "Sub-Section"
            else: type_display = type_str.capitalize()

            id_str = match.group(2)

            transformed_item = {
                "type": type_display,
                "id": id_str,
                "title": None,
                "children": [],
                "articles": []
            }

            # Check if the value is a dictionary (it should be)
            if isinstance(value, dict):
                # Extract title if present at this level
                if "title" in value:
                    transformed_item["title"] = value["title"]

                # Extract articles if present at this level
                if "articles" in value:
                    # Ensure articles is a list (defensive coding)
                    if isinstance(value["articles"], list):
                        transformed_item["articles"] = value["articles"]
                    else:
                        print(f"Warning: Expected list for articles under {key}, found {type(value['articles'])}")


                # Recursively transform children (excluding 'title' and 'articles' keys)
                children_dict = {k: v for k, v in value.items() if k not in ["title", "articles"]}
                if children_dict: # Only recurse if there are actual child structures
                    transformed_item["children"] = convert_to_array_structure(children_dict)

                # Clean up empty lists if no children/articles were found
                if not transformed_item["children"]:
                    del transformed_item["children"]
                if not transformed_item["articles"]:
                    del transformed_item["articles"]

            else:
                print(f"Warning: Expected dictionary value for key {key}, found {type(value)}")

            output_list.append(transformed_item)

        # Handle articles list if it appears directly under the current dict
        # (The previous parser puts it *inside* the lowest structure, so this might not be hit often)
        elif key == "articles" and isinstance(value, list):
             # If articles exist directly at this level, add them to the last structural item found,
             # or handle them differently if needed (e.g., add a separate 'articles' entry to output_list).
             # For simplicity now, we assume the original parser nested them correctly.
             # If this case needs handling, logic would go here.
             pass

        elif key == "title":
             # Title should have been handled within the structural item processing
             pass

        else:
            print(f"Warning: Unhandled key '{key}' during transformation.")


    # Attempt to sort the output list based on the original key structure/numbering
    # This is a secondary sort, primary order depends on iteration order of input_dict
    output_list.sort(key=lambda item: sort_key((f"{item['type'].upper()}_{item['id']}", None))[1] if 'id' in item else 999)
    # Refine sort key extraction for sorting the final list if necessary

    return output_list


# --- Main execution ---
input_filename = "CRR.md"
output_filename = "CRR.json"

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        md_content = f.read()

    parsed_data = parse_regulation(md_content)
    transformed_data = copy.deepcopy(parsed_data)
    
# Transform the enacting_terms part
    if "enacting_terms" in transformed_data and isinstance(transformed_data["enacting_terms"], dict):
        transformed_data["enacting_terms"] = convert_to_array_structure(transformed_data["enacting_terms"])
    else:
        raise ValueError("enacting_terms is not a dictionary or not found in the parsed document.")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)

    print(f"Successfully parsed '{input_filename}' and saved to '{output_filename}'")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

