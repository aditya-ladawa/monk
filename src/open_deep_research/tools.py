from pathlib import Path
from typing import Dict, Optional, List
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing_extensions import Annotated
import os
import json
from PyPDF2 import PdfReader
import fitz  # PyMuPDF

# Fixed working directory
WORKING_DIRECTORY = Path.home() / "Aditya/y_projects/monitizor_coach/outputs"

# --- Markdown Tools ---

@tool
def create_md_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File name to save the Markdown outline, e.g. 'outline.md'."],
) -> Annotated[str, "Path of the saved Markdown outline file."]:
    """Create and save a Markdown outline as a numbered list."""
    file_path = WORKING_DIRECTORY / "roadmaps" / file_name
    if not file_name.endswith(".md"):
        file_path = file_path.with_suffix(".md")
    with file_path.open("w", encoding="utf-8") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return str(file_path)


@tool
def read_md_document(
    file_name: Annotated[str, "File name of the Markdown document to read, e.g. 'doc.md'."],
    start: Annotated[Optional[int], "The start line (0-indexed)."] = None,
    end: Annotated[Optional[int], "The end line (exclusive)."] = None,
) -> str:
    """Read the specified Markdown document."""
    file_path = WORKING_DIRECTORY / "roadmaps" / file_name
    if not file_name.endswith(".md"):
        file_path = file_path.with_suffix(".md")
    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()
    return "".join(lines[start or 0:end])


@tool
def write_md_document(
    content: Annotated[str, "Markdown text content to be written into the document."],
    file_name: Annotated[str, "File name to save the Markdown document, e.g. 'roadmap.md'."],
) -> Annotated[str, "Path of the saved Markdown document file."]:
    """Create and save a Markdown text document."""
    file_path = WORKING_DIRECTORY / "roadmaps" / file_name
    if not file_name.endswith(".md"):
        file_path = file_path.with_suffix(".md")
    with file_path.open("w", encoding="utf-8") as file:
        file.write(content)
    return str(file_path)


@tool
def edit_md_document(
    file_name: Annotated[str, "File name of the Markdown document to be edited."],
    inserts: Annotated[Dict[int, str], "Line number (1-indexed) to text insert mapping."],
) -> Annotated[str, "Path of the edited Markdown document file."]:
    """Edit a Markdown document by inserting text at specific line numbers."""
    file_path = WORKING_DIRECTORY / "roadmaps" / file_name
    if not file_name.endswith(".md"):
        file_path = file_path.with_suffix(".md")

    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line_number, text in sorted(inserts.items()):
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with file_path.open("w", encoding="utf-8") as file:
        file.writelines(lines)

    return str(file_path)

# --- Python Code Execution Tool ---

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "Python code to execute for generating charts/images."]
):
    """Execute Python code and return stdout. Used for data visualization or computation."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


# --- JSON Tools ---

@tool
def write_json_document(
    content: Annotated[str, "JSON string content to be written."],
    file_name: Annotated[str, "File name to save the JSON, e.g. 'data.json'."],
) -> Annotated[str, "Path of the saved JSON document."]:
    """Save a valid JSON string to file."""
    file_path = WORKING_DIRECTORY / "user_progress" / file_name
    if not file_name.endswith(".json"):
        file_path = file_path.with_suffix(".json")
    try:
        parsed = json.loads(content)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(parsed, file, indent=2)
        return str(file_path)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {repr(e)}"


@tool
def read_json_document(
    file_name: Annotated[str, "File name of the JSON document to read."],
) -> str:
    """Read a JSON file and return pretty-printed content."""
    file_path = WORKING_DIRECTORY / "user_progress" / file_name
    if not file_name.endswith(".json"):
        file_path = file_path.with_suffix(".json")
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Failed to read JSON: {repr(e)}"


# --- File Parsing Tool ---

@tool
def parse_progress_data(
    file_paths: Annotated[List[str], "List of paths to .md, .pdf, or .json files"]
) -> dict:
    """
    Parses the content of given file paths (.md, .pdf, .json) and returns a truncated preview (first 2000 characters) for each.
    If format is unsupported or error occurs during parsing, returns the error message.
    """
    results = {}
    for path in file_paths:
        try:
            if path.endswith(".json"):
                with open(path, "r") as f:
                    data = json.load(f)
                results[path] = json.dumps(data, indent=2)[:2000]

            elif path.endswith(".md"):
                with open(path, "r") as f:
                    results[path] = f.read()[:2000]

            elif path.endswith(".pdf"):
                reader = PdfReader(path)
                text = "".join([page.extract_text() or "" for page in reader.pages])
                results[path] = text[:2000]

            else:
                results[path] = f"Unsupported format: {path}"
        except Exception as e:
            results[path] = f"Error parsing {path}: {repr(e)}"
    return results


def parse_pdf(file) -> str:
    """Extract text from a file-like PDF stream."""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text.append(page.get_text())
    pdf_document.close()
    return "\n".join(text)
