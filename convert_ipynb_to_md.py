import os
import nbformat
from nbconvert import MarkdownExporter

def clean_output_cells(notebook):
    """
    Clean output cells from the notebook to remove execution results,
    keeping only the source code and markdown content.
    """
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            # Clear all outputs (stdout, stderr, display_data, etc.)
            cell.outputs = []
            # Clear execution count
            cell.execution_count = None
    return notebook

def convert_ipynb_to_md(notebook_path, output_folder):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Clean output cells before conversion
    notebook = clean_output_cells(notebook)

    # Extract title from the first Markdown cell
    title = None
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            for line in cell.source.splitlines():
                if line.startswith('# '):  # First header
                    title = line.lstrip('# ').strip()
                    break
            if title:
                break

    if not title:
        raise ValueError("No title found in the notebook.")

    # Determine nav_order from filename
    filename = os.path.basename(notebook_path)
    prefix = filename.split('_')[0]
    try:
        nav_order = int(prefix) + 1
    except ValueError:
        raise ValueError("Filename does not start with a valid number prefix.")

    # Create YAML header
    yaml_header = f"""---
layout: default
title: {title}
parent: Tutorials
nav_order: {nav_order}
---
"""

    # Convert notebook to Markdown
    markdown_exporter = MarkdownExporter()
    body, _ = markdown_exporter.from_notebook_node(notebook)

    # Combine YAML header and Markdown content
    markdown_content = yaml_header + "\n" + body

    # Save to .md file
    output_path = os.path.join(output_folder, filename.replace('.ipynb', '.md'))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"Converted: {notebook_path} -> {output_path}")

notebook_folder = "tutorials"
output_folder = "docs/tutorials"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(notebook_folder):
    if file_name.endswith(".ipynb"):
        print(f'Converting {file_name}')
        convert_ipynb_to_md(os.path.join(notebook_folder, file_name), output_folder)