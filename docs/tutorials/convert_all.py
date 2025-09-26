import os
import jupytext


def convert_md_to_ipynb(tutorials_folder, colab_base_url=None):
    # Iterate through all files in the tutorials folder
    for file_name in os.listdir(tutorials_folder):
        if file_name.endswith(".md"):
            md_path = os.path.join(tutorials_folder, file_name)
            ipynb_path = os.path.join(tutorials_folder, 'notebooks', file_name.replace(".md", ".ipynb"))

            # Convert .md to .ipynb
            markdown = jupytext.read(md_path)
            jupytext.write(markdown, ipynb_path)
            print(f"Converted: {md_path} -> {ipynb_path}")

            # Optionally generate "Open in Colab" link
            if colab_base_url:
                colab_link = f"{colab_base_url}/{ipynb_path.replace(os.sep, '/')}"
                print(f"Open in Colab: {colab_link}")


# Example usage
tutorials_folder = "."
colab_base_url = "https://colab.research.google.com/github/exampleuser/myrepo/blob/main"  # Replace with your repo URL
convert_md_to_ipynb(tutorials_folder, colab_base_url)