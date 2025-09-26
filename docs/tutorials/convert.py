import jupytext

markdown = jupytext.read("data_preparation.md")
jupytext.write(markdown, "output.ipynb")