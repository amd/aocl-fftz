# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# ##############################################################################

# Configuration file for the Sphinx documentation builder.

import os
# -- Project information -----------------------------------------------------
project = 'AOCL-FFTZ'
copyright = '2026, Advanced Micro Devices, Inc'
author = 'Advanced Micro Devices, Inc'
version = '5.2.2'
release = '5.2.2'

extensions = ['breathe', 'myst_parser']
pwd = os.path.dirname(os.path.abspath(__file__))
pwd = os.path.join(pwd, '..')
pwd = os.path.join(pwd,'xml')
breathe_projects = {"fftz": pwd}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_template']

# tells the myst_parser to generate labels for heading anchors for h1, h2, and h3 level headings (corresponding to #, ##, and ### in markdown)
myst_heading_anchors = 3

# Exclude orphaned files that aren't part of any toctree
exclude_patterns = [
    'fftz_readme.rst'
]

# Override toctree section numbers so a branch can display as 1, 1.1, 1.2, ...
# (docname -> number string). Adjust or set to {} to use Sphinx default numbering.
toc_secnumber_overrides = {
    "data_structures_index": "1",
    "typedefs": "1.1",
    "problem_descriptor": "1.2",
    "data_layout_conventions": "1.3",
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'rocm_docs_theme'
html_theme_options = {
    "link_main_doc": False,
    "flavor": "local",
    "repository_provider" : None,
}