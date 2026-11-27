# R figure environment

The archived figures were generated with:

- R 4.5.2
- ggplot2 4.0.1
- ggrepel 0.9.7
- jsonlite 2.0.0
- patchwork 1.3.2
- ragg 1.5.0
- grid 4.5.2 (base/recommended R component)

PDF generation uses the Cairo graphics backend and therefore requires an R
build with Cairo/Pango support. Times New Roman must be installed and visible
to both R/Cairo and XeLaTeX.

The figure scripts produce six main and one supplementary PDF/TIFF pair. TIFF output is 600 dpi
or higher with LZW compression. `scripts/make_interface_figures_s2.R` renders the implemented
PepSol Explorer interface from privacy-safe synthetic evidence.
`scripts/make_graphical_abstract.R` produces the
525-pixel graphical abstract and its vector cover-letter version. Matplotlib is
not used.
