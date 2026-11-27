#!/usr/bin/env Rscript
## Figure 1: overview of the review, drawn as a flow chart.
##
## S28. Figure 1 has been through three forms. A six-step gate diagram whose text
## contained exactly one domain word, so it read as a template any prediction
## paper could reuse; a banded overview that carried the article's real topics but
## was a layout, not a flow; and this, which keeps the topics and draws them as
## the route the review actually took. Two branches leave one declaration, and
## each terminates where the evidence stopped it.
##
## Named tools and internal feature-block labels stay out; the domain terms a
## first-time reader needs are in.
##
## Authored at 180 mm and placed at \textwidth (172 mm), scale 0.956.

suppressPackageStartupMessages({library(grid); library(ragg)})

out_dir <- file.path(dirname(dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE)[1])))), "figures")
if (!dir.exists(out_dir)) out_dir <- "figures"

ff <- "Arial"
ink <- "#20252B"; muted <- "#5A6570"; hair <- "#C8D0D6"
stop_col <- "#D55E00"; go_col <- "#0072B2"
PLACEMENT_SCALE <- 172 / 180
HEAD_PT <- 9.4; BODY_PT <- 8.6

box <- function(cx, cy, w, h, head, body = NULL, col = ink, fill = "#F5F7F8",
                border = hair, lwd = 0.9) {
  grid.roundrect(x = unit(cx, "npc"), y = unit(cy, "npc"),
                 width = unit(w, "npc"), height = unit(h, "npc"),
                 r = unit(1.6, "pt"), gp = gpar(fill = fill, col = border, lwd = lwd))
  if (is.null(body)) {
    grid.text(head, x = unit(cx, "npc"), y = unit(cy, "npc"),
              gp = gpar(fontfamily = ff, fontsize = HEAD_PT, fontface = "bold",
                        col = col, lineheight = 1.10))
  } else {
    grid.text(head, x = unit(cx, "npc"), y = unit(cy + h * 0.20, "npc"),
              gp = gpar(fontfamily = ff, fontsize = HEAD_PT, fontface = "bold",
                        col = col, lineheight = 1.10))
    grid.text(body, x = unit(cx, "npc"), y = unit(cy - h * 0.21, "npc"),
              gp = gpar(fontfamily = ff, fontsize = BODY_PT, col = muted,
                        lineheight = 1.12))
  }
}
down <- function(x, y0, y1, col = muted) {
  grid.lines(x = unit(c(x, x), "npc"), y = unit(c(y0, y1), "npc"),
             arrow = arrow(length = unit(3.8, "pt"), type = "closed"),
             gp = gpar(col = col, lwd = 1.05, fill = col))
}
elbow <- function(x0, y0, x1, y1, col = muted) {
  grid.lines(x = unit(c(x0, x0, x1), "npc"), y = unit(c(y0, y1, y1), "npc"),
             gp = gpar(col = col, lwd = 1.05))
  grid.lines(x = unit(c(x1 - sign(x1 - x0) * 0.001, x1), "npc"), y = unit(c(y1, y1), "npc"),
             arrow = arrow(length = unit(3.8, "pt"), type = "closed"),
             gp = gpar(col = col, lwd = 1.05, fill = col))
}

LX <- 0.245; RX <- 0.755; CX <- 0.500
BW <- 0.435

draw_figure <- function() {
  grid.newpage(); grid.rect(gp = gpar(fill = "white", col = NA))

  ## 1. the problem
  box(CX, 0.930, 0.960, 0.118,
      "One word, several different experiments",
      paste("aqueous concentration  ·  cell-free soluble fraction  ·  recombinant expression",
            "production status  ·  LC-MS observability", sep = "\n"))
  grid.text("labeled \"solubility\", but not interchangeable",
            x = unit(CX, "npc"), y = unit(0.855, "npc"),
            gp = gpar(fontfamily = ff, fontsize = BODY_PT, col = muted))
  down(CX, 0.840, 0.812)

  ## 2. the declaration both routes start from
  box(CX, 0.775, 0.520, 0.062, "Declare which experiment the label means",
      col = ink, fill = "white", border = ink, lwd = 1.1)

  ## split
  grid.lines(x = unit(c(LX, RX), "npc"), y = unit(c(0.720, 0.720), "npc"),
             gp = gpar(col = muted, lwd = 1.05))
  grid.lines(x = unit(c(CX, CX), "npc"), y = unit(c(0.744, 0.720), "npc"),
             gp = gpar(col = muted, lwd = 1.05))
  down(LX, 0.720, 0.688); down(RX, 0.720, 0.688)

  ## 3. the two bodies of evidence
  box(LX, 0.632, BW, 0.088, "Public direct-peptide measurements",
      "three sources, tens to a few hundred records each")
  box(RX, 0.632, BW, 0.088, "One separately declared endpoint",
      "recombinant soluble expression, ~11,000 records")
  down(LX, 0.588, 0.556); down(RX, 0.588, 0.556)

  ## 4. what each route did next
  box(LX, 0.500, BW, 0.088, "Audit comparability",
      "assay, scaffold and chemistry, entity by entity")
  box(RX, 0.500, BW, 0.088, "Freeze the split before fitting",
      "grouped by sequence identity, then locked")
  down(LX, 0.456, 0.424); down(RX, 0.456, 0.424)

  ## 5. left terminates; right compares
  box(LX, 0.368, BW, 0.088, "no pooled benchmark",
      "sources reported separately, not merged",
      col = stop_col, fill = "#FBECE6", border = stop_col, lwd = 1.1)
  box(RX, 0.368, BW, 0.088, "Compare against a transparent baseline",
      "paired effect, calibration, applicability")
  down(RX, 0.324, 0.292)

  ## 6. the bounded result
  box(RX, 0.236, BW, 0.088, "a measured gain, bounded to that endpoint",
      "not aqueous solubility, not external validation",
      col = go_col, fill = "#EAF3F8", border = go_col, lwd = 1.1)

  ## 6b. exploratory compression: retained gain, deployment gates unmet
  elbow(RX - BW / 2 - 0.012, 0.236, LX + BW / 2 + 0.012, 0.236)
  box(LX, 0.236, BW, 0.088, "Retain the gain in a smaller model",
      "70% of teacher gain retained",
      col = go_col, fill = "#EAF3F8", border = go_col, lwd = 1.1)

  down(RX, 0.192, 0.160)

  ## 7. what reaches the reader
  box(RX, 0.104, BW, 0.088, "Local screening tool",
      "reports the endpoint and its applicability limit",
      col = ink, fill = "white", border = ink, lwd = 1.1)
  grid.text("Endpoint evidence: Figures 4–5\nPepSol Explorer (Figure 3) and distillation (Figure 6)",
            x = unit(LX, "npc"), y = unit(0.104, "npc"),
            gp = gpar(fontfamily = ff, fontsize = BODY_PT, col = muted,
                      lineheight = 1.25))
}

W <- 180; H <- 138
pdf_file  <- file.path(out_dir, "Okada_Fig1.pdf")
tiff_file <- file.path(out_dir, "Okada_Fig1.tiff")
cairo_pdf(pdf_file, width = W / 25.4, height = H / 25.4, family = ff, bg = "white")
draw_figure(); dev.off()
ragg::agg_tiff(tiff_file, width = W, height = H, units = "mm", res = 600,
               compression = "lzw", background = "white", scaling = 1)
draw_figure(); dev.off()
message("wrote Okada_Fig1 (review flow chart, ", W, " x ", H, " mm)")
