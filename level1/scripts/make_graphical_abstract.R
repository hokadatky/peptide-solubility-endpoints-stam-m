suppressPackageStartupMessages({
  library(grid)
  library(ragg)
})

# Submission PNG is the primary deliverable. Design at its actual 525-pixel
# width; physical print sizes alone cannot guarantee legibility in that file.
# Author-requested repair, 25 October 2026: retain the three-stage narrative,
# enlarge the labels, and replace the miniature app score with a clear diagram.
# All numerical values below are reported in the frozen revised manuscript.
# No fitting, recalculation, or changed scientific claim is involved.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Run this file with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
package_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)

W_PX <- 525
H_PX <- 350
W_MM <- 180
H_MM <- W_MM * H_PX / W_PX
MIN_TEXT_PX <- 12
ff <- "Arial"
navy <- "#103A50"
ink <- "#16384A"
muted <- "#425B69"
blue <- "#0072B2"
blue_pale <- "#EAF3F9"
orange <- "#A34A00"
amber_pale <- "#FFF3DF"
rule <- "#C9D7DF"
white <- "#FFFFFF"

# Logical coordinates are pixels in the final PNG, measured from top left.
# Both the vector and raster devices use the same physical canvas and layout.
# Each source point size is derived from its intended final pixel size.
pt <- function(px) px * (W_MM / 25.4 * 72) / W_PX
ux <- function(x) unit(x / W_PX, "npc")
uy <- function(y) unit(1 - y / H_PX, "npc")
rect <- function(x, y, w, h, fill, border = NA, radius = 0) {
  args <- list(x = ux(x + w / 2), y = uy(y + h / 2),
               width = unit(w / W_PX, "npc"), height = unit(h / H_PX, "npc"),
               gp = gpar(fill = fill, col = border, lwd = 0.8))
  if (radius > 0) {
    do.call(grid.roundrect, c(args, list(r = unit(pt(radius), "pt"))))
  } else do.call(grid.rect, args)
}
line <- function(x, y, col = ink, lwd = 1, end_arrow = FALSE) {
  grid.lines(ux(x), uy(y), gp = gpar(col = col, fill = col, lwd = lwd),
             arrow = if (end_arrow) arrow(type = "closed", length = unit(pt(4), "pt")) else NULL)
}
text_bounds <- list()
txt <- function(label, x, y, size = 12, col = ink, face = "plain",
                width = NULL, height = NULL, just = "centre") {
  stopifnot(size >= MIN_TEXT_PX)
  g <- textGrob(label, x = ux(x), y = uy(y), just = just,
                gp = gpar(fontfamily = ff, fontsize = pt(size), col = col,
                          fontface = face, lineheight = 1.18))
  actual_w <- convertWidth(grobWidth(g), "npc", valueOnly = TRUE) * W_PX
  actual_h <- convertHeight(grobHeight(g), "npc", valueOnly = TRUE) * H_PX
  if (!is.null(width) && actual_w > width + 0.1)
    stop(sprintf("Label exceeds its box: %s (%.1f > %.1f px)", label, actual_w, width))
  if (!is.null(height) && actual_h > height + 0.1)
    stop(sprintf("Label exceeds its height: %s (%.1f > %.1f px)", label, actual_h, height))
  text_bounds[[length(text_bounds) + 1]] <<- data.frame(
    label = gsub("\n", " / ", label), font_px = size,
    x = x, y = y, width_px = actual_w, height_px = actual_h)
  grid.draw(g)
}
card <- function(x, step, title, tint, accent) {
  rect(x, 69, 157, 237, white, rule, 5)
  rect(x + 1, 70, 155, 34, tint)
  line(c(x + 1, x + 156), c(104, 104), accent)
  txt(paste(step, title), x + 78.5, 87, 14, accent, "bold", width = 140)
}

draw_ga <- function() {
  text_bounds <<- list()
  grid.newpage()
  grid.rect(gp = gpar(fill = white, col = NA))
  rect(0, 0, W_PX, 58, navy)
  txt("Solubility-related prediction", 262.5, 20, 20, white,
      "bold", width = 501)
  txt("An endpoint-aware protein language model workflow",
      262.5, 43, 12, white, width = 501)

  card(12, "1", "DEFINE", blue_pale, blue)
  txt("Protein soluble\nexpression", 90.5, 130, 15, ink, "bold",
      width = 141, height = 40)
  txt("E. coli", 90.5, 164, 13, muted, "italic", width = 141)
  # Schematic expression vial; it does not encode a concentration or result.
  rect(79, 183, 23, 9, blue, radius = 2)
  rect(73, 194, 35, 53, blue_pale, blue, 5)
  rect(76, 221, 29, 21, "#90C6DF", radius = 2)
  grid.circle(ux(c(82, 99, 89)), uy(c(228, 233, 239)),
              r = unit(pt(2), "pt"), gp = gpar(fill = blue, col = NA))
  txt("Distinct from aqueous\npeptide solubility", 90.5, 280, 12, ink,
      width = 141, height = 33)

  card(184, "2", "TEST", blue_pale, blue)
  txt("ROC-AUC", 262.5, 121, 13, ink, "bold", width = 141)
  txt("B1  \u2192  B1 + ProtT5", 262.5, 143, 12, muted, width = 141)
  txt("Within development", 262.5, 171, 12, muted, width = 141)
  txt("0.715 \u2192 0.787", 262.5, 194, 18, blue, "bold", width = 141)
  line(c(199, 326), c(214, 214), rule)
  txt("External NESG", 262.5, 232, 12, muted, width = 141)
  txt("0.537 \u2192 0.540", 262.5, 255, 18, ink, "bold", width = 141)
  txt("Gain not confirmed", 262.5, 286, 12, orange, "bold", width = 141)

  card(356, "3", "APPLY", amber_pale, orange)
  txt("PepSol Explorer", 434.5, 126, 15, ink, "bold", width = 141)
  # Local interface diagram, deliberately without a tiny synthetic score.
  rect(371, 147, 127, 72, navy, radius = 5)
  rect(376, 152, 117, 60, blue_pale, radius = 2)
  txt("B0 / B1 / ProtT5", 434.5, 170, 12, ink, "bold", width = 111)
  txt("Check domain", 434.5, 195, 12, ink, width = 111)
  line(c(367, 502), c(224, 224), navy, 2.2)
  txt("Private evaluation", 434.5, 246, 12, muted, width = 141)
  txt("No distilled student", 434.5, 269, 12, ink, width = 141)
  txt("Distillation exploratory", 434.5, 290, 12, orange,
      width = 141)

  line(c(172, 181), c(200, 200), muted, 1.4, TRUE)
  line(c(344, 353), c(200, 200), muted, 1.4, TRUE)
  txt("Development gain +0.072 (95% CI 0.065 to 0.079)",
      262.5, 331, 12, muted, width = 501)
}

# Editable, scalable original for production and local review.
grDevices::cairo_pdf(file.path(package_dir, "GraphicalAbstract1_vector.pdf"),
                     width = W_MM / 25.4, height = H_MM / 25.4,
                     family = ff, bg = white)
draw_ga()
dev.off()

# Journal-format upload: no final text smaller than 12 native image pixels.
ragg::agg_png(file.path(package_dir, "GraphicalAbstract1.png"),
              width = W_PX, height = H_PX, units = "px", res = 300,
              background = white, scaling = W_PX / (W_MM / 25.4 * 300))
draw_ga()
dev.off()

# Larger direct render for zooming/production, not an upscaled 525-pixel image.
ragg::agg_png(file.path(package_dir, "GraphicalAbstract1_highres.png"),
              width = W_PX * 4, height = H_PX * 4, units = "px", res = 300,
              background = white, scaling = W_PX * 4 / (W_MM / 25.4 * 300))
draw_ga()
dev.off()

message("Created 525 x 350 px submission PNG, 2100 x 1400 px direct render, and vector PDF")
