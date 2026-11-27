suppressPackageStartupMessages({
  library(grid)
  library(jsonlite)
  library(ragg)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Run this file with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
package_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
input_file <- file.path(package_dir, "analysis_addon", "results", "pepsol_explorer_s2_demo.json")
out_dir <- file.path(package_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

payload <- jsonlite::fromJSON(input_file)
examples <- payload$examples
protein <- examples[examples$id == "protein_like_seed_20260813", , drop = FALSE]
short <- examples[examples$id == "short_synthetic", , drop = FALSE]
ph <- payload$ph_profile
auc <- payload$internal_group_controlled_oof_metrics

font_family <- "Arial"
ink <- "#102A3D"
muted <- "#587084"
line <- "#DCE6EC"
canvas <- "#F4F8FA"
white <- "#FFFFFF"
blue <- "#176B91"
teal <- "#168C7A"
steel <- "#71808C"
amber <- "#B66B18"
red <- "#B74646"
pale_blue <- "#EDF6F8"
pale_teal <- "#F1FAF7"
pale_amber <- "#FFF9EF"
pale_red <- "#FDEAEA"

box <- function(x, y, w, h, fill = white, col = line, lwd = 0.8, radius = 0.012) {
  grid.roundrect(
    x = unit(x + w / 2, "npc"), y = unit(y + h / 2, "npc"),
    width = unit(w, "npc"), height = unit(h, "npc"),
    r = unit(radius, "npc"), gp = gpar(fill = fill, col = col, lwd = lwd)
  )
}

## Legibility floor. These figures are authored 179.9 mm wide and placed at
## \textwidth (172 mm), a 0.956 scale, so the floor must be set in AUTHORED points
## such that authored * 0.956 >= 7.0 pt on the page. 7.0 authored would render at
## 6.7 pt and fail the journal minimum.
PLACEMENT_SCALE <- 172 / 179.9
MIN_LABEL_PT <- ceiling(7.0 / PLACEMENT_SCALE * 10) / 10   # 7.4
MAIN_LABEL_PT <- MIN_LABEL_PT + 0.4

txt <- function(label, x, y, size = 8, col = ink, face = "plain",
                just = c("left", "centre"), lineheight = 0.95, rot = 0) {
  size <- max(size, MIN_LABEL_PT)
  grid.text(
    label, x = unit(x, "npc"), y = unit(y, "npc"), just = just, rot = rot,
    gp = gpar(fontfamily = font_family, fontsize = size, col = col,
              fontface = face, lineheight = lineheight)
  )
}

main_txt <- function(label, x, y, size = 8, col = ink, face = "plain",
                     just = c("left", "centre"), lineheight = 0.95, rot = 0) {
  txt(label, x, y, size = max(size, MAIN_LABEL_PT), col = col, face = face,
      just = just, lineheight = lineheight, rot = rot)
}

line_seg <- function(x0, y0, x1, y1, col = line, lwd = 0.8, arrow = FALSE) {
  arrow_spec <- if (arrow) grid::arrow(type = "closed", length = unit(1.5, "mm")) else NULL
  grid.segments(unit(x0, "npc"), unit(y0, "npc"), unit(x1, "npc"), unit(y1, "npc"),
                arrow = arrow_spec, gp = gpar(col = col, lwd = lwd))
}

pill <- function(label, x, y, w, fill, col, size = 6.7) {
  box(x, y, w, 0.042, fill = fill, col = fill, radius = 0.021)
  txt(label, x + w / 2, y + 0.021, size = size, col = col, face = "bold",
      just = "centre")
}

draw_mark <- function(cx, cy, scale = 1) {
  theta <- seq(0, 2 * pi, length.out = 80)
  x <- cx + scale * 0.014 * sin(theta) * (0.85 + 0.15 * cos(theta))
  y <- cy - scale * 0.004 + scale * 0.018 * cos(theta)
  grid.polygon(unit(c(cx, x), "npc"), unit(c(cy + scale * 0.026, y), "npc"),
               gp = gpar(fill = blue, col = blue))
  line_seg(cx - scale * 0.008, cy - scale * 0.003,
           cx + scale * 0.008, cy - scale * 0.003, col = white, lwd = 1.5)
  grid.points(unit(c(cx - scale * 0.008, cx, cx + scale * 0.008), "npc"),
              unit(rep(cy - scale * 0.003, 3), "npc"), pch = 21,
              size = unit(scale * 0.012, "npc"),
              gp = gpar(fill = white, col = white))
}

panel_label <- function(letter, x, y) {
  txt(paste0("(", letter, ")"), x, y, size = 8.2, face = "bold", just = c("left", "top"))
}

metric_tile <- function(x, y, w, title, value, note = NULL) {
  box(x, y, w, 0.078, fill = white, col = line, radius = 0.009)
  txt(title, x + 0.009, y + 0.058, size = 5.2, col = muted, face = "bold",
      lineheight = 0.88)
  txt(value, x + 0.009, y + 0.021, size = 8.1, col = ink, face = "bold")
  if (!is.null(note)) txt(note, x + w - 0.008, y + 0.029, size = 5.4, col = muted,
                          just = c("right", "centre"))
}

draw_score_bars <- function(x, y, w, h, values, labels, colors, title = NULL) {
  if (!is.null(title)) txt(title, x, y + h + 0.022, size = 6.5, face = "bold")
  n <- length(values)
  gap <- h / n
  for (i in seq_len(n)) {
    yy <- y + h - (i - 0.5) * gap
    txt(labels[i], x, yy, size = 5.9, col = muted)
    bar_x <- x + 0.105
    bar_w <- w - 0.145
    box(bar_x, yy - 0.013, bar_w, 0.026, fill = "#E8EFF3", col = "#E8EFF3", radius = 0.013)
    box(bar_x, yy - 0.013, bar_w * values[i] / 100, 0.026,
        fill = colors[i], col = colors[i], radius = 0.013)
    txt(sprintf("%.1f", values[i]), x + w, yy, size = 6.4, col = colors[i],
        face = "bold", just = c("right", "centre"))
  }
}

source(file.path(package_dir, "scripts", "draw_project_translation_s25.R"))

draw_ph_chart <- function(x, y, w, h) {
  box(x, y, w, h, fill = white, col = line, radius = 0.012)
  txt("Estimated net charge across pH", x + 0.018, y + h - 0.035, size = 7.1, face = "bold")
  plot_x0 <- x + 0.055; plot_x1 <- x + w - 0.028
  plot_y0 <- y + 0.055; plot_y1 <- y + h - 0.080
  x_map <- function(v) plot_x0 + (v - 3) / 8 * (plot_x1 - plot_x0)
  y_map <- function(v) plot_y0 + (v + 25) / 90 * (plot_y1 - plot_y0)
  for (tick in c(-20, 0, 20, 40, 60)) {
    line_seg(plot_x0, y_map(tick), plot_x1, y_map(tick), col = "#E8EFF3", lwd = 0.55)
    txt(as.character(tick), plot_x0 - 0.010, y_map(tick), size = 5.0, col = muted,
        just = c("right", "centre"))
  }
  for (tick in c(3, 5, 7, 9, 11)) {
    txt(as.character(tick), x_map(tick), plot_y0 - 0.020, size = 5.0, col = muted, just = "centre")
  }
  line_seg(plot_x0, plot_y0, plot_x1, plot_y0, col = muted, lwd = 0.7)
  line_seg(plot_x0, plot_y0, plot_x0, plot_y1, col = muted, lwd = 0.7)
  for (identifier in unique(ph$id)) {
    rows <- ph[ph$id == identifier, ]
    color <- if (identifier == "short_synthetic") blue else teal
    grid.lines(unit(x_map(rows$pH), "npc"), unit(y_map(rows$net_charge), "npc"),
               gp = gpar(col = color, lwd = 1.6))
    grid.points(unit(x_map(rows$pH), "npc"), unit(y_map(rows$net_charge), "npc"),
                pch = 21, size = unit(0.010, "npc"), gp = gpar(fill = white, col = color, lwd = 1))
  }
  # The legend sits below the tick labels, not across them.
  line_seg(x + 0.020, y + 0.008, x + 0.040, y + 0.008, col = blue, lwd = 1.6)
  txt("short synthetic", x + 0.047, y + 0.008, size = 5.1, col = blue)
  line_seg(x + 0.155, y + 0.008, x + 0.175, y + 0.008, col = teal, lwd = 1.6)
  txt("protein-like synthetic", x + 0.182, y + 0.008, size = 5.1, col = teal)
}

draw_auc_chart <- function(x, y, w, h) {
  box(x, y, w, h, fill = white, col = line, radius = 0.012)
  txt("Methods & limits: internal group-controlled OOF", x + 0.018, y + h - 0.035,
      size = 7.1, face = "bold")
  base_y <- y + 0.105
  max_y <- y + h - 0.085
  labels <- c("B0", "B1", "B1 + ProtT5")
  values <- auc$roc_auc
  colors <- c(blue, teal, "#E2A444")
  xs <- seq(x + 0.075, x + w - 0.075, length.out = 3)
  for (tick in c(0.5, 0.6, 0.7, 0.8)) {
    yy <- base_y + (tick - 0.5) / 0.34 * (max_y - base_y)
    line_seg(x + 0.045, yy, x + w - 0.025, yy, col = "#E8EFF3", lwd = 0.55)
    txt(sprintf("%.1f", tick), x + 0.035, yy, size = 5.0, col = muted, just = c("right", "centre"))
  }
  for (i in seq_along(values)) {
    bh <- (values[i] - 0.5) / 0.34 * (max_y - base_y)
    grid.rect(unit(xs[i], "npc"), unit(base_y + bh / 2, "npc"),
              width = unit(0.060, "npc"), height = unit(bh, "npc"),
              gp = gpar(fill = colors[i], col = colors[i]))
    txt(sprintf("%.3f", values[i]), xs[i], base_y + bh + 0.018, size = 5.8,
        col = colors[i], face = "bold", just = "centre")
    txt(labels[i], xs[i], base_y - 0.025, size = 5.1, col = muted, just = "centre")
  }
  box(x + 0.025, y + 0.010, w - 0.050, 0.040, fill = pale_amber, col = pale_amber, radius = 0.008)
  txt("Internal discrimination only\nNo external validation of deployment artifact",
      x + w / 2, y + 0.031, size = 5.2, col = amber, face = "bold",
      just = "centre", lineheight = 0.90)
}

draw_supplement <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = white, col = NA))
  txt("PepSol Explorer interface contract: inspect, compare, export, and stop when unsupported",
      0.025, 0.975, size = 10.0, face = "bold", just = c("left", "top"))
  txt("Publication rendering of implemented tabs using the privacy-safe synthetic S2 evidence record",
      0.025, 0.940, size = 6.8, col = muted, just = c("left", "top"))

  panel_label("a", 0.025, 0.900)
  draw_ph_chart(0.040, 0.535, 0.440, 0.355)
  panel_label("b", 0.515, 0.900)
  draw_auc_chart(0.530, 0.535, 0.440, 0.355)

  panel_label("c", 0.025, 0.480)
  box(0.040, 0.115, 0.440, 0.355, fill = canvas, col = line, radius = 0.012)
  txt("Batch & export", 0.058, 0.438, size = 7.2, face = "bold")
  columns <- c("ID", "length", "B0", "B1", "ProtT5", "AD")
  col_x <- c(0.060, 0.191, 0.246, 0.295, 0.345, 0.408)
  for (i in seq_along(columns)) txt(columns[i], col_x[i], 0.397, size = 5.2, col = muted, face = "bold")
  line_seg(0.058, 0.380, 0.462, 0.380, col = line, lwd = 0.8)
  row1 <- c("protein_like...", "300", "80.1", "91.3", "94.7", "IN")
  row2 <- c("short_synthetic", "16", "0.0", "100.0", "100.0", "OOD")
  for (i in seq_along(row1)) txt(row1[i], col_x[i], 0.350, size = 5.1,
                                  col = if (i == 6) teal else ink,
                                  face = if (i == 6) "bold" else "plain")
  for (i in seq_along(row2)) txt(row2[i], col_x[i], 0.316, size = 5.1,
                                  col = if (i == 6) red else muted,
                                  face = if (i == 6) "bold" else "plain")
  box(0.058, 0.245, 0.124, 0.044, fill = blue, col = blue, radius = 0.007)
  txt("Scores: CSV", 0.120, 0.267, size = 5.2, col = white, face = "bold", just = "centre")
  box(0.190, 0.244, 0.124, 0.044, fill = teal, col = teal, radius = 0.007)
  txt("pH profile: CSV", 0.252, 0.267, size = 5.2, col = white, face = "bold", just = "centre")
  box(0.322, 0.245, 0.140, 0.044, fill = "#E2A444", col = "#E2A444", radius = 0.007)
  txt("Report: JSON", 0.392, 0.267, size = 5.2, col = white, face = "bold", just = "centre")
  box(0.058, 0.136, 0.404, 0.086, fill = pale_teal, col = "#D7E7E3", radius = 0.009)
  txt("JSON carries app version, model hash\nand run metadata.", 0.072, 0.198,
      size = 5.5, col = teal, face = "bold", just = c("left", "top"), lineheight = 0.90)
  txt("OOD rows stay auditable, not quantitative decisions.", 0.072, 0.153,
      size = 5.1, col = muted)

  panel_label("d", 0.515, 0.480)
  box(0.530, 0.115, 0.440, 0.355, fill = canvas, col = line, radius = 0.012)
  txt("Scientific boundaries and release integrity", 0.548, 0.438, size = 7.2, face = "bold")
  boundaries <- c(
    "Aggregation proxy is not an experimental concentration",
    "B0/B1/ProtT5 refer only to the E. coli expression endpoint",
    "0-100 values are rankings, not calibrated probabilities",
    "Modified/non-canonical chemistry fails closed",
    "External validation of the artifact remains future work"
  )
  for (i in seq_along(boundaries)) {
    pill_y <- 0.397 - (i - 1) * 0.043
    grid.points(unit(0.555, "npc"), unit(pill_y, "npc"), pch = 21,
                size = unit(0.011, "npc"), gp = gpar(fill = amber, col = amber))
    txt(boundaries[i], 0.568, pill_y, size = 5.4, col = ink)
  }
  box(0.548, 0.112, 0.404, 0.090, fill = white, col = line, radius = 0.009)
  txt("SOFTWARE QA", 0.562, 0.182, size = 5.5, col = blue, face = "bold")
  txt("12 tests | parity 1e-12 | hash/privacy gates\nClean core installation",
      0.562, 0.154, size = 5.1, col = muted, just = c("left", "top"), lineheight = 0.90)
  txt("These gates verify delivery integrity, not scientific external generalization.",
      0.500, 0.060, size = 6.0, col = red, face = "bold", just = "centre")
}

main_pdf <- file.path(out_dir, "Okada_Fig6.pdf")
main_tiff <- file.path(out_dir, "Okada_Fig6.tiff")
main_png <- file.path(out_dir, "Okada_Fig6.png")
supp_pdf <- file.path(out_dir, "Okada_FigS1.pdf")
supp_tiff <- file.path(out_dir, "Okada_FigS1.tiff")

grDevices::cairo_pdf(main_pdf, width = 82 / 25.4, height = 150 / 25.4,
                     family = font_family, bg = white)
draw_main()
dev.off()
ragg::agg_tiff(main_tiff, width = 82, height = 150, units = "mm", res = 600,
               compression = "lzw", background = white, scaling = 1)
draw_main()
dev.off()
ragg::agg_png(main_png, width = 82, height = 150, units = "mm", res = 300,
              background = white, scaling = 1)
draw_main()
dev.off()

message("Created ", main_pdf, ", ", main_tiff, " and ", main_png)
