suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(ragg)
  library(grid)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Run this file with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
package_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
result_dir <- file.path(package_dir, "analysis_addon", "results")
analysis_path <- file.path(
  result_dir,
  "stam_s6_released_aggregate_diagnostic_analysis_20260816.json"
)
analysis <- jsonlite::fromJSON(analysis_path, simplifyVector = TRUE)
auc_analysis <- jsonlite::fromJSON(file.path(result_dir, "aggregate_internal_auc_s1.json"), simplifyVector = TRUE)

stopifnot(
  identical(analysis$schema, "stam_s6_released_aggregate_diagnostic_analysis_v1"),
  identical(analysis$status, "completed_no_refit_no_resampling"),
  identical(analysis$privacy$contains_row_level_data, FALSE),
  identical(analysis$privacy$contains_sequences, FALSE),
  identical(analysis$scientific_computation_counts$model_fits, 0L),
  identical(analysis$scientific_computation_counts$external_evaluations, 0L),
  grepl("not direct aqueous solubility", analysis$endpoint, fixed = TRUE)
)

gain <- as.numeric(analysis$primary_contrast$point_gain)
ci95 <- as.numeric(analysis$primary_contrast$ci95)
margin_ref <- as.numeric(analysis$primary_contrast$practical_margin)
stopifnot(
  isTRUE(all.equal(gain, 0.07200193518843745, tolerance = 1e-14)),
  isTRUE(all.equal(ci95, c(0.06497865020719498, 0.07912424631381525), tolerance = 1e-14)),
  isTRUE(all.equal(margin_ref, 0.02, tolerance = 1e-14))
)

font_family <- "Arial"
ink <- "#20252B"
grey <- "#6B7280"
light_grey <- "#D9DEE3"
# Shared semantic palette (also used by make_figure_distillation.R):
# B0/B1 are descriptor/order-aware grey; ESM-2 is blue; ProtT5 is green.
b0_grey <- "#A5ADB6"
b1_grey <- "#5E6873"
esm2_blue <- "#0072B2"
esm2_blue_light <- "#56B4E9"
prott5_green <- "#009E73"
prott5_green_dark <- "#006A52"
blue <- esm2_blue
sky <- esm2_blue_light
orange <- "#E69F00"

theme_s6 <- function(base_size = 8.0) {
  theme_classic(base_size = base_size, base_family = font_family) +
    theme(
      text = element_text(family = font_family, colour = ink),
      axis.text = element_text(size = base_size, colour = ink),
      axis.title = element_text(size = base_size + 0.3, colour = ink),
      axis.line = element_line(linewidth = 0.35, colour = ink),
      axis.ticks = element_line(linewidth = 0.3, colour = ink),
      legend.position = "none",
      plot.tag = element_text(
        family = font_family, face = "bold", size = 9.0,
        colour = ink, hjust = 0, vjust = 1
      ),
      plot.tag.position = c(0, 1),
      plot.margin = margin(8, 7, 5, 10, "pt")
    )
}

# (b) Primary paired contrast and conditional interval.
primary_df <- data.frame(y = 1, gain = gain, lower = ci95[1], upper = ci95[2])
p_a <- ggplot(primary_df, aes(x = gain, y = y)) +
  geom_vline(xintercept = 0, linewidth = 0.45, colour = grey) +
  geom_vline(xintercept = margin_ref, linewidth = 0.5, linetype = "dotted", colour = orange) +
  geom_segment(aes(x = lower, xend = upper, yend = y), linewidth = 1.25, colour = prott5_green) +
  geom_point(shape = 18, size = 4.6, colour = prott5_green) +
  annotate(
    "text", x = 0.066, y = 1.23,
    label = sprintf("+%.3f", gain),
    family = font_family, fontface = "bold", size = 3.75, colour = prott5_green
  ) +
  annotate(
    "text", x = 0.060, y = 0.76,
    label = sprintf("95%% paired CI\n[%.3f, %.3f]", ci95[1], ci95[2]),
    family = font_family, size = 2.65, colour = ink, hjust = 0.5, lineheight = 0.92
  ) +
  annotate(
    "text", x = margin_ref + 0.002, y = 1.45,
    label = "+0.020 plan reference\nnot a utility threshold",
    family = font_family, size = 2.60, colour = orange, hjust = 0, lineheight = 0.92
  ) +
  scale_x_continuous(
    limits = c(-0.005, 0.090),
    breaks = c(0, 0.02, 0.05, 0.08),
    labels = c("0", "+0.020", "+0.050", "+0.080"),
    expand = c(0, 0)
  ) +
  scale_y_continuous(limits = c(0.52, 1.58), breaks = NULL) +
  labs(x = "B1 + ProtT5 − B1 ROC-AUC", y = NULL, tag = "(b)") +
  theme_s6()

# (a) ROC curves are the lead result panel. Keep the legend as a model glossary;
# the plotted AUCs are reported in the adjacent text and need not be repeated here.
fpr <- as.numeric(unlist(auc_analysis$roc$fpr_grid))
roc_names <- c(B0 = "B0", B1 = "B1", `B1 + ProtT5` = "B1_plus_ProtT5")
roc_df <- do.call(rbind, lapply(names(roc_names), function(lbl) {
  key <- c(B0 = "B0_TRANSPARENT_LOGIT", B1 = "B1_ORDER_LOGIT", `B1 + ProtT5` = "B1_ORDER_PLUS_PROTT5_PLS_LOGIT")[[lbl]]
  data.frame(fpr = fpr, tpr = as.numeric(unlist(auc_analysis$roc[[key]])), arm = lbl)
}))
roc_df$arm <- factor(roc_df$arm, levels = names(roc_names))
p_roc <- ggplot(roc_df, aes(fpr, tpr, colour = arm, linetype = arm)) +
  geom_abline(slope = 1, intercept = 0, colour = light_grey, linewidth = 0.45) +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = c(B0 = b0_grey, B1 = b1_grey, `B1 + ProtT5` = prott5_green),
    labels = c("B0 transparent",
               "B1 order-aware",
               "B1 + ProtT5")) +
  scale_linetype_manual(values = c(B0 = "solid", B1 = "dashed", `B1 + ProtT5` = "solid"), guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), expand = c(0, 0)) +
  labs(x = "False-positive rate", y = "True-positive rate", tag = "(a)", colour = NULL) +
  theme_s6() + theme(aspect.ratio = 1,
                     legend.position = c(.98, .03),
                     legend.justification = c(1, 0),
                     legend.direction = "vertical",
                     legend.text = element_text(size = 7.4),
                     legend.title = element_blank(), legend.key.width = unit(10, "pt"),
                     # Pulled up and closed down: the space released here goes to
                     # (b), whose bar differences need vertical extent to read.
                     plot.margin = margin(2, 7, 0, 10, "pt"))

# The former absolute-AUC lollipop and redundant legend numerals were removed;
# the curves remain the direct visual comparison and exact values remain in text.

# S23: panel (c), the secondary within-development diagnostics, was removed at the
# author's request. The five diagnostics are still reported in full -- the body
# text states that PR-AUC, Brier and log loss moved favourably while the
# calibration slope did NOT, and Supplementary Table S3 tabulates all five. The
# anti-cherry-picking guarantee therefore survives in the text and the supplement,
# but it is no longer visible in the figure itself; tests/test_s10 now enforces it
# there instead.

# (d) Exploratory representation-stacking ablation under a DIFFERENT model (HistGBM).
# Vertical bars begin at ROC-AUC 0 so their absolute scale is not exaggerated. These AUCs
# are not comparable to the logistic values in (a); the panel carries its own BASE baseline.
abl <- jsonlite::fromJSON(
  file.path(result_dir, "exploratory_histgbm_plm_ablation_20260805.json"),
  simplifyVector = TRUE
)
stopifnot(identical(abl$schema, "exploratory_histgbm_plm_ablation_v1"),
          isTRUE(abl$no_new_computation))
abl_df <- abl$arms
abl_df$label <- factor(
  abl_df$label,
  levels = c("No pLM (BASE)", "ESM-2", "ProtT5", "ProtT5 + ESM-2")
)
steel <- c("No pLM (BASE)" = b0_grey, "ESM-2" = esm2_blue,
           "ProtT5" = prott5_green, "ProtT5 + ESM-2" = prott5_green_dark)
p_ablation <- ggplot(abl_df, aes(x = label, y = auc, fill = label)) +
  geom_hline(yintercept = 0.50, linewidth = 0.42, linetype = "dotted", colour = grey) +
  geom_col(width = 0.74, colour = "white", linewidth = 0.35) +
  scale_fill_manual(values = steel, guide = "none") +
  scale_x_discrete(
    labels = c("Base", "ESM-2", "ProtT5", "Stacked"),
    expand = expansion(add = 0.42)
  ) +
  # Anchored at chance, not at zero. The meaningful floor for ROC-AUC is 0.50,
  # which this panel already marks with the dotted line; on a 0-1 axis the four
  # arms differ by about a tenth of the axis and are indistinguishable at any
  # panel size. The axis top is the observed maximum plus a small margin, so the
  # flattening from ESM-2 to ProtT5 to the stacked arm is visible, which is the
  # panel's point.
  scale_y_continuous(
    breaks = c(0.50, 0.60, 0.70, 0.80),
    labels = c("0.50\n(chance)", "0.60", "0.70", "0.80"),
    expand = c(0, 0)
  ) +
  labs(x = NULL, y = "Grouped OOF ROC-AUC", tag = "(b)") +
  # Clipped at the panel edge: the columns are drawn from zero, so without
  # clipping they run under the axis and across the arm labels.
  coord_cartesian(ylim = c(0.50, 0.85), clip = "on") +
  theme_s6() +
  theme(
    axis.text.x = element_text(size = 7.4, angle = 25, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 7.4),
    # The tag is pinned above the panel and the left margin widened, so it
    # cannot collide with the 1.00 tick label.
    plot.tag.position = c(0.004, 1.055),
    plot.margin = margin(11, 6, 14, 10, "pt")
  )


# S23: a single column, (a) over (b). patchwork aligns the plotting regions, so
# the two y axes sit on one vertical line even though "True-positive rate" and
# "Grouped OOF ROC-AUC" have different label widths. The heights give (a) the
# room its square aspect needs; (b) keeps enough for the rotated arm labels.
figure <- (p_roc / p_ablation) +
  plot_layout(heights = c(2.1, 1)) +
  plot_annotation(
    theme = theme(
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(4, 5, 3, 5, "pt")
    )
  )

out_dir <- file.path(package_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
pdf_file <- file.path(out_dir, "Okada_Fig4.pdf")
tiff_file <- file.path(out_dir, "Okada_Fig4.tiff")
png_file <- file.path(out_dir, "Okada_Fig4.png")

ggsave(
  pdf_file, figure, width = 120, height = 175, units = "mm",
  device = cairo_pdf, family = font_family, bg = "white", limitsize = FALSE
)
ragg::agg_tiff(
  tiff_file, width = 120, height = 175, units = "mm", res = 600,
  compression = "lzw", background = "white", scaling = 1
)
print(figure)
dev.off()
ragg::agg_png(
  png_file, width = 120, height = 175, units = "mm", res = 300,
  background = "white", scaling = 1
)
print(figure)
dev.off()

message("Created ", pdf_file)
message("Created ", tiff_file)
message("Created ", png_file)
