suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(jsonlite)
  library(patchwork)
  library(ragg)
  library(ggrepel)
  library(stringr)
})
set.seed(20260816)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Run this file with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
package_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
out_dir <- file.path(package_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

font_family <- "Arial"
ink <- "#243746"
navy <- "#2F5573"
teal <- "#4C8C87"
gold <- "#B88A3B"
rose <- "#A65F5F"
slate <- "#71808C"
paper <- "#FAFAF7"
mist <- "#EAF0F2"
mint <- "#E6F0ED"
sand <- "#F3ECDD"
blush <- "#F3E6E4"

save_grid_figure <- function(filename, draw_fun, width_mm = 180, height_mm = 105) {
  pdf_file <- file.path(out_dir, paste0(filename, ".pdf"))
  tiff_file <- file.path(out_dir, paste0(filename, ".tiff"))
  grDevices::cairo_pdf(pdf_file, width = width_mm / 25.4,
                       height = height_mm / 25.4, family = font_family,
                       bg = "white")
  draw_fun()
  dev.off()
  ragg::agg_tiff(tiff_file, width = width_mm, height = height_mm,
                 units = "mm", res = 600, compression = "lzw",
                 background = "white", scaling = 1)
  draw_fun()
  dev.off()
}

rounded_box <- function(x, y, w, h, label, fill = "white", col = ink,
                        fontsize = 9.5, fontface = "plain", radius = 0.015) {
  grid.roundrect(x = unit(x, "npc"), y = unit(y, "npc"),
                 width = unit(w, "npc"), height = unit(h, "npc"),
                 r = unit(radius, "npc"),
                 gp = gpar(fill = fill, col = col, lwd = 1.05))
  grid.text(label, x = unit(x, "npc"), y = unit(y, "npc"),
            gp = gpar(fontfamily = font_family, fontsize = fontsize,
                      fontface = fontface, col = ink),
            just = "centre")
}

arrow_line <- function(x0, y0, x1, y1, col = slate, lwd = 1.2) {
  grid.segments(unit(x0, "npc"), unit(y0, "npc"),
                unit(x1, "npc"), unit(y1, "npc"),
                arrow = arrow(type = "closed", length = unit(2.0, "mm")),
                gp = gpar(col = col, lwd = lwd))
}

draw_fig1 <- function() {
  grid.newpage()
  pushViewport(viewport(gp = gpar(fontfamily = font_family)))
  grid.text("(a)", 0.025, 0.965, just = c("left", "top"),
            gp = gpar(fontfamily = font_family, fontsize = 12, fontface = "bold", col = ink))
  grid.text("One word, several measurement operators", 0.08, 0.94,
            just = c("left", "top"),
            gp = gpar(fontfamily = font_family, fontsize = 11.2, fontface = "bold", col = ink))

  rounded_box(0.12, 0.72, 0.18, 0.12, "Chemical\nidentity", mist, navy, 10, "bold")
  rounded_box(0.12, 0.54, 0.18, 0.12, "Aqueous\nconditions", mint, teal, 10, "bold")
  rounded_box(0.12, 0.36, 0.18, 0.12, "Protocol\nand time", sand, gold, 10, "bold")

  rounded_box(0.39, 0.54, 0.20, 0.24, "Measurement\noperator", "#FFFFFF", ink, 11, "bold")
  arrow_line(0.215, 0.72, 0.285, 0.60)
  arrow_line(0.215, 0.54, 0.285, 0.54)
  arrow_line(0.215, 0.36, 0.285, 0.48)

  endpoint_y <- c(0.77, 0.655, 0.54, 0.425, 0.31)
  endpoint_labs <- c("Equilibrium", "Kinetic / apparent", "Soluble recovery",
                     "Pipeline status", "Phase / process proxy")
  endpoint_fill <- c(mist, mist, mint, mint, blush)
  endpoint_col <- c(navy, navy, teal, teal, rose)
  for (i in seq_along(endpoint_y)) {
    rounded_box(0.75, endpoint_y[i], 0.27, 0.085, endpoint_labs[i],
                endpoint_fill[i], endpoint_col[i], 9.3, "plain", 0.012)
    arrow_line(0.49, 0.54, 0.605, endpoint_y[i], col = "#9AA5AD", lwd = 0.9)
  }

  grid.lines(x = unit(c(0.02, 0.98), "npc"), y = unit(c(0.205, 0.205), "npc"),
             gp = gpar(col = "#D2D9DD", lwd = 0.9))
  grid.text("(b)", 0.025, 0.185, just = c("left", "top"),
            gp = gpar(fontfamily = font_family, fontsize = 12, fontface = "bold", col = ink))
  grid.text("The model claim cannot be broader than its recorded inputs and endpoint", 0.08, 0.175,
            just = c("left", "top"),
            gp = gpar(fontfamily = font_family, fontsize = 11.2, fontface = "bold", col = ink))
  rounded_box(0.18, 0.075, 0.26, 0.10, "Canonical sequence\nProtT5 / ESM-2", mist, navy, 9.6, "bold")
  rounded_box(0.50, 0.075, 0.25, 0.10, "Endpoint-specific\npredictor", mint, teal, 9.6, "bold")
  rounded_box(0.82, 0.075, 0.27, 0.10, "Condition-specific\nmodel output", sand, gold, 9.6, "bold")
  arrow_line(0.31, 0.075, 0.37, 0.075)
  arrow_line(0.625, 0.075, 0.68, 0.075)
  popViewport()
}

# S10 editorial decision: the endpoint diagram is redundant with Table 1 and
# the adjacent prose. Its source is retained for provenance, but it is no
# longer rendered as a main figure.

audit_dir <- file.path(package_dir, "audit")
camsol_audit <- fromJSON(file.path(audit_dir, "camsolptm_endpoint_audit_20260804.json"))
stapled_audit <- fromJSON(file.path(audit_dir, "stapled_peptides_endpoint_audit_20260804.json"))
pegasus_audit <- fromJSON(file.path(audit_dir, "pegasus_endpoint_audit_20260804.json"))

audit_counts <- c(
  as.integer(stapled_audit$solubility_value_types$exact_numeric),
  as.integer(pegasus_audit$rows_with_numeric_solubility),
  as.integer(camsol_audit$total_rows_with_experimental_endpoint)
)
stopifnot(identical(audit_counts, c(234L, 40L, 34L)))

expected_scale <- c(
  "CamSol-PTM" = 34L, "Stapled series" = 234L, "PEGASUS" = 40L, "eSOL" = 3173L,
  "SoluProt training" = 11436L, "PeptideBERT / PROSO-derived" = 18453L,
  "PLM_Sol UESolDS" = 78031L
)

audit <- data.frame(
  source = factor(c("Stapled-peptide\nseries", "PEGASUS\ncyclic peptides", "CamSol-PTM\nthree scaffolds"),
                  levels = c("Stapled-peptide\nseries", "PEGASUS\ncyclic peptides", "CamSol-PTM\nthree scaffolds")),
  n = audit_counts,
  colour = c(slate, teal, navy)
)

p2a <- ggplot(audit, aes(x = n, y = source, fill = I(colour))) +
  geom_col(width = 0.60, colour = "white", linewidth = 0.4) +
  geom_text(aes(label = n), hjust = -0.20, family = font_family,
            size = 3.7, colour = ink) +
  scale_x_continuous(limits = c(0, 270), breaks = c(0, 100, 200),
                     expand = expansion(mult = c(0, 0.01))) +
  labs(x = "Exact numerical outcomes retained", y = NULL,
       title = "(a) Direct experimental outcomes") +
  theme_minimal(base_family = font_family, base_size = 10.5) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 11.5, colour = ink, hjust = 0),
        axis.text = element_text(colour = ink), axis.title = element_text(colour = ink),
        plot.margin = margin(8, 16, 8, 8))

stapled_value_types <- stapled_audit$solubility_value_types
stapled_status <- data.frame(
  status = factor(c("Exact numeric", "Missing", "Categorical"),
                  levels = c("Categorical", "Missing", "Exact numeric")),
  n = c(as.integer(stapled_value_types$exact_numeric),
        as.integer(stapled_value_types$missing),
        as.integer(stapled_value_types$unparsed_text))
)
stopifnot(sum(stapled_status$n) == as.integer(stapled_audit$total_included_rows))
stapled_status$label <- sprintf("%d\n(%.1f%%)", stapled_status$n,
                                100 * stapled_status$n / sum(stapled_status$n))

p2b <- ggplot(stapled_status, aes(x = n, y = status, fill = status)) +
  geom_col(width = 0.56, colour = "white", linewidth = 0.45) +
  geom_text(aes(label = gsub("\\n", " ", label)), hjust = -0.12,
            family = font_family, size = 3.30, colour = ink) +
  scale_fill_manual(values = c("Exact numeric" = teal, "Missing" = "#DCE2E5",
                               "Categorical" = gold),
                    guide = "none") +
  scale_x_continuous(limits = c(0, 370), breaks = c(0, 100, 200, 300),
                     expand = expansion(mult = c(0, 0.01))) +
       labs(x = sprintf("Rows (n=%d)", as.integer(stapled_audit$total_included_rows)),
       y = NULL, fill = NULL,
       title = "(b) Stapled-series outcomes") +
  theme_minimal(base_family = font_family, base_size = 10.5) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        axis.text.y = element_text(colour = ink), axis.text.x = element_text(colour = ink),
        axis.title = element_text(colour = ink),
        plot.title = element_text(face = "bold", size = 11.5, colour = ink),
        plot.margin = margin(8, 8, 2, 8))

canonical <- data.frame(
  source = factor(c("CamSol-PTM", "Stapled series", "PEGASUS"),
                  levels = rev(c("CamSol-PTM", "Stapled series", "PEGASUS"))),
  faithful = c(as.integer(camsol_audit$canonical_one_letter_sequence_strings),
               as.integer(stapled_audit$representation_support_counts$canonical_linear_one_letter_only),
               as.integer(pegasus_audit$canonical_one_letter_sequence_count)),
  total = c(as.integer(camsol_audit$sequence_table_rows),
            as.integer(stapled_audit$total_included_rows - stapled_audit$rows_without_chemical_name),
            as.integer(pegasus_audit$data_rows))
)
canonical$percent <- 100 * canonical$faithful / canonical$total
canonical$label <- sprintf("%d/%d (%.1f%%)", canonical$faithful, canonical$total,
                           canonical$percent)
stopifnot(identical(canonical$faithful, c(1L, 0L, 0L)))

p2c <- ggplot(canonical, aes(x = percent, y = source)) +
  geom_segment(aes(x = 0, xend = percent, yend = source), linewidth = 2.6,
               colour = navy, lineend = "round") +
  geom_point(size = 3.6, shape = 21, stroke = 0.9, fill = "white", colour = navy) +
  geom_text(aes(x = pmax(percent + 3.2, 4.0), label = label), hjust = 0,
            family = font_family, size = 3.35, colour = ink) +
  scale_x_continuous(limits = c(0, 31), breaks = c(0, 10, 20, 30),
                     labels = c("0", "10", "20", "30"),
                     expand = expansion(mult = c(0.05, 0))) +
  labs(x = "Bare canonical one-letter strings (%)", y = NULL,
       title = "(c) Bare canonical one-letter strings") +
  theme_minimal(base_family = font_family, base_size = 10.5) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        axis.text = element_text(colour = ink), axis.title = element_text(colour = ink),
        plot.title = element_text(face = "bold", size = 11.5, colour = ink),
        plot.margin = margin(8, 12, 8, 8))

pegasus_projection <- data.frame(
  issue = factor(c("N-methylated rows", "Chemical distinctions lost\nby sequence projection"),
                 levels = rev(c("N-methylated rows",
                                "Chemical distinctions lost\nby sequence projection"))),
  n = c(as.integer(pegasus_audit$rows_with_N_methylation_count_greater_than_zero),
        as.integer(pegasus_audit$chemical_distinctions_lost_by_sequence_projection)),
  total = as.integer(pegasus_audit$data_rows)
)
pegasus_projection$percent <- 100 * pegasus_projection$n / pegasus_projection$total
pegasus_projection$label <- sprintf("%d/%d (%.1f%%)", pegasus_projection$n,
                                    pegasus_projection$total,
                                    pegasus_projection$percent)

p2d <- ggplot(pegasus_projection, aes(x = percent, y = issue)) +
  geom_col(width = 0.48, fill = slate, colour = "white", linewidth = 0.4) +
  geom_text(aes(label = label), hjust = -0.12, family = font_family,
            size = 3.35, colour = ink) +
  scale_x_continuous(limits = c(0, 125), breaks = c(0, 20, 40, 60, 80, 100),
                     labels = c("0", "20", "40", "60", "80", "100"), expand = c(0, 0)) +
  labs(x = "Share of measured structures (%)", y = NULL,
       title = "(d) PEGASUS input loss") +
  theme_minimal(base_family = font_family, base_size = 10.5) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        axis.text = element_text(colour = ink), axis.title = element_text(colour = ink),
        plot.title = element_text(face = "bold", size = 11.5, colour = ink),
        plot.margin = margin(8, 24, 8, 8))

p2 <- (p2a + p2b) / (p2c + p2d) +
  plot_layout(widths = c(1, 1), heights = c(1, 1)) +
  plot_annotation(theme = theme(plot.background = element_rect(fill = "white", colour = NA)))
# S10 editorial decision: these audit counts are already stated exactly in the
# main text. The four-panel audit graphic is retained as reproducible source,
# but is no longer rendered as a main figure.

landscape_audit <- fromJSON(file.path(audit_dir, "literature_dataset_scale_20260805.json"),
                            simplifyDataFrame = TRUE)$records
landscape_audit$endpoint <- factor(
  landscape_audit$endpoint,
  levels = rev(c("Direct peptide measurement", "Cell-free soluble fraction",
                 "Recombinant-expression status", "Production-stage status",
                 "Integrated production status"))
)
landscape_audit$evidence_type <- factor(
  landscape_audit$evidence_type,
  levels = c("Audited direct source", "Classical protein benchmark", "pLM-era benchmark")
)
stopifnot(identical(sort(as.integer(landscape_audit$records)),
                    sort(c(34L, 40L, 234L, 3173L, 11436L, 18453L, 78031L))))

landscape_audit$display_y <- as.numeric(landscape_audit$endpoint)

# The two smallest direct-measurement counts are deliberately offset by a
# fixed, outcome-independent amount so their marks remain distinguishable.
#
# S26, at the author's request: the S16 design left the marks unlabelled and
# recited every dataset name and count in the caption. Naming each mark in the
# plotting field is what a reader actually needs, so the labels move onto the
# dots with ggrepel and the caption drops the recitation. The figure and the
# caption are still generated from one source, `literature_dataset_scale`, and a
# contract test now checks that the rendered labels match that file rather than
# checking that they are absent.
landscape_audit$display_y[landscape_audit$records == 34] <-
  landscape_audit$display_y[landscape_audit$records == 34] - 0.13
landscape_audit$display_y[landscape_audit$records == 40] <-
  landscape_audit$display_y[landscape_audit$records == 40] + 0.13

landscape_audit$point_label <- sprintf("%s (%s)", landscape_audit$label,
  formatC(landscape_audit$records, format = "d", big.mark = ","))

p3 <- ggplot(landscape_audit, aes(x = records, y = display_y)) +
  geom_hline(yintercept = seq_len(5), colour = "#E1E6E9", linewidth = 0.42) +
  geom_vline(xintercept = c(0, 100, 1000, 10000, 100000),
             colour = "#E8ECEE", linewidth = 0.38) +
  geom_point(
    aes(fill = evidence_type), shape = 21, size = 4.2, stroke = 0.55,
    colour = "white"
  ) +
  ggrepel::geom_text_repel(
    aes(label = point_label), family = font_family, size = 2.95, colour = ink,
    # ggrepel's xlim/ylim are applied in transformed panel space; on this
    # pseudo-log axis they pushed every label to the right edge and clipped it.
    # The headroom added to the y scale below does the same job safely.
    seed = 20260908, min.segment.length = 0.20, segment.size = 0.32,
    segment.colour = "#98A2AA", box.padding = 0.38, point.padding = 0.24,
    max.overlaps = Inf, direction = "both", nudge_y = 0.30
  ) +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10, sigma = 10),
    limits = c(0, 260000),
    breaks = c(0, 100, 1000, 10000, 100000),
    labels = c("0", "100", "1,000", "10,000", "100,000"),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = seq_len(5),
    labels = c(
      "Integrated status", "Production status", "Recombinant expression",
      "Cell-free fraction", "Direct measurement"
    ),
    limits = c(0.55, 5.85), expand = c(0, 0)
  ) +
  scale_fill_manual(
    values = c(
      "Audited direct source" = rose,
      "Classical protein benchmark" = gold,
      "pLM-era benchmark" = teal
    ),
    guide = "none"
  ) +
  labs(x = "Reported records (pseudo-log scale)", y = NULL) +
  theme_classic(base_family = font_family, base_size = 10.0) +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.x = element_line(colour = ink, linewidth = 0.42),
    axis.ticks.x = element_line(colour = ink, linewidth = 0.35),
    axis.text = element_text(colour = ink),
    axis.text.y = element_text(size = 9.0, hjust = 1),
    axis.title.x = element_text(size = 9.4, colour = ink),
    plot.margin = margin(8, 10, 7, 8)
  ) +
  coord_cartesian(clip = "off")

ggsave(file.path(out_dir, "Okada_Fig3.pdf"), p3, width = 180, height = 96,
       units = "mm", device = cairo_pdf, family = font_family, bg = "white")
ragg::agg_tiff(file.path(out_dir, "Okada_Fig3.tiff"), width = 180, height = 96,
               units = "mm", res = 600, compression = "lzw", background = "white")
print(p3)
dev.off()

message("Created the S16 numbered minimalist dataset-scale Figure 2 in ", out_dir)
