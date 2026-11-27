# Restored original mobile interface drawing; evidence remains the synthetic record.
draw_phone <- function() {
  # S18: Figure 5 is the collaborator-facing mobile screen alone. The separate
  # input panel, the outer container frame and the "(b) UI / query -> bound"
  # captions were removed at the author's request; the input surface and the
  # remaining tabs stay in Supplementary Figure S1.
  #
  # The canvas is 82 x 150 mm and the figure is placed at 82 mm, a scale of 1.0,
  # so every authored point size below is also its on-page size.
  # Draw into the compositor viewport.
  grid.rect(gp = gpar(fill = white, col = NA))

  # Device body and screen.
  box(0.045, 0.020, 0.910, 0.960, fill = ink, col = ink, lwd = 1.2, radius = 0.028)
  box(0.075, 0.055, 0.850, 0.890, fill = white, col = white, radius = 0.020)
  box(0.400, 0.962, 0.200, 0.011, fill = "#5E7280", col = "#5E7280", radius = 0.006)

  # Header.
  main_txt("PepSol Explorer 1.1.0", 0.108, 0.905, size = 7.8, col = blue, face = "bold")
  box(0.660, 0.884, 0.235, 0.042, fill = pale_teal, col = pale_teal, radius = 0.021)
  main_txt("IN DOMAIN", 0.7775, 0.905, size = 7.4, col = teal, face = "bold",
           just = "centre")

  # Query.
  box(0.105, 0.740, 0.790, 0.118, fill = pale_blue, col = "#D7E9EF", radius = 0.014)
  main_txt("QUERY", 0.130, 0.833, size = 7.4, col = blue, face = "bold")
  main_txt("E. coli soluble expression", 0.130, 0.797, size = 7.8, face = "bold")
  main_txt("300 aa  ·  local input", 0.130, 0.764, size = 7.4,
           col = teal, face = "bold")
  line_seg(0.500, 0.733, 0.500, 0.712, col = blue, lwd = 1.1, arrow = TRUE)

  # Endpoint-matched score views.
  box(0.105, 0.435, 0.790, 0.268, fill = white, col = line, radius = 0.014)
  main_txt("SCORE VIEWS", 0.130, 0.672, size = 7.4, col = blue, face = "bold")
  score_x <- c(0.130, 0.395, 0.660)
  score_fill <- c("#EEF1F3", pale_blue, pale_teal)
  score_col <- c(steel, blue, teal)
  score_lab <- c("B0", "B1", "+ ProtT5")
  score_val <- c(
    protein$soluprot_transparent_score_0_100,
    protein$soluprot_order_aware_score_0_100,
    protein$soluprot_prott5_score_0_100
  )
  for (i in seq_along(score_x)) {
    box(score_x[i], 0.518, 0.235, 0.126, fill = score_fill[i], col = score_fill[i],
        radius = 0.012)
    main_txt(score_lab[i], score_x[i] + 0.1175, 0.612, size = 7.4,
             col = score_col[i], face = "bold", just = "centre")
    main_txt(sprintf("%.1f", score_val[i]), score_x[i] + 0.1175, 0.560, size = 9.4,
             col = score_col[i], face = "bold", just = "centre")
  }
  main_txt("Model scores · not concentration", 0.500, 0.474, size = 7.4,
           col = amber, face = "bold", just = "centre")
  line_seg(0.500, 0.428, 0.500, 0.407, col = blue, lwd = 1.1, arrow = TRUE)

  # Bounded interpretation and the explicit stop boundary.
  box(0.105, 0.170, 0.790, 0.228, fill = pale_amber, col = "#F0DEB8", radius = 0.014)
  main_txt("BOUNDED INTERPRETATION", 0.130, 0.365, size = 7.4, col = amber, face = "bold")
  main_txt("Use for E. coli expression", 0.130, 0.327, size = 7.4, face = "bold")
  main_txt("STOP for other endpoints", 0.130, 0.292, size = 7.4, col = red, face = "bold")
  box(0.130, 0.196, 0.740, 0.062, fill = pale_red, col = pale_red, radius = 0.016)
  main_txt("OUT OF DOMAIN  →  STOP", 0.500, 0.227, size = 7.4, col = red,
           face = "bold", just = "centre")

  main_txt("No network transfer", 0.500, 0.140, size = 7.4, col = muted,
           face = "bold", just = "centre")
  main_txt("CSV / JSON export", 0.500, 0.108,
           size = 7.4, col = muted, just = "centre")
  box(0.395, 0.072, 0.210, 0.010, fill = "#9EAFB9", col = "#9EAFB9", radius = 0.005)
}

# Native 82 x 150 mm phone preserved at the same physical size.
draw_main <- function() {
 grid.newpage()
 draw_phone()
}
