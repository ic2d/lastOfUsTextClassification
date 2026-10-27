# ------------------------------------------------------------------
# Figures of the paper (ggplot2). Run from the repository root:
#   Rscript figures/make_figures.R [path/to/results/analysis] [output/folder]
# ------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(forcats)
  library(stringr)
  library(fontawesome)
  library(rsvg)
  library(png)
})

args <- commandArgs(trailingOnly = TRUE)
RESULTS <- if (length(args) > 0) args[1] else "results/analysis"
OUT <- if (length(args) > 1) args[2] else "figures"

COLUMN <- 3.33   # inches, sigconf column width
TEXT <- 7.0      # inches, sigconf text width

DATASETS <- c(user_reviews_g1 = "user_reviews_g1 (The Last of Us)",
              user_reviews_g2 = "user_reviews_g2 (The Last of Us Part II)")
FEATURES <- c("BOW", "TFIDF", "WESG", "WECBOW", "ROBERTA")
BASELINES <- c("LEXICON", "PRETRAINED")
FEATURE_LABELS <- c(BOW = "BOW", TFIDF = "TF-IDF", WESG = "WE-SG", WECBOW = "WE-CBOW", ROBERTA = "RoBERTa-emb")
FEATURE_COLORS <- c(BOW = "#1b9e77", TFIDF = "#d95f02", WESG = "#7570b3", WECBOW = "#e7298a", ROBERTA = "#666666")
CLASS_COLORS <- c(negative = "#b2182b", positive = "#2166ac")

theme_paper <- function(base_size = 8) {
  theme_bw(base_size = base_size) +
    theme(panel.grid.minor = element_blank(),
          strip.background = element_rect(fill = "grey92"),
          legend.position = "bottom",
          legend.margin = margin(0, 0, 0, 0),
          legend.box.margin = margin(-4, 0, 0, 0))
}

save_pdf <- function(plot, name, width, height) {
  ggsave(file.path(OUT, paste0(name, ".pdf")), plot, width = width, height = height, device = cairo_pdf)
}

read_results <- function(file) read_csv(file.path(RESULTS, file), show_col_types = FALSE)

pretty_setup <- function(setup) {
  setup |>
    str_replace("^(LEXICON|PRETRAINED)_", "") |>
    str_replace_all(c("^TFIDF_" = "TF-IDF_", "^WESG_" = "WE-SG_", "^WECBOW_" = "WE-CBOW_", "^ROBERTA_" = "RoBERTa-emb_")) |>
    str_replace("_", " + ")
}

# ------------------------------------------------------------------
# Figure 1: experimental pipeline
# ------------------------------------------------------------------

# Font Awesome icon as a raster grob (rendered at high resolution)
fa_grob <- function(name, colour = "grey20", size = 512) {
  svg <- as.character(fa(name, fill = colour))
  bitmap <- png::readPNG(rsvg::rsvg_png(charToRaw(svg), height = size))
  grid::rasterGrob(bitmap, interpolate = TRUE)
}

icon <- function(name, x, y, half = 0.2, colour = "grey20") {
  annotation_custom(fa_grob(name, colour), xmin = x - half, xmax = x + half, ymin = y - half, ymax = y + half)
}

# Okabe-Ito colorblind-safe palette
OKABE_ITO <- c(blue = "#0072B2", orange = "#E69F00", green = "#009E73", purple = "#CC79A7",
               vermillion = "#D55E00", sky = "#56B4E9")

plot_pipeline <- function() {
  w <- 1.7; h <- 1.15; y0 <- 2.0; yb <- 0.5
  steps <- tribble(
    ~x, ~title, ~detail, ~icon, ~fill, ~ink,
    1, "Reviews", "TLOU I and II", "comments", OKABE_ITO[["blue"]], "white",
    3, "Labeling", "user score", "tags", OKABE_ITO[["blue"]], "white",
    5, "Preprocessing", "negations kept", "broom", OKABE_ITO[["blue"]], "white",
    7, "Features", "BOW, TF-IDF, embeddings", "table-cells", OKABE_ITO[["orange"]], "black",
    9, "Classifiers", "5 algorithms", "brain", OKABE_ITO[["orange"]], "black",
    11, "Evaluation", "BAC, F1, tests", "chart-column", OKABE_ITO[["green"]], "white"
  )
  flow <- tibble(x = seq(1, 9, 2) + w / 2, xend = seq(3, 11, 2) - w / 2, y = y0, yend = y0)
  baseline <- list(x = 7, half = 2.4)
  # baselines receive the raw (labeled) text and go straight to the evaluation
  lexicon <- tribble(
    ~x, ~xend, ~y, ~yend,
    3, 3, y0 - h / 2, yb,
    3, baseline$x - baseline$half, yb, yb,
    baseline$x + baseline$half, 11, yb, yb,
    11, 11, yb, y0 - h / 2
  )
  # RoBERTa embeddings are also computed from the raw text, skipping the preprocessing
  raw_text <- tribble(
    ~x, ~xend, ~y, ~yend,
    3, 3, y0 + h / 2, 2.95,
    3, 7, 2.95, 2.95,
    7, 7, 2.95, y0 + h / 2
  )
  arrow_head <- arrow(length = unit(1.6, "mm"), type = "closed")
  plot <- ggplot() +
    annotate("rect", xmin = 5.95, xmax = 10.05, ymin = 1.1, ymax = 2.78, fill = "#FFF4DC",
             colour = OKABE_ITO[["orange"]], linetype = "dashed", linewidth = 0.4) +
    annotate("text", x = 8, y = 1.24, size = 2.2, fontface = "italic", colour = "grey20",
             label = "10 x 10-fold CV, fitted on training folds") +
    geom_rect(data = steps, aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y0 - h / 2, ymax = y0 + h / 2, fill = fill),
              colour = NA) +
    geom_text(data = steps, aes(x = x, y = y0 - 0.12, label = title, colour = ink), size = 2.6, fontface = "bold") +
    geom_text(data = steps, aes(x = x, y = y0 - 0.38, label = detail, colour = ink), size = 2.0) +
    annotate("rect", xmin = baseline$x - baseline$half, xmax = baseline$x + baseline$half,
             ymin = yb - 0.3, ymax = yb + 0.3, fill = OKABE_ITO[["purple"]]) +
    annotate("text", x = baseline$x + 0.5, y = yb, size = 2.6, fontface = "bold",
             label = "Baselines: TextBlob, VADER, RoBERTa") +
    geom_segment(data = flow, aes(x = x, xend = xend, y = y, yend = yend), arrow = arrow_head,
                 linewidth = 0.4, colour = "grey25") +
    geom_segment(data = lexicon[c(1, 3), ], aes(x = x, xend = xend, y = y, yend = yend),
                 colour = OKABE_ITO[["purple"]], linewidth = 0.5) +
    geom_segment(data = lexicon[c(2, 4), ], aes(x = x, xend = xend, y = y, yend = yend),
                 colour = OKABE_ITO[["purple"]], linewidth = 0.5, arrow = arrow_head) +
    geom_segment(data = raw_text[1:2, ], aes(x = x, xend = xend, y = y, yend = yend),
                 colour = OKABE_ITO[["blue"]], linewidth = 0.4, linetype = "longdash") +
    geom_segment(data = raw_text[3, ], aes(x = x, xend = xend, y = y, yend = yend),
                 colour = OKABE_ITO[["blue"]], linewidth = 0.4, linetype = "longdash", arrow = arrow_head) +
    annotate("label", x = 4.6, y = 2.95, size = 2.0, label = "raw text", fill = "white",
             colour = OKABE_ITO[["blue"]], label.padding = unit(0.5, "mm"), label.r = unit(0, "mm"), linewidth = 0) +
    scale_fill_identity() +
    scale_colour_identity() +
    coord_cartesian(xlim = c(0, 12), ylim = c(0.12, 3.08), expand = FALSE) +
    theme_void()
  # one icon per step (same colour as its text), plus the icons of the baselines
  for (i in seq_len(nrow(steps))) {
    plot <- plot + icon(steps$icon[i], steps$x[i], y0 + 0.27, half = 0.19, colour = steps$ink[i])
  }
  plot +
    icon("spell-check", baseline$x - baseline$half + 0.3, yb, half = 0.16, colour = "black") +
    icon("robot", baseline$x - baseline$half + 0.72, yb, half = 0.16, colour = "black")
}

# ------------------------------------------------------------------
# Figure 2: most discriminative terms per class (weighted log-odds)
# ------------------------------------------------------------------

plot_terms <- function(n = 12) {
  terms <- read_results("terms.csv") |>
    group_by(dataset, cls) |>
    slice_max(abs(z), n = n) |>
    ungroup() |>
    mutate(dataset = factor(DATASETS[dataset], levels = DATASETS),
           key = paste(term, dataset, sep = "___"),
           key = fct_reorder(key, z))
  levels(terms$dataset) <- sub(" \\(", "\n(", levels(terms$dataset))
  ggplot(terms, aes(x = z, y = key, fill = cls)) +
    geom_col(width = 0.75) +
    geom_vline(xintercept = 0, linewidth = 0.3) +
    facet_wrap(~dataset, ncol = 2, scales = "free") +
    scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
    scale_fill_manual(values = CLASS_COLORS, name = "More frequent in") +
    labs(x = "z-score of the weighted log-odds ratio", y = NULL) +
    theme_paper(7)
}

# ------------------------------------------------------------------
# Figure 3: BAC of every setup, with the lexicon baselines
# ------------------------------------------------------------------

plot_bac <- function() {
  summary <- read_results("summary.csv") |>
    mutate(dataset = factor(DATASETS[dataset], levels = DATASETS))
  ml <- summary |>
    filter(!feature %in% BASELINES) |>
    mutate(feature = factor(feature, levels = FEATURES),
           algorithm = factor(algorithm, levels = c("SVM", "RF", "DT", "KNN", "GNB")))
  baselines <- summary |> filter(feature %in% BASELINES)
  ggplot(ml, aes(x = algorithm, y = bac_mean, colour = feature)) +
    geom_hline(data = baselines, aes(yintercept = bac_mean, linetype = algorithm), colour = "grey30") +
    geom_hline(yintercept = 0.5, colour = "grey70", linewidth = 0.3) +
    geom_pointrange(aes(ymin = bac_mean - bac_std, ymax = bac_mean + bac_std),
                    position = position_dodge(width = 0.6), size = 0.2) +
    facet_wrap(~dataset, ncol = 2) +
    scale_colour_manual(values = FEATURE_COLORS, labels = FEATURE_LABELS, name = "Features") +
    scale_linetype_manual(values = c(TextBlob = "dashed", VADER = "dotted", RoBERTa = "solid"), name = "Baselines") +
    labs(x = "Algorithm", y = "BAC (mean and sd)") +
    theme_paper(8)
}

# ------------------------------------------------------------------
# Figure 4: average ranks (Friedman) with the Nemenyi critical difference
# ------------------------------------------------------------------

plot_ranks <- function(top = 12) {
  friedman <- read_results("friedman.csv")
  ranks <- read_results("ranks.csv") |>
    group_by(dataset) |>
    slice_min(avg_rank, n = top) |>
    ungroup() |>
    left_join(friedman |> select(dataset, cd), by = "dataset") |>
    mutate(best = ave(avg_rank, dataset, FUN = min),
           group = ifelse(avg_rank - best <= cd, "no", "yes"),
           baseline = str_detect(setup, "^(LEXICON|PRETRAINED)_"),
           dataset = factor(DATASETS[dataset], levels = DATASETS),
           key = fct_rev(fct_reorder(paste(setup, dataset, sep = "___"), avg_rank)))
  ggplot(ranks, aes(x = avg_rank, y = key)) +
    geom_rect(data = distinct(ranks, dataset, best, cd),
              aes(xmin = best, xmax = best + cd, ymin = -Inf, ymax = Inf),
              inherit.aes = FALSE, fill = "grey88") +
    geom_point(aes(shape = baseline, colour = group), size = 1.6) +
    facet_wrap(~dataset, ncol = 2, scales = "free") +
    scale_y_discrete(labels = function(x) pretty_setup(sub("___.*$", "", x))) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 17), labels = c("ML setup", "Baseline"), name = NULL) +
    scale_colour_manual(values = c(no = "black", yes = "grey55"), guide = "none") +
    labs(x = "Average rank over the 100 test folds (lower is better)", y = NULL) +
    theme_paper(7)
}

# ------------------------------------------------------------------
# Figure 5: mean confusion matrices of the top-2 setups per dataset
# ------------------------------------------------------------------

plot_confusion <- function() {
  cm <- read_results("confusion.csv") |>
    group_by(dataset, setup, true) |>
    mutate(rate = count / sum(count)) |>
    ungroup() |>
    mutate(panel = paste0(dataset, "\n", pretty_setup(setup)),
           true = factor(c("neg.", "pos.")[true + 1], levels = c("pos.", "neg.")),
           pred = factor(c("neg.", "pos.")[pred + 1], levels = c("neg.", "pos.")),
           label = sprintf("%.0f\n(%.1f%%)", count, 100 * rate))
  ggplot(cm, aes(x = pred, y = true, fill = rate)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = label, colour = rate > 0.5), size = 2.3, lineheight = 0.85) +
    facet_wrap(~panel, ncol = 2) +
    scale_fill_gradient(low = "grey95", high = "#2166ac", limits = c(0, 1), guide = "none") +
    scale_colour_manual(values = c(`FALSE` = "black", `TRUE` = "white"), guide = "none") +
    labs(x = "Predicted label", y = "True label") +
    theme_paper(7) +
    theme(panel.grid = element_blank())
}

# ------------------------------------------------------------------
# Figure 6: barcode of the predictions (seed 1) against the true labels
# ------------------------------------------------------------------

plot_barcode <- function() {
  barcode <- read_results("barcode.csv")
  # reviews sorted by true class and then by how many setups misclassify them
  order <- barcode |>
    group_by(dataset, row) |>
    summarise(truth = label[setup == "truth"], wrong = sum(label[setup != "truth"] != truth), .groups = "drop") |>
    group_by(dataset) |>
    arrange(truth, wrong, .by_group = TRUE) |>
    mutate(position = row_number() / n()) |>
    ungroup()
  barcode <- barcode |>
    left_join(order, by = c("dataset", "row")) |>
    mutate(setup = ifelse(setup == "truth", "Truth", pretty_setup(setup)),
           class = factor(ifelse(label == 1, "positive", "negative"), levels = c("negative", "positive")),
           dataset = factor(DATASETS[dataset], levels = DATASETS))
  barcode <- barcode |>
    group_by(dataset) |>
    mutate(width = 1 / n_distinct(row)) |>
    ungroup() |>
    mutate(setup = factor(setup, levels = c(rev(setdiff(unique(setup), "Truth")), "Truth")))
  ggplot(barcode, aes(x = position, y = setup, fill = class)) +
    geom_tile(aes(width = width), height = 0.85) +
    facet_wrap(~dataset, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = CLASS_COLORS, name = "Predicted / true class") +
    scale_x_continuous(expand = c(0, 0), labels = scales::percent) +
    labs(x = "Reviews, sorted by true class (share of the dataset)", y = NULL) +
    theme_paper(7) +
    theme(panel.grid = element_blank())
}

# ------------------------------------------------------------------
# Figure 7: error rate of the best setup per user score
# ------------------------------------------------------------------

plot_errors_by_score <- function() {
  errors <- read_results("errors_by_score.csv") |>
    mutate(cls = ifelse(score >= 7, "positive", "negative"),
           dataset = factor(paste0(DATASETS[dataset], "\nbest: ", pretty_setup(setup)),
                            levels = unique(paste0(DATASETS[dataset], "\nbest: ", pretty_setup(setup)))))
  ggplot(errors, aes(x = factor(score), y = error_rate, fill = cls)) +
    geom_col(width = 0.75) +
    geom_text(aes(label = n), vjust = -0.3, size = 1.9) +
    facet_wrap(~dataset, ncol = 1) +
    scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.15))) +
    scale_fill_manual(values = CLASS_COLORS, name = "Class") +
    labs(x = "Score given by the user", y = "Error rate (seed 1)") +
    theme_paper(7)
}

# ------------------------------------------------------------------
# Heatmap of the mean BAC of every (feature, algorithm) setup
# ------------------------------------------------------------------

plot_heatmap <- function() {
  ALGORITHMS <- c("SVM", "RF", "DT", "KNN", "GNB")
  BASELINE_LABELS <- c(TextBlob = "TextBlob", VADER = "VADER", RoBERTa = "SiEBERT")
  summary <- read_results("summary.csv") |>
    mutate(dataset = factor(DATASETS[dataset], levels = DATASETS))
  ml <- summary |>
    filter(!feature %in% BASELINES) |>
    mutate(row = FEATURE_LABELS[feature], x = match(algorithm, ALGORITHMS), width = 1,
           group = "ML setups") |>
    group_by(dataset) |>
    mutate(best = bac_mean == max(bac_mean)) |>
    group_by(dataset, algorithm) |>
    mutate(best_algorithm = bac_mean == max(bac_mean)) |>
    ungroup()
  # the baselines do not depend on the features nor on the algorithms: one wide cell each
  baselines <- summary |>
    filter(feature %in% BASELINES) |>
    mutate(row = BASELINE_LABELS[algorithm], x = mean(seq_along(ALGORITHMS)), width = length(ALGORITHMS),
           group = "Baselines", best = FALSE, best_algorithm = FALSE)
  heat <- bind_rows(ml, baselines) |>
    mutate(row = factor(row, levels = rev(c(FEATURE_LABELS, BASELINE_LABELS))),
           group = factor(group, levels = c("ML setups", "Baselines")))
  # one plot per block, so that the algorithm axis sits right below the ML setups
  block <- function(data) {
    ggplot(data, aes(x = x, y = row, fill = bac_mean)) +
      geom_tile(aes(width = width), colour = "white", linewidth = 0.6) +
      geom_tile(data = filter(data, best), fill = NA, colour = "black", linewidth = 1.2) +
      geom_text(aes(label = sprintf("%.3f", bac_mean), colour = bac_mean < 0.58,
                    fontface = ifelse(best_algorithm, "bold", "plain")), size = 2.3) +
      facet_grid(group ~ dataset) +
      scale_fill_gradient2(low = "#D01C8B", mid = "white", high = "#00C5CD", midpoint = 0.75, limits = c(0.5, 1), name = "BAC") +
      scale_colour_manual(values = c(`FALSE` = "black", `TRUE` = "white"), guide = "none") +
      scale_x_continuous(breaks = seq_along(ALGORITHMS), labels = ALGORITHMS, expand = c(0, 0)) +
      scale_y_discrete(expand = c(0, 0), drop = TRUE) +
      labs(x = NULL, y = NULL) +
      theme_paper(8) +
      theme(panel.grid = element_blank(), legend.key.width = unit(1.2, "cm"), legend.key.height = unit(0.25, "cm"),
            legend.title = element_text(vjust = 0.9))
  }
  top <- block(filter(heat, group == "ML setups")) + labs(x = "Algorithm")
  bottom <- block(filter(heat, group == "Baselines")) +
    theme(strip.text.x = element_blank(), strip.background.x = element_blank(),
          axis.text.x = element_blank(), axis.ticks.x = element_blank())
  patchwork::wrap_plots(top, bottom, ncol = 1, heights = c(5, 3), guides = "collect") &
    theme(legend.position = "right", legend.key.width = unit(0.3, "cm"), legend.key.height = unit(1.2, "cm"),
          legend.title = element_text(vjust = 1), legend.box.margin = margin(0, 0, 0, 4))
}

# ------------------------------------------------------------------
# Heatmap of the mean BAC per training labels (score, TextBlob, VADER);
# every model is evaluated against the score labels
# ------------------------------------------------------------------

plot_heatmap_labels <- function() {
  ALGORITHMS <- c("SVM", "RF", "DT", "KNN", "GNB")
  LABELS <- c(SCORE = "Trained with score labels", TEXTBLOB = "Trained with TextBlob labels",
              VADER = "Trained with VADER labels")
  heat <- read_results("summary_labels.csv") |>
    mutate(dataset = factor(DATASETS[dataset], levels = DATASETS),
           labels = factor(LABELS[labels], levels = LABELS),
           feature = factor(FEATURE_LABELS[feature], levels = rev(FEATURE_LABELS)),
           algorithm = factor(algorithm, levels = ALGORITHMS)) |>
    group_by(dataset, labels) |>
    mutate(best = bac_mean == max(bac_mean)) |>
    ungroup()
  # the baselines need no training labels: one column each, spanning all the feature rows
  baselines <- read_results("summary.csv") |>
    filter(feature %in% BASELINES) |>
    mutate(dataset = factor(DATASETS[dataset], levels = DATASETS),
           baseline = factor(c(TextBlob = "TextBlob", VADER = "VADER", RoBERTa = "SiEBERT")[algorithm],
                             levels = c("TextBlob", "VADER", "SiEBERT")),
           group = "Baselines")
  fill_scale <- scale_fill_gradient2(low = "#D01C8B", mid = "white", high = "#00C5CD", midpoint = 0.75,
                                     limits = c(0.5, 1), name = "BAC")
  text_colour <- scale_colour_manual(values = c(`FALSE` = "black", `TRUE` = "white"), guide = "none")
  heat_theme <- theme_paper(7) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(size = 6))
  ml <- ggplot(heat, aes(x = algorithm, y = feature, fill = bac_mean)) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_tile(data = filter(heat, best), fill = NA, colour = "black", linewidth = 1) +
    geom_text(aes(label = sprintf("%.2f", bac_mean), colour = bac_mean < 0.58,
                  fontface = ifelse(best, "bold", "plain")), size = 2) +
    facet_grid(dataset ~ labels) +
    fill_scale + text_colour +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(x = "Algorithm", y = NULL) +
    heat_theme +
    theme(strip.text.y = element_blank(), strip.background.y = element_blank())
  base <- ggplot(baselines, aes(x = baseline, y = 3, fill = bac_mean)) +
    geom_tile(height = length(FEATURE_LABELS), colour = "white", linewidth = 0.6) +
    geom_text(aes(label = sprintf("%.2f", bac_mean), colour = bac_mean < 0.58), size = 2) +
    facet_grid(dataset ~ group, labeller = labeller(dataset = function(x) sub(" \\(", "\n(", x))) +
    fill_scale + text_colour +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, 5.5), expand = c(0, 0)) +
    labs(x = NULL, y = NULL) +
    heat_theme +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  patchwork::wrap_plots(ml, base, nrow = 1, widths = c(15, 4.2), guides = "collect") &
    theme(legend.position = "right", legend.key.width = unit(0.3, "cm"), legend.key.height = unit(1.2, "cm"))
}

# ------------------------------------------------------------------
# ------------------------------------------------------------------

save_pdf(plot_pipeline(), "pipeline", TEXT, 1.8)
save_pdf(plot_terms(), "terms", COLUMN, 3.4)
save_pdf(plot_bac(), "bac", TEXT, 2.6)
save_pdf(plot_ranks(), "ranks", TEXT, 2.3)
save_pdf(plot_heatmap(), "heatmap", TEXT, 3.3)
if (file.exists(file.path(RESULTS, "summary_labels.csv"))) save_pdf(plot_heatmap_labels(), "heatmap_labels", TEXT, 3.6)
save_pdf(plot_confusion(), "confusion", COLUMN, 3.0)
save_pdf(plot_barcode(), "barcode", TEXT, 2.0)
save_pdf(plot_errors_by_score(), "errors_by_score", COLUMN, 2.6)
cat("Figures saved in", OUT, "\n")
