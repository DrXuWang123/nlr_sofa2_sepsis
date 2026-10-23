

M <- 5L
MAXIT <- 15L
MISSING_THRESHOLD <- 0.50
# 以下为明确记录的实现选项，并非稿件额外声明的方法参数。
PMM_DONORS <- 5L
PMM_MATCHTYPE <- 1L
RIDGE <- 1e-4

CLINICAL <- c(
  "sex", "age", "race", "admission_type", "hypertension", "heart_failure",
  "malignancy", "charlson_index", "bmi", "smoking", "alcohol", "infection_site",
  "cortisone_used", "crrt_used", "vasoactive_used", "advanced_Resp_support",
  "alb_baseline", "alt_baseline", "bun_baseline", "inr_baseline",
  "glucose_baseline", "hgb_baseline", "lactate_baseline", "wbc_baseline",
  "aki_onset_by72h", "balance_24h_baseline", "MAP_median_baseline",
  "HR_median_baseline", "SpO2_median_baseline", "Temp_max_baseline"
)
FEATURES <- list(
  baseline = c(
    "sofa2_total_max_baseline", "logNLR_baseline", "sofa_TWA", "sofa_slope",
    "sofa_variability_SD", "logNLR_TWA", "logNLR_slope", "logNLR_variability_SD"
  ),
  latest = c(
    "sofa_t_latest", "logNLR_t_latest", "sofa_TWA_t_latest", "sofa_slope_t_latest",
    "sofa_SD_t_latest", "logNLR_TWA_t_latest", "logNLR_slope_t_latest", "logNLR_SD_t_latest"
  )
)

# 种子来自当前保留的各 pool completed.json；相同种子不代表逐值复现保证。
POOLS <- data.frame(
  pool = c("baseline_mimic", "baseline_eicu", "latest_mimic", "latest_eicu"),
  family = c("baseline", "baseline", "latest", "latest"),
  rows = c(5286L, 15735L, 5286L, 15735L),
  seed = c(449972869L, 995354905L, 1326134983L, 1058479334L),
  stringsAsFactors = FALSE
)

fail <- function(...) stop(..., call. = FALSE)
write_table <- function(x, path) {
  write.csv(x, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
}

prepare_pool <- function(input_dir, cfg) {
  input_path <- file.path(input_dir, paste0(cfg$pool, "_predictors_for_mice.csv"))
  spec_path <- file.path(input_dir, paste0(cfg$family, "_mice_specification.csv"))
  spec <- read.csv(spec_path, stringsAsFactors = FALSE, check.names = FALSE)
  expected <- c(FEATURES[[cfg$family]], CLINICAL)
  if (!all(c("variable", "type") %in% names(spec)) ||
      anyNA(spec[, c("variable", "type")]) || anyDuplicated(spec$variable) ||
      !identical(spec$variable, expected) ||
      !all(spec$type %in% c("numeric", "categorical"))) {
    fail("变量配置必须按既有顺序包含本 family 的 8 个特征及 30 个临床变量。")
  }
  # 先按字符读取，再显式转换，避免非法数值被静默变成缺失。
  x <- read.csv(input_path, colClasses = "character", check.names = FALSE,
                na.strings = c("", "NA", "NaN", "NULL"), stringsAsFactors = FALSE)
  if (anyDuplicated(names(x)) || !setequal(names(x), expected)) {
    fail("输入必须仅含配置中的 38 个预测变量，不得加入 ID、结局等其他列。")
  }
  if (nrow(x) != cfg$rows) fail("输入队列行数不符合当前最终分析输入：", cfg$pool)
  x <- x[, expected, drop = FALSE] # 只重排列；绝不筛选、排序或删除患者行。
  for (v in spec$variable[spec$type == "numeric"]) {
    y <- suppressWarnings(as.numeric(x[[v]]))
    if (any(!is.na(x[[v]]) & is.na(y)) || any(!is.na(y) & !is.finite(y))) {
      fail("存在非法数值或无穷值：", v)
    }
    x[[v]] <- y
  }
  for (v in spec$variable[spec$type == "categorical"]) {
    # Unknown 等已编码类别保留；不擅自改成 NA。所有因子均为无序因子。
    x[[v]] <- factor(x[[v]], ordered = FALSE)
  }
  missingness <- data.frame(
    variable = names(x), type = spec$type,
    n_missing = vapply(x, function(z) sum(is.na(z)), integer(1)),
    missing_fraction = vapply(x, function(z) mean(is.na(z)), numeric(1)),
    row.names = NULL
  )
  missingness$excluded <- missingness$missing_fraction > MISSING_THRESHOLD
  keep <- !missingness$excluded
  x <- x[, keep, drop = FALSE]
  spec <- spec[keep, , drop = FALSE]
  if (ncol(x) < 2L) fail("排除 >50% 缺失变量后，可用变量不足。")

  has_missing <- vapply(x, anyNA, logical(1))
  constant <- vapply(x, function(z) length(unique(z[!is.na(z)])) < 2L, logical(1))
  if (any(constant & has_missing)) {
    fail("部分缺失变量仅有一个观测水平，无法按预定方法拟合：",
         paste(names(x)[constant & has_missing], collapse = ", "))
  }
  method <- setNames(rep("", ncol(x)), names(x))
  for (v in names(x)[has_missing]) {
    method[[v]] <- if (is.numeric(x[[v]])) "pmm" else if (nlevels(x[[v]]) == 2L) "logreg" else "polyreg"
  }
  # 使用同一 pool 的其余预测变量。MIMIC-IV / eICU 不拼接，供体不跨队列。
  predictor_matrix <- mice::make.predictorMatrix(x)
  diag(predictor_matrix) <- 0
  predictor_matrix[, constant] <- 0
  predictor_matrix[!has_missing, ] <- 0
  list(data = x, spec = spec, missingness = missingness, method = method,
       predictor_matrix = predictor_matrix, input_path = input_path, spec_path = spec_path)
}

run_pool <- function(prepared, cfg, output_root) {
  out <- file.path(output_root, cfg$pool)
  if (file.exists(out) || dir.exists(out)) fail("输出目录已存在，请另选新目录：", out)
  if (!dir.create(out, recursive = TRUE)) fail("无法创建输出目录：", out)
  x <- prepared$data
  write_table(prepared$missingness, file.path(out, "missingness_before_imputation.csv"))
  write_table(prepared$spec, file.path(out, "retained_variable_specification.csv"))
  write_table(data.frame(input_row = seq_len(nrow(x))), file.path(out, "row_order.csv"))
  # 空字符串方法表示该变量无缺失，无需插补，但仍可作为其他变量的预测因子。
  write_table(data.frame(variable = names(x), method = unname(prepared$method)),
              file.path(out, "requested_methods.csv"))

  imp <- mice::mice(
    data = x, m = M, maxit = MAXIT, method = prepared$method,
    predictorMatrix = prepared$predictor_matrix, where = is.na(x),
    visitSequence = names(x)[prepared$method != ""],
    seed = cfg$seed, printFlag = TRUE,
    donors = PMM_DONORS, matchtype = PMM_MATCHTYPE, ridge = RIDGE,
    remove.constant = TRUE, remove.collinear = TRUE,
    nnet.MaxNWts = 100000, nnet.maxit = 200
  )
  saveRDS(imp, file.path(out, "data_mice_object.rds"))
  write_table(data.frame(variable = names(imp$method), method = unname(imp$method)),
              file.path(out, "data_mice_methods.csv"))
  write_table(data.frame(variable = rownames(imp$predictorMatrix), imp$predictorMatrix,
                         check.names = FALSE, row.names = NULL),
              file.path(out, "effective_predictor_matrix.csv"))
  events <- imp$loggedEvents
  if (is.null(events)) {
    events <- data.frame(it = integer(), im = integer(), dep = character(),
                         meth = character(), out = character())
  }
  write_table(events, file.path(out, "data_mice_logged_events.csv"))
  writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
  if (nrow(events) > 0L) {
    warning("mice 产生 loggedEvents；请结合实际方法和预测矩阵人工检查：", out, call. = FALSE)
  }

  # 先检查所有完成数据；全部通过后才导出五套 CSV。
  completed <- lapply(seq_len(M), function(i) mice::complete(imp, action = i))
  checks <- list()
  for (i in seq_len(M)) {
    z <- completed[[i]]
    if (!identical(dim(z), dim(x)) || !identical(names(z), names(x)) ||
        !identical(rownames(z), rownames(x))) fail("完成数据的行列结构发生变化。")
    for (v in names(x)) {
      observed <- !is.na(x[[v]])
      original_values <- x[[v]][observed]
      new_values <- z[[v]][observed]
      observed_ok <- if (is.factor(x[[v]])) {
        identical(as.character(original_values), as.character(new_values))
      } else isTRUE(all.equal(original_values, new_values, tolerance = 0, check.attributes = FALSE))
      donor_ok <- if (prepared$method[[v]] == "pmm") {
        all(z[[v]][!observed] %in% original_values)
      } else TRUE
      levels_ok <- if (is.factor(x[[v]])) {
        all(as.character(z[[v]]) %in% levels(x[[v]]))
      } else TRUE
      checks[[length(checks) + 1L]] <- data.frame(
        imputation = i, variable = v,
        no_missing = !anyNA(z[[v]]), observed_values_preserved = observed_ok,
        pmm_donor_check = donor_ok, categorical_levels_check = levels_ok
      )
    }
  }
  checks <- do.call(rbind, checks)
  write_table(checks, file.path(out, "completion_checks.csv"))
  if (!all(as.matrix(checks[, 3:6]))) {
    fail("完成数据检查未通过；未导出完成数据 CSV。请检查 completion_checks 和 loggedEvents：", out)
  }
  for (i in seq_len(M)) {
    z <- completed[[i]]
    z[] <- lapply(z, function(v) if (is.factor(v)) as.character(v) else v)
    write_table(z, file.path(out, paste0("data_imputation_", i, ".csv")))
  }
  # 此记录只在未来实际执行且检查通过时生成；不表示已验证收敛或复现稿件数值。
  jsonlite::write_json(list(
    script_status = "new_draft_not_historical_source",
    pool = cfg$pool, seed = cfg$seed, completed_imputations = M, iterations = MAXIT,
    rows = nrow(x), variables = ncol(x),
    excluded_variables = prepared$missingness$variable[prepared$missingness$excluded],
    observed_values_preserved = all(checks$observed_values_preserved),
    pmm_donor_checks_passed = all(checks$pmm_donor_check),
    no_missing_values = all(checks$no_missing), logged_events = nrow(events),
    convergence_assessed = FALSE,
    implementation = list(donors = PMM_DONORS, matchtype = PMM_MATCHTYPE, ridge = RIDGE,
                          remove_constant = TRUE, remove_collinear = TRUE,
                          nnet_MaxNWts = 100000, nnet_maxit = 200),
    inputs_md5 = as.list(tools::md5sum(c(prepared$input_path, prepared$spec_path))),
    mice_version = as.character(utils::packageVersion("mice"))
  ), file.path(out, "completed.json"), pretty = TRUE, auto_unbox = TRUE)
  message("完成：", cfg$pool, "；诊断及人工收敛检查仍需结合 data_mice_object.rds。")
  invisible(imp)
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) < 2L || length(args) > 3L) {
    fail("用法：Rscript run_mice_draft.R <项目根目录> <新的输出根目录> [pool名或all]")
  }
  for (pkg in c("mice", "jsonlite")) {
    if (!requireNamespace(pkg, quietly = TRUE)) fail("请先安装 R 包：", pkg)
  }
  project_root <- normalizePath(args[[1]], mustWork = TRUE)
  input_dir <- file.path(project_root, "manuscript",
                         "new_feature_sets重做（static和dynamic不同时纳入）", "input")
  output_root <- path.expand(args[[2]])
  selected <- if (length(args) == 3L) args[[3]] else "all"
  if (!selected %in% c("all", POOLS$pool)) fail("未知 pool：", selected)
  pools <- if (selected == "all") POOLS else POOLS[POOLS$pool == selected, , drop = FALSE]
  targets <- file.path(output_root, pools$pool)
  if (any(file.exists(targets) | dir.exists(targets))) fail("至少一个 pool 输出目录已存在；请指定新的输出根目录。")
  # 四个输入先全部检查；不进行跨队列合并，也不按结局删行。
  prepared <- lapply(seq_len(nrow(pools)), function(i) prepare_pool(input_dir, pools[i, , drop = FALSE]))
  for (i in seq_len(nrow(pools))) run_pool(prepared[[i]], pools[i, , drop = FALSE], output_root)
  invisible(NULL)
}

# source() 仅加载函数；显式用 Rscript 执行才开始插补。
if (sys.nframe() == 0L) main()
