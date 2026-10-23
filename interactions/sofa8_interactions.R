# Reconstructed 2026-10-21 from the retained steroid interaction code and final Table S6b specification.
# Original cluster-robust Rubin Wald and 3-df natural-spline sensitivity.
# VCOV below computes sandwich::vcovCL(type='HC0', cadjust=TRUE) explicitly.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
source <- normalizePath(args[[1]],mustWork=TRUE)
out <- args[[2]]
endpoint <- args[[3]]
stopifnot(endpoint %in% c('hospital_expire_flag','icu_death'))
for (d in c('tables','qa','logs','code')) dir.create(file.path(out,d),recursive=TRUE,showWarnings=FALSE)
clinical <- c('sex','age','race','admission_type','hypertension','heart_failure','malignancy','charlson_index','bmi','smoking','alcohol','infection_site','crrt_used','vasoactive_used','advanced_Resp_support','alb_baseline','alt_baseline','bun_baseline','inr_baseline','glucose_baseline','hgb_baseline','lactate_baseline','wbc_baseline','aki_onset_by72h','balance_24h_baseline','MAP_median_baseline','HR_median_baseline','SpO2_median_baseline','Temp_max_baseline')
categorical <- c('sex','race','admission_type','smoking','alcohol','infection_site','aki_onset_by72h')
sets <- list(baseline=list(sofa=c('sofa2_total_max_baseline','sofa_TWA','sofa_slope','sofa_variability_SD'),nlr_static='logNLR_baseline',nlr=c('logNLR_TWA','logNLR_slope','logNLR_variability_SD')),latest=list(sofa=c('sofa_t_latest','sofa_TWA_t_latest','sofa_slope_t_latest','sofa_SD_t_latest'),nlr_static='logNLR_t_latest',nlr=c('logNLR_TWA_t_latest','logNLR_slope_t_latest','logNLR_SD_t_latest')))
vcov_cluster <- function(fit,ids) {
 x <- model.matrix(fit);keep <- !is.na(coef(fit));x <- x[,keep,drop=FALSE]
 w <- fit$weights;scores <- x*as.numeric(residuals(fit,type='response'))
 sums <- rowsum(scores,as.character(ids));g <- nrow(sums)
 bread <- summary(fit)$cov.unscaled;bread <- bread[colnames(x),colnames(x),drop=FALSE]
 v <- bread %*% crossprod(sums) %*% bread*g/(g-1)
 dimnames(v) <- list(colnames(x),colnames(x));v
}
pool <- function(fits,ds) {
 terms <- Reduce(intersect,lapply(fits,function(f)names(coef(f))[is.finite(coef(f))]))
 estimates <- sapply(fits,function(f)coef(f)[terms]);ubar <- Reduce('+',lapply(seq_along(fits),function(i)vcov_cluster(fits[[i]],ds[[i]]$group_id)[terms,terms,drop=FALSE]))/length(fits)
 list(beta=rowMeans(estimates),vcov=ubar+(1+1/length(fits))*cov(t(estimates)),terms=terms)
}
tests <- list();coefficients <- list();diagnostics <- list();formulae <- character();counter <- 0;warning_rows <- list()
for (family in c('baseline')) for(cohort in c('mimic','eicu')) {
 raw <- read.csv(file.path(source,'input',paste0(cohort,'_analysis_raw.csv')),check.names=FALSE,stringsAsFactors=FALSE,colClasses=c(patient_key='character',group_id='character'))
 ds <- lapply(1:5,function(i) {
  d <- read.csv(file.path(source,'imputations',paste0(family,'_',cohort),paste0('data_imputation_',i,'.csv')),check.names=FALSE,stringsAsFactors=FALSE)
  stopifnot(nrow(d)==nrow(raw));d$group_id <- raw$group_id;d$sofa8 <- as.integer(raw$sofa2_total_max_baseline > 8)
  for(v in categorical) d[[v]] <- factor(d[[v]])
  for(v in c(sets[[family]]$sofa,sets[[family]]$nlr_static,sets[[family]]$nlr)) {
   mu <- mean(raw[[v]],na.rm=TRUE);sd <- sd(raw[[v]],na.rm=TRUE);stopifnot(sd>0)
   d[[v]] <- (d[[v]]-mu)/sd
  }
  for(y in c(endpoint)) d[[y]] <- raw[[y]]
  d
 })
 for(outcome in c(endpoint)) for(scope in c('main')) for(specification in c('linear','natural_spline_3df')) {
  s <- sets[[family]];nlr_terms <- if(specification=='linear')s$nlr else sprintf('splines::ns(%s, df=3)',s$nlr)
  rhs <- c('cortisone_used','sofa8',s$sofa,s$nlr_static,nlr_terms,paste0('sofa8:',nlr_terms),if(scope=='clinical')clinical)
  formula <- as.formula(paste(outcome,'~',paste(rhs,collapse='+')))
  fits <- lapply(seq_along(ds),function(i)withCallingHandlers(glm(formula,data=ds[[i]],family=binomial()),warning=function(w) {
   warning_rows[[length(warning_rows)+1]] <<- data.frame(family=family,cohort=cohort,outcome=outcome,specification=specification,imputation=i,message=conditionMessage(w))
   invokeRestart('muffleWarning')
  }))
  p <- pool(fits,ds);positions <- grep('^sofa8:',p$terms)
  stopifnot(length(positions)==if(specification=='linear')3 else 9)
  v <- p$vcov[positions,positions,drop=FALSE];q <- p$beta[positions];ev <- eigen((v+t(v))/2,symmetric=TRUE,only.values=TRUE)$values;stopifnot(all(is.finite(v)),min(ev)>0)
  chi <- as.numeric(t(q)%*%solve(v,q));stopifnot(is.finite(chi),chi>=0)
  counter <- counter+1
  tests[[counter]] <- data.frame(family=family,cohort=cohort,outcome=outcome,scope=scope,specification=specification,statistic=chi,df=length(positions),p_value=pchisq(chi,length(positions),lower.tail=FALSE),covariance_min_eigenvalue=min(ev),covariance_condition=max(ev)/min(ev),test='Patient cluster HC0 Rubin Wald chi-square')
  se <- sqrt(diag(p$vcov));coefficients[[counter]] <- data.frame(family=family,cohort=cohort,outcome=outcome,scope=scope,specification=specification,term=p$terms,estimate=p$beta,standard_error=se,OR=exp(p$beta),CI_low=exp(p$beta-qnorm(.975)*se),CI_high=exp(p$beta+qnorm(.975)*se),p_value=2*pnorm(-abs(p$beta/se)))
  diagnostics[[counter]] <- do.call(rbind,lapply(seq_along(fits),function(i)data.frame(family=family,cohort=cohort,outcome=outcome,scope=scope,specification=specification,imputation=i,n=nobs(fits[[i]]),converged=fits[[i]]$converged,aliased=sum(!is.finite(coef(fits[[i]]))))))
  formulae <- c(formulae,paste(family,cohort,outcome,scope,specification,paste(deparse(formula),collapse=' '),sep=' | '))
  cat('INTERACTION',family,cohort,outcome,scope,specification,'P=',tests[[counter]]$p_value,'\n')
 }
}
write.csv(do.call(rbind,tests),file.path(out,'tables','sofa8_interaction_joint_tests.csv'),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(out,'tables','sofa8_interaction_coefficients.csv'),row.names=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(out,'qa','interaction_model_diagnostics.csv'),row.names=FALSE)
writeLines(formulae,file.path(out,'code','interaction_formulae.txt'))
writeLines(capture.output(sessionInfo()),file.path(out,'logs','interaction_R_session.txt'))
write.csv(if(length(warning_rows))do.call(rbind,warning_rows) else data.frame(message=character()),file.path(out,'qa','interaction_model_warnings.csv'),row.names=FALSE)
stopifnot(all(do.call(rbind,diagnostics)$converged),all(do.call(rbind,diagnostics)$aliased==0))
