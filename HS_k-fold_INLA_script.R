# Validation statistics using k fold cross validation
## Model cross validation: k-fold cross validation (5-fold, 10-fold, 20-fold CV)
    #  Correlation coefficient
    #  RMSE

### Install/load packages
packages <- c("tidyverse", "INLA", "INLAutils", "devtools", "sp",
              "raster", "dismo")

# if(length(setdiff(packages, rownames(installed.packages()))) > 0) { 
#   install.packages(setdiff(packages, rownames(installed.packages()))) }
# 
# #For INLA!!
# if(length(setdiff("INLA", rownames(installed.packages()))) > 0){
#   install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
# }
lapply(packages, require, character.only = TRUE)

### Custom functions
# This is for stacking data for INLA
# 1. data = training data; 2. dp = prediction/validation data; 3. cov_list = list of environmental variables 
stack_data <- function(data, dp, cov_list){
  # stack for estimation stk.e
  df <- data.frame(Intercept = 1, subset(data, select = cov_list))
  stk.e <- inla.stack(
    tag = "est",
    data = list(y = data$CASES, numtrials = data$N),
    A = list(1, Ae),
    effects = list(df, spatial.field = iset)
  )
  
  # stack for prediction stk.p
  df_p <- data.frame(Intercept = 1, subset(dp, select = cov_list))
  stk.p <- inla.stack(
    tag = "pred",
    data = list(y = dp$CASES, numtrials = dp$N),
    A = list(1, Ap),
    effects = list(df_p, spatial.field = iset))
  
  # stk.full has stk.e and stk.p
  stk.full <- inla.stack(stk.e, stk.p)
  
  return(stk.full)
}



### Loading the data

load("data/selected_covs_files_1km.RData")
data <- eth_prev2  # main dataset with observation and environmental covariates
data$PREV <- data$CASES/data$N * 100
m <- getData(name = "GADM", country = "ETH", level = 0)

varlist <- c("slope", "isothermality", "precp_seasonality", "NDVI_2003_11")  # list of selected covariates

### Model specs: formula, triangulation mesh and SPDE
formula0 <- as.formula(paste("y ~ 0 + Intercept +", paste(varlist, collapse =  "+"), "+ f(spatial.field, model = spde)"))
data$observed <- data$CASES

# Mesh construction
coords <-  cbind(data$LONG, data$LAT)
bdry <- inla.sp2segment(m)
bdry$loc <- inla.mesh.map(bdry$loc)
mesh1 <- inla.mesh.2d(
  loc = coords, boundary = bdry, 
  max.edge = c(0.5, 5),
  cutoff = 0.03
)

# Spde and spatial field
spde <- inla.spde2.matern(mesh1, alpha=2)
iset <- inla.spde.make.index(name = "spatial.field", spde$n.spde)

### Model validation
# Generates RMSE and correlation ceofficient
# Generates validation data with observed and predicted values

fold <- 2 # specify fold
model <- 2 # specify model
variables = 2
pb = txtProgressBar(min = 0, max = fold, initial = 0) 
output <- matrix(ncol=variables, nrow=fold)
colnames(output) <- c("RMSE", "Correlation-coefficient")
output <- data.frame(output)

set.seed(12345) # for reproducibility of the training and validation sets
kf <- kfold(nrow(data), k = fold) # assigning observations to each folds

old <- Sys.time()
for(i in 1:fold) {
  # fold changes in each loop
  test <- data[kf == i, ] 
  train <- data[kf != i, ]
  test$CASES <- NA  # replace observations of the test dataset with NA
  test_coords <- coords[kf == i,]
  train_coords <- coords[kf != i,]
  Ae <- inla.spde.make.A(mesh=mesh1,loc=as.matrix(train_coords));dim(Ae)
  Ap <- inla.spde.make.A(mesh = mesh1, loc = test_coords);dim(Ap)
  # using custom function to prapred the INLA stack
  stk.full <- stack_data(data = train, dp = test, cov_list = varlist) # 
  inla.setOption(num.threads = 8)
  p.res <- inla(formula0,
            family = "zeroinflated.binomial.1", Ntrials = numtrials,
            data = inla.stack.data(stk.full, spde = spde),
            control.family = list(link = "logit"),
            control.compute = list(dic = TRUE, waic = TRUE,
                                   cpo = TRUE, config = TRUE,
                                   openmp.strategy="huge"),
            control.predictor = list(
              compute = TRUE, link = 1,
              A = inla.stack.A(stk.full)
            )
  )
  index.pred <- inla.stack.index(stk.full, "pred")$data
  obs_prev <- test$PREV  # observed prevalence
  pred_prev <- p.res$summary.fitted.values[index.pred, "mean"] # predicted prevalence
  pred_prev <- pred_prev * 100 
  tmp.sd = p.res$summary.fitted.values[index.pred,"sd"]
  validation = list()
  validation$res = obs_prev - pred_prev  # residuals
  validation$rmse = sqrt(mean(validation$res^2, na.rm=TRUE)) # calculate RMSE
  validation$cor = cor(obs_prev, pred_prev, 
                       use="pairwise.complete.obs",
                       method="spearman") # calculate correlation coefficient
  output[i, ] <- c(validation$rmse, validation$cor)
  
  if(i == 1){
      object_cor <- data.frame(fold = i, pred_prev, obs_prev)
    }
    else{
      object_cor <- bind_rows(object_cor, data.frame(fold = i, pred_prev, obs_prev))
    }
    setTxtProgressBar(pb,i)
    cat("\nIteration = ", i, "\n")
    print(Sys.time() - old)
}

# write.csv(output, paste0("data/validation_stats_", "model_",model, "_", "fold_", fold,".csv"))
# write.csv(object_cor, paste0("data/obs_pred_val_","model_",model, "_", "fold_", fold,".csv"))

val_stats <- data.frame(RMSE = mean(output$RMSE), Corr_coff = mean(output$Correlation.coefficient))
# write.csv(val_stats, "docs/valid_cor.csv")
save(output, object_cor, val_stats, file = "docs/validation_stats.RData")

### OUTPUT/INFO
# This is the script to run k fold cross-validation for different `k`.
# The output that we are interested in are:
# 1. `output` - dataframe with RMSE and R-squared values for each folds
# 2. `object_cor` - dataframe with observed and predicted values for each folds
