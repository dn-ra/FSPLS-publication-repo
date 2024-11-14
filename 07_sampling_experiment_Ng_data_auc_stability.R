library(dplyr)
library(ROCR)
library(ggplot2)
library(glmnet)
library(lars)
library(pROC)

source('~/git/fspls/fspls_lachlan/fspls.R')
pv_thresh = 0.01
refit=FALSE


ng_data <- readRDS('/data/gpfs/projects/punim1597/Projects/fspls_benchmarking/publication_code/input/ng_data/ng_data.prepd.Rds')

ng_X <- ng_data$data
ng_y <- ng_data$y

set.seed(42)
ng_train_idx <- caret::createDataPartition(y = ng_y, p = 0.8, list = F)


#X variances
vars_X <- ng_X %>% log1p() %>% {apply(., 2, var)} %>% sort(decreasing = T)

select_vars <- vars_X[1:10000]

train_ng_unnormalised <- ng_X[ng_train_idx, ] 
ng_million_lib_sizes_train <- apply(train_ng_unnormalised, 1, sum) / 1e6
ng_tpms_train <- sweep(train_ng_unnormalised, 1, STATS = ng_million_lib_sizes_train, FUN = '/')

test_ng_unnormalised <- ng_X[-ng_train_idx, ]
ng_million_lib_sizes_test <- apply(test_ng_unnormalised, 1, sum) / 1e6
ng_tpms_test <- sweep(test_ng_unnormalised, 1, STATS = ng_million_lib_sizes_test, FUN = '/')
ng_tpms_test_trimmed <- ng_tpms_test[,names(select_vars)]

train_ng_y <- ng_y[ng_train_idx]
test_ng_y <- ng_y[-ng_train_idx]

#number of iterations at each level
n_iter = 20
samp_sizes = round(seq(0,1, 0.05) * length(train_ng_y))[-1]

results_table_fspls <- data.frame()
results_table_stagewise <- data.frame()

#for each sample size
for (i in samp_sizes) {
  print(paste('running iteration of sample size',i))
  #sample training data n_iter times
  samp_size_aucs <- sapply(1:n_iter, function(iter_num) {
    samp_i <- sample.int(nrow(ng_tpms_train), size = i, replace = F)
    
    
    trainx <- ng_tpms_train[samp_i,names(select_vars)] %>% log1p()
    trainy <- train_ng_y[samp_i]
    
    fspls_fold_data <- list(data = as.matrix(trainx), y = as.matrix(trainy))
    fspls_test_data <- list(data = as.matrix(ng_tpms_test_trimmed), y = as.matrix(test_ng_y))
    
    #train model
    result_fspls = tryCatch({
      fspls_model_ng <- trainModel(trainOriginal = fspls_fold_data, pv_thresh = pv_thresh, testOriginal = 
                                      fspls_test_data, refit =refit, max = 10)
      test_auc <- tail(fspls_model_ng$eval, n = 1)[8]
      #store number of variables in the name of the auc output
      names(test_auc) <- length(fspls_model_ng$variables)
      test_auc
    }, error = function(error_condition) {
      print(error_condition)
      NA
    })
    
    result_stagewise = tryCatch({
      
      fs <- lars(as.matrix(trainx), as.matrix(trainy), type = 'forward.stagewise', max.steps = as.numeric(names(result_fspls)), normalize = F)
      test.met <- predict(fs, as.matrix(ng_tpms_test_trimmed))
      stagewise_auc <-  pROC::auc(test_ng_y, test.met$fit[,as.numeric(names(result_fspls))+1])
      stagewise_auc[1]
      
    }, error = function(error_condition) {
      print(error_condition)
      NA
    })
    
    c(fspls = result_fspls, stagewise = result_stagewise)
  })
  #bind results
  results_table_fspls <- rbind(results_table_fspls, samp_size_aucs[1,])
  results_table_stagewise <- rbind(results_table_stagewise, samp_size_aucs[2,])
}


saveRDS(list(fspls = results_table_fspls, stagewise = results_table_stagewise),'ng_downsample_experiment_aucs.Rds')
