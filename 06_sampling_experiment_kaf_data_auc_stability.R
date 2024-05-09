library(dplyr)
library(ROCR)
library(ggplot2)
library(glmnet)
library(lars)
library(pROC)

source('~/git/fspls/fspls_lachlan/fspls.R')
pv_thresh = 0.01
refit=FALSE


kaf_data <- readRDS('../input/kaforou_data/kaforou_data.prepd.Rds')

kaf_X <- kaf_data$data[kaf_data$TB_idx,]
kaf_y <- kaf_data$y[kaf_data$TB_idx]

set.seed(42)
kaf_train_idx <- caret::createDataPartition(y = kaf_y, p = 0.8, list = F)

kaf_y <- as.factor(kaf_y) %>% as.numeric()-1

#filter variables. Choosing top 10000 features correlated with outcome
var_sigs <- apply(kaf_X[kaf_train_idx,], 2, function(x) glm(kaf_y[kaf_train_idx] ~ x) %>% {summary(.)$coefficients[2,4] })

select_vars <- sort(var_sigs)[1:1000]

train_kaf <- kaf_X[kaf_train_idx, names(select_vars)] 
test_kaf <- kaf_X[-kaf_train_idx, names(select_vars)] %>% scale()

train_kaf_y <- kaf_y[kaf_train_idx]
test_kaf_y <- kaf_y[-kaf_train_idx]

#number of iterations at each level
n_iter = 20
samp_sizes = round(seq(0,1, 0.05) * length(train_kaf_y))[-1]



results_table_fspls <- data.frame()
results_table_stagewise <- data.frame()

#for each sample size
for (i in samp_sizes) {
  print(paste('running iteration of sample size',i))
  #sample training data n_iter times
  samp_size_aucs <- sapply(1:n_iter, function(iter_num) {
    samp_i <- sample.int(nrow(train_kaf), size = i, replace = F)
    trainx <- train_kaf[samp_i,] %>% scale()
    trainy <- train_kaf_y[samp_i]

    fspls_fold_data <- list(data = as.matrix(trainx), y = as.matrix(trainy))
    fspls_test_data <- list(data = as.matrix(test_kaf), y = as.matrix(test_kaf_y))

    #train model
    result_fspls = tryCatch({
      fspls_model_kaf <- trainModel(trainOriginal = fspls_fold_data, pv_thresh = pv_thresh, testOriginal = 
                                  fspls_test_data, refit =refit, max = 10)
      test_auc <- tail(fspls_model_kaf$eval, n = 1)[8]
      #store number of variables in the name of the auc output
      names(test_auc) <- length(fspls_model_kaf$variables)
      test_auc
    }, error = function(error_condition) {
      NA
    })
    
    result_stagewise = tryCatch({
           
      fs <- lars(as.matrix(trainx), as.matrix(trainy), type = 'forward.stagewise', max.steps = as.numeric(names(result_fspls)), normalize = F)
      test.met <- predict(fs, as.matrix(test_kaf))
      stagewise_auc <-  pROC::auc(test_kaf_y, test.met$fit[,as.numeric(names(result_fspls))+1])
      stagewise_auc[1]
      
    }, error = function(error_condition) {
      NA
    })
      
    c(fspls = result_fspls, stagewise = result_stagewise)
    })
  #bind results
results_table_fspls <- rbind(results_table_fspls, samp_size_aucs[1,])
results_table_stagewise <- rbind(results_table_stagewise, samp_size_aucs[2,])
}


saveRDS(list(fspls = results_table_fspls, stagewise = results_table_stagewise),'output/kaf_downsample_experiment_aucs.Rds')
