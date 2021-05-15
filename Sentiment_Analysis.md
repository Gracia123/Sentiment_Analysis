STAT841\_6.4
================

\#a)

``` r
library(quanteda)
```

    ## Package version: 2.1.2

    ## Parallel computing: 2 of 8 threads used.

    ## See https://quanteda.io for tutorials and examples.

    ## 
    ## Attaching package: 'quanteda'

    ## The following object is masked from 'package:utils':
    ## 
    ##     View

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(e1071)
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ tibble  3.0.5     ✓ purrr   0.3.4
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()
    ## x purrr::lift()   masks caret::lift()

``` r
library(MASS)
```

    ## 
    ## Attaching package: 'MASS'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
set.seed(7419)
filepath <- "/Users/graciagodwin/Desktop/sentiment labelled sentences"
data<-read.table(paste0(filepath,"/amazon_cells_labelled.txt"), sep="\t", 
                 header=FALSE, stringsAsFactor = FALSE)
colnames(data) <- c('Text','Review')
text<-data$Text
toks <- tokens(x=text,what="word",remove_punct=TRUE,remove_symbols=TRUE,
               remove_numbers=TRUE,remove_url=TRUE)
toks<-tokens_tolower(toks)
toks<-tokens_wordstem(toks)
sw <- stopwords("english")
toks<-tokens_remove(toks, sw)
dtm <- dfm(toks,tolower=TRUE,stem=TRUE,remove=sw)
sparsity(dtm)
```

    ## [1] 0.9914446

``` r
summary(colSums(dtm))
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   1.000   1.000   3.843   3.000 175.000

``` r
tf_total <- colSums(dtm)
to_keep <- which(tf_total > median(tf_total))
dtm_reduced <- dfm_keep(dtm, pattern = names(to_keep), valuetype = "fixed", 
                        verbose = TRUE)
```

    ## kept 643 features

    ## 

``` r
dfreq <- apply(dtm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   2.000   3.000   5.865   6.000  87.000

``` r
#Threshold = 5
dtm_5<-dfm_trim(dtm_reduced, min_docfreq = 5)
df<-convert(dtm_5, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count) <- paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count) <- paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator) <- paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator) <- paste(colnames(df_test_indicator), "_c", sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred1<-round(count_pred)
pred1<-as.factor(pred1)
result1<-as.factor(df_test_count$label_c)
count_cm1<-confusionMatrix(pred1,result1)
print(paste0("Random forest - Count N-grams - Threshold = 5 - Accuracy = ",
             count_cm1$overall[1]))
```

    ## [1] "Random forest - Count N-grams - Threshold = 5 - Accuracy = 0.663101604278075"

``` r
pred2<-round(indicator_pred)
pred2<-as.factor(pred2)
result2<-as.factor(df_test_indicator$label_c)
indicator_cm1<-confusionMatrix(pred2,result2)
print(paste0("Random forest - Indicator N-grams - Threshold = 5 - Accuracy = ",indicator_cm1$overall[1]))
```

    ## [1] "Random forest - Indicator N-grams - Threshold = 5 - Accuracy = 0.679144385026738"

*Comments:* From the above accuracies we can observe that for both the
N-gram count and N-gram indicators models, their accuracies only vary by
1% with indicator performing better than count. This could be attributed
to the fact that, since each unigram occurs with a low frequency, the
model is unable to distinguish between common and rare words (due to
threshold = 5), for N-gram count. For N-gram indicator, it doesn’t have
to worry about this as it only consists of 1’s and 0’s. Hence, the
indicator N-gram model performs very slightly better than the count
N-gram model.

\#b)

``` r
#Threshold = 3
dtm_3<-dfm_trim(dtm_reduced, min_docfreq = 3)
df<-convert(dtm_3, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count)<-paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count)<-paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator)<-paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator)<-paste(colnames(df_test_indicator),"_c",sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred3<-round(count_pred)
pred3<-as.factor(pred3)
result3<-as.factor(df_test_count$label_c)
count_cm2<-confusionMatrix(pred3,result3)
print(paste0("Random forest - Count N-grams - Threshold = 3 - Accuracy = ",
             count_cm2$overall[1]))
```

    ## [1] "Random forest - Count N-grams - Threshold = 3 - Accuracy = 0.668449197860963"

``` r
pred4<-round(indicator_pred)
pred4<-as.factor(pred4)
result4<-as.factor(df_test_indicator$label_c)
indicator_cm2<-confusionMatrix(pred4,result4)
print(paste0("Random forest - Indicator N-grams - Threshold = 3 - Accuracy = ",indicator_cm2$overall[1]))
```

    ## [1] "Random forest - Indicator N-grams - Threshold = 3 - Accuracy = 0.684491978609626"

``` r
#Threshold = 7
dtm_7<-dfm_trim(dtm_reduced, min_docfreq = 7)
df<-convert(dtm_7, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count)<-paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count)<-paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator)<-paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator)<-paste(colnames(df_test_indicator),"_c",sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred5<-round(count_pred)
pred5<-as.factor(pred5)
result5<-as.factor(df_test_count$label_c)
count_cm3<-confusionMatrix(pred5,result5)
print(paste0("Random forest - Count N-grams - Threshold = 7 - Accuracy = ",
             count_cm3$overall[1]))
```

    ## [1] "Random forest - Count N-grams - Threshold = 7 - Accuracy = 0.625668449197861"

``` r
pred6<-round(indicator_pred)
pred6<-as.factor(pred6)
result6<-as.factor(df_test_indicator$label_c)
indicator_cm3<-confusionMatrix(pred6,result6)
print(paste0("Random forest - Indicator N-grams - Threshold = 7 - Accuracy = ",
             indicator_cm3$overall[1]))
```

    ## [1] "Random forest - Indicator N-grams - Threshold = 7 - Accuracy = 0.625668449197861"

``` r
#Threshold = 9
dtm_9<-dfm_trim(dtm_reduced, min_docfreq = 9)
df<-convert(dtm_9, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count)<-paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count)<-paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator)<-paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator)<-paste(colnames(df_test_indicator),"_c",sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred7<-round(count_pred)
pred7<-as.factor(pred7)
result7<-as.factor(df_test_count$label_c)
count_cm4<-confusionMatrix(pred7,result7)
print(paste0("Random forest - Count N-grams - Threshold = 9 - Accuracy = ",
             count_cm4$overall[1]))
```

    ## [1] "Random forest - Count N-grams - Threshold = 9 - Accuracy = 0.593582887700535"

``` r
pred8<-round(indicator_pred)
pred8<-as.factor(pred8)
result8<-as.factor(df_test_indicator$label_c)
indicator_cm4<-confusionMatrix(pred8,result8)
print(paste0("Random forest - Indicator N-grams - Threshold = 9 - Accuracy = ",indicator_cm4$overall[1]))
```

    ## [1] "Random forest - Indicator N-grams - Threshold = 9 - Accuracy = 0.598930481283422"

``` r
#install.packages("knitr")
library(knitr)
Table1<-data.frame(Threshold=c(3,5,7,9),
                   Accuracy_count=c(count_cm2$overall[1],count_cm1$overall[1],
                                    count_cm3$overall[1],count_cm4$overall[1]),
                   Accuracy_indicator=c(indicator_cm2$overall[1],
                                        indicator_cm1$overall[1],
                                        indicator_cm3$overall[1],
                                        indicator_cm4$overall[1])
)
kable(Table1)
```

| Threshold | Accuracy\_count | Accuracy\_indicator |
|----------:|----------------:|--------------------:|
|         3 |       0.6684492 |           0.6844920 |
|         5 |       0.6631016 |           0.6791444 |
|         7 |       0.6256684 |           0.6256684 |
|         9 |       0.5935829 |           0.5989305 |

\#c)

``` r
#With stem + without stop words
print(paste0("Count N-grams - With stemming, without stop words - Accuracy= ",
             count_cm1$overall[1]))
```

    ## [1] "Count N-grams - With stemming, without stop words - Accuracy= 0.663101604278075"

``` r
print(paste0("Indicator N-grams - With stemming, without stop words - Accuracy=",
             indicator_cm1$overall[1]))
```

    ## [1] "Indicator N-grams - With stemming, without stop words - Accuracy=0.679144385026738"

``` r
#With stem + with stop words
toks <- tokens(x=text,what="word",remove_punct=TRUE,remove_symbols=TRUE,
               remove_numbers=TRUE,remove_url=TRUE)
toks<-tokens_tolower(toks)
toks<-tokens_wordstem(toks)
#sw <- stopwords("english")
#toks<-tokens_remove(toks, sw)
dtm <- dfm(toks,tolower=TRUE,stem=TRUE)
sparsity(dtm)
```

    ## [1] 0.9876043

``` r
summary(colSums(dtm))
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   1.000   1.000   6.666   4.000 514.000

``` r
tf_total <- colSums(dtm)
to_keep <- which(tf_total > median(tf_total))
dtm_reduced <- dfm_keep(dtm, pattern = names(to_keep), valuetype = "fixed", 
                        verbose = TRUE)
```

    ## kept 733 features

``` r
dfreq <- apply(dtm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    1.00    2.00    4.00    8.55    8.00  174.00

``` r
#Threshold = 5
dtm_5<-dfm_trim(dtm_reduced, min_docfreq = 5)
df<-convert(dtm_5, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count) <- paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count) <- paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator) <- paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator) <- paste(colnames(df_test_indicator), "_c", sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred9<-round(count_pred)
pred9<-as.factor(pred9)
result9<-as.factor(df_test_count$label_c)
count_cm5<-confusionMatrix(pred9,result9)
print(paste0("Count N-grams - With stemming, with stop words - Accuracy= ",
             count_cm5$overall[1]))
```

    ## [1] "Count N-grams - With stemming, with stop words - Accuracy= 0.647058823529412"

``` r
pred10<-round(indicator_pred)
pred10<-as.factor(pred10)
result10<-as.factor(df_test_indicator$label_c)
indicator_cm5<-confusionMatrix(pred10,result10)
print(paste0("Count N-grams - With stemming, with stop words - Accuracy= ", 
             indicator_cm5$overall[1]))
```

    ## [1] "Count N-grams - With stemming, with stop words - Accuracy= 0.657754010695187"

``` r
#Without stem + without stop words
toks <- tokens(x=text,what="word",remove_punct=TRUE,remove_symbols=TRUE,
               remove_numbers=TRUE,remove_url=TRUE)
toks<-tokens_tolower(toks)
#toks<-tokens_wordstem(toks)
sw <- stopwords("english")
toks<-tokens_remove(toks, sw)
dtm <- dfm(toks,tolower=TRUE,remove=sw)
sparsity(dtm)
```

    ## [1] 0.9930643

``` r
summary(colSums(dtm))
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   1.000   1.000   3.005   2.000 162.000

``` r
tf_total <- colSums(dtm)
to_keep <- which(tf_total > median(tf_total))
dtm_reduced <- dfm_keep(dtm, pattern = names(to_keep), valuetype = "fixed", 
                        verbose = TRUE)
```

    ## kept 683 features

``` r
dfreq <- apply(dtm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   2.000   3.000   5.098   5.000  85.000

``` r
#Threshold = 5
dtm_5<-dfm_trim(dtm_reduced, min_docfreq = 5)
df<-convert(dtm_5, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count) <- paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count) <- paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator) <- paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator) <- paste(colnames(df_test_indicator), "_c", sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred11<-round(count_pred)
pred11<-as.factor(pred11)
result11<-as.factor(df_test_count$label_c)
count_cm6<-confusionMatrix(pred11,result11)
print(paste0("Count N-grams - Without stemming,without stop words - Accuracy= ",
             count_cm6$overall[1]))
```

    ## [1] "Count N-grams - Without stemming,without stop words - Accuracy= 0.6524064171123"

``` r
pred12<-round(indicator_pred)
pred12<-as.factor(pred12)
result12<-as.factor(df_test_indicator$label_c)
indicator_cm6<-confusionMatrix(pred12,result12)
print(paste0("Count N-grams - Without stemming,without stop words - Accuracy= ", 
             indicator_cm6$overall[1]))
```

    ## [1] "Count N-grams - Without stemming,without stop words - Accuracy= 0.631016042780749"

``` r
#Without stem + with stop words
toks <- tokens(x=text,what="word",remove_punct=TRUE,remove_symbols=TRUE,
               remove_numbers=TRUE,remove_url=TRUE)
toks<-tokens_tolower(toks)
#toks<-tokens_wordstem(toks)
#sw <- stopwords("english")
#toks<-tokens_remove(toks, sw)
dtm <- dfm(toks,tolower=TRUE)
sparsity(dtm)
```

    ## [1] 0.9896817

``` r
summary(colSums(dtm))
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   1.000   1.000   5.416   3.000 514.000

``` r
tf_total <- colSums(dtm)
to_keep <- which(tf_total > median(tf_total))
dtm_reduced <- dfm_keep(dtm, pattern = names(to_keep), valuetype = "fixed", 
                        verbose = TRUE)
```

    ## kept 789 features

``` r
dfreq <- apply(dtm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   2.000   3.000   7.787   7.000 174.000

``` r
#Threshold = 5
dtm_5<-dfm_trim(dtm_reduced, min_docfreq = 5)
df<-convert(dtm_5, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]
colnames(df_train_count) <- paste(colnames(df_train_count), "_c", sep = "")
colnames(df_test_count) <- paste(colnames(df_test_count), "_c", sep = "")
colnames(df_train_indicator) <- paste(colnames(df_train_indicator), "_c",sep="")
colnames(df_test_indicator) <- paste(colnames(df_test_indicator), "_c", sep="")
rf_count <- randomForest(df_train_count$label_c~., data=df_train_count)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
rf_indicator<-randomForest(df_train_indicator$label_c~.,data=df_train_indicator)
```

    ## Warning in randomForest.default(m, y, ...): The response has five or fewer
    ## unique values. Are you sure you want to do regression?

``` r
count_pred<- predict(rf_count, df_test_count)
indicator_pred<- predict(rf_indicator, df_test_indicator)
pred13<-round(count_pred)
pred13<-as.factor(pred13)
result13<-as.factor(df_test_count$label_c)
count_cm7<-confusionMatrix(pred13,result13)
print(paste0("Count N-grams - Without stemming,with stop words - Accuracy= ",
             count_cm7$overall[1]))
```

    ## [1] "Count N-grams - Without stemming,with stop words - Accuracy= 0.679144385026738"

``` r
pred14<-round(indicator_pred)
pred14<-as.factor(pred14)
result14<-as.factor(df_test_indicator$label_c)
indicator_cm7<-confusionMatrix(pred14,result14)
print(paste0("Count N-grams - Without stemming,with stop words - Accuracy= ", 
             indicator_cm7$overall[1]))
```

    ## [1] "Count N-grams - Without stemming,with stop words - Accuracy= 0.700534759358289"

``` r
Table2<-data.frame(Condition=c('With stem + without stop words',
                               'With stem + with stop words',
                               'Without stem + without stop words',
                               'Without stem + with stop words'),
                   Accuracy_count=c(count_cm1$overall[1],count_cm5$overall[1],
                                    count_cm6$overall[1],count_cm7$overall[1]),
                   Accuracy_indicator=c(indicator_cm1$overall[1],
                                        indicator_cm5$overall[1],
                                        indicator_cm6$overall[1],
                                        indicator_cm7$overall[1])
)
kable(Table2)
```

| Condition                         | Accuracy\_count | Accuracy\_indicator |
|:----------------------------------|----------------:|--------------------:|
| With stem + without stop words    |       0.6631016 |           0.6791444 |
| With stem + with stop words       |       0.6470588 |           0.6577540 |
| Without stem + without stop words |       0.6524064 |           0.6310160 |
| Without stem + with stop words    |       0.6791444 |           0.7005348 |

\#d)

``` r
library(glmnet)
```

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## Loaded glmnet 4.1

``` r
text<-data$Text
toks <- tokens(x=text,what="word",remove_punct=TRUE,remove_symbols=TRUE,
               remove_numbers=TRUE,remove_url=TRUE)
toks<-tokens_tolower(toks)
toks<-tokens_wordstem(toks)
sw <- stopwords("english")
toks<-tokens_remove(toks, sw)
dtm <- dfm(toks,tolower=TRUE,stem=TRUE,remove=sw)
sparsity(dtm)
```

    ## [1] 0.9914446

``` r
summary(colSums(dtm))
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   1.000   1.000   3.843   3.000 175.000

``` r
tf_total <- colSums(dtm)
to_keep <- which(tf_total > median(tf_total))
dtm_reduced <- dfm_keep(dtm, pattern = names(to_keep), valuetype = "fixed", 
                        verbose = TRUE)
```

    ## kept 643 features

    ## 

``` r
dfreq <- apply(dtm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   2.000   3.000   5.865   6.000  87.000

``` r
#Threshold = 5
dtm_5_lr<-dfm_trim(dtm_reduced, min_docfreq = 5)
df<-convert(dtm_5_lr, to = "data.frame")
df<-subset(df,select=-c(doc_id))
df$num_words <- rowSums(df)
df_indicator<-df
df_indicator[,-which(names(df_indicator) %in% c('num_words'))]<-
  ifelse(df_indicator[,-which(names(df_indicator) %in% c('num_words'))]>=1,1,0)
df<-cbind(df,label=data$Review)
smp_siz <- floor(0.5*nrow(df))
index <- sample(seq_len(nrow(df)),size = smp_siz)
df_train_count <- df[index, ]
df_test_count <- df[-index, ]
df_indicator<-cbind(df_indicator,label=data$Review)
df_train_indicator<-df_indicator[index,]
df_test_indicator<-df_indicator[-index,]

lr_count<-glm(df_train_count$label~.,data=df_train_count,family="binomial")
```

    ## Warning: glm.fit: algorithm did not converge

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
lr_indicator<-glm(df_train_indicator$label~.,data=df_train_indicator,
                  family="binomial")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
count_pred1<- predict(lr_count, df_test_count)
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
count_pred1 <- ifelse(count_pred1 >= 0.5, 1, 0)
indicator_pred1<- predict(lr_indicator, df_test_indicator)
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
indicator_pred1<- ifelse(indicator_pred1 <= 0.5, 1, 0)

pred15<-round(count_pred1)
pred15<-as.factor(pred15)
result15<-as.factor(df_test_count$label)
count_cm8<-confusionMatrix(pred15,result15)
print(paste0("Logistic regression - Count N-grams - Threshold = 5 - Accuracy= ",
             count_cm8$overall[1]))
```

    ## [1] "Logistic regression - Count N-grams - Threshold = 5 - Accuracy= 0.620320855614973"

``` r
pred16<-round(indicator_pred1)
pred16<-as.factor(pred16)
result16<-as.factor(df_test_indicator$label)
indicator_cm8<-confusionMatrix(pred16,result16)
print(paste0("Logistic regression - Indicator N-grams - Threshold = 5 - Accuracy = ",indicator_cm8$overall[1]))
```

    ## [1] "Logistic regression - Indicator N-grams - Threshold = 5 - Accuracy = 0.39572192513369"

``` r
Table3<-data.frame(LR_accuracy_count=c(count_cm8$overall[1]),
                   LR_accuracy_indicator=c(indicator_cm8$overall[1])
)
kable(Table3)
```

|          | LR\_accuracy\_count | LR\_accuracy\_indicator |
|:---------|--------------------:|------------------------:|
| Accuracy |           0.6203209 |               0.3957219 |
