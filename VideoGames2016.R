###################################
# Preparation & Set up environment
###################################

# Load and install necessary packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")

# Download csv file from Github
mydata <- read.csv("https://raw.githubusercontent.com/IvyYW/Video_Games_2016/master/Video_Games_2016.csv",header=TRUE)

# Drop NA, unused data and convert year to numeric
mydata_clean <- mydata %>% drop_na() %>% 
    mutate(Year_of_Release = 1979 + as.numeric(Year_of_Release), User_Score = as.numeric(as.character(User_Score))) %>%
    select(-NA_Sales, -EU_Sales, -JP_Sales, -Other_Sales, -Publisher, -Developer) %>% 
    filter(Year_of_Release %in% c(1996:2016)) 

# Create Validation Set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = mydata_clean$User_Score, times = 1, p = 0.1, list = FALSE)
VideoGames <- mydata_clean[-test_index,]
validation <- mydata_clean[test_index,]

#Create train and test sets
set.seed(66, sample.kind="Rounding")

test_index <- createDataPartition(y = VideoGames$User_Score, times = 1, p = 0.1, list = FALSE)
train_set<- VideoGames[-test_index,]
test_set <- VideoGames[test_index,]

# Computate the RMSE function
RMSE <- function(true_rating, pred_rating){
  sqrt(mean((true_rating - pred_rating)^2))
}

############################
# Data Exploration
###########################

#First look at the dataset
summary(train_set)
head(train_set)
hist(train_set$User_Score)

#Look at coeficient & p-value of variables 
train_fit <- train_set %>% select(-Name) 
fit <- lm (User_Score~. , train_fit)
summary(fit)

#Look at coeficient & p-value of numeric variables
fit <- lm(User_Score~Year_of_Release + Global_Sales + Critic_Score + Critic_Count + User_Count , train_fit)
summary(fit)

# Bloxplot Platfrom v User_Score
Platform_Box <- train_set %>% ggplot(aes(x=User_Score, y = Platform)) + 
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) + 
  stat_boxplot(geom = 'errorbar') 

# Boxplot Rating v User_Score
Rating_Box <- train_set %>% ggplot(aes(x=User_Score, y = Rating)) + 
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) + 
  stat_boxplot(geom = 'errorbar') 

# Boxplot Rating v User_Score
Genre_Box <- train_set %>% 
  ggplot(aes(x = Genre, y = User_Score)) +
  geom_boxplot(aes(color = Genre), show.legend = FALSE) 

# Line Year v Average User_Score
Year_Line <- train_set %>% group_by(Year_of_Release) %>% 
  summarise(avg_score = mean(User_Score)) %>%
  ggplot(aes(x=Year_of_Release, y = avg_score)) + 
  geom_line() 

User_Count_point <- train_set %>% ggplot(aes(x = User_Count, y = User_Score)) +
  geom_point() +
  geom_smooth(method = "lm", span = 0.3)

Critic_Score_point <- train_set %>% ggplot(aes(x = Critic_Score, y = User_Score)) +
  geom_point() +
  geom_smooth(method = "lm", span = 0.3)


##################################
# Model 1: Average Sales -> Y = mu
##################################

# compute average User_Score
mu <- mean(train_set$User_Score)
# compute predicted value
pred_1 <- rep(mu, nrow(test_set))
# compute RMSE
rmse_1 <- RMSE(test_set$User_Score, pred_1)
# add value to results table
rmse_results <- tibble(method = "Average Score", RMSE = rmse_1)

#####################################################
# Model 2: Average Sales + Platform -> Y = mu + b_p
#####################################################

#compute b_p: platfrom effect
Platform_effect <- train_set %>% group_by(Platform) %>% 
  summarise(b_p = mean(User_Score) - mu) 
train_effect <- train_set %>% left_join(Platform_effect, by = "Platform")

# compute predicted value
pred_2 <- test_set %>% mutate(mu = mu) %>% 
  left_join(Platform_effect, by = "Platform") %>%
  mutate(pred = mu + b_p) %>% pull(pred)

#comput RMSE and update result table
rmse_2 <- RMSE(test_set$User_Score, pred_2)
rmse_results <- bind_rows(rmse_results, tibble(method="Platform Effect", RMSE = rmse_2))
rmse_results %>% kable()

############################################
# Model 3: Average Sales + Platform + Genre -- Y = mu + b_p + b_g
############################################

#compute b_g: genre effect
Genre_effect <- train_effect %>% group_by(Genre) %>% summarise(b_g = mean(User_Score) - mu - mean(b_p))
train_effect <- train_effect %>% left_join(Genre_effect, by = "Genre")

# compute predicted value
pred_3 <- test_set %>% mutate(mu = mu) %>% 
  left_join(Platform_effect, by = "Platform") %>%
  left_join(Genre_effect, by = "Genre") %>%
  mutate(pred = mu + b_p +b_g ) %>% pull(pred)

#comput RMSE and update result table
rmse_3 <- RMSE(test_set$User_Score, pred_3)
rmse_results <- bind_rows(rmse_results, tibble(method="Platfrom + Genre Effect", RMSE = rmse_3))
rmse_results %>% kable()

############################################
# Model 3: Average Sales + Platform + Genre + Rating -- Y = mu + b_p + b_g + b_r
############################################

#compute b_r: rating effect
Rating_effect <- train_effect %>% group_by(Rating) %>% summarise(b_r = mean(User_Score) - mu - mean(b_p) - mean(b_g))
train_effect <- train_effect %>% left_join(Rating_effect, by = "Rating")

# compute predicted value
pred_3 <- test_set %>% mutate(mu = mu) %>% 
  left_join(Platform_effect, by = "Platform") %>%
  left_join(Genre_effect, by = "Genre") %>%
  left_join(Rating_effect, by = "Rating") %>%
  mutate(pred = mu + b_p +b_g + b_r ) %>% pull(pred)

#comput RMSE and update result table
rmse_4 <- RMSE(test_set$User_Score, pred_3)
rmse_results <- bind_rows(rmse_results, tibble(method="Platform + Genre + Rating Effect", RMSE = rmse_4))
rmse_results %>% kable()

############################################
# Model 5: lm (All numeric)
############################################

# fit linear model for numeric values
fit <- lm(User_Score ~ Year_of_Release + Global_Sales + User_Count + Critic_Count + Critic_Score, train_set)

# compute predicted value
pred_5 <- predict(fit, test_set)

#comput RMSE and update result table
rmse_5 <- RMSE(test_set$User_Score, pred_5)
rmse_results <- bind_rows(rmse_results, tibble(method="lm: All numeric", RMSE = rmse_5))
rmse_results %>% kable()

############################################
# Model 6: lm (All)
############################################

# fit linear model for all values
fit <- lm(User_Score ~ Platform + Genre + Rating + Global_Sales + User_Count + Critic_Count + Critic_Score, train_set)

# compute predicted value
pred_6 <- predict(fit, test_set)

#comput RMSE and update result table
rmse_6 <- RMSE(test_set$User_Score, pred_6)
rmse_results <- bind_rows(rmse_results, tibble(method="lm: All", RMSE = rmse_6))
rmse_results %>% kable()

############################################
# Model 7: random forest
############################################

train_rf <- train_set %>% select(-Name) 

# fit random forest model with minimum OOB
fit_rf <- randomForest(User_Score~., train_rf, ntree = 100)

# compute predicted value
pred_7 <- predict(fit_rf, test_set)

#comput RMSE and update result table
rmse_7 <- RMSE(test_set$User_Score, pred_7)
rmse_results <- bind_rows(rmse_results, tibble(method="randomForest", RMSE = rmse_7))
rmse_results %>% kable()

#Plot minimum OOB
plot(fit_rf)
# variable importance 
importance(fit_rf)
#Plot variable importance
varImpPlot(fit_rf)

############################################
# Model 8: Ridge Regression 
############################################

#Prep matrix for the model
x_var <- data.matrix(train_set[, -c(1,8)])
y_var <- train_set$User_Score
lambda_seq <- 10^seq(-2, 5, length = 100)

# fit the model for minimum lambda
fit_cv <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
lambda_min <- fit_cv$lambda.min

# fit ridge regression model with min lambda
fit_ridge <- glmnet(x_var, y_var, alpha = 0, lambda  = lambda_min)

# prep test set - remove name and User_score
test_x <- data.matrix(test_set[, -c(1,8)])
# compute predicted values
pred_8 <- predict(fit_ridge, s = lambda_min, newx = test_x)
# compute RMSE and update results table
rmse_8 <- RMSE(test_set$User_Score, pred_8)
rmse_results <- bind_rows(rmse_results, tibble(method="Ridge R", RMSE = rmse_8))
rmse_results %>% kable()



############################################
# Model 9: Lasso Regression 
############################################

# fit the model for minimum lambda
fit_la <- cv.glmnet(x_var, y_var, alpha = 1)
lambda_min <- fit_la$lambda.min

# fit lasso regression model with min lambda
fit_lasso <- glmnet(x_var, y_var, alpha = 1, lambda  = lambda_min)
# prep test set - remove name and User_score
test_x <- data.matrix(test_set[, -c(1,8)])
# compute predicted values
pred_9 <- predict(fit_lasso, s = lambda_min, newx = test_x)
# compute RMSE and update results table
rmse_9 <- RMSE(test_set$User_Score, pred_9)
rmse_results <- bind_rows(rmse_results, tibble(method="Lasso R", RMSE = rmse_9))
rmse_results %>% kable()

##########################################
# Validation with Random Forest
#########################################

# compute predicted value for validation set
pred_vali <- predict(fit_rf, validation)

#comput RMSE and update result table
rmse_vali <- RMSE(validation$User_Score, pred_vali)
rmse_results <- bind_rows(rmse_results, tibble(method="randomForest: Validation", RMSE = rmse_vali))
rmse_results %>% kable()



