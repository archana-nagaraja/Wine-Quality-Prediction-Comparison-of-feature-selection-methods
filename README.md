**INTRODUCTION:-**

Dataset used : Wine Quality based on physiochemical tests - https://archive.ics.uci.edu/ml/datasets/Wine+Quality. 
The dataset is related to red and white wine samples of the Portuguese "Vinho Verde" wine. The main dataset includes 2 smaller datasets – one for red-wine samples and one for white-wine samples. 
This project uses the red-wine samples dataset alone, containing 1599 instances, 11 input variables and 1 output variable. 
Values for input variables are obtained from objective tests. Values for output variable is based on sensory data. Values are calculated as median of at least 3 ratings from wine-experts.

Input variables (based on physicochemical tests):
1 - fixed acidity	2 - volatile acidity		3 - citric acid		4 - residual sugar
5 – chlorides		6 - free sulfur dioxide		7 - total sulfur dioxide		8 - density
9 – pH		10 – sulphates		11 - alcohol

Output variable (based on sensory data): 
12 - quality (score between 0 and 10)

Sources: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
             	[Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
               [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

**Goal of the project**: Predict the output/target variable - wine-quality(range 3-8). Also test feature-selection methods to analyze how the different feature combinations perform. 

Implementation Details: 
The wine_quality_main.py file reads the csv file with red-wine data into a dataframe. Feature selection methods - Pearson correlation, Recursive Feature Elimination(RFE) and L1 Regularization / LassoCV are applied to obtain list of features and these features are then used to apply Random Forest Regressor to predict wine quality. 
Results of each method are visualized using matplotlib.

apply_rfe.py file uses DecisionTreeRegressor to apply recursive feature elimination and identify important features. 

**List of functions:** 

apply_pearson_correlation() – creates the correlation matrix and selects features based on the correlation-threshold specified.

apply_rfe() – applies the recursive-feature-elimination using decision-trees to select features.

apply_lasso() – applies LassoCV to select features.

apply_random_forest() – applies the random-forest regressor to predict the wine-quality using the feature-list provided. Computes r2, rmse and accuracy metrics.

plot_results() – plots accuracy vs method for pearson-correlation vs RFE vs LassoCV vs All-features

plot_shply() – plots accuracy vs feature-excluded for Shapley-method vs All-features

Project Presentation : https://drive.google.com/file/d/1Pb6k2rlFXOvOjSfmhXbqP8CgiZmC8_jw/view?usp=sharing



