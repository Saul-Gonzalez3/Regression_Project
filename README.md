Zillow Machine Learning Regression Project


Project Description:
Predicting propery tax assessed values of Single Family Properties


Project Goal:
To find the best features and ML model to predict home values.


Initial Thoughts
After an birds-eye review of the data and then a glance from a univariate perspective, I came up with a few hypotheses based on the goal:
    - 'Total Sqft' will have the largest impact on the target 'home value. 
    - 'Bedrooms' will have some impact on the target, but noticeably less than 'Total_Sqft'. Not a great variable to use by itself for prediction. 
    - 'Bathrooms' will have the least impact on the target due to bathrooms having the smallest amount of square footage in a home, reducing impact on home value.


The Plan
1. Create all the files I will need to make a functioning project (.py and .ipynb files).
2. Create a .gitignore file and ignore my env.py file.
3. Start by acquiring data from the codeup database and document all my initial acquisition steps in the wrangle.py file.
4. Using the prepare file, clearn the data and split it into train, validatate, and test sets.
5. Explore the data. (Focus on the main main questions)
6. Answer all the questions with statistical testing.
7. Identify variables key to predicting property tax assessed values of Single Family Properties. 
8. Document findings (include 4 visuals)
9. Add important finding to the final notebook.
10. Create csv file of test predictions on best performing model.


Data Dictionary
|**Feature**|**Definition**|
|----|----|
|`home_value`| Represents the price of a home|
|`total_sqft`| Represent the size of a property's total square footage.|
|`bedrooms`| Indicate how many bedrooms a property has.|
|`bathrooms`| Indicate how many bathrooms a property has.|



Steps to Reproduce My Work:
1. Clone this repo.
2. Acquire the telco data from the Codeup SQL database.
3. Put the data in the file containing the cloned repo.
4. Run your notebook.


Takeaways and Conclusions

- The highest value results for pair-wise features from the heatmap seem to be spot-on for what features were determined to be best for predicting home value.
- Total_sqft and bathrooms were the two best features found to determine home_value
- Polynomial Regression was the best model out of 5 models tested.


Recommendations:
There are a lot more features that should be looked at, here is my recommendation for future prediction efforts to determine home value:

Acquire as many features as possible presented in the original data and proceed to conduct Select K Best or Recursive Feature Elimination, along with personal insight,to determine which features should be fed into the modeling process. 