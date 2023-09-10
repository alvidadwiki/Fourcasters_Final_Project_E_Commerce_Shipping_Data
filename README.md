## Fourcasters
#### Our team and their roles : 
- Atthoriq Putra Pangestu (Data Scientist)
- Emir Akbar (Data Analyst)
- Darell Timothy Tarigan (Data Analyst)
- Nur Baiti Listyaningrum (Data Analyst)
- Reny Rafiqah (Business Intelligence Analyst)
- Alvida Dwiki Chairunnisa (Business Analyst)

# E-Commerce Shipping Data

## Prerequisites
Download data [here](https://www.kaggle.com/datasets/prachi13/customer-analytics).

## Getting Started
The dataset used for model building contained 10999 observations of 12 variables.

The data contains the following information:

- ID: ID Number of Customers.
- Warehouse block: The Company have big Warehouse which is divided in to block such as A,B,C,D,E.
- Mode of shipment:The Company Ships the products in multiple way such as Ship, Flight and Road.
- Customer care calls: The number of calls made from enquiry for enquiry of the shipment.
- Customer rating: The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).
- Cost of the product: Cost of the Product in US Dollars.
- Prior purchases: The Number of Prior Purchase.
- Product importance: The company has categorized the product in the various parameter such as low, medium, high.
- Gender: Male and Female.
- Discount offered: Discount offered on that specific product.
- Weight in gms: It is the weight in grams.
- Reached on time: It is the target variable, where **1** Indicates that the product has **NOT reached on time** and **0** indicates it has **reached on time**.

## **Problem Statement** 
#### Late Shipments : 
Currently, our e-commerce On-Time Delivery Rate only reaches 40%. In e-commerce, on-time delivery is very important for customer convenience, where 32% of customers complain about delivery delays according to the results of the survey "Biggest Drawbacks of E Commerce Purchases (Worldwide, 2022)" by Koen van Gelder via Statista. The survey also confirms that shipping delays are one of the top 4 problems faced by e-commerce users.

## **What will this Project do?** 
#### Goal : 
Increase On-Time Delivery Rate by recommending solutions related to the factors causing delays so as to improve customer shopping convenience through our e-commerce.

#### Objective : 
1. Create machine learning models that can predict late or on-time deliveries
2. Identify factors that cause delivery delays
3. Recommend solutions and business actions related to the factors causing delivery delays


## Exploratory Data Analysis
### Descriptive Statistics
#### Observation results:
- There are no missing values across all columns.
- There are no duplicates in data
- The **Discount_offered** variable indicate a right-skewed distribution (Need to confirm later on with histograms), high variance, and potential outlier based on the max and 75% percentile distance
- **Prior_purchases** variable has potential outliers
- Other columns looks 'normal' enough based on their numbers
- Most products (about 68%) are delivered using the ship method.
- Most deliveries are late (about 60%).
- All columns seem fine based on their numbers (no strange or catchy values).

### Univariate Analysis
#### Observation results:
##### Numerical Features:
- The numerical features that has outliers are **Prior_Purchases** and **Dicount_offered**.
- The **Discount_offered** feature has the most outliers.
- No outliers found in other features.
- The **Cost_of_the_product** distribution is approximately normal
- **Weight_in_gms** feature have a bimodal distribution.
- The **Discount_offered** feature has a positively skewed distribution.
- The following variables contains discrete values and needed their own bins:
    1. Customer_care_calls
    2. Prior_purchases
- The **Customer_care_calls** feature has a normal distribution, The highest customer care calls is at 4 calls with 3557 customers.
- The **Prior_purchases** feature has a positively skewed distribution, the highest Prior_purchases is at 3 prior purchases with 3955 customers.

##### Categorical Features:
- The visualization of categorical features, it is known that most shipments comes from **Warehouse_block** F.
  - Strangely, warehouse block E is missing from the data
- Based on the **Mode_of_Shipment**, the majority of order is delivered using the ship method.
- **Product_importance** shows that the most shipments are "Low", followed by "Medium" and the least are "High".
- There are fewer on time deliveries (4436) than late deliveries (6563) based on the **Delivery_status** feature.
- The distribution of values inside **Customer_rating** and **Gender** is similar (no dominant values).


### Multivariate Analysis
#### Observation results:
##### Correlation Heatmap:
- Discount_offered has moderate positive relationship with our target (r = 0.4), meaning that higher discount lead to higher probability of late deliveries. This feature may need to be retained for modeling.
- Weight_in_gms has weak negative relationship with our target (r = -0.27), meaning that the heavier a product is, the higher the probability of late deliveries. This feature may need to be retained for modeling.

As for features vs. features, the correlation doesn't show any strong relationships (above 0.8). Let's investigate further using VIF

##### Multicollinearity Check with VIF:

There's no VIF that exceeds 5. Generally, VIF above 5 means that particular variable can't be used for modelling and the multicollinearity is severe. But as we can see from the above results, we will not drop or unuse any of the features up to this point. We'll investigate later on with feature importance for selecting best features for the model.

Let's continue the multivariate exploration with pair plot

##### Pair Plot:
The late product shipment tend to cluster at:
- Higher amount of discount offered
- Several medium weight range and higher weight range
- We can further explore to the boxplot of numerical features vs. target

##### Box Plot:
Now, we can further confirm that:
- Late deliveries always happened at a higher amount of discount offered (> 10% of discount)
- Late deliveries always occured at the weight range of 2000 - 4000 gr and more than 6000 gr
- Other numerical variables doesn't show much difference between on time and late deliveries

Indication of good features list
1. **Discount_offered**
2. **Weight_in_gms**

##### Categorical (Warehouse Block) vs. Target
- All deliveries mostly are late across all warehouse blocks
- When we compare the similar late and on-time delivery proportions for each warehouse block, we can see that the warehouse block itself does not significantly influence whether a shipment will be late or on time.

##### Categorical (Mode of Shipment) vs. Target
- All deliveries mostly are late across all mode of shipment.
- We can see that mode of shipment does not significantly impact delivery lateness, as each mode of shipment shows a similar rate of on time and late deliveries.

##### Categorical (Customer Rating) vs. Target
- All deliveries mostly are late across all customer ratings.
- Based on the proportions, we can see that there is not much differences between customer rating of 1 to 5. This means that customer rating doesn't influence the status of a delivery (whether it will late or not).

##### Categorical (Product Importance) vs. Target
- All deliveries mostly are late across all product importance.
- Looking at the proportions, it's visible that products categorized as high importance are more likely to experience late deliveries compared to those categorized as low and medium importance. This indicates that product importance is one of the factors influencing whether a delivery will be late or not.

##### Categorical (Gender) vs. Target
- All deliveries mostly are late across all genders
- We can see that there's no significant effect of gender to late deliveries because the proportion of late and on time deliveries across all gender is highly similar

**Indication of good features list**
- Discount_offered
- Weight_in_gms
- Product_importance

**A) What is the correlation between each feature and label. Which features are most relevant and should be kept?**
- Discount_offered
- Weight_in_gms
- Product_importance

**B) What is the correlation between features, are there any interesting patterns? What needs to be done with that feature?**
- Based on the correlation heatmap, we could see that several features have weak to moderate relationship with other features. But after confirming further with VIF, we don't need to remove or drop any feature for now.

## Business Insight + Recommendation
### Business Insight

Customers who ordered and applied a discount of more than 10% are experiencing late deliveries. This means that a higher amount of discount is associated with late deliveries. The higher the discount offered, the higher the probability of the delivery status will be late. Thus, recommendations regarding the amount of discount offered need to be given.

High importance products category tend to experience more late deliveries than other product importance levels. It is important to identify and overcome the causes of this lateness in detail (need more data about the detail about the characteristics of "High" importance goods) to improve the delivery performance of high importance category products

High importance products category tend to experience more late deliveries than other product importance levels. It is important to identify and overcome the causes of this lateness in detail (need more data about the detail about the characteristics of "High" importance goods) to improve the delivery performance of high importance category products

### Business Recommendations
This section provides insights and recommendations based on the current dataset. For more accurate and specific follow-up actions, additional data may be required, depending on the type of factors that cause late deliveries.

1. **Discount factor**: There is a tendency that the larger the discount given, the more likely the order is to be late. This could be due to several factors, such as high demand for products with large discounts or possible impacts on the delivery process. Interim recommendations that can be made:

    - **Implementing a Discount Threshold Policy (to be simulated later):** Simulate the impact of setting a maximum discount threshold (10%) for orders. Test whether limiting discounts to this threshold can reduce late deliveries. This policy can help balance between attracting customers through discounts and ensuring on-time delivery.

    - **Shipping System:** Consider optimizing the shipping system/routes for products with large discounts, so that items always arrive on time for products with large discounts.

    - **Continuously Monitor:** Monitor late deliveries of products with large discounts specifically and strive to minimize them.

Note: Limiting discounts may seem counterintuitive. However, here are some of the things we can gain from this policy:

- Reduce Late Deliveries: As has been observed, higher discounts are associated with late deliveries. Limiting discounts can help ensure that the company meets its delivery commitments and maintains a better on-time delivery rate, which is essential for customer satisfaction.

- Improve Profit Margins: By limiting discounts to 10%, the company can maintain better profit margins on its products. Higher discounts can significantly reduce profits.

2. **Product Importance Influence:** Products categorized as "High" are more likely to be late. This suggests that the "High" category may require special attention to meet delivery schedules. We need additional data on the characteristics of "High" importance products to analyze in more detail. However, there are some interim suggestions that can be made:

    - **Dedicated Team Assignment:** Form a dedicated team for the delivery of "high" importance products. This team will have the resources and expertise necessary to ensure on-time delivery for "high" importance items.

    - **Premium Shipping Services:** Use shipping companies with a good reputation and track record, especially for "High" importance products. This ensures on-time and fast delivery, even if there is a potential for late delivery.

3. **Weight Factor:** It is seen that some weight ranges (2000-4000 gr and > 6000 gr) of products tend to be late. This could be due to the complexity of the process of shipping several items with the mentioned weight ranges or the need for different shipping methods. We need more data on the characteristics of products that have weights in those ranges. Here are some interim recommendations that can be implemented:

    - **Further Analysis:** Identify in more detail why products with the aforementioned weights are late. Are there any problems in the delivery process that need to be fixed? Consider adjusting the shipping or logistics process specifically for these products so that they can be shipped more efficiently and on time. Ensure that customers are given realistic delivery estimates for products with certain weights.

    - **Product Packaging:** One of the reasons why some items with certain weights are prone to being late can be caused by the packaging of the item itself:

         **Packaging Strength:** Evaluate whether the packaging used for products in that range is strong enough to withstand the weight and handling during shipping. Weak packaging can cause product damage and order delays.

         **Seal Quality:** Make sure the sealing of these packages is secure. Insufficient sealing can cause items to shift during transportation, potentially leading to delays and damage.

         **Weight Label Accuracy:** Make sure the accuracy of the weight label on the package is correct. The wrong weight labeling process can result in the selection of an inappropriate transportation method, leading to late delivery (e.g., items with heavy weight labels are shipped using small modes of transportation, and vice versa).

Overall, it is important for businesses to continuously monitor and improve logistics and delivery processes to ensure timely and reliable service to customers. These recommendations should be a starting point for addressing potential problems and optimizing on-time delivery rates.