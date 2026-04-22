## B1. Problem Formulation

### B1(a) — ML Problem Formulation

The target variable is items_sold — the number of items sold in a particular store in a particular month under a particular promotion.

The candidate input features are:
- Store characteristics: location type, store size, footfall,  competition density
- Promotion type: which of the five promotions is being run
- Time features: month, season, whether it is a festival period or weekend
- Historical performance: how the store performed in previous months under each promotion type

This is a regression problem because the target is a continuous number, not a category. The model predicts how many items will be sold, and the promotion with the highest predicted
value is recommended for that store and month.


### B1(b) — Why items sold is a better target than revenue

Revenue is affected by price, and prices change according to seasonal discounts, supplier cost changes, and promotion mechanics all distort the revenue figure in ways that have nothing
to do with how well a promotion actually drove customer behaviour.

Items sold is a cleaner signal. It directly measures whether customers responded to a promotion by picking up more products, regardless of what those products cost.

The broader principle this illustrates is that the target variable should measure what the business actually wants to influence, not a metric that is easy to track but is contaminated by other
factors. In real-world projects, a poorly chosen target variable can cause a model to optimise for the wrong thing entirely — producing predictions that look good on paper but drive bad
business decisions.


### B1(c) — Alternative to a single global model

A single global model assumes all 50 stores behave the same way, which they do not. A BOGO promotion might work well in an urban store with high footfall but fail in a rural store where customers
buy in smaller quantities.

An alternate approach is to train separate models per store segment — for example one model for urban stores, one for semi-urban, and one for rural. This allows each model to learn the promotion
response patterns specific to that store type without being diluted by data from stores that behave very differently.

If individual store-level data is sufficient, a model per store is better. A middle ground is to use location type and store size as features and let the model learn the interactions
naturally if data is available.


## B2. Data and EDA Strategy

### B2(a) — Joining the tables

The four tables would be joined as follows. The transactions table is the base — it contains one row per transaction with a store ID, a date, and a promotion ID. Store attributes are joined onto it
using store ID. Promotion details are joined using promotion ID. The calendar table is joined using the transaction date to bring in the weekend and festival flags. 

The grain of the final modelling dataset is one row per store per month. Transactions would be aggregated by summing items sold and averaging basket size. Store attributes and calendar features such as number of
festival days in that month would be attached at this level.

### B2(b) — EDA Strategy

At least four analyses would be performed before modelling.

First, a distribution of items sold across all stores and months to understand the range and spot any outliers. Stores with unusually high or low sales might indicate data quality issues
or genuinely different store profiles that need separate handling.

Second, average items sold by promotion type across all stores. This shows whether certain promotions consistently outperform others overall, and flags whether the signal is strong enough
to model. If all five promotions produce similar sales, the problem becomes harder.

Third, average items sold by promotion type broken down by location type. This directly tests whether urban, semi-urban, and rural stores respond differently to the same promotion —
which would confirm the need for segmented models rather than a single global one.

Fourth, a time series plot of items sold per month across the three years. This reveals whether there are seasonal patterns such as peaks in festival months or year-end periods. Any strong
seasonality would need to be captured through month and festival features in the model.

Findings from these charts would directly influence which features to engineer, whether to segment the model, and whether any stores or months need to be excluded from training.

### B2(c) — Promotion imbalance in the dataset

If 80% of transactions occurred without any promotion, the model sees very few examples of promoted behaviour during training. It may learn to predict sales as if no promotion is running most of
the time, making it poor at distinguishing which promotion works best when one is actually deployed.

To address this, the training data would be filtered or reweighted to focus on records where a promotion was active, since those are the only situations the model needs to reason
about at prediction time. Alternatively, promotion type could be treated as a required input feature with no-promotion as one valid category, and the model evaluated specifically on
its performance during promoted periods rather than overall.


## B3. Model Evaluation and Deployment

### B3(a) — Train-test split and evaluation metrics

With three years of monthly store-level data, the split should be time based. The first two years would form the training set and the most recent year would be held out as the test set.

A random split is inappropriate because the data is ordered by time. A random split would allow the model to train on future months and predict past ones, which is not possible in real
deployment. The time based split honestly reflects the situation where the model is always predicting the future from past data.

Three metrics would be used. RMSE measures the average prediction error in the same units as items sold — a lower RMSE means the model is closer to the actual sales figures on average. MAE is
similar but less sensitive to large errors, giving a more intuitive sense of how many items off the predictions typically are. Both should be evaluated specifically on promoted months
rather than the full dataset.

In the business context, an RMSE of 30 items means the model's promotion recommendations could be off by around 30 items sold per store per month on average — the marketing team can use this
to set realistic expectations around the recommendations.

### B3(b) — Explaining different recommendations for the same store

The model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount in March. To investigate this, the feature importances from the trained model would be examined
to identify which features are driving the predictions most strongly.

The likely explanation is seasonal. December typically brings festival periods, higher footfall, and a customer mindset oriented towards gifting and reward — conditions where a loyalty
promotion that encourages repeat visits performs well. March has none of these conditions, and customers may respond better to an immediate price reduction through a flat discount.

To communicate this to the marketing team, a simple table would be produced showing the key feature values for Store 12 in December versus March — month, festival flag, footfall estimate,
and competition density — alongside the predicted items sold for each promotion option in each month. This makes it clear that the model is not being inconsistent. It is responding to genuine
differences in the store's context between the two months, and the top features explain exactly which differences matter most.

### B3(c) — Deployment and monitoring

The trained model would be saved to disk using a standard serialisation library such as pickle or joblib. The full pipeline including the preprocessor would be saved together
so that new data is transformed.

At the start of each month, the previous month's store data would be collected and assembled into the same format as the training data — same columns, same aggregation level, same
feature engineering steps. This prepared dataset would be fed into the saved pipeline, which would output a predicted items sold figure for each promotion option for each store.
The promotion with the highest predicted value for each store becomes the recommendation for that month.

For monitoring, actual items sold would be recorded after each month and compared against the model's predictions. If the prediction error starts increasing consistently over several
months, this signals that the model is no longer capturing current store behaviour — known as model drift. Additional triggers for retraining would include a new promotion type
being introduced, a significant change in store footfall patterns, or a macroeconomic event that shifts customer spending behaviour. When any of these occur, the model would
be retrained on the most recent available data and redeployed using the same pipeline structure.