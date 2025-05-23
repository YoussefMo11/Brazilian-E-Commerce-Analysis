#Data Upload
import pandas as pd
import numpy as np
orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"])

order_items = pd.read_csv("olist_order_items_dataset.csv")

#Merge order\_items with orders to get status and dates.
order_data = pd.merge(order_items, orders, on="order_id", how="inner")

#1.Total Revenue (for delivered orders)
delivered = order_data[order_data["order_status"] == "delivered"]
total_revenue = (delivered["price"] + delivered["freight_value"]).sum()
print("Total Revenue:", round(total_revenue, 2))

#2.Expected Revenue (for approved orders)
approved = order_data[order_data["order_status"] == "approved"]
expected_revenue = (approved["price"] + approved["freight_value"]).sum()
print("Expected Revenue:", round(expected_revenue, 2))

#3.Net Profit
net_profit = (delivered["price"] - delivered["freight_value"]).sum()
print("Net Profit:", round(net_profit, 2))

#4.Gross Margin %
gross_margin = (net_profit / total_revenue) * 100 if total_revenue != 0 else 0
print("Gross Margin %:", round(gross_margin, 2))

#Gross Margin % by Month

#Calculate monthly net profit.
delivered["net_profit"] = delivered["price"] - delivered["freight_value"]
monthly = delivered.groupby("month").agg({
    "price": "sum",
    "freight_value": "sum",
    "net_profit": "sum"})
monthly["gross_margin"] = (monthly["net_profit"] / (monthly["price"] + monthly["freight_value"])) * 100
#Visualization
import matplotlib.pyplot as plt
monthly["gross_margin"].plot(kind="bar", color="teal", title="Gross Margin % by Month")
plt.ylabel("Gross Margin %")
plt.xlabel("Month")
plt.tight_layout()
plt.show()

#5.Average Order Value (AOV)
delivered_orders_count = delivered["order_id"].nunique()
AOV = total_revenue / delivered_orders_count if delivered_orders_count != 0 else 0
print("Average Order Value (AOV):", round(AOV, 2))

#Average Order Value (AOV) by Month:

#Create a month column.
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M")
#Calculate monthly AOV (Average Order Value).
aov_by_month = delivered.groupby("month").apply(lambda x: (x["price"] + x["freight_value"]).sum() / x["order_id"].nunique())
## Visualization
import matplotlib.pyplot as plt

aov_by_month.plot(kind="line", marker="o", title="Average Order Value by Month")
plt.ylabel("AOV (BRL)")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.show()

#6.Revenue by Product Category
products = pd.read_csv("olist_products_dataset.csv")
#Merge order_items with products to get the category.
order_items_products = pd.merge(order_items, products, on="product_id", how="left")
order_items_products_orders = pd.merge(order_items_products, orders, on="order_id", how="inner")
#Only delivered orders.
delivered_data = order_items_products_orders[order_items_products_orders["order_status"] == "delivered"]
#Revenue by Product Category
revenue_by_category = delivered_data.groupby("product_category_name")[["price", "freight_value"]].sum()
revenue_by_category["total_revenue"] = revenue_by_category["price"] + revenue_by_category["freight_value"]
revenue_by_category = revenue_by_category.sort_values("total_revenue", ascending=False)
print("\nTop Categories by Revenue:")
print(revenue_by_category.head())
revenue_by_category.to_csv('output1.csv', index=True)

top_categories = revenue_by_category.head(10)
top_categories["total_revenue"].plot(kind="bar", figsize=(10,5), title="Top 10 Categories by Revenue", color="blue")
plt.ylabel("Revenue (BRL)")
plt.xlabel("Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#7.Revenue per Seller
sellers = pd.read_csv("olist_sellers_dataset.csv")

order_items_sellers = pd.merge(order_items, sellers, on="seller_id", how="left")
order_items_sellers_orders = pd.merge(order_items_sellers, orders, on="order_id", how="inner")
delivered_sellers = order_items_sellers_orders[order_items_sellers_orders["order_status"] == "delivered"]

#Revenue per Seller
revenue_per_seller = delivered_sellers.groupby("seller_id")[["price", "freight_value"]].sum()
revenue_per_seller["total_revenue"] = revenue_per_seller["price"] + revenue_per_seller["freight_value"]
revenue_per_seller = revenue_per_seller.sort_values("total_revenue", ascending=False)
print("\nTop Sellers by Revenue:")
print(revenue_per_seller.head())
revenue_per_seller.to_csv('revenue_per_seller_output.csv', index=True)

top_sellers = revenue_per_seller.head(10)
top_sellers["total_revenue"].plot(kind="bar", figsize=(10,5), title="Top 10 Sellers by Revenue", color="blue")
plt.ylabel("Revenue (BRL)")
plt.xlabel("Seller ID")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#8.Revenue per Customer
customers = pd.read_csv("olist_customers_dataset.csv")

order_items_orders_customers = pd.merge(order_items, orders, on="order_id", how="inner")
order_items_orders_customers = pd.merge(order_items_orders_customers, customers, on="customer_id", how="left")

delivered_customers = order_items_orders_customers[order_items_orders_customers["order_status"] == "delivered"]

#Revenue per Customer
revenue_per_customer = delivered_customers.groupby("customer_id")[["price", "freight_value"]].sum()
revenue_per_customer["total_revenue"] = revenue_per_customer["price"] + revenue_per_customer["freight_value"]
revenue_per_customer = revenue_per_customer.sort_values("total_revenue", ascending=False)
print("\nTop Customers by Revenue:")
print(revenue_per_customer.head())
revenue_per_customer.to_csv('revenue_per_customer_output.csv', index=True)

top_customers = revenue_per_customer.head(10)
top_customers.plot(kind="bar", figsize=(10,5), title="Top 10 Customers by Revenue", color="skyblue")
plt.ylabel("Revenue (BRL)")
plt.xlabel("Customer ID")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#9.Percentage of orders where the shipping cost is higher than the product price.
freight_gt_price = delivered[delivered["freight_value"] > delivered["price"]]
percent_freight_gt_price = (len(freight_gt_price) / len(delivered)) * 100 if len(delivered) != 0 else 0
print("\n% Orders where Freight > Product Price:", round(percent_freight_gt_price, 2), "%")

#10.Monthly Revenue Forecast
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M")
monthly_revenue = delivered.groupby("month")[["price", "freight_value"]].sum().sum(axis=1)
monthly_revenue.index = monthly_revenue.index.to_timestamp()
df_prophet = monthly_revenue.reset_index()
df_prophet.columns = ['ds', 'y'] 

pip install prophet
from prophet import Prophet

model = Prophet()
model.fit(df_prophet)
#Create future periods.
future = model.make_future_dataframe(periods=6, freq='M')
#Predict the values.
forecast = model.predict(future)
#Display the predictions alongside the actual values.
import matplotlib.pyplot as plt

fig = model.plot(forecast)
plt.title("Monthly Revenue Forecast with Prophet", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Revenue (BRL)", fontsize=12)
plt.plot(forecast['ds'], forecast['yhat_upper'], color='lightblue', label='Upper Bound', linewidth=2)
plt.plot(forecast['ds'], forecast['yhat_lower'], color='lightcoral', label='Lower Bound', linewidth=2)
plt.plot(forecast['ds'], forecast['yhat'], color='darkgreen', label='Forecast', linewidth=3)
plt.legend()
plt.show()

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
forecast.to_csv('monthly_revenue_forecast.csv', index=False)

#11.Number of Unique Customers
customer_orders = pd.merge(orders, customers, on="customer_id", how="left")
delivered_customers = customer_orders[customer_orders["order_status"] == "delivered"]
#Total Number of Unique Customers
total_customers = customers["customer_id"].nunique()
print("Total Unique Customers:", total_customers)

#12.Returning Customers
repeat_customers = delivered_customers["customer_id"].value_counts()
repeat_customers_count = (repeat_customers > 1).sum()
print("Repeat Customers:", repeat_customers_count)
non_repeat_customers_count = (repeat_customers == 1).sum()

labels = ['Repeat Customers', 'One-time Customers']
sizes = [repeat_customers_count, non_repeat_customers_count]
colors = ['blue', 'skyblue']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Repeat vs One-time Customers')
plt.axis('equal')  
plt.show()

#13.Customer Retention Rate 
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])  
orders['year_month'] = orders['order_purchase_timestamp'].dt.to_period('M')  

retained_customers = orders.groupby('year_month')['customer_id'].nunique()  
retention_rate = retained_customers / retained_customers.shift(1)
print(retention_rate)
retention_rate_cleaned = retention_rate.dropna()

plt.figure(figsize=(12, 6))
plt.plot(retention_rate_cleaned.index.astype(str), retention_rate_cleaned, label='Retention Rate', color='skyblue')
plt.title('Customer Retention Rate Over Time')
plt.xlabel('Month')
plt.ylabel('Retention Rate')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(retention_rate_cleaned)
retention_rate_cleaned.to_csv('retention_rate_cleaned_output.csv', index=True)

#14.Customer Churn Rate
retention_rate = retention_rate.clip(lower=0, upper=1)
churn_rate = 1 - retention_rate
churn_rate = churn_rate.clip(lower=0)
print(churn_rate)
churn_rate.to_csv('churn_rate_output.csv', index=True)

plt.figure(figsize=(12, 6))
plt.plot(churn_rate.index.astype(str), churn_rate, label='Churn Rate', color='skyblue')
plt.title('Customer Churn Rate Over Time')
plt.xlabel('Month')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#15.Customer Lifetime Value (CLV)
clv['first_order'] = pd.to_datetime(clv['first_order'])
clv['last_order'] = pd.to_datetime(clv['last_order'])
clv['retention_months'] = ((clv['last_order'] - clv['first_order']).dt.days) / 30
clv['retention_months'] = clv['retention_months'].replace(0, 1)
clv['CLV'] = clv['total_revenue'] * clv['retention_months']
print(clv.head())
clv.to_csv('clv_output.csv', index=False)

plt.figure(figsize=(10, 6))
sns.histplot(clv['CLV'], bins=30, kde=True, color='blue')
plt.title('Customer Lifetime Value (CLV) Distribution')
plt.xlabel('CLV')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

#16.Top Customer Locations
top_states = customers['customer_state'].value_counts().head(10)
print(top_states)
top_states.to_csv('top_states_output.csv', index=True)
plt.figure(figsize=(10, 6))
top_states.plot(kind='bar', color='skyblue')
plt.title('Top 10 Customer states')
plt.xlabel('states')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
#based on cites
top_cites= customers['customer_city']. value_counts().head(10)
print(top_cites)
top_cites.to_csv('top_cites_output.csv', index=True)
plt.figure(figsize=(10, 6))
top_cites.plot(kind='bar', color='skyblue')
plt.title('Top 10 Customer cites')
plt.xlabel('cites')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#17. Average Review Score
order_reviews = pd.read_csv('olist_order_reviews_dataset_df.csv')
avg_score = order_reviews['review_score'].mean()
print(f"Average Review Score: {avg_score:.2f}")

plt.figure(figsize=(6, 4))
plt.bar(['Average Review Score'], [avg_score], color='skyblue')
plt.ylim(0, 5)
plt.ylabel('Review Score')
plt.title('Average Review Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#18.% of 5-Star Reviews
five_star_pct = (order_reviews['review_score'] == 5).mean() * 100
print(f"% of 5-Star Reviews: {five_star_pct:.2f}%")
 
labels = ['5-Star Reviews', 'Other Reviews']
sizes = [five_star_pct, 100 - five_star_pct]
colors = ['blue', 'lightgray']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Percentage of 5-Star Reviews')
plt.axis('equal')
plt.show()

#19.Top Products with Low Ratings (1–2 stars)
low_reviews = order_reviews[order_reviews['review_score'].isin([1, 2])]
low_products = low_reviews.merge(order_items, on='order_id')
top_low_products = low_products['product_id'].value_counts().head(10)
print(top_low_products)
top_low_products.to_csv('top_low_products_output.csv', index=True)

plt.figure(figsize=(10, 6))
top_low_products.plot(kind='bar', color='darkblue')
plt.title('Top 10 Products with Low Ratings (1-2 stars)')
plt.xlabel('Product ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#20.Average Review Score per Seller 
reviews_seller = order_reviews.merge(order_items, on='order_id')
avg_score_seller = reviews_seller.groupby('seller_id')['review_score'].mean().sort_values()
print(avg_score_seller.head(10))
avg_score_seller.to_csv('avg_score_seller_output.csv', index=True)

plt.figure(figsize=(10, 6))
avg_score_seller.head(10).plot(kind='bar', color='darkblue')
plt.title('Average Review Score per Seller')
plt.xlabel('Seller ID')
plt.ylabel('Average Review Score')
plt.tight_layout()
plt.show()

#21.Average Review Score per Product Category 
reviews_products = order_reviews.merge(order_items, on='order_id').merge(products, on='product_id')
avg_score_category = reviews_products.groupby('product_category_name')['review_score'].mean().sort_values()
print(avg_score_category.head(10))
avg_score_category.to_csv('avg_score_category_output.csv', index=True)

plt.figure(figsize=(10, 6))
avg_score_category.plot(kind='bar', color='darkblue')
plt.title('Average Review Score per Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Review Score')
plt.tight_layout()
plt.show()

#22.Average Delivery Time
delivered = orders.dropna(subset=['order_delivered_customer_date'])
delivered['delivery_days'] = (delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']).dt.days
avg_delivery_time = delivered['delivery_days'].mean()
print(f"Average Delivery Time: {avg_delivery_time:.2f} days")

plt.figure(figsize=(10, 6))
sns.histplot(orders['delivery_time'], bins=30, kde=True, color='blue')
plt.title('Average Delivery Time Distribution')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

#23.Late Delivery Rate
delivered['late'] = delivered['order_delivered_customer_date'] > delivered['order_estimated_delivery_date']
late_rate = delivered['late'].mean() * 100
print(f"Late Delivery Rate: {late_rate:.2f}%")

labels = ['On Time', 'Late']
sizes = [100 - late_rate,late_rate]
colors = ['lightgreen', 'salmon']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Late Delivery Rate')
plt.axis('equal')  
plt.show()

#24.Delivery Delay (Avg. Days Late)
delivered_late = delivered[delivered['late']]
delivered_late['delay_days'] = (delivered_late['order_delivered_customer_date'] - delivered_late['order_estimated_delivery_date']).dt.days
avg_delay_days = delivered_late['delay_days'].mean()
print(f"Avg. Delay Days (Late Deliveries): {avg_delay_days:.2f} days")

orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
delayed_orders = orders[
    (orders['order_delivered_customer_date'].notna()) &
    (orders['order_estimated_delivery_date'].notna()) &
    (orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date'])]
delayed_orders['delay_days'] = (delayed_orders['order_delivered_customer_date'] - delayed_orders['order_estimated_delivery_date']).dt.days
avg_delay = delayed_orders['delay_days'].mean()

plt.figure(figsize=(10, 6))
sns.histplot(delayed_orders['delay_days'], bins=30, kde=True, color='BLUE')
plt.axvline(avg_delay, color='red', linestyle='--', linewidth=2, label=f'Avg Delay = {avg_delay:.2f} days')
plt.title('Distribution of Delivery Delays (Late Orders Only)', fontsize=14)
plt.xlabel('Days Late', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#25.Average Time from Order to Delivery 
avg_total_time = (delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']).dt.days.mean()
print(f"Average Time from Order to Delivery: {avg_total_time:.2f} days")

import matplotlib.pyplot as plt
import seaborn as sns
delivered['order_purchase_timestamp'] = pd.to_datetime(delivered['order_purchase_timestamp'])
delivered['order_delivered_customer_date'] = pd.to_datetime(delivered['order_delivered_customer_date'])
delivered['delivery_time'] = (delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']).dt.days
avg_total_time = delivered['delivery_time'].mean()

plt.figure(figsize=(10, 6))
sns.histplot(delivered['delivery_time'], bins=30, kde=True, color='skyblue')
plt.axvline(avg_total_time, color='red', linestyle='--', linewidth=2, label=f'Average = {avg_total_time:.2f} days')
plt.title('Time from Order to Delivery (in Days)')
plt.xlabel('Days')
plt.ylabel('Number of Orders')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#26.Shipping Time
import pandas as pd
import matplotlib.pyplot as plt

orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"])

order_items = pd.read_csv("olist_order_items_dataset.csv")
merged = pd.merge(order_items, orders, on="order_id", how="inner")
delivered_items = merged[merged["order_status"] == "delivered"].copy()

delivered_items["order_delivered_customer_date"] = pd.to_datetime(delivered_items["order_delivered_customer_date"])
delivered_items["order_delivered_carrier_date"] = pd.to_datetime(delivered_items["order_delivered_carrier_date"])

shipping_data = orders.dropna(subset=["order_delivered_customer_date", "order_delivered_carrier_date"]).copy()
shipping_data["shipping_time"] = (shipping_data["order_delivered_customer_date"] - shipping_data["order_delivered_carrier_date"]).dt.days
avg_shipping_time = shipping_data["shipping_time"].mean()
print(f"Average Shipping Time: {avg_shipping_time:.2f} days")

shipping_data["shipping_time"].plot(
    kind="hist", bins=20, color="cornflowerblue", edgecolor="black", title="Distribution of Shipping Time")
plt.xlabel("Shipping Time (Days)")
plt.ylabel("Number of Orders")
plt.grid(True)
plt.tight_layout()
plt.show()

#27.Order Handling Time
handling_data = orders.dropna(subset=["order_approved_at", "order_purchase_timestamp"]).copy()
handling_data["order_handling_time"] = (handling_data["order_approved_at"] - handling_data["order_purchase_timestamp"]).dt.total_seconds() / 3600 
avg_handling_time = handling_data["order_handling_time"].mean()
print(f"Average Order Handling Time: {avg_handling_time:.2f} hours")

#28.% of Orders with Missing Delivery Date
total_orders = len(orders)
#Customer Delivery Date
missing_customer = orders['order_delivered_customer_date'].isna().sum()
percent_missing_customer = (missing_customer / total_orders) * 100
print(f"% of Orders Missing Customer Delivery Date: {percent_missing_customer:.2f}%")
#Carrier Delivery Date
missing_carrier = orders['order_delivered_carrier_date'].isna().sum()
percent_missing_carrier = (missing_carrier / total_orders) * 100
print(f"% of Orders Missing Carrier Delivery Date: {percent_missing_carrier:.2f}%")
#Estimated Delivery Date
missing_estimated = orders['order_estimated_delivery_date'].isna().sum()
percent_missing_estimated = (missing_estimated / total_orders) * 100
print(f"% of Orders Missing Estimated Delivery Date: {percent_missing_estimated:.2f}%")

percent_missing_customer = (missing_customer / total_orders) * 100
percent_missing_carrier = (missing_carrier / total_orders) * 100
percent_missing_estimated = (missing_estimated / total_orders) * 100
labels = ['Customer Delivery Date', 'Carrier Delivery Date', 'Estimated Delivery Date']
percentages = [percent_missing_customer, percent_missing_carrier, percent_missing_estimated]
plt.figure(figsize=(8, 6))
plt.bar(labels, percentages, color=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Percentage of Orders Missing Delivery Dates')
plt.xlabel('Delivery Date Type')
plt.ylabel('Percentage of Orders (%)')
plt.tight_layout()
plt.show()

#29.Popular Payment Methods
payments = pd.read_csv("olist_order_payments_dataset.csv")
orders_payments = pd.merge(orders, payments, on="order_id", how="left")

payment_counts = orders_payments['payment_type'].value_counts()
payment_percentages = (payment_counts / payment_counts.sum()) * 100
print(payment_percentages.round(2).astype(str) + " %")

payment_counts = orders_payments['payment_type'].value_counts()
payment_percentages = (payment_counts / payment_counts.sum()) * 100
payment_percentages_filtered = payment_percentages[payment_percentages.index != 'not_defined']
plt.figure(figsize=(8, 6))
plt.pie(payment_percentages_filtered,
        labels=payment_percentages_filtered.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Pastel1.colors)
plt.title("Popular Payment Methods (%)")
plt.axis('equal')   
plt.tight_layout()
plt.show()

#30.Average Number of Installments 
average_installments = orders_payments['payment_installments'].mean()
print(f"Average Number of Installments: {average_installments:.2f}")
#Average Number of Installments by Payment Type
avg_installments_by_payment = orders_payments.groupby('payment_type')['payment_installments'].mean().round(2)
print("Average Number of Installments by Payment Type:")
print(avg_installments_by_payment)

avg_installments_by_payment = orders_payments.groupby('payment_type')['payment_installments'].mean().round(2)
avg_installments_by_payment = avg_installments_by_payment[avg_installments_by_payment.index != 'not_defined']
plt.figure(figsize=(8, 5))
avg_installments_by_payment.sort_values(ascending=False).plot(kind='bar', color='skyblue')

plt.title("Average Number of Installments by Payment Type")
plt.xlabel("Payment Type")
plt.ylabel("Average Installments")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#31.Installment Use Rate
total_orders = len(orders_payments)
installment_orders = orders_payments[orders_payments['payment_installments'] > 1].shape[0]
installment_use_rate = (installment_orders / total_orders) * 100
print(f"Installment Use Rate: {installment_use_rate:.2f}%")

#32.Average Payment per Installment 
orders_payments['payment_per_installment'] = orders_payments['payment_value'] / orders_payments['payment_installments']
average_payment_per_installment = orders_payments['payment_per_installment'].mean()
print(f"Average Payment per Installment: {average_payment_per_installment:.2f}")

#33.Total Payments Received
total_payments_received = orders_payments['payment_value'].sum()
print(f"Total Payments Received (Sum of all payment_value): {total_payments_received:,.2f} BRL")

#34.Revenue per Payment Method 
revenue_by_payment_type = orders_payments.groupby('payment_type')['payment_value'].sum().sort_values(ascending=False)
revenue_by_payment_type = revenue_by_payment_type.round(2)
print("Revenue per Payment Method (in BRL):")
revenue_by_payment_type = revenue_by_payment_type[revenue_by_payment_type.index != 'not_defined']
print(revenue_by_payment_type)

plt.figure(figsize=(8, 5))
revenue_by_payment_type.plot(kind='bar', color='skyblue')
plt.title("Revenue per Payment Method")
plt.xlabel("Payment Type")
plt.ylabel("Total Revenue (BRL)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#35.Total Number of Orders 
total_orders = orders['order_id'].nunique()
print(f"Total Number of Orders: {total_orders}")

#36.Number of Orders by Category 
products = pd.read_csv('olist_products_dataset.csv')
merged_items_products = order_items.merge(products, on='product_id', how='left')
merged_data = merged_items_products.merge(orders[['order_id']], on='order_id', how='left')
orders_by_category = merged_data.groupby('product_category_name')['order_id'].nunique().sort_values(ascending=False)
print("Number of Orders by Category:")
print(orders_by_category)
orders_by_category.to_csv('orders_by_category_output.csv', index=True)

orders_by_category.head(10).sort_values().plot(kind='barh', figsize=(8, 5), color='slateblue')
plt.title("Top 10 Categories by Number of Orders")
plt.xlabel("Number of Orders")
plt.ylabel("Product Category")
plt.tight_layout()
plt.show()

#37.Canceled Orders 
canceled_orders_count = orders[orders['order_status'] == 'canceled'].shape[0]
print(f"Number of Canceled Orders: {canceled_orders_count}")

order_status_counts = orders['order_status'].value_counts()
plt.figure(figsize=(8, 5))
order_status_counts.plot(kind='pie', color='steelblue')
plt.title('Number of Orders by Status')
plt.xlabel('Order Status')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#38.Pending Orders 
pending_orders = orders[orders['order_status'].isin(['created', 'approved'])]
pending_orders_count = pending_orders.shape[0]
print(f"Number of Pending Orders: {pending_orders_count}")

#39.Delivered Orders by Month 
delivered_orders = orders[orders['order_status'] == 'delivered'].copy()
delivered_orders['order_delivered_customer_date'] = pd.to_datetime(delivered_orders['order_delivered_customer_date'])
delivered_orders['delivery_month'] = delivered_orders['order_delivered_customer_date'].dt.to_period('M')
monthly_deliveries = delivered_orders.groupby('delivery_month').size()
print("Delivered Orders by Month:")
print(monthly_deliveries)
monthly_deliveries.to_csv('monthly_deliveries_output.csv', index=True)

monthly_deliveries.index = monthly_deliveries.index.to_timestamp()
plt.figure(figsize=(10, 5))
monthly_deliveries.plot(kind='line', marker='o', color='blue')
plt.title('Delivered Orders by Month')
plt.xlabel('Month')
plt.ylabel('Number of Delivered Orders')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#40.Top 10 Most Ordered Products 
top_products = order_items['product_id'].value_counts().head(10)
print("Top 10 Most Ordered Products (by product_id):")
print(top_products)

merged_products = order_items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
top_product_categories = merged_products['product_category_name'].value_counts().head(10)
print("Top 10 Most Ordered Product Categories:")
print(top_product_categories)
top_product_categories.to_csv('top_product_categories_output.csv', index=True)

top_product_categories.sort_values().plot(kind='barh', figsize=(8, 5), color='cornflowerblue')
plt.title('Top 10 Most Ordered Product Categories')
plt.xlabel('Number of Orders')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()

#41.Top 10 Most Reviewed Products 
order_reviews = pd.read_csv('olist_order_reviews_dataset_df.csv')
reviews_with_product = order_reviews.merge(order_items[['order_id', 'product_id']], on='order_id', how='left')
most_reviewed_products = reviews_with_product['product_id'].value_counts().head(10)
print("Top 10 Most Reviewed Products (by product_id):")
print(most_reviewed_products)

most_reviewed_with_names = most_reviewed_products.reset_index().merge(
    products[['product_id', 'product_category_name']],
    left_on='product_id', right_on='product_id', how='left')
most_reviewed_with_names = most_reviewed_with_names[['product_category_name', 'product_id', 0]]
most_reviewed_with_names.columns = ['product_id', 'review_count', 'product_category_name']
print("Top 10 Most Reviewed Product Categories:")
print(most_reviewed_with_names)
most_reviewed_with_names.to_csv('most_reviewed_with_names_output.csv', index=False)

most_reviewed_with_names.set_index('product_category_name')['count'].sort_values().plot(
    kind='barh', figsize=(8, 5), color='cornflowerblue')
plt.title('Top 10 Most Reviewed Product Categories')
plt.xlabel('Number of Reviews')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()

#42.Orders with Product Return Risk 
reviews_with_product = order_reviews.merge(order_items[['order_id', 'product_id']], on='order_id', how='left')
low_rating_orders = reviews_with_product[reviews_with_product['review_score'] < 3]
low_rating_orders_count = low_rating_orders.shape[0]
print(f"Number of Orders with Product Return Risk: {low_rating_orders_count}")

#43.Number of Sellers 
number_of_sellers = order_items['seller_id'].nunique()
print(f"Number of Sellers: {number_of_sellers}")

#44.Top Sellers by Revenue or Orders 
#by orders
top_sellers_by_orders = order_items['seller_id'].value_counts().head(10)
print("Top 10 Sellers by Number of Orders:")
print(top_sellers_by_orders)
top_sellers_by_orders.to_csv('top_sellers_by_orders_output.csv', index=True)

top_sellers_by_orders.sort_values().plot(kind='barh', figsize=(8, 5), color='teal')
plt.title('Top 10 Sellers by Number of Orders')
plt.xlabel('Number of Orders')
plt.ylabel('Seller ID')
plt.tight_layout()
plt.show()

#by revenue
top_sellers_by_revenue = order_items.groupby('seller_id')['price'].sum().sort_values(ascending=False).head(10)
print("Top 10 Sellers by Revenue:")
print(top_sellers_by_revenue)
top_sellers_by_revenue.to_csv('top_sellers_by_revenue_output.csv', index=True)

top_sellers_by_revenue.sort_values().plot(kind='barh', figsize=(8, 5), color='teal')
plt.title('Top 10 Sellers by Revenue')
plt.xlabel('Total Revenue (R$)')
plt.ylabel('Seller ID')
plt.tight_layout()
plt.show()

#45.Orders by Region/State 
customers = pd.read_csv("olist_customers_dataset.csv")
orders_with_state = orders.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')
orders_by_state = orders_with_state['customer_state'].value_counts().sort_values(ascending=False)
print("Number of Orders by Customer State:")
print(orders_by_state)
orders_by_state.to_csv('orders_by_state_output.csv', index=True)

orders_by_state.plot(kind='bar', figsize=(10, 6), color='steelblue')
plt.title('Number of Orders by Customer State')
plt.xlabel('Customer State')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#46.Monthly Orders Overview 
orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
orders["order_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
orders["status_group"] = orders["order_status"].apply(lambda x:
    "Delivered" if x == "delivered" else
    "Canceled" if x == "canceled" else
    "Pending" if x in ["created", "approved"] else
    "Other")
monthly_status = orders.groupby(["order_month", "status_group"]).size().unstack(fill_value=0)
monthly_status["Total"] = monthly_status.sum(axis=1)
print(monthly_status.tail())
monthly_status.to_csv('monthly_status_output.csv', index=True)

monthly_status[["Delivered", "Canceled", "Pending"]].plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    color=["#4da6ff", "#1f77b4", "#08306b"] )
plt.title("Monthly Orders Overview", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#47.% Revenue Reconciliation 
total_revenue = payments['payment_value'].sum()
expected_revenue = order_items['price'].sum()
revenue_reconciliation_percent = (total_revenue / expected_revenue) * 100
print(f"Revenue Reconciliation %: {revenue_reconciliation_percent:.2f}%")

#48.Interactive Filters 
import pandas as pd
import plotly.express as px

orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
payments = pd.read_csv("olist_order_payments_dataset.csv")

df = orders.merge(payments, on='order_id', how='left')

df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

fig = px.bar(df, 
             x="order_month", 
             color="order_status", 
             title="Orders by Month and Status",
             labels={"order_month": "Month", "order_status": "Order Status"},
             category_orders={"order_status": df["order_status"].unique()})
fig.update_layout(barmode="stack")
fig.show()
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {'label': 'All Status', 'method': 'relayout', 'args': [{'barmode': 'stack'}]},
            {'label': 'Delivered', 'method': 'relayout', 'args': [{'barmode': 'stack'}]},
            {'label': 'Canceled', 'method': 'relayout', 'args': [{'barmode': 'stack'}]},],
        'direction': 'down',
        'showactive': True,
        'active': 0,
        'x': 0.17,    
        'xanchor': 'left',
        'y': 1.15,    
        'yanchor': 'top'}])

fig.update_layout(
    sliders=[{
        'steps': [{
            'args': [
                [f'{month}'],
                {
                    'xaxis.range': [df["order_month"].min(), df["order_month"].max()]
                }
            ],
            'label': month,
            'method': 'relayout'
        } for month in df['order_month'].unique()]
    }])
fig.update_traces(marker=dict(color=["#4da6ff", "#1f77b4", "#08306b"]))
fig.write_html("orders_by_month_status.html")



