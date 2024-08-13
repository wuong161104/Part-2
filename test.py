import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/vuong/Python/PART 2/sale_table.csv')
# Calculate the percentage of each payment method
payment_method_counts = df['PaymentMethod'].value_counts()
payment_method_percent = payment_method_counts / payment_method_counts.sum() * 100

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(payment_method_percent, labels=payment_method_percent.index, autopct='%1.1f%%', startangle=140)
plt.title('Payment method ratio')
plt.show()



