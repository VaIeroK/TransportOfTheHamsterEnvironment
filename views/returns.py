import pandas as pd
import json

df = pd.read_csv('views/result.csv')
df['mean_values'] = df[['quantity_car', 'average_speed_car', 'quantity_van', 'average_speed_van', 'quantity_bus',
                        'average_speed_bus']].mean(axis=1)
df['total_quantity_car'] = df['quantity_car'].sum()
df['total_quantity_van'] = df['quantity_van'].sum()
df['total_quantity_bus'] = df['quantity_bus'].sum()
avg_df = df[['average_speed_car', 'average_speed_van', 'average_speed_bus']]
df_sorted = df.sort_values(by='mean_values', ascending=False)
df_sorted.drop('mean_values', axis=1, inplace=True)
json_data = df_sorted.to_json(orient='records', indent=4)
json_data_avg = avg_df.to_json(orient='records', indent=4)
processed_data_avg = json.loads(json_data_avg)
processed_data = json.loads(json_data)


async def get_data(skip, take):
    global processed_data
    return processed_data[skip:skip + take]
