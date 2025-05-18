import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([[1, 5], [3, 2], [8, 4], [7, 14]])

print(f'Data (before scaling):\n{data}')
scaler = StandardScaler()
z_scaledData = scaler.fit_transform(data.reshape(-1, 1)) 
print(f'Mean (before scaling): {scaler.mean_}')
print(f'Standard deviation (before scaling): {scaler.var_**.5}')

print(f'Data (after scaling):\n{z_scaledData.flatten()}')
print(f'Mean (after scaling): {np.mean(z_scaledData)}')
print(f'Standard deviation (after scaling): {np.std(z_scaledData)}')

dataPoint2scale = np.array([10])
dataPoint_scaled = scaler.transform(dataPoint2scale.reshape(-1, 1))
print(f'Data point before scaling: {dataPoint2scale}')
print(f'Data point after scaling: {dataPoint_scaled}')
print(f'Data point after reverting to original value:{dataPoint_scaled*(scaler.var_**.5)+scaler.mean_}')

revert2originalValues = z_scaledData * (scaler.var_**.5) + scaler.mean_
print(f'\nData after reverting to original values:\n{revert2originalValues.flatten()}')

z = (data - np.mean(data)) / np.std(data)
print(f'\nData after scaling using eqn: z=(x-mean)/std:\n{z}')
originalData = z * np.std(data) + np.mean(data)
print(f'\nData after reverting to original values using eqn: x=(z*std)+mean:\n{originalData}')

