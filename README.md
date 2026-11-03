# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 03-11-2026
# Name: S.Harika
# reg no: 212224240155
# DATASET: Tesla stock price Dataset
# S/W read: colab

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data=pd.read_csv('/content/tesla.csv.zip')
data.head()

data['date']=pd.to_datetime(data['date'])
data.set_index('date',inplace=True)
data['open_diff']=data['open']-data['open'].shift(1)

result = seasonal_decompose(data['open'], model='additive', period=12)
data['open_sea_diff']=result.resid

data['open_log'] = np.log(data['open'])
data['open_log_diff']=data['open_log']-data['open_log'].shift(1)

result = seasonal_decompose(data['open_log_diff'].dropna(), model='additive', period=12)
data['open_log_seasonal_diff']=result.resid

plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data['open'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('date')
plt.ylabel('open')

plt.subplot(6, 1, 2)
plt.plot(data['open_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('date')
plt.ylabel('open')

plt.subplot(6, 1, 3)
plt.plot(data['open_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('date')
plt.ylabel('open')

plt.subplot(6, 1, 4)
plt.plot(data['open_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(No of open)')

plt.subplot(6, 1, 5)
plt.plot(data['open_log_diff'], label='Log Transformation and Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Date')
plt.ylabel('RDiff(Log(No of open))')

plt.figure(figsize=(16, 16))
plt.subplot(6, 1, 6)
plt.plot(data['open_log_seasonal_diff'], label='Log Transformation and regular Differencing and Seasonal Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing and Seasonal Differencing')
plt.xlabel('Date')
plt.ylabel('SDiff(RDiff(Log(No of open)))')

plt.tight_layout()
plt.show()

data.plot(kind='line')

```


### OUTPUT:

ORIGINAL DATASET:

<img width="733" height="193" alt="image" src="https://github.com/user-attachments/assets/411a359a-787e-48a3-95b8-78abf4280a71" />

REGULAR DIFFERENCING:

<img width="760" height="202" alt="image" src="https://github.com/user-attachments/assets/9eed9b75-3fa0-4539-ac28-11569af6d8e2" />


SEASONAL ADJUSTMENT:

<img width="725" height="177" alt="image" src="https://github.com/user-attachments/assets/98056213-f4bb-4ea1-9e70-e30f08958fa5" />


LOG TRANSFORMATION:

<img width="752" height="222" alt="image" src="https://github.com/user-attachments/assets/bf2c5283-af48-4cf9-a0ef-9930c5859b62" />

LOG TRANSFORMATION and REGULAR DIFFERENCING and SEASONAL DIFFERENCING:


<img width="852" height="190" alt="image" src="https://github.com/user-attachments/assets/fb6146da-52b9-49ad-bc85-01e882665038" />

OVERALL VIEW:


<img width="760" height="567" alt="image" src="https://github.com/user-attachments/assets/11d46f22-8254-4ede-b3ed-17ff6342edfb" />



### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
