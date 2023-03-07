import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df_auto = pd.read_csv('auto.csv', names = headers) # Зчитування даних у data frame
print(df_auto.head())

df_auto.replace("?", np.nan, inplace = True) # Заміна ? на NaN
print(df_auto.head(5))

missing_data = df_auto.isnull() # Пошук пропущених данних і створення data frame з True/False
print(missing_data.head(5))

for column in missing_data.columns.values.tolist(): # Підрахування пропущених даних у кожному стовпці, вивід для наглядності
    print(column)
    print (missing_data[column].value_counts())
    print("")

avg_norm_loss = df_auto["normalized-losses"].astype("float").mean(axis=0) # Визначення середнього значення для стовпця
df_auto["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True) # Заміна NaN на середнє значення

avg_bore=df_auto['bore'].astype('float').mean(axis=0) # Визначення середнього значення для стовпця
df_auto["bore"].replace(np.nan, avg_bore, inplace=True) # Заміна NaN на середнє значення

avg_stroke=df_auto['stroke'].astype('float').mean(axis=0) # Визначення середнього значення для стовпця
df_auto["stroke"].replace(np.nan, avg_stroke, inplace=True) # Заміна NaN на середнє значення

avg_horsepower = df_auto['horsepower'].astype('float').mean(axis=0) # Визначення середнього значення для стовпця
df_auto['horsepower'].replace(np.nan, avg_horsepower, inplace=True) # Заміна NaN на середнє значення

avg_peakrpm=df_auto['peak-rpm'].astype('float').mean(axis=0) # Визначення середнього значення для стовпця
df_auto['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True) # Заміна NaN на середнє значення

print(df_auto['num-of-doors'].value_counts().idxmax()) # Визначення найпоширенішого виду кільксоті дверей

df_auto["num-of-doors"].replace(np.nan, "four", inplace=True) # Заміна NaN на найпоширеніший вид кільксоті дверей

df_auto.dropna(subset=["price"], axis=0, inplace=True) # Виділення рядків у яких відсітні значення ціни машини

df_auto.reset_index(drop=True, inplace=True) # Скидання індекс, тому що видалено два рядки

# Зміна типів данних для деяких стовпців
df_auto[["bore", "stroke"]] = df_auto[["bore", "stroke"]].astype("float")
df_auto[["normalized-losses"]] = df_auto[["normalized-losses"]].astype("int")
df_auto[["price"]] = df_auto[["price"]].astype("float")
df_auto[["peak-rpm"]] = df_auto[["peak-rpm"]].astype("float")

df_auto['city-L/100km'] = 235/df_auto["city-mpg"] # Перетворення витрат пального в л/100км у новий стовпець

df_auto["highway-mpg"] = 235/df_auto["highway-mpg"]
df_auto.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)# Перейменувати назву стовпця з "highway-mpg" на "highway-L/100km"

df_auto['length'] = df_auto['length']/df_auto['length'].max() # Нормалізація стовпця length
df_auto['width'] = df_auto['width']/df_auto['width'].max() # Нормалізація стовпця width
df_auto['height'] = df_auto['height']/df_auto['height'].max() # Нормалізація стовпця height


df_auto["horsepower"]=df_auto["horsepower"].astype(int, copy=True)

plt.hist(df_auto["horsepower"], bins = 3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# plt.savefig('1.png')
plt.show()

# Перетворення в індикаторні змінні, використовувати категоріальні змінні для регресійного аналізу
dummy_variable_1 = pd.get_dummies(df_auto["fuel-type"])
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
df_auto = pd.concat([df_auto, dummy_variable_1], axis=1)
df_auto.drop("fuel-type", axis = 1, inplace=True)


dummy_variable_2 = pd.get_dummies(df_auto['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
df_auto = pd.concat([df_auto, dummy_variable_2], axis=1)
df_auto.drop('aspiration', axis = 1, inplace=True)

print(df_auto.columns)

df_auto.to_csv('clean_df_auto.csv')