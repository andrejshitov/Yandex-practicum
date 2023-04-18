#!/usr/bin/env python
# coding: utf-8

# <div padding: 40px">
#     
# <div>
# <h1>Анализ активности пользователей в чатах Пачки и Слак <a class="tocSkip"> </h1>

# # Описание проекта

# ## Заказчик 
# Яндекс практиткум

# ## Проблемма
# Проблема – низкая обратная связь на важные посты, публикуемые в чатах.

# ## Цель исследования
# Цель проекта - проанализировать активность в чатах Пачки и Слак. 
# * Выявить паттерны, динамику, цикличность в течение дня, недели, месяца, года. Визуализировать свои находки. 
# * Понять, когда активность студентов в чатах наибольшая, и когда лучше публиковать посты/анонсы, чтобы получить больше откликов.
# * Проанализировать различные типы каналов, когорт, групп. оценить в каких общения больше, в каких меньше, и как они различаются от когорты к когорте, от канала к каналу

# ##  Описание данных
# -	Unnamed: 0  -  Индекс
# -	client_msg_id -  id сообщения 
# -	type – тип поста 
# -	user – id пользователя 
# -	ts – дата поста
# -	latest_reply – дата ответа
# -	team -  факт вхождения в неизвестную группу. Возможно что то другое.
# -	thread_ts – дата треда
# -	subtype – метка действий пользователя
# -	channel - канал
# -	file_date – дата файла
# -	attachments – прикрепленные файлы
# -	reactions – реакции
# -	text_len – длина текста сообщения
# -	text_words – количество слов в сообщении
# 
# 

# # Загрузка данных

# In[1]:


#импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime as dt
import time
import numpy as np
import warnings
import folium
import json
import geopandas as gpd
from folium import Choropleth
from folium import Map, Marker
from folium.plugins import MarkerCluster
import yaml
import ast
import plotly.io as pio
pio.renderers.default = "svg"
svg_renderer = pio.renderers["svg"]
svg_renderer.scale = 1.2


# In[2]:


#унифицируем размер графиков
sns.set(rc={'figure.figsize':(10, 5)})
#настраиваем отображение колонок (полностью)
pd.set_option('display.max_columns', None)
#настраиваем отображение формата значений
pd.options.display.float_format = '{:,.2f}'.format
#вывод значений без сокращений
pd.set_option('display.max_colwidth', -1)
#убираем системные предупреждения
warnings.filterwarnings('ignore')


# In[3]:


#открываем и сохраняем как датафрейм файл с данными:
df = pd.read_csv('chat_data_clean.csv', parse_dates=['ts'])
df.head()


# In[4]:


df.info()


# In[5]:


# приводим типы данных к нужному формату
df['ts'] = pd.to_datetime(df['ts'], unit='s').dt.round('1s')
df['latest_reply'] = pd.to_datetime(df['latest_reply'], unit='s').dt.round('1s')
df['thread_ts'] = pd.to_datetime(df['thread_ts'], unit='s').dt.round('1s')
df['file_date'] = pd.to_datetime(df['file_date'])
df['attachments'] = df['attachments'].astype('bool')
df = df.drop(columns=['Unnamed: 0'])
df.head()


# In[6]:


df.info()


# In[7]:


# Изучаем процент пропущенных значений
pd.DataFrame(df.isna().mean()*100).sort_values(by=0)[::-1].style.background_gradient('coolwarm')


# <h4>Вывод: <a class="tocSkip"> </h4>  
#     Итого, в таблице 26533 записей, и 14 столбцов. 7 столбцов - текстовые, 4 - с датами и временем, 2-целочисленные, 1 - булевый. 7 столбцов не имеют пропусков. Остальные семь имеют пропуски от 10% до 91%. Больше всего пропусков в столбцах с реакциями и временем ответа 84 и 91 процент. 

# # Предобработка данных

# Начнем со столбцов не имеющих пропусков

# ## text_len (Длина текста сообщения)

# In[8]:


# Построим гимтограмму и сводную таблицу
df.hist('text_len', bins=100, range=(1,700))
plt.title('Длина текста')
plt.xlabel('Длина текста, симв.')
plt.ylabel('Кол-во сообщений')
print(df.pivot_table(index= 'text_len', values= 'type', aggfunc= 'count').sort_values(by = 'type', ascending=False).rename(columns= {'type': 'кол-во сообщений'}))
df['text_len'].describe()


# In[9]:


# имеем 211 сообщений с длиной текста - 0. посмотрим на них
df.query('text_len == 0').head()


# Расперделение данных выглядит нормально, за исключением нулевых значений. Их скорее всего нужно будет удалить. Наибольшее кол-во сообщений с длиной 35 символов. Медианное кол-во символов в выборке 80. 

# ## text_words (количество слов в сообщении)
# Сделаем аналогично как в предыдущем пункте

# In[10]:


# Построим гимтограмму и сводную таблицу
df.hist('text_words', bins=100, range=(0,100))
plt.title('Кол-во слов')
plt.xlabel('Кол-во слов, шт.')
plt.ylabel('Кол-во сообщений')
print(df.pivot_table(index= 'text_words', values= 'type', aggfunc= 'count').sort_values(by = 'type', ascending=False).rename(columns= {'type': 'кол-во сообщений'}))
df['text_words'].describe()


# In[11]:


# имеем 211 сообщений с кол-вом слов - 0. посмотрим на них
df.query('text_words == 0')


# В целом данный столбец повторяет предыдущий. Те же 211 пустых сообщений. График так же похож на предыдущий.  Наибольшее кол-во сообщений с кол-вом слов - 5. Медианное кол-во слов в выборке 11.

# ## channel (канал)

# In[12]:


# Посмотрим на данные
df['channel'].unique()


# Из данного столбца можно извлечь полезные данные. Выделим столбец - тип канала, наименование группы. 

# In[13]:


# Создадим столбец "тип канала" и заполним его...для этого сделаем функцию
def new_column(column_name, cloumn_value = ['other']):
    df[column_name] = ' '
    df['channel'] = df['channel'].str.replace('-','_')
    d = cloumn_value
    for i in d:
        df.loc[df['channel'].str.contains(i, na=False), column_name] = i
    df.loc[df[column_name] == ' ', column_name] = 'other'
new_column('channel_type', cloumn_value = ['project', 'library', 'b2g', 'apps', 'bus', 'complaints', 'study', 'telecom', 'tutorial', 'digital_professions','digitalprof', 'info', 'teamwork', 'exerciser', 'masterskaya', 'mentors', 'student_feedback', 'teach_me', 'community', 'academ', 'цифровые_профессии', 'students_feedback'])
df.groupby('channel_type')['type'].count().sort_values(ascending=False)


# In[14]:


# Аналогичным образом создадим и заполним колонку тип професии
new_column('profession_type', cloumn_value = ['da', 'ds', 'dl', 'sql', 'de'])
df.groupby('profession_type')['type'].count().sort_values(ascending=False)


# In[15]:


# Выделим номер группы
df['group'] = df['channel'].str.extract(pat = '([_]..)')
df['group'] = df['group'].str.replace('_','')
s = ['an', 'lo', 'co', 'ed','bc', 'pl', 'pr', 'we', 'ra', 'ex', 'in', 'te', 'ac']
for i in s:
    df['group'] = df['group'].str.replace(i,'other')
df['group'] = df['group'].str.replace('d','')
df.groupby('group')['type'].count().sort_values(ascending=False)


# In[16]:


# создадим столбец с именем когорты
df['chorot_name'] = df['profession_type'] + "_" + df['group']
df.loc[df['chorot_name'].str.contains('other', na=False), 'chorot_name'] = "other"
df.groupby('chorot_name')['type'].count().sort_values(ascending=False)


# Из поля "cannel" удалось вычленить данные по названию канала, профессии и номер когорты.

# ## ts (дата поста)
# Из даты поста создадим колонки с годом, месяцем, днем недели, временем суток

# In[17]:


# выделим год
df['message_year'] = df['ts'].dt.year
df.groupby('message_year')['type'].count().sort_values(ascending=False)


# In[18]:


# Выделим месяц
df['message_month'] = df['ts'].dt.month
df['message_month'] = df['message_year'].astype('str') + df['message_month'].astype('str')
df.groupby('message_month')['type'].count().sort_values(ascending=False)


# In[19]:


# Выделим день
df['message_day'] = df['ts'].dt.date


# In[20]:


# Выделим день недели
df['message_weekday'] = df['ts'].dt.weekday
v = [0,1,2,3,4,5,6]
w = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
df['message_weekday'] =df['message_weekday'].replace(v,w)
df.groupby('message_weekday')['type'].count().sort_values(ascending=False)


# In[21]:


# выделим время и создадим категории (утро, день, вечер, ночь)
df['message_time'] = df['ts'].dt.time
df['time_category'] = " "
df.loc[(df['message_time'] >= dt.time(6,0,0)) & (df['message_time'] < dt.time(9,0,0)), 'time_category'] = "утро"
df.loc[(df['message_time'] >= dt.time(9,0,0)) & (df['message_time'] < dt.time(12,0,0)), 'time_category'] = "день_до_обеда"
df.loc[(df['message_time'] >= dt.time(12,0,0)) & (df['message_time'] < dt.time(17,0,0)), 'time_category'] = "день_после_обеда"
df.loc[(df['message_time'] >= dt.time(17,0,0)) & (df['message_time'] < dt.time(21,0,0)), 'time_category'] = "вечер"
df.loc[(df['message_time'] >= dt.time(21,0,0)) & (df['message_time'] < dt.time(23,59,59)), 'time_category'] = "ночь"
df.loc[(df['message_time'] >= dt.time(0,0,0)) & (df['message_time'] < dt.time(6,0,0)), 'time_category'] = "ночь"
df.groupby('time_category')['type'].count().sort_values(ascending=False)


# Из поля с временем и датой поста удалось выделить год, месяц, день недели, и время дня. Это понадобится в дальнейшем исследовании

# ## reactions (реакции)
# Извлечем количество реакций на каждое сообщение

# In[22]:


# не правильно работает
#df['number_of_reactions'] = df['reactions'].str.split(':').str[-1]
#df['number_of_reactions'] = df['number_of_reactions'].str.replace(r'\D+','', regex=True).str[-1].fillna(0).astype(int)
#df['number_of_reactions']
#df.groupby('number_of_reactions')['type'].count().sort_values(ascending=False)


# In[23]:


# Функция извлечения и подсчета количеста реакций
sal_l = []
for i in df.index:
    sal = df.iloc[i,11]
    sal = str(sal).replace("'",'"')
    if sal != 'nan':
        sal1 = ast.literal_eval(sal)
        sal3=0
        for j in range(0,10):
            try:
                sal2= sal1[j]["count"]
            except:
                sal2=0
            sal3=sal3+sal2
        sal_l.append(sal3)
    else:
        sal_l.append(0)
df['reactions_qty'] = sal_l
df['reactions_qty'] = df['reactions_qty'].astype(int)
df.head(30)


# Из поля reactions удалось выделить количество реакций на сообщения

# ##  Создадим новый столбец user_type (тип юзера - студент или преподаватель)
# Наши вычисления будут базироваться на предположении, что в каналах "project", "exerciser", "info", "library" создавать трэды могут только преподаватели.

# In[24]:


# создадим новую колонку
df['user_type'] = 'student'
# отфильтруем сообщения старта тредов и нужные каналы
df.loc[
    (df['thread_ts'] == df['ts'])&
    (df["channel_type"].eq("project")|
     df["channel_type"].eq("exerciser")|
     df["channel_type"].eq("info")|
     df["channel_type"].eq("library")),
    'user_type'
] = 'teacher'
# присвоим всем найденым user тип - преподаватель
e = df.loc[df['user_type'] == 'teacher']['user'].unique()
for i in e:
    df.loc[df['user'] == i, 'user_type'] = 'teacher'
df.groupby('user_type')['type'].count().sort_values(ascending=False)


# ## subtype (метка действий пользователя)

# In[25]:


# посмотрим на данные
df.groupby('subtype')['type'].count().sort_values(ascending=False)


# In[26]:


# посчитаем долю служебных сообщений
d = df.groupby('subtype')['type'].count().reset_index()
d.loc[(d['subtype'] != 'bot_message')&(d['subtype'] != 'thread_broadcast')]['type'].sum() / len(df['type'])


# In[27]:


# Удалим служебные сообщения
df = df.loc[(df['subtype'] == 'bot_message')|(df['subtype'] == 'thread_broadcast')|(df['subtype'].isna())]
df.info()


# Не смотря на то , что служебные сообщения составляют 20% от всех сообщений, их придется удалить. Так как в дальнейшем исследовании они будут мешать.

# # Исследовательский анализ данных

# ## За какой период представлены данные

# In[28]:


# Сделаем сводную таблицу по месяцам и построим по ней график
data_by_month = df.groupby('message_month')['type'].count().reset_index().rename(columns ={'type':'qty_message'})
display(data_by_month)
data_by_month.info()
sns.lineplot(data = data_by_month, x='message_month', y='qty_message')
plt.title('Количество сообщений по месяцам')
plt.xlabel('Месяцы')
plt.ylabel('Кол-во сообщений')


# Данные представлены за ноябрь - декабрь 2022 года. Посмотрим более детально

# In[29]:


#cltkftv сводную таблицу по дням и построим по ней график
data_by_day = df.pivot_table(index = 'message_day', columns = 'user_type',values = 'type', aggfunc = 'count')
sns.lineplot(data = data_by_day)
plt.title('Количество сообщений по дням')
plt.xlabel('Дни')
plt.ylabel('Кол-во сообщений')


# Наблюдается резкий всплеск активности первого декабря и такое же резкое падение 15ого декабря у студентов и преподавателей

# ## Даты страрта трэдов
# Посмотрим на даты старта трэдов. Насколько они корелируютс с началом активности студентов.

# In[30]:


# Выделим день старта трэда
df['tread_day'] = df['thread_ts'].dt.date
# выделим время старта трэда
df['tread_time'] = df['thread_ts'].dt.time
# Сгрупируем дату и время трэдов
tred_data = df.pivot_table(index = ['tread_day', 'tread_time']).reset_index()
# Посчитаем сколько трэдов стартовало за каждый день
tred_data1= tred_data.groupby('tread_day')['tread_time'].count()
sns.lineplot(data = tred_data1)
plt.title('Количество стартовавших трэдов по датам')
plt.xlabel('Даты')
plt.ylabel('Кол-во трэдов')


# <h4>Вывод: <a class="tocSkip"> </h4>
#     Активность студентов напрямую корелирует с началом трэдов. Это может свидетельствовать о начале учебы у данных групп 1 ого декабря 2022 года. Так как трэды открываются именно тогда. И всплеск активности студентов соответственно тогда же. 

# ## Активность в трэдах

# In[31]:


# посчитаем разницу в минутах между сообщением и началом трэда
df['time_diff'] = (df['ts'] - df['thread_ts']) / np.timedelta64 ( 1 , 'm')


# In[32]:


# посчитаем количество сообщений в каждом трэде минимальное, максимальное и среднее время ответа
df1 = df.loc[df['time_diff'] != 0]
tread_activity = (
    df1.groupby('thread_ts').
    agg({'time_diff':['count' , 'min', 'max']}).
    droplevel(0, axis=1).
    reset_index().
    sort_values(by = ['count'], ascending=False).
    rename(columns= {'thread_ts': 'трэды', 'count': 'Кол-во сообщений', 'min': 'Время до первого сообщения','max': 'Время до последнего сообщения'})
)
tread_activity


# In[33]:


tread_activity.describe()


# In[34]:


# построим график распределения сообщений по трэдам
tread_activity.hist('Кол-во сообщений', bins=50, range=(1,50))
plt.title('Распределение сообщений по трэдам')
plt.xlabel('Кол-во сообщений')
plt.ylabel('Кол-во трэдов')


# In[35]:


# построим график распределения времени до первого сообщения  по трэдам
tread_activity.hist('Время до первого сообщения', bins=100, range=(1,200))
plt.title('Распределение времени до первого сообщения по трэдам')
plt.xlabel('Времени до первого сообщения, мин')
plt.ylabel('Кол-во трэдов')


# In[36]:


# построим график распределения времени до последнего сообщения  по трэдам
tread_activity.hist('Время до последнего сообщения', bins=100, range=(1,3000))
plt.title('Распределение до последнего сообщения по трэдам')
plt.xlabel('Время до последнего сообщения, мин')
plt.ylabel('Кол-во трэдов')


# <h4>Вывод: <a class="tocSkip"> </h4>
# 
#     Всего в выгрузке 1540 трэдов. Большая часть из них состоит из одного - двух сообщений, по 300 штук. Медианное количество сообщений в трэдах - 3. Таких 150шт. Первая квартиль - 2 сообщения. Третья квартиль - 8 сообщений.  
#     Минимальное время до первого сообщения менее 1ой минуты. Таких трэдов около 140. Медианное время до первого сообщения 75 мин. 1я квартиль - 9минут 3я квартиль - 679минут.  
#     Минимальное время до последнего сообщения так же менее 1ой минуты. Таких трэдов более 50ти. В них всего одно сообщение. Медианное время до последнего сообщения 835 мин. 1я квартиль - 107 минут 3я квартиль - 3000 минут

# ## Высоконагруженные трэды
# Рассмотрим отдельно "Высоконагруженные" трэды. Будем считать таковыми, трэды с количеством сообщений более 50и.

# In[37]:


# Выделим высоконагруженные трэды ... более 50 сообщений
high_perfomance_tread = tread_activity.loc[tread_activity['Кол-во сообщений'] > 50]
high_perfomance_tread


# In[38]:


high_perfomance_tread.describe()


# In[39]:


# построим график распределения количества сообщений по нагруженым трэдам
high_perfomance_tread.hist('Кол-во сообщений', bins=64, range=(50,300))
plt.title('Распределение сообщений по нагруженным трэдам')
plt.xlabel('Кол-во сообщений')
plt.ylabel('Кол-во трэдов')


# In[40]:


# построим график распределения времени до первого сообщения  по нагруженным трэдам
high_perfomance_tread.hist('Время до первого сообщения', bins=50, range=(1,1000))
plt.title('Распределение времени до первого сообщения по трэдам')
plt.xlabel('Времени до первого сообщения, мин')
plt.ylabel('Кол-во трэдов')


# In[41]:


# построим график распределения времени до последнего сообщения  по нагруженным трэдам
high_perfomance_tread.hist('Время до последнего сообщения', bins=100, range=(10000,40000))
plt.title('Распределение до последнего сообщения по трэдам')
plt.xlabel('Время до последнего сообщения, мин')
plt.ylabel('Кол-во трэдов')


# <h4>Вывод: <a class="tocSkip"> </h4>
# Выделили нагруженные трэды, с количеством сообщений более 50и. Таких получилось 64 штуки. Но только в двух из них первое сообщение появилось в течение первого часа. В основном в таких трэдах первые сообщения появляются в течение первых суток. "Замолкают" же такие трэды в основном через 17 дней. 

# ## К каким каналам относятся высоконагруженные трэды

# In[42]:


# Сделаем сводную количества сообщений в когортах и разных каналах
high_perfomance_tread['трэды'] = pd.to_datetime(high_perfomance_tread['трэды'], unit='s')
df2 = df1.merge(high_perfomance_tread, how='inner', left_on='thread_ts', right_on='трэды')
high_perfomance_channel = df2.pivot_table(index='chorot_name' , columns = 'channel_type', values = 'type', aggfunc = 'count').fillna(0).astype('int')
fig_2 = sns.heatmap(high_perfomance_channel,  annot= True,  fmt=" d")
fig_2.set_title('Тепловая карта каналов и групп высоконагруженных когорт')
fig_2.set_xlabel('Название каналов')
fig_2.set_ylabel('Название когорт')


# <h4>Вывод: <a class="tocSkip"> </h4>
# Из тепловой карты становится видно, что самые нагруженные каналы это "exerciser" и "project". И самые активные профессии это da и ds. Самая активная группа da_59 (570 сообщений в exerciser и 232 в project), на втором месте ds_55 (676 сообщений в project). Необходимо внимательнее посмотртреть на группы в которых в одном канале высокая активность, в другом низкая. Например ds_55

# ## Распределение активности по дням недели

# In[43]:


# Сделаем сводную распределения сообщений по дням недели
message_by_day = df.groupby(['message_weekday', 'user_type'])['type'].count().reset_index().rename(columns= {'message_weekday': 'День недели', 'type': 'Кол-во сообщений'})
display(message_by_day)
fig_1 = px.bar(message_by_day, x='День недели', y='Кол-во сообщений', color="user_type", title='Количество сообщений по дням')
fig_1.update_xaxes(tickangle=45)
fig_1.update_layout(xaxis_title="День недели", yaxis_title="Кол-во сообщений")
fig_1.show() 


# <h4>Вывод: <a class="tocSkip"> </h4>
#     Подсчет количества сообщений по дням привел к следующим результатам. Наибольшая активность, как преподавателей, так и студентов наблюдается в Понедельник. Более 4к сообщений от студентов и более 1,5к сообщений от преподавателей. Самая низкая активность в Субботу и Воскресенье, по 1.3к сообщений от студентов и около 300 у преподпвателей. 
#     По остальным дням активность преподавателей распределяется равномерно. 

# ## Распределение активности по времени суток

# In[44]:


# Сделаем сводную распределения по времени суток
message_by_time = df.groupby(['time_category', 'user_type'])['type'].count().reset_index().rename(columns= {'time_category': 'Время суток', 'type': 'Кол-во сообщений'})
display(message_by_time)
fig_1 = px.bar(message_by_time, x='Время суток', y='Кол-во сообщений', color="user_type", title='Количество сообщений по времени суток')
fig_1.update_xaxes(tickangle=45)
fig_1.update_layout(xaxis_title="Время суток", yaxis_title="Кол-во сообщений")
fig_1.show() 


# <h4>Вывод: <a class="tocSkip"> </h4>
#     Подсчет количества сообщений по времени суток выявил, что основная активность у преподавателей и студентов утром и днем после обеда.

# ## Распредеделение активности по дням недели и по суткам 

# In[45]:


# Сделаем сводную по дням недени и времени суток
message_by_day_time = df.pivot_table(index = ['time_category', 'user_type'], columns = 'message_weekday', values = 'type', aggfunc= 'count').fillna(0).astype('int')
fig_3 = sns.heatmap(message_by_day_time,  annot= True,  fmt=" d")
fig_3.set_title('Тепловая карта количества сообщений от дня недели и времени суток')
fig_3.set_xlabel('Дни недели')
fig_3.set_ylabel('Время суток')


# Подсчет количества сообщений в зависимости от дня недели и суток привел к следущим результатам:  
#   * Понедельник утро - максимальная активность студентов и преподавателей.
#   * Понедельник после обеда - высокая активность студентов. Активность преподавателей средняя.
#   * Любой другой день после обеда - Высокая активность у студентов, низкая у преподавателей.
#   * В остальное время активность у студентов и преподавателей - низкая.
#   * Суббота и Воскресенье - низкая активность весь день.

# ## Распредеделение активности по дням недели и по суткам у "Активных" когорт

# In[46]:


# выделим активные когорты (более 500 сообщений)
high_activity_chorot = df.groupby('chorot_name')['type'].count().reset_index().sort_values(by = 'type', ascending=False)
high_activity_chorot = high_activity_chorot.loc[(high_activity_chorot['chorot_name'] != 'other')&(high_activity_chorot['type'] > 500)]
high_activity_chorot


# In[47]:


df3 = df.merge(high_activity_chorot, how='inner', left_on='chorot_name', right_on='chorot_name')
message_by_day_time_1 = df3.pivot_table(index = ['time_category', 'user_type'], columns = 'message_weekday', values = 'type_x', aggfunc= 'count').fillna(0).astype('int')
fig_3 = sns.heatmap(message_by_day_time_1,  annot= True,  fmt=" d")
fig_3.set_title('Тепловая карта количества сообщений от дня недели и времени суток Активных когорт')
fig_3.set_xlabel('Дни недели')
fig_3.set_ylabel('Время суток')


# Принципиально ничего не поменялось. Тепловая карта активных груп повторяет общую картину.

# ## Проверка активности студентов

# In[48]:


# Посчитаем количество сообщений и реакций у студентов
df4 = df.query('user_type == "student"').groupby('user').agg({'type':'count' , 'reactions_qty':'sum'}).reset_index().rename(columns={'type':'message_qnt'})
display(df4.sort_values(by = 'reactions_qty'))
# введем категории активности студентов
df4['student_activity'] = 'Средне_активный'
df4.loc[(df4['message_qnt'] > 100)|(df4['reactions_qty'] > 50), 'student_activity']='Активный'
df4.loc[(df4['message_qnt'] < 10)&(df4['reactions_qty'] < 10), 'student_activity']='Пассивный'
layout = go.Layout(title='График соотношения активности студентов')
fig = go.Figure(data = [go.Pie(labels = df4['student_activity'])], layout = layout)
fig


# <h4>Вывод: <a class="tocSkip"> </h4>
#     Выявилась еще одна неожиданная особенность. 83% всех студентов пассивно ведут себя в чатах (то  есть, за все время исследования написали менее 10 сообщений и при этом поставили менее 10 реакций) А активных студентов, которые более 100 сообщений или поставили более 50 реакций, менее 1%.  

# ## Проверка активности преподавателей.

# In[49]:


# Проанализируем преподавателей так же как и со студентов. 
df5 = df.query('user_type == "teacher"').groupby('user').agg({'type':'count' , 'reactions_qty':'sum'}).reset_index().rename(columns={'type':'message_qnt'})
display(df5.sort_values(by = 'reactions_qty'))
df5['student_activity'] = 'Средне_активный'
df5.loc[(df5['message_qnt'] > 100)|(df5['reactions_qty'] > 50), 'student_activity']='Активный'
df5.loc[(df5['message_qnt'] < 10)&(df5['reactions_qty'] < 10), 'student_activity']='Пассивный'
layout = go.Layout(title='График соотношения активности преподавателей')
fig = go.Figure(data = [go.Pie(labels = df5['student_activity'])], layout = layout)
fig


# <h4>Вывод: <a class="tocSkip"> </h4>
#     У преподавателей ситуация выглядит лучше, чем у студентов. Половина преподавателей - активны в чатах. Но тем не менее 56% преподователей написали менее 10и сообщений.

# ## Распределение сообщений по количеству слов

# In[50]:


# посчитаем длину сообщений
qty_words = df.groupby(['user_type','text_words'])['type'].count().reset_index().rename(columns={'text_words':'qty_words','type':'qty_message'})
display(qty_words.describe())
fig_3 = px.bar(qty_words, x='qty_words', y='qty_message', color="user_type", title='Соотношение Кол-ва сообщений к кол-ву слов')
fig_3.update_xaxes(tickangle=45)
fig_3.update_xaxes(range=[0 , 300])
fig_3.update_layout(xaxis_title="Кол-во слов", yaxis_title="Кол-во сообщений")
fig_3.show() 


# 
#     И студенты и преподаватели пишут в основном короткие сообщения. От 2ух до 10 слов. Таких сообщений по 600 шт у студентов и по 150шт у преподавателей. Выведем более наглядный график.

# In[51]:


# Заведем категории до 10 слов - короткое сообщение, 10 - 50 слов нормальное сообщение, более 50 длинное
df['message_category'] = 'Нормальное сообщение'
df.loc[(df['text_words'] > 50), 'message_category']='Длинное сообщение'
df.loc[(df['text_words'] < 10), 'message_category']='Короткое сообщение'
df_5 = df.query('user_type == "student"')
df_6 = df.query('user_type == "teacher"')
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
#layout = go.Layout(title='График соотношения длины сообщений')
fig.add_trace(go.Pie(labels = df_5['message_category']), 1, 1)
fig.add_trace(go.Pie(labels = df_6['message_category']), 1, 2)
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(height=800, width=1000, 
                  title_text="Соотношение кол-ва сообщений к длине сообщений",
                  annotations=[dict(text='Студенты', x=0.15, y=0.5, font_size=20, showarrow=False), dict(text='Тренеры', x=0.85, y=0.5, font_size=20, showarrow=False)])

fig.show()


# <h4>Вывод: <a class="tocSkip"> </h4> 
#     По данным графикам становится видно что и студенты и преподаватели в половине случаев пишут средней длины сообщения (от 10 до 50 слов) или короткие - менее 10 слов (менее 30% случаев). Длинные сообщения преподаватели пишуть чаще в 16% случаев против 10% случаев у студентов.

# ## Анализ особо длинных сообщений

# In[52]:


# Выделим длинные сообщения (более 200 слов)
df_7 =  df.query('text_words > 200')


# ### Кто пишет

# In[53]:


layout = go.Layout(title='Авторы особо длинных сообщений')
fig = go.Figure(data = [go.Pie(labels = df_7['user_type'])], layout = layout)
fig


# Авторы длинных сообщений поделились пополам. В половине случаев студенты, в половине - преподаватели

# ### Когда пишут 

# In[54]:


message_by_day_time_2 = df_7.pivot_table(index = ['time_category', 'user_type'], columns = 'message_weekday', values = 'type', aggfunc= 'count').fillna(0).astype('int')
fig_3 = sns.heatmap(message_by_day_time_2,  annot= True,  fmt=" d")
fig_3.set_title('Тепловая карта количества сообщений от дня недели и времени суток. Длинные сообщеня')
fig_3.set_xlabel('Дни недели')
fig_3.set_ylabel('Время суток')


# Традиционно, авторы длинных сообщений активны в понедельник весь день кроме вечера и ночи. Преподаватели активны еще в четверг. 

# ### Популярны ли данные сообщения

# In[55]:


# построим сводную и график количества реакций на длинные сообщения
reactions_long_mess = df_7.groupby(['user_type', 'reactions_qty'])['type'].count().fillna(0).reset_index()
reactions_long_mess
fig_1 = px.bar(reactions_long_mess, x='reactions_qty', y='type', color="user_type", title='Количество реакций длинных сообщений')
fig_1.update_layout(xaxis_title="Кол-во реакций", yaxis_title="Кол-во сообщений")
fig_1.show() 


# НЕТ. Данные сообщения не собирают много реакций. в большинстве случаев пользователи на них вообще не реагируют. На отдельные сообщения преподавателей есть достаточно бурная реакция. 

# <h4>Вывод: <a class="tocSkip"> </h4> 
#     Анализ особо длинных сообщений показал, Что пишут их как преподаватели, иак и студенты. В пропорции 50/50. На сообщения преподавателей реакций больше. И пишутся они примерно в то же время как и остальные сообщения(Воскресенье - Понедельник)

# # Общий вывод

# В качестве объекта исследования была получена выгрузка данных из месенджера Slak. В таблице 26533 записей, и 14 столбцов. 7 столбцов - текстовые, 4 - с датами и временем, 2-целочисленные, 1 - булевый. 7 столбцов не имеют пропусков. Остальные семь имеют пропуски от 10% до 91%. Больше всего пропусков в столбцах с реакциями и временем ответа 84 и 91 процент.
# 
# Была выполнена предобработка данных. Были выделенны данные: названия групп, типы каналов, года, месяцы, дни недель сообщений. Введены категории определяющие время суток,  длину сообщения. Пользователей разделили на студентов и преподавателей. Удалены служебные сообщения.
# 
# В изучаемом датафрейме представлены данные о сообщениях с начала ноября 2022 года по конец декабря 2022 года. Большинство трэдов стартуют 1 декабря. Активность в трэдах длится до 15 декабря. Всего в выгрузке 1540 трэдов. Большая часть из них состоит из одного - двух сообщений, по 300 штук. Медианное количество сообщений в трэдах - 3. Таких 150шт. Первая квартиль - 2 сообщения. Третья квартиль - 8 сообщений.  
# Минимальное время до первого сообщения менее 1ой минуты. Таких трэдов около 140. Медианное время до первого сообщения 75 мин. 1я квартиль - 9минут 3я квартиль - 679минут.  
# Минимальное время до последнего сообщения так же менее 1ой минуты. Таких трэдов более 50ти. В них всего одно сообщение. Медианное время до последнего сообщения 835 мин. 1я квартиль - 107 минут 3я квартиль - 3000 минут.  
# Выделили нагруженные трэды, с количеством сообщений более 50и. Таких получилось 64 штуки. Но только в двух из них первое сообщение появилось в течение первого часа. В основном в таких трэдах первые сообщения появляются в течение первых суток. "Замолкают" же такие трэды в основном через 17 дней.
# Нагруженные трэды относятся к каналам "exerciser" и "project". И самые активные профессии это da и ds. Самая активная группа da_59 (570 сообщений в exerciser и 232 в project), на втором месте ds_55 (676 сообщений в project). Необходимо внимательнее посмотртреть на группы в которых в одном канале высокая активность, в другом низкая. Например ds_55.  
# Подсчет количества сообщений по дням недели привел к следующим результатам:
#  - Понедельник утро - максимальная активность студентов и преподавателей.
#  - Понедельник после обеда - высокая активность студентов. Активность преподавателей средняя.
#  - Любой другой день после обеда - Высокая активность у студентов, низкая у преподавателей.
#  - В остальное время активность у студентов и преподавателей - низкая.
#  - Суббота и Воскресенье - низкая активность весь день.  
# По остальным дням активность преподавателей распределяется равномерно.  
# Выявилась еще одна неожиданная особенность. 83% всех студентов пассивно ведут себя в чатах (то есть, за все время исследования написали менее 10 сообщений и при этом поставили менее 10 реакций) А активных студентов, которые более 100 сообщений или поставили более 50 реакций, менее 1%.  
# У преподавателей ситуация выглядит лучше, чем у студентов. Половина преподавателей - активны в чатах. Но тем не менее 56% преподователей написали менее 10и сообщений.  
# Анализ особо длинных сообщений не выявил каких либо паттернов. 

# # Рекомендации

# В связи с тем, что исследование показало отвутствие активности в чатах 50% преподавателей, необходимо провести отделльное исследование какие преподаватели должны быть активными в чате и разработать систему мотивации активности в чатах.   
# 
# Такой же подход можно применить к студентам. 80% студентов не активны в чатах. Система мотивации значительно увеличит данный показатель. Например поставь 100 реакций или собери 100 лайков и получи скидку на следующий месяц.
# 
# Что касается времени рассылки важных сообщений - Понедельник утро для этого тот самый момент. 
