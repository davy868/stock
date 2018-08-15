#coding:utf-8
from sqlalchemy import create_engine
import tushare as ts
import pymysql
import csv
import numpy as np
import time
import pandas as pd

stock_number = []
with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "r") as csvfile:
	reader = csv.reader(csvfile)
	for item in reader:
		stock_number = item
# for i in range(len(stock_number)):
# 	stock_number[i]=stock_number[i][0:-2]


df = ts.get_realtime_quotes('601398')


conn=pymysql.connect(host='127.0.0.1',port=3306,user='root',passwd='Luo0046sq',db='mysql',charset='utf8',cursorclass=pymysql.cursors.DictCursor)
cur=conn.cursor()
cur.execute("USE stock_5min_from20151201")

load_command=[]

for i in range(len(stock_number)):

	load_command.append("select * from stock_5min_from20151201." + stock_number[i]+ " into outfile '/var/lib/mysql-files/"+stock_number[i]+".csv' fields terminated by ',' enclosed by '"+'"'+ "'"+' lines terminated by '+"'"+"\r\n';")


print  'OK'

for i in range(len(stock_number)):
	cur.execute(load_command[i])
#	cur.execute("select * from stock_5min_from20151201.600000 into outfile \'/var/lib/mysql-files/600000.csv\' fields terminated by \',\' enclosed by \'\"\' lines terminated by \'\r\n\';")
# ret= cur.fetchall()

# stock_number_database=['']
# for numb in ret:
# 	stock_number_database.append(str(numb.values())[3:9])
#
#
# cons=ts.get_apis()
#
# #df=ts.bar('600000',conn=cons,freq='5min',start_date='2016-05-01',end_date='2018-06-05')
#
# engine = create_engine('mysql+pymysql://root:Luo0046sq@localhost/stock_5min_from20151201?charset=utf8')
#
# start=['2015-12-21','2016-01-06','2016-01-21','2016-02-06','2016-02-21','2016-03-06','2016-03-21','2016-04-06','2016-04-21',
#        '2016-05-06','2016-05-21','2016-06-06','2016-06-21','2016-07-06','2016-07-21','2016-08-06','2016-08-21','2016-09-06',
#        '2016-09-21','2016-10-06','2016-10-21','2016-11-06','2016-11-21','2016-12-06','2016-12-21','2017-01-06','2017-01-21',
#        '2017-02-06','2017-02-21','2017-03-06','2017-03-21','2017-04-06','2017-04-21','2017-05-06','2017-05-21','2017-06-06',
#        '2017-06-21','2017-07-06','2017-07-21','2017-08-06','2017-08-21','2017-09-06','2017-09-21','2017-10-06','2017-10-21',
#        '2017-11-06','2017-11-21','2017-12-06','2017-12-21','2018-01-06','2018-01-21','2018-02-06','2018-02-21','2018-03-06',
#        '2018-03-21','2018-04-06','2018-04-21','2018-05-06','2018-05-21','2018-06-06','2018-06-21']
#
# for i in range(len(stock_number)):
# 	#t1=time.time()
# 	if stock_number[i] not in stock_number_database:
#
# 		# df = ts.get_k_data(stock_number[i], ktype='5',start='2015-12-18',end='2018-06-05')
# 		# while
# 		# if len(df.index)>16:
# 		# 	data=df.drop(df.index[0:16])
# 		# 	data.reset_index(drop=True,inplace=True)
#
# 		v=0
# 		for u in range(len(start)):
# 			data=ts.get_k_data(stock_number[i], ktype='5', start=start[u], end='2018-06-05')
# 			if len(data.index)>1:
# 				v=u+1
# 				break
#
# 		for j in range(v,len(start)):
# 			df = ts.get_k_data(stock_number[i], ktype='5', start=start[j], end='2018-06-05')
# 			t=0
# 			if len(df.index)>1 :
# 				while data.ix[data.index[-1]][0][0:16] != df.ix[df.index[t]][0][0:16]:
# 					t=t+1
# 					if t>639:
# 						break
# 				if t<640:
# 					df.drop(df.index[0:t+1],inplace=True)
# 					data=data.append(df,ignore_index=True)
# 				else:
# 					data = data.append(df, ignore_index=True)
#
# 		data.to_sql(stock_number[i], engine)
# 	#print time.time() - t1
#
#
# 	#df.drop(df.index[0:16],inplace=True)

conn.close()
cur.close()

print 'OK'