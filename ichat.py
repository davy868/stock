#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import time
import itchat
from itchat.content import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import operator
import csv
import math
from matplotlib import pyplot
import gc
from sqlalchemy import create_engine
import tushare as ts
import pymysql
import pandas as pd
ISOTIMEFORMAT='%Y-%m-%d %X'

@itchat.msg_register([TEXT, PICTURE, MAP, CARD, NOTE, SHARING, RECORDING, ATTACHMENT, VIDEO])
def text_reply(msg):
    if msg['Type'] == 'Text':
        reply_content = msg['Text']
    elif msg['Type'] == 'Picture':
        reply_content = r"图片: " + msg['FileName']
    elif msg['Type'] == 'Card':
        reply_content = r" " + msg['RecommendInfo']['NickName'] + r" 的名片"
    elif msg['Type'] == 'Map':
        x, y, location = re.search("<location x=\"(.*?)\" y=\"(.*?)\".*label=\"(.*?)\".*", msg['OriContent']).group(1,
                                                                                                                    2,
                                                                                                                    3)
        if location is None:
            reply_content = r"位置: 纬度->" + x.__str__() + " 经度->" + y.__str__()
        else:
            reply_content = r"位置: " + location
    elif msg['Type'] == 'Note':
        reply_content = r"通知"
    elif msg['Type'] == 'Sharing':
        reply_content = r"分享"
    elif msg['Type'] == 'Recording':
        reply_content = r"语音"
    elif msg['Type'] == 'Attachment':
        reply_content = r"文件: " + msg['FileName']
    elif msg['Type'] == 'Video':
        reply_content = r"视频: " + msg['FileName']
    else:
        reply_content = r"消息"

    friend = itchat.search_friends(userName=msg['FromUserName'])
    itchat.send(r"Friend:%s -- %s    "
                r"Time:%s    "
                r" Message:%s" % (friend['NickName'], friend['RemarkName'], time.ctime(), reply_content),
                toUserName='filehelper')

    # itchat.send(r"我已经收到你在【%s】发送的消息【%s】稍后回复。--微信助手(Python版)" % (time.ctime(), reply_content),
    #             toUserName=msg['FromUserName'])

itchat.auto_login()

stock_number = []
with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "r") as csvfile:
	reader = csv.reader(csvfile)
	for item in reader:
		stock_number = item

stock_pairs = []
with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_hello_world.csv", "r") as csvfile:
	reader = csv.reader(csvfile)
	for item in reader:
		stock_pairs.append(item[0:2])

stock_pairs_leader=[]
stock_pairs_follower=[]
for i in range(len(stock_pairs)):
	stock_pairs_leader.append(stock_number[int(float(stock_pairs[i][0]))])
	stock_pairs_follower.append(stock_number[int(float(stock_pairs[i][1]))])

now_time=time.strftime( ISOTIMEFORMAT, time.localtime() )

stock_numer_real=np.zeros((len(stock_pairs),3),dtype=np.float32)
for i in range(len(stock_pairs)):
	stock_numer_real[i][0]=stock_pairs_leader[i]
	stock_numer_real[i][1] = stock_pairs_follower[i]

for i in range(len(stock_numer_real)):
	for j in range(len(stock_numer_real)):
		if stock_numer_real[j][0]==stock_numer_real[i][0]:
			stock_numer_real[i][2]+=1
stock_real_number_index=np.argsort(stock_numer_real[:,2])
stock_numer_real_sorted=np.zeros_like(stock_numer_real)
for i in range(len(stock_real_number_index)):
	stock_numer_real_sorted[i][0:3]=stock_numer_real[stock_real_number_index[i]][0:3]

# print np.average(stock_numer_real_sorted[:,2])
#
# print list(set(stock_numer_real_sorted[:,0]))

while now_time[11:16] != '10:10':
	time.sleep(20)
	now_time = time.strftime(ISOTIMEFORMAT, time.localtime())


stock_number_today_follower=[]
for i in range(0,len(stock_pairs_leader),20):
	df = ts.get_realtime_quotes(stock_pairs_leader[i:i+20])
	for t in range(df['pre_close'].size):
		if (float(df['high'][t])+0.00001)/(float(df['pre_close'][t])+0.00001) > 1.05 and  (float(df['high'][t])+0.00001)/(float(df['pre_close'][t])+0.00001) < 1.109 :
			temp_increae=(float(df['high'][t])+0.00001)/(float(df['pre_close'][t])+0.00001)
			for j in range(len(stock_numer_real_sorted)):
				if float(stock_pairs_leader[i+t])==stock_numer_real_sorted[j][0]:
					stock_number_today_follower.append(
						'F:' + stock_pairs_follower[i + t] + ' ' + 'L:' + stock_pairs_leader[i + t] + '-'+str(int(stock_numer_real_sorted[j][2])) +' '+ str(
							round(temp_increae * 100 - 100, 1)) + ' ' + str(1010 - i - t))
					break

itchat.send(r"%s today's binggo stock is" % (time.ctime()),'filehelper')
for i in range(len(stock_number_today_follower)):
	itchat.send(str(i)+'th '+stock_number_today_follower[i],'filehelper')

itchat.send(r"today's binggo stock is over,sum is %d" % (len(stock_number_today_follower)),'filehelper')

print 'OK'

# # coding:utf-8
# import itchat,time,sys,xlwt
#
# file = xlwt.Workbook()
# table = file.add_sheet('info',cell_overwrite_ok=True)
#
#
#
# # 登录-持续
# itchat.auto_login()
# print(u"logged")
# # 获取好友列表
# friends = itchat.get_friends(update=True)[0:]
#
# male = female = other = 0
#
# for i in friends[1:]:
#     sex = i["Sex"]
#     if sex == 1:
#         male += 1
#     elif sex == 2:
#         female += 1
#     else:
#         other += 1
# total = len(friends[1:])
#
# table.write(0,5,u'【made by junzi】')
# table.write(0,7,u'【共'+str(len(friends)-1)+u'位朋友，'+str(male)+u'位男性朋友，'+str(female)+u'位女性朋友，另外'+str(other)+u'位不明性别】')
# table.write(0,0,u' 【昵称】')
# table.write(0,1,u' 【备注名】')
# table.write(0,2,u' 【省份】')
# table.write(0,3,u' 【城市】')
# table.write(0,4,u' 【签名】')
#
# a=0
#
# for i in friends:
#
#
#     table.write(a+1,0,i['NickName'])
#     table.write(a+1,1,i['RemarkName'])
#     table.write(a+1,2,i['Province'])
#     table.write(a+1,3,i['City'])
#     table.write(a+1,4,i['Signature'])
#     if i['RemarkName'] == u'':
#         table.write(a+1,1,u'[ ]')
#     if i['Province'] == u'':
#         table.write(a+1,2,u'[ ]')
#     if i['City'] == u'':
#         table.write(a+1,3,u'[ ]')
#     if i['Signature'] == u'':
#         table.write(a+1,4,u'[ ]')
#
#
#
#     a=a+1
#     print(a)
#
#
# # qm=raw_input("file name >>>:")
# aaa='weixin_'+time.strftime("%Y%m%d", time.localtime())+'.xls'
# file.save(aaa)
# itchat.send('made by junzi','filehelper')
# itchat.send('@%s@%s' % ('fil',aaa), 'filehelper')
# print ("over")
