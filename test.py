import tushare as ts

stockcode=ts.get_today_all()["code"]

for code in stockcode:
    df=ts.get_k_data(code)
    print df

print "OK"