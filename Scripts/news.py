from GoogleNews import GoogleNews

googlenews = GoogleNews()
googlenews = GoogleNews(period ='7d')
googlenews.search('Greece')
result=googlenews.result()
for x in result:
    print("_"*50)
    print("Title--",x['title'])
    print("Date/Time--",x['date'])
    print("description--",x['desc'])
    print("Link--",x['link'])
    