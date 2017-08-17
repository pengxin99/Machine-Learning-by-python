import urllib.request
import os
import random


# 打开网页函数
def url_open(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')

    response = urllib.request.urlopen(url)
    html = response.read()

    return html


# 寻找文章的地址函数
def find_paper_links(url):
    html = url_open(url).decode('utf-8')
    paper_addrs = []

    a = html.find('<a href="/sinat_')

    while a != -1:
        b = html.find('title="阅读次数">', a , a + 255)
        if b != -1:
            paper_addrs.append( html[a+16:b-2])
        else:
            b = a + 9
        a = html.find('<a href="/sinat_',b)

    return paper_addrs



# 访问网址的主函数
def visit_url():
	# 总的访问量
    count = 0

    url = 'http://blog.csdn.net/sinat_34022298?viewmode=contents'
    links = find_paper_links(url)
    for i in range(len(links)):
        page_url = 'http://blog.csdn.net/sinat_' + links[i]
        print(page_url)
        # 访问网页
        url_open(page_url)
        count += 1
        print("this is NO: " + str(count))
        
# 主函数入口
if __name__  ==  '__main__':
    visit_url()
    