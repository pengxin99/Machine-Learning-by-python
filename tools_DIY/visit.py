import urllib.request
import os
import random


# 打开网页
def url_open(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')

    response = urllib.request.urlopen(url)
    html = response.read()

    return html


# 找到文章的地址
def find_imgs(url):
    html = url_open(url).decode('utf-8')
    img_addrs = []

    a = html.find('<a href="/sinat_')

    while a != -1:
        b = html.find('title="阅读次数">', a , a + 255)
        if b != -1:
            img_addrs.append( html[a+16:b-2])
        else:
            b = a + 9
        a = html.find('<a href="/sinat_',b)

    return img_addrs


# 保存图片
def save_imgs(folder,img_addrs):
    i = 1
    for each in img_addrs:
        img = url_open(each)
#        pic_name = str(i) + '.jpg'
        pic_name = each.split('/')[-1]
        with open(pic_name,'wb') as f:
            f.write(img)
        i += 1
    return i


# 下载图片的主函数
def download_pic(folder = 'mm',pages = 10):
   #os.mkdir(folder)
    #os.chdir(folder)
    count = 0

    url = 'http://blog.csdn.net/sinat_34022298?viewmode=contents'
    link = find_imgs(url)
    for i in range(len(link) - 1):
        
        page_url = 'http://blog.csdn.net/sinat_' + link[i]
        print(page_url)
        url_open(page_url)
        i += 1
        print("this is NO: " + str(i))
        
# 主函数入口
if __name__  ==  '__main__':
    download_pic()
    