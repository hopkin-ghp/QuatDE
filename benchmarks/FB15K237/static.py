import os

def static():
    path = './train2id.txt'
    head_dic = {}
    tail_dic = {}
    cnt = 0
    with open(path, 'r') as f:
        for raw in f.readlines():
            raw = raw.strip().split(' ')
            try:
                head, tail, r = raw
                if head not in head_dic.keys():
                    head_dic[head] = 1
                else:
                    head_dic[head] = head_dic[head] + 1

                if tail not in tail_dic.keys():
                    tail_dic[tail] = 1
                else:
                    tail_dic[tail] = tail_dic[tail] + 1
            except:
                cnt = cnt + 1
                continue

    head_cnt = 0
    head_mean = 0

    tail_cnt = 0
    tail_mean = 0

    for key in head_dic.keys():
        head_cnt = head_cnt + head_dic[key]
    for key in tail_dic.keys():
        tail_cnt = tail_cnt + tail_dic[key]

    head_mean = head_cnt / len(head_dic.keys())
    tail_mean = tail_cnt / len(tail_dic.keys())

    print(len(head_dic.keys()))
    print(head_mean)
    print(len(tail_dic.keys()))
    print(tail_mean)

def url2name(number):
    with open('./entity2id.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            a, b = line.strip().split('	')
            if int(b) == number:
                str = a
                print(str)
                break
    url_path = './fb2w.nt'
    # f_write = open('./entity2id.txt', 'w')
    print(str)

    with open(url_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('	')
            if len(line) == 3:
                line.pop(1)
                m, url = line
                m = '/' + m.split('/')[-1].replace('>', '').replace('.', '/')
                url = url[1:-3]
                if m == str:
                    print('\t'.join([m, url]))
                    break

def pachong(url):
    from lxml import etree
    import requests
    import urllib.parse
    import urllib3
    from bs4 import BeautifulSoup
    import re
    import random
    import time
    import xlwt
    requesHeader = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'max-age=0',
        'Host': 'www.guidelines-registry.org',
        'Proxy-Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'http://www.guidelines-registry.org/guid?lang=zh_CN&page=2&limit=10',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'
    }
    requesHeader1 = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Cookie': 'SESSION=NzBkZjlhMmYtYjEyNS00NDUyLTg1ZDktOTlkZDYxYTYyMTVi',
        'Host': 'www.guidelines-registry.org',
        'Referer': 'http://www.guidelines-registry.org/guid?lang=zh_CN',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'
    }
    http = urllib3.PoolManager(num_pools=10, headers=requesHeader1)
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('Sheet1')
    resp2 = http.request('GET', url)
    print(resp2.data)
    soup2 = BeautifulSoup(resp2.data, "html.parser")
    print(soup2)


if __name__ == "__main__":
    # static()  # 统计数据集信息

    # 5438 69 30 (tootsie, BAFTA Award for Best Makeup and Hair, award)
    # 268 5438 32(Warner Bros., toorsie, film)
    # 268 4427 32
    # 268 618 32
    # 268 7480 32
    # 268 1504 32
    # 10747 268 8
    # 268 2973 32
    # 268 7261 32

    # 6651 637 13（Winnie the Pooh， 法国， 电影地址）
    # 6651 293 13
    # 6651 816 13 (,danmai,)
    # 6651  68 13
    # 3494 6651 6（Jim Cummings， Winnie the Pooh， 出演）
    # 6651 3028 13
    # 6651 1960 26（， 声音编辑器，电影摄制组角色）

    # 6571 1155 8 (Brenda Strong, Alfre Woodard, 奖项)
    # 1155 6571 3 (Alfre Woodard，Brenda Strong, 获得)
    # 2346 6571 8 (Lyndsy Fonseca, Brenda Strong, 奖项)
    # 6571 2346 8(Brenda Strong, Brenda Strong, 奖项)
    # 3023 6571 8 Kathryn Joosten
    # 6571 7554 6 (Brenda Strong, Starship Troopers(movie),扮演)
    # 7554 1098 26 (Starship Troopers(movie), )

    # 8605 2974 84 Priest, Michael De Luca, produce by
    # 291 2974 136 Austin Powers: The Spy Who Shagged Me， Michael De Luca, executive_produced_by
    # 9022 2974 136 The Mask， Michael De Luca, executive_produced_by
    # 9022 136 31 The Mask，action film， /film/film/genre
    # 3504 9022 6 Ben Stein，The Mask，扮演
    # 9022 1885 90 The Mask, streaming media(流媒体),film distribution medium
    # 69 9022 7 BAFTA Award for Best Makeup and Hair, The Mask, nominated for

    # 1224 7261 102 Brett Ratner, Rush Hour
    # 2181 660 102 Rob Zombie, Grindhouse
    # 8883 9143 102 John Landis, Coming to America
    # 7557 3822 102 Ron Howard, Willow
    # 4204 7539 102
    # 8531 5744 102
    # 5121 3066 102
    # 2561 11402 102 Ridley Scott, Blade Runner
    # 2561 3508 102 Ridley Scott, Legend
    # 1239 761 102
    # 5510 1003 102
    # 8647 4583 102
    # 1917 2460 102 Steven Spielberg, 1941 1-n
    # 1917 3950 102 Schindler's List
    # 1917 4282 102 Steven Spielberg, Indiana Jones and the Kingdom of the Crystal Skull
    # 1917 928 159 Steven Spielberg, Universal Pictures, 雇佣公司 n-1
    # 879 1917 71 Jewish people， Steven Spielberg, 民族  1-n
    # 1127 1917 47 Academy Award for Best Director， Steven Spielberg, award_winner n-n
    # 1420 1917 45 Bill Clinton，  Steven Spielberg, friendship n-n
    # 1420 141 43
    # 1420 12777 224 Fayetteville, location of ceremony n-1
    # 1420 44 82 Christianity, religion n-1
    # 1420 4489 236 Arkansas, jurisdiction of office n-1
    # 1420 236 179 president, government position held n-n
    #
    # 5243 1917 47
    # 1917 3950 38
    # 1917 4272 38
    # 1917 1420 45
    # 1917 1127 19
    # 1846 1917 47 Golden Globe Cecil B. DeMille Award
    # 1917 1846 19

    # 5328 1371 192 Orson Welles Rita Hayworth
    # 13858 13158 192 Suhasini Maniratnam Mani Ratnam
    # 262 2454 192 Patricia Arquette Thomas Jane
    # 3888 12165 192 Mary Steenburgen Malcolm McDowell

    ll = [1917, 2460, 3950, 4282]
    if True:
        number = 12165
        url2name(number)
    else:
        for number in ll:
            url2name(number)