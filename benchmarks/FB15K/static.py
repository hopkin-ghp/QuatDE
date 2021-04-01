import os

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

