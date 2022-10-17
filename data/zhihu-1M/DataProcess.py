import json
import os
import pickle
import sys


def loadfile(filename):
    """ load a file, return a generator. """
    fp = open(filename, 'r', encoding='utf-8')
    for i, line in enumerate(fp):
        yield line.strip('\r\n')
        if i % 100000 == 0:
            print('loading %s(%s)' % (filename, i), file=sys.stderr)
    fp.close()
    print('load %s succ' % filename, file=sys.stderr)


def writetofileLine(data, dfile):
    with open(dfile, 'w') as f:
        for i in range(len(data)):
            item = data[i]
            for j in range(len(item)):
                f.write(str(item[j]) + '\t')
            f.write('\n')


def generate_dataset(filename, historynum, exposenum):
    """ load rating data and split it to training set and test set """
    usernum = 0
    itemnum = 0
    Hisitem = []
    Expitem = []
    User = dict()
    sequence2user = dict()
    usermap = dict()
    itemmap = dict()

    imap_file = './' + '/imap.json'
    umap_file = './' + '/umap.json'
    his_file = './' + '/history.txt'
    exp_file = './' + '/expose.txt'
    intera_file = './' + '/interaction.txt'

    for line in loadfile(filename):
        user, itemLen, ilist, queryNum, queryList = line.split('\t')
        # print('user : ' + user)
        itemlist = ilist.split(',')

        if user in usermap:
            userid = usermap[user]
        else:
            userid = usernum
            usermap[user] = usernum
            User[userid] = []
            usernum += 1
        for i in range(len(itemlist)):
            _item, _showtime, _readtime = itemlist[i].split('|')
            _showtime = int(_showtime)
            _readtime = int(_readtime)
            if (_item in itemmap):
                itemid = itemmap[_item]
            else:
                itemid = itemnum
                itemmap[_item] = itemid
                itemnum += 1

            User[userid].append([itemid, _showtime, _readtime])
    with open(imap_file, 'w') as f:
        json.dump(itemmap, f)

    with open(umap_file, 'w') as f:
        json.dump(usermap, f)
    for userid in User.keys():  # 1. order by showtime (up)
        User[userid].sort(key=lambda x: x[1])
    UserList = []
    sequenceId = 0
    for userid in User.keys():  # for each user
        user = User[userid]
        item = [interac[0] for interac in user]
        UserList.append(item)
        UserHisitem = []
        for ans in range(len(user)):  # 2.seek out the answers which have readtime
            _readtime = user[ans][2]
            if _readtime != 0:
                UserHisitem.append(user[ans])
        if len(UserHisitem) >= historynum:
            for i in range(len(UserHisitem)):  # start at i
                if len(UserHisitem) - i >= historynum:
                    tmphis = []
                    tmpexp = []
                    maxlasttme = 0
                    expStart = -1
                    for j in range(i, i + historynum):  # [i,i+historynum)
                        maxlasttme = max(maxlasttme, UserHisitem[j][2])
                        tmphis.append(UserHisitem[j][0])
                    for j in range(len(user)):  # find expose Start position
                        if user[j][1] > maxlasttme:
                            expStart = j
                            break
                    if expStart != -1:  # gain exposedata
                        tot = 0
                        for j in range(expStart, len(user)):
                            tmpexp.append(user[j][0])
                            tot += 1
                            if tot == exposenum:
                                break
                        if len(tmpexp) == exposenum:
                            sequence2user[sequenceId] = userid
                            tmphis.reverse()
                            Hisitem.append(tmphis)
                            Expitem.append(tmpexp)
                            sequenceId += 1
    with open(f'processed/sequence2user.pkl', 'wb') as f:
        pickle.dump(sequence2user, f)
    writetofileLine(UserList, intera_file)
    writetofileLine(Hisitem, his_file)
    writetofileLine(Expitem, exp_file)


if __name__ == '__main__':
    datafile = os.path.join('zhihu1M.txt')
    historynum = 5
    exposenum = 10
    generate_dataset(datafile, historynum, exposenum)
