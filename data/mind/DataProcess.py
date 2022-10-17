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
    His = dict()
    Exp = dict()
    sequence2user = dict()
    usermap = dict()
    itemmap = dict()
    his_file = './' + '/history.txt'
    exp_file = './' + '/expose.txt'
    intera_file = './' + '/interaction.txt'
    impre_tot = 0
    for line in loadfile(filename):
        _, user, _, hislist, explist = line.split('\t')
        explist = explist.split(' ')
        explist = [i[:-2] for i in explist]
        hislist = hislist.split(' ')
        impre_tot += len(explist)
        # print('user : ' + user)
        if user in usermap:
            userid = usermap[user]
            His[userid] = []
        else:
            userid = usernum
            usermap[user] = usernum
            His[userid] = []
            Exp[userid] = []
            usernum += 1
        for i in range(len(hislist)):
            _item = hislist[i]
            if (_item in itemmap):
                itemid = itemmap[_item]
            else:
                itemid = itemnum
                itemmap[_item] = itemid
                itemnum += 1
            His[userid].append(itemid)

        for i in range(len(explist)):
            _item = explist[i]
            if (_item in itemmap):
                itemid = itemmap[_item]
            else:
                itemid = itemnum
                itemmap[_item] = itemid
                itemnum += 1
            Exp[userid].append(itemid)
    print(impre_tot)
    print(len(itemmap))
    UserList = []
    sequenceId = 0
    for userid in His.keys():  # for each user
        his = His[userid]
        exp = Exp[userid]
        items = [item for item in his] + [item for item in exp]
        UserList.append(items)
        UserHisitem = his
        UserHisitem.reverse()  # his reverse
        UserExpitem = exp
        if len(UserHisitem) >= historynum and len(UserExpitem) >= exposenum:
            for i in range(5):  # start at i (begin at 0)
                if len(UserHisitem) - i >= historynum and len(UserExpitem) - i >= exposenum:
                    tmphis = []
                    tmpexp = []
                    for j in range(i, i + historynum):  # [i,i+historynum)
                        tmphis.append(UserHisitem[j])
                    for j in range(i, i + exposenum):  # [i,i+historynum)
                        tmpexp.append(UserExpitem[j])
                    if len(tmphis) == historynum and len(tmpexp) == exposenum:
                        sequence2user[sequenceId] = userid
                        Hisitem.append(tmphis)
                        Expitem.append(tmpexp)
                        sequenceId += 1
    with open(f'processed/sequence2user.pkl', 'wb') as f:
        pickle.dump(sequence2user, f)
    writetofileLine(UserList, intera_file)
    writetofileLine(Hisitem, his_file)
    writetofileLine(Expitem, exp_file)


if __name__ == '__main__':
    datafile = os.path.join('behaviors.tsv')
    historynum = 5
    exposenum = 10
    generate_dataset(datafile, historynum, exposenum)
