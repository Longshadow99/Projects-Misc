file1 = open("vote-dem.txt","r")
rwdata = []
for x in file1:
    temp = file1.readline()
    check = True
    for y in temp:
        if "/" in y:
            check = False
    if check:
        xm = x.replace('\n','')
        rwdata.append(xm)
del rwdata[0]
pos = 0
pos2 = 0
prcdata = []
for x in rwdata:
    prcdata.append([])
for x in rwdata:
    pos = 0
    for y in x:
        if pos == (len(x) - 1):
            if y == '1':
                prcdata[pos2].append(1)
            else:
                prcdata[pos2].append(-1)
        elif y + x[pos + 1] == 10:
            prcdata[pos2].append(2)
        else:
            prcdata[pos2].append(1)
        pos += 1
    pos2 += 1
            
        
print(rwdata)
print(prcdata)
        