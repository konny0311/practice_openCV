import time

def testfunc2(rangelist):
    l = []
    append = l.append
    for t in rangelist:
        append(t)

def testfunc3(rangelist):
    l = [t for t in rangelist]

s = time.time()
l = range(0,50000000)
testfunc3(l)
print(time.time() - s)
