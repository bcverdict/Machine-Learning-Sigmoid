from sympy import symbols, diff
from math import e
import matplotlib.pyplot as plt
import numpy as np
import time
def getInputFile():
    while True:
        try:
            inputFile = input("Enter input file name: ")
            inputFile = open(inputFile)
        except IOError:
            print('Could not find file')
            print('-------------------')
            continue
        else:
            break
    return inputFile
def getOutputFile():
    outputFile = input("Enter output file name: ")
    open(outputFile, 'w').close()
    return outputFile
def format(full):
    for i in range(len(full)):
        line = full[i].split()
        commentFound = False
        for y in range(len(line)):
            word = line[y]
            for z in word:
                if z=="!":
                    line[y]=word[:word.find("!")]
                    commentFound = True
                    break
            if commentFound:
                full[i]=' '.join(line)
                break
    addition = 0
    i = 0
    while True:
        if full[i].split() == []:
            full.pop(i)
        else:
            i = i + 1
        if i == len(full):
            break
    for x in range(len(full)):
        full[x]=full[x].split()
    full[0]=full[0][1:len(full[0])-1]
    full[1]=full[1][1:len(full[1])-1]
    return full
def sigmoid(x):
    return "(1/(1+e**(-("+str(x)+"))))"
def dc(weight1,weight2,bias,d):
    x, y, b = symbols('x y b',real=True)
    return round(d.subs(x,weight1).subs(y,weight2).subs(b,bias),2)
def strCostSig():
    cost = ''
    for i in data:
        z = i[0]+"*x"+"+"+i[1]+"*y+b"
        pred = sigmoid(z)
        cost = cost + "(("+pred+"-"+i[2]+")**2)+"
    return cost[:len(cost)-1]
def partials():
    x, y, b = symbols('x y b',real=True)
    f = strCostSig()
    return diff(eval(f),x),diff(eval(f),y), diff(eval(f),b)
def update(w1,w2,b,dw1,dw2,db):
    alpha = 0.01
    w1 = w1 - alpha * dc(w1,w2,b,dw1)
    w2 = w2 - alpha * dc(w1,w2,b,dw2)
    b = b - alpha * dc(w1,w2,b,db)
    return w1,w2,b
def main():
    w1 = 0.01
    w2 = 0.3
    bias = 0.5
    x, y, b = symbols('x y b',real=True)
    cost_arr = []
    point_arr = []
    for i in range(2000): #iterations
        #last_time = time.time()
        w1, w2, bias = update(w1,w2,bias,dw1,dw2,db)
        #print('Took {} seconds'.format(time.time()-last_time))
        
        #--------
        f = strCostSig()
        evalF = eval(f)
        cost = evalF.subs(x,w1).subs(y,w2).subs(b,bias)
        cost_arr.append(cost)
        #--------
        point = 0
        num_true = 0
        for j in data:
            point = w1*float(j[0])+w2*float(j[1])+bias
            point = 1/(1 + np.exp(-1*point)) 
            if point >=0.5:
                point = 1
            else:
                point = 0
            if int(point)==int(j[2]):
                num_true+=1
        point_arr.append(num_true)
        #--------
            
    plt.xlabel('Iterations')
    cost_line, = plt.plot(cost_arr, label="cost")
    correct_line, = plt.plot(point_arr, label="# correctly classified")

    plt.legend(handles=[correct_line, cost_line])
    plt.savefig('cost-point')
    plt.show()
    
    print('w1:',w1)
    print('w2:',w1)
    print('b:',bias)

inputFile = getInputFile()
full = format(inputFile.readlines())
data = full[2:len(full)]
dw1, dw2, db = partials()
main()

#input file must have 3 inputs per line
