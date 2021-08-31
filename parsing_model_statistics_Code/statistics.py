import json
import pandas as pd
import string

df=pd.read_csv('/Users/fsadiq21/PycharmProjects/pythonProject1/original_testing_data.csv',encoding='utf-8' , header= None ,names=['message'] , skiprows=1)

#print(df.head())

#data=list(df)
#print(data)

#data=df.applymap(str)
#converting df to string
data=df.to_string()

#print(data)
#print(type(data))

#creating tokens
e=data.split()
print('Tokens:')
print(len(e))
#print(e)
tokenn=[]
count = 0
for i in e:
    #print(i)
    if i not in tokenn:
        tokenn.append(i)
        count=count+ 1

print("Vocabulary:")
print(len(tokenn))
#print(count)

utilitylist=[]
splitbypipe =[]
utility =[]
f=data.split('|')
#print(f)

for j in f:
    #print(j)
    splitbypipe.append(j)

print('Total utility')
print(len(splitbypipe))
for k in splitbypipe:
        l = k.strip().split(' ', 1)[0]
        if l not in utility:
            utility.append(l)
            #print(utility)
utilitylist.append(utility)
#print(utility)
#print(utilitylist)
print('Unique Utility:')
print(len(utility))

#to find log for line

n=df['message'].values.tolist()
#print(type(n))

finallist =[]
#counter = 0
#g=data.split('|')
for m in n:
    glist = []
    s=m.split("|")
    #print(s)
    for z in s:

        xx = z.strip().split(' ', 1)[0]
        glist.append(xx)

    #print(glist)
    finallist.append((glist))
#print(finallist)
print("Overall number of utilities per log")
print(len(finallist))

#Utiltiy per log
sum =0
for p in finallist:
    #counter=0
    #print(p)
    #print(len(p))
    sum = sum +len(p)
print('overall sum of utilities per log')
print(sum)

avg =sum/len(finallist)
print('Avg Utility per log')
print(avg)




#print(g)


