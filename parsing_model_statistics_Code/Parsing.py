import json #import json file
import glob #import glob
import pandas as pd
import re
import string


# read all json file from directory
read_files = glob.glob("/Users/fsadiq21/PycharmProjects/pythonProject1/jsonfiles2/*.json")
#msglist = []
def Parsing(files):
    msglist=[]
    for i in files:
        #print(i)
        with open(i ,'r') as honeylogs:
         #print(honeylogs)
         honeypotlogs = json.load(honeylogs)
         #print(honeypotlogs)
         for records in honeypotlogs:
             for k,v in records.items():
                 #print(k,v)
                 #msglist = []
                 for messages in v:
                     #print(messages['message'])

                     msg = messages['message']
                     #msglist = []
                     #print(len(msg))
                     for i in msg.split():

                      #word='cat'
                      #if word in i:
                          #print(msg)
                        if i.startswith("CMD") or i.startswith("Command"): #extracting msgs with only CMD and Command Found
                          msglist.append(msg)
                          #print(msglist)
    return msglist

a=Parsing(read_files)
#print(type(a))
#print(a)
#print(len(a))
#removing duplicates
#function to remove duplicate and truncate long strings to only 1024
def duplicate_removal(listofmsgs):
    after_duplicates =[]
    for c in listofmsgs:
        if c not in after_duplicates:
            after_duplicates.append(c[:800])
        #print(len(after_duplicates))
    return after_duplicates
after_duplicates1=duplicate_removal(a)
#print(after_duplicates1)
print(len(a))
print(len(after_duplicates1))


#function to remove exceptions  and make it coherent
def replacefunc(afterduplist):
    afterexcept = []
    for g in afterduplist:
        ic1 = g.replace('CMD:', '')
        ic = ic1.replace('Command not found: ', '')
        j1 = ic.replace('Command found:', '')
        j2=j1.replace('/bin/busybox' , '')
        j21=j2.replace('""', '')
        j22=j21.replace('////><@/\,#""%^!0123456789' , '')
        j2 = j22.replace('LC_ALL=C', '')
        j4 = j2.replace('""', '')
        j5 = j4.replace('$(', '|')
        j = j5.replace('123456', '')
        l1 = j.replace('install', '|')
        l = l1.replace('{print $4,$5,$6,$7,$8,$9;}', '')
        m = l.replace(';', '|')
        pattern = r'\s'
        mn=re.sub(pattern, " " , m)
        pattern1 = r'\"'
        mm = re.sub(pattern1, " ", mn)
        pattern2 = r'\\'
        mm1 = re.sub(pattern2, " ", mm)

        afterexcept.append(mm1)
    #print(afterexcept)
    return afterexcept

coherentmsgs=replacefunc(after_duplicates1)
#print(coherentmsgs)

df = pd.DataFrame(coherentmsgs)
df.to_csv("originaldata.csv" , index=False , header=["message"])

#function to tokenize data and extract only list of commands from it.
def data_prep(list_of_clean_msgs):
    newlist =[]

    for ic in list_of_clean_msgs:

        finallist = []
        splist=[]
        for nn in ic.split("|"):
            splist.append(nn)
        #print(splist)


        #print(len(cd))
        #for xy in range(len(a)):
        #print(xy)
        for z in splist:
            xx=z.strip().split(' ' , 1)[0]
            finallist.append(xx)
        #print(finallist)
        newlist.append(finallist)
        #print(newlist)
    #print(newlist)
    #print(len(newlist))
    #print(len(finallist))
#print(type(finallist))
    return newlist
command_data=data_prep(coherentmsgs)
#print(command_data)
#data in gpt2 format

#Function for training

"""def data_gpt2_format(afterdup_msgs , cleanmsgs):
    data_total = []
    for ib in range(len(afterdup_msgs)):
        #for jj in range(len(cleanmsgs[ib])):
            #line = afterdup_msgs[ib] + ", "+ "Utility" + ": " + cleanmsgs[ib][jj] + "<|endoftext|>"
            line = afterdup_msgs[ib] + "," + "Used utilities"+ ":\n" + ' '.join(cleanmsgs[ib]) +"<|endoftext|>"
            data_total.append(line)
    #print(data_total)
    return data_total
finaldata=data_gpt2_format(coherentmsgs,command_data)
print(finaldata)
df = pd.DataFrame(finaldata)
df.to_csv("trainingdata.csv" , index=False , header=["message"])"""

#function for testing

def data_gpt2_format(afterdup_msgs , cleanmsgs):
    data_total = []
    for ib in range(len(afterdup_msgs)):
        #for jj in range(len(cleanmsgs[ib])):
            #line = afterdup_msgs[ib] + ", "+ "Utility" + ": " + cleanmsgs[ib][jj] + "<|endoftext|>"
            line = afterdup_msgs[ib] + "," + "Used utilities"+ ":" +";"+ ' '.join(cleanmsgs[ib]) +"<|endoftext|>"
            data_total.append(line)
    #print(data_total)
    return data_total
finaldata=data_gpt2_format(coherentmsgs,command_data)
#print(finaldata)
df = pd.DataFrame(finaldata)
df.to_csv("testingdata1.csv" , index=False , header=["message"])
















