# A General library for Bayesian network of discrete states
import numpy.random as random
import numpy
import math
import itertools
import bisect
from collections import Counter
import pickle

# savebn()
# save the raw BN object to a pickle file

def savebn(filename):
    output = open(filename, 'wb')
    pickle.dump(bn, output)
    output.close()
    return

# loadbn()
# load in the raw BN object from a pickle file

def loadbn(filename):
    bn = pickle.load(file(filename))
    return bn

# parsebn()
# parse the Bayesnet
# takes a general name and try to find three files: .state, .topo, and .param 

def parsebn(name):

    try:
        fstate = file(name+'.state')
        ftopo = file(name+'.topo')
    except:
        print 'File not found'
        return

    try:
        fparam = file(name+'.param')
    except:
        print 'Parameter file not found, new one created'
        fparam = file(name+'.param','a')
        fparam.close()
        fparam = file(name+'.param')

    bn = {}

# read in the legal states of each node

    for line in fstate:
        ll = line.split()
        bn[ll[0]] = {}
        bn[ll[0]]['st'] = ll[1:]
    fstate.close()

# read in the topology of each node and its parents

    for line in ftopo:
        ll = line.split()
        bn[ll[0]]['pa'] = ll[1:]
    ftopo.close()    

# generate a blank parameter set

    for node in bn:
        bn[node]['cp'] = {}
        nodestates = bn[node]['st'][:]
        arguments = []
        for par in bn[node]['pa']:
            parentstates = bn[par]['st']
            arguments.append(parentstates)
        for states in itertools.product(*arguments):
            parstatestr = '_'.join(states)
            bn[node]['cp'][parstatestr] = {}
            for nodestate in nodestates:
                bn[node]['cp'][parstatestr][nodestate] = 0.0000001

# read in the parameter set

    for line in fparam:
        ll = line.split()
        prob = float(ll[-1])
        nodestate = ll[1]
        states = ll[2:len(ll)-1]
        parstatestr = '_'.join(states)
        if parstatestr in bn[ll[0]]['cp'] and nodestate in bn[ll[0]]['cp'][parstatestr]:
            bn[ll[0]]['cp'][parstatestr][nodestate] = prob
        else:
            print 'illegal state'
            return
    fparam.close()
    return bn

# childlist()
# create a child list given a bayesnet

def childlist(bn):
    cl = {}
    for node in bn:
        cl[node] = []
    for node in bn:
        parents = bn[node]['pa']
        for par in parents:
            cl[par].append(node)
    return cl

# tsort()
# performs topological sort given a bn

def tsort(bn):
# recursuve function visit()
    def visit(node):
        if node in tempmark:
            return
        if node in unmarked:
            unmarked.remove(node)
            tempmark.append(node)
            for cn in child[node]:
                visit(cn)
            tempmark.remove(node)
            sortlist.append(node)
# create the children list 
    child = childlist(bn)
    sortlist = []
    unmarked = bn.keys()
    tempmark = []
    while len(unmarked) > 0:
        visit(unmarked[0])
    return sortlist[::-1]
 
# bnnormalize()
# enforcing all the conditional probs to sum to unity

def bnnormalize(bn):
    for node in bn:
        for parentstates in bn[node]['cp']:
            sum = 0.0
            for nodestate in bn[node]['cp'][parentstates]:
                sum += bn[node]['cp'][parentstates][nodestate] 
            for nodestate in bn[node]['cp'][parentstates]:
                if sum > 0.0:
                    bn[node]['cp'][parentstates][nodestate] = bn[node]['cp'][parentstates][nodestate] / sum
    return

# scdf()
# calculating CDF for each CPD

def scdf(bn):
    bnnormalize(bn)
    cdf = {}
    for node in bn:
        cdf[node] = {}
        for parstates in bn[node]['cp']:
            cprob = 0.0
            cdf[node][parstates] = []
            for nodestate in bn[node]['st']:
                if nodestate in bn[node]['cp'][parstates]:
                    prob = bn[node]['cp'][parstates][nodestate]
                else:
                    prob = 0.0
                cprob += prob
                cdf[node][parstates].append(cprob)
    return cdf

# samplebn()
# draw a sample from a BayesNet

def samplebn(bn):
    nodelist = tsort(bn)
    state = dict.fromkeys(nodelist)
    for node in nodelist:
        parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
        if parstates in bn[node]['cp']:
            localpdf = bn[node]['cp'][parstates]
            lstate = localpdf.keys()
            lprobs = localpdf.values()
            state[node] = random.choice(lstate,1,p=lprobs)[0]
        else:
            lstate = bn[node]['st']
            state[node] = random.choice(lstate,1)[0]
    return state

# masssamplebn()
# draw a bunch of samples from a BayesNet, output as a dataset

def masssamplebn(bn,num):
    nodelist = tsort(bn)
    dataset = []
    for i in range(num):
        state = dict.fromkeys(nodelist)
        for node in nodelist:
            parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
            if parstates in bn[node]['cp']:
                localpdf = bn[node]['cp'][parstates]
                lstate = localpdf.keys()
                lprobs = localpdf.values()
                state[node] = random.choice(lstate,1,p=lprobs)[0]
            else:
                lstate = bn[node]['st']
                state[node] = random.choice(lstate,1)[0]
        dataset.append((state,1.0))
    return dataset

# samplecondbn()
# draw a sample from a BayesNet by rejection algorithm

def samplecondbn(bn,cond,maxtrial=10000):
    nodelist = tsort(bn)
    match = False
    trial = 0
    while not match and trial < maxtrial:
        state = dict.fromkeys(nodelist)
        for node in nodelist:
            parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
            if parstates in bn[node]['cp']:
                localpdf = bn[node]['cp'][parstates]
                lstate = localpdf.keys()
                lprobs = localpdf.values()
                state[node] = random.choice(lstate,1,p=lprobs)[0]
            else:
                lstate = bn[node]['st']
                state[node] = random.choice(lstate,1)[0]
        trial += 1
        match = True
        for condnode in cond:
            match = match and state[condnode] == cond[condnode] 
    if not match:
        state = {}
    return state

# masssamplecondbn()
# draw a bunch of samples from a BayesNet, output as a dataset, by rejection algorithm

def masssamplecondbn(bn,cond,num,maxtrial=10000):
    nodelist = tsort(bn)
    cdf = scdf(bn)
    dataset = []
    for i in range(num):
        match = False
        trial = 0
        while not match and trial < maxtrial:
            state = dict.fromkeys(nodelist)
            for node in nodelist:
                parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
                if parstates in bn[node]['cp']:
                    localpdf = bn[node]['cp'][parstates]
                    lstate = localpdf.keys()
                    lprobs = localpdf.values()
                    state[node] = random.choice(lstate,1,p=lprobs)[0]
                else:
                    lstate = bn[node]['st']
                    state[node] = random.choice(lstate,1)[0]
            trial += 1
            match = True
            for condnode in cond:
                match = match and state[condnode] == cond[condnode]
        dataset.append((state,1.0))
    return dataset

# gibbssamplecondbn()
# draw a bunch of samples from a BayesNet given a condition using the Gibbs sampling, output as a dataset

def gibbssamplecondbn(bn,cond,num,state={}):
# initialize states
   if state == {}:
       state = samplebn(bn)
   statenodes = state.keys()
   statenodes.sort()
   bnnodes = bn.keys()
   bnnodes.sort()
   if statenodes != bnnodes:
       state = samplebn(bn)
# mutate the state vector to be consistent with the condition
   condnodes = cond.keys()
   activenodes = bn.keys()
   for condnode in condnodes:
       state[condnode] = cond[condnode]
       activenodes.remove(condnode)
   dataset = []
   for i in range(num):
       for node in activenodes:
           pdf = []
           for nodestate in bn[node]['st']:
               state[node] = nodestate
               pdf.append(querybn(bn,state))
           norm = sum(pdf)
           cdf = []
           for i in range(len(pdf)):
               if i > 0:
                   cdf.append(cdf[i-1]+pdf[i]/norm)
               elif i == 0:
                   cdf.append(pdf[0]/norm)
           dice = random.random()
           state[node] = bn[node]['st'][bisect.bisect(cdf,dice)]
       dataset.append((state,1.0)) 
   return dataset

# querybn()
# given a bn and a state dictionary, returns a joint prob

def querybn(bn,state,alpha=1e-8):
    logprob = 0.0
    logalpha = numpy.log(alpha)
    nodelist = bn.keys()
    for node in nodelist:
        parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
        if parstates in bn[node]['cp']:
            if state[node] in bn[node]['cp'][parstates]:
                logprob += numpy.log(bn[node]['cp'][parstates][state[node]])
            else:
                logprob += logalpha
        else:
            logprob += logalpha
    return numpy.exp(logprob)

# querybndelta1()
# given a bn and a state dictionary, returns a joint prob

def querybndelta1(bn,state,nodelist,alpha=1e-8):
    logprob = 0.0
    logalpha = numpy.log(alpha)
    for node in nodelist:
        parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
        if parstates in bn[node]['cp']:
            if state[node] in bn[node]['cp'][parstates]:
                logprob += numpy.log(bn[node]['cp'][parstates][state[node]])
            else:
                logprob += logalpha
        else:
            logprob += logalpha
    return numpy.exp(logprob)

# querybn1()
# given a bn and a state dictionary, returns a joint prob

def querybn(bn,state,alpha=1e-8):
    prob = 1.0
    nodelist = bn.keys()
    for node in nodelist:
        parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
        if parstates in bn[node]['cp']:
            if state[node] in bn[node]['cp'][parstates]:
                prob = prob * bn[node]['cp'][parstates][state[node]]
            else:
                prob = prob * alpha
        else:
            prob = prob * alpha
    return prob

# querybncontribute()
# given a bn and a state dictionary, returns list of individually contributions
# to the joint density

def querybncontribute(bn,state,alpha=1e-8):
    logprob = {}
    logalpha = numpy.log(alpha)
    nodelist = bn.keys()
    for node in nodelist:
        parstates = '_'.join([state[parnode] for parnode in bn[node]['pa']])
        if parstates in bn[node]['cp']:
            if state[node] in bn[node]['cp'][parstates]:
                logprob[node] = numpy.log(bn[node]['cp'][parstates][state[node]])
            else:
                logprob[node] = logalpha
        else:
            logprob[node] = logalpha
    return logprob

# querymargbn()
# given a bn and a state dictionary, returns a marginalized joint prob

def querymargbn(bn,state):
    cstate = state.copy()
    missing = list(set(bn.keys()) - set(state.keys()))
    nmiss = len(missing)
    prob = 0.0
    
    arguments = []
    for mnode in missing:
        arguments.append(bn[mnode]['st'])

    for mstates in itertools.product(*arguments):
        for i in range(nmiss):
            cstate[missing[i]] = mstates[i]
        prob += querybn(bn,cstate)
            
    return prob

# querybncond()
# given a bn and a state dictionary, returns a conditional prob

def querybncond(bn,state,cond):
    jp = querymargbn(bn,state)
    mp = querymargbn(bn,cond)
    if mp > 0.0:
        return jp / mp
    else:
        return 0.0

# datacompact()
# given a dataset, returns a compacted/aggregrated dataset

def datacompact(dataset):
    newdata = {}
    for data in dataset:
        state = data[0]
        count = data[1]
        statestr = '_'.join(state.keys())+':'+'_'.join(state.values())
        if statestr not in newdata:
            newdata[statestr] = count
        else:
            newdata[statestr] += count
    newdataset = []
    for statestr in newdata:
        ss = statestr.split(':')
        newdataset.append((dict(zip(ss[0].split('_'),ss[1].split('_'))),newdata[statestr]))
    return newdataset

# loglikelihood()
# given a bn and a dataset, returns a log likelihood
# dataset format: a list of tuple of state dict and counts
# [(state1,count1),(state2,count2),(state3,count3)....]

def loglikelihood(bn,dataset):
    value = 0.0
    compleng = len(bn)
    for data in dataset:
        if data[1] > 0:
            if len(data[0]) == compleng:
                value += data[1] * math.log(querybn(bn,data[0]))
            else:
                value += data[1] * math.log(querymargbn(bn,data[0]))
    return value

# loglikelihoodc()
# given a bn and a dataset of complete observations, returns a log likelihood

def loglikelihoodc(bn,dataset):
    value = 0.0
    for data in dataset:
        if data[1] > 0:
            value += data[1] * math.log(querybn(bn,data[0]))
    return value

# partloglikelihood()
# given a bn, a dataset, and condition, returns a partial log likelihood for Gibbs/MH sampling

def partloglikelihood(bn,dataset,cond):
    value = 0.0
    compleng = len(bn)
    for data in dataset:
        proceed = True
        for item in cond:
            if item in data[0]:
                proceed = proceed and (cond[item] == data[0][item])
        if proceed:    
            if len(data[0]) == compleng:
                value += data[1] * math.log(querybn(bn,data[0]))
            else:
                value += data[1] * math.log(querymargbn(bn,data[0]))
    return value

# bicbn()
# returns the Bayes information crit given a dataset and a model

def bicbn(bn,dataset):
    value = 0.0
    sample = 0.0
    for data in dataset:
        sample += data[1]
        value += data[1] * math.log(querymargbn(bn,data[0]))
    nodelist = bn.keys()
    params = 0
    for node in nodelist:
        for parstates in bn1[node]['cp']:
            params += len(bn1[node]['cp'][parstates]) - 1
    return -2.0*value + params * math.log(sample)

# imputebn1()
# given a bn and an incomplete dataset list, returns a imputated dataset (slower)

def imputebn1(bn,dataset):
    dataset1 = []
    for data in dataset:
        state = data[0]
        count = data[1]
        missing = list(set(bn.keys()) - set(state.keys()))
        nmiss = len(missing)

        if nmiss == 0:
            dataset1.append((state,count))
        else:
            arguments = []
            for mnode in missing:
                arguments.append(bn[mnode]['st'])

            for mstates in itertools.product(*arguments):
                cstate = state.copy()
                for i in range(nmiss):
                    cstate[missing[i]] = mstates[i]
                    cprob = querybn(bn,cstate) / querymargbn(bn,state)
                    dataset1.append((cstate,cprob*count))

#    dataset1 = datacompact(dataset1)
    return dataset1 

# imputebn()
# given a bn and an incomplete dataset list, returns a imputated dataset

def imputebn(bn,dataset):
    dataset1 = []
    for data in dataset:
        state = data[0]
        count = data[1]
        missing = list(set(bn.keys()) - set(state.keys()))
        nmiss = len(missing)

        if nmiss == 0:
            dataset1.append((state,count))
        else:
            arguments = []
            for mnode in missing:
                arguments.append(bn[mnode]['st'])

            marginal = 0.0
            temp = []
            for mstates in itertools.product(*arguments):
                cstate = state.copy()
                for i in range(nmiss):
                    cstate[missing[i]] = mstates[i]
                jp = querybn(bn,cstate)
                marginal += jp
                temp.append((cstate,jp*count))
            for item in temp:
                dataset1.append((item[0],item[1]/marginal))

#    dataset1 = datacompact(dataset1)
    return dataset1

# compdatabn()
# given a bn and an incomplete dataset list, returns completed dataset by Gibbs sampling

def compdatabn(bn,dataset):
    dataset1 = []
    for data in dataset:
        state = data[0]
        count = data[1]
        missing = list(set(bn.keys()) - set(state.keys()))
        nmiss = len(missing)

        if nmiss == 0:
            dataset1.append((state,count))
        else:
            arguments = []
            for mnode in missing:
                arguments.append(bn[mnode]['st'])

            marginal = 0.0
            missstates = []
            temp = []
            for mstates in itertools.product(*arguments):
                cstate = state.copy()
                missstates.append(mstates)
                for i in range(nmiss):
                    cstate[missing[i]] = mstates[i]
                jp = querybn(bn,cstate)
                temp.append(jp)
                marginal += jp
            pdf = [p/marginal for p in temp]
            samples = random.multinomial(count,pdf)

            for m in range(len(missstates)):
                cstate = state.copy()
                for i in range(nmiss):
                    cstate[missing[i]] = missstates[m][i]
                dataset1.append((cstate,samples[m]))

#    dataset1 = datacompact(dataset1)
    return dataset1


# copybn()
# make a carbon copy of a bn dict

def copybn(bn):
    nodelist = bn.keys()
    bn1 = dict.fromkeys(nodelist)

    for node in nodelist:
        bn1[node] = {}
        bn1[node]['st'] = []
        bn1[node]['st'] = bn[node]['st'][:]
        bn1[node]['pa'] = []
        bn1[node]['pa'] = bn[node]['pa'][:]
        bn1[node]['cp'] = {}
        for parstates in bn[node]['cp']:
            bn1[node]['cp'][parstates] = {}
            for nodestate in bn[node]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = bn[node]['cp'][parstates][nodestate]
    return bn1 

# renamenode()
# rename a node in bn from name1 to name2, whole dict needs to be rebuilt because the dict key cannot be changed

def renamenode(bn,name1,name2):
    nodelist = bn.keys()
    nodelist = [name2 if inode == name1 else inode for inode in nodelist]
    bn1 = dict.fromkeys(nodelist)

    for node in nodelist:
        if node == name2:
            node1 = name1
        else:
            node1 = node
        bn1[node] = {}
        bn1[node]['st'] = []
        bn1[node]['st'] = bn[node1]['st'][:]
        bn1[node]['pa'] = []
        palist = bn[node1]['pa'][:]
        palist = [name2 if inode == name1 else inode for inode in palist]
        bn1[node]['pa'] = palist
        bn1[node]['cp'] = {}
        for parstates in bn[node1]['cp']:
            bn1[node]['cp'][parstates] = {}
            for nodestate in bn[node1]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = bn[node1]['cp'][parstates][nodestate]
    return bn1

# renamestate()
# rename a state of inode in bn from name1 to name2,whole dict needs to be rebuilt because the dict key cannot be changed

def renamestate(bn,inode,name1,name2):
    nodelist = bn.keys()
    bn1 = dict.fromkeys(nodelist)

    for node in nodelist:
        bn1[node] = {}
        bn1[node]['st'] = []
        if node == inode:
            slist = bn[node]['st'][:]
            slist = [name2 if istate == name1 else istate for istate in slist]
            bn1[node]['st'] = slist
        else:
            bn1[node]['st'] = bn[node]['st'][:]
        bn1[node]['pa'] = []
        bn1[node]['pa'] = bn[node]['pa'][:]
        bn1[node]['cp'] = {}
        for parstates in bn[node]['cp']:
            parstates1 = parstates
            if inode in bn1[node]['pa']:
                inodepost = bn1[node]['pa'].index(inode)
                pslist = parstates.split('_')
                if pslist[inodepost] == name1:
                    pslist[inodepost] = name2
                parstates1 = '_'.join(pslist)
            bn1[node]['cp'][parstates1] = {}
            if node == inode:
                for nodestate in bn[node]['cp'][parstates]:
                    if nodestate == name1:
                        bn1[node]['cp'][parstates1][name2] = bn[node]['cp'][parstates][nodestate]
                    else:
                        bn1[node]['cp'][parstates1][nodestate] = bn[node]['cp'][parstates][nodestate]
            else:
                for nodestate in bn[node]['cp'][parstates]:
                    bn1[node]['cp'][parstates1][nodestate] = bn[node]['cp'][parstates][nodestate]
    return bn1

# messupbn()
# assign random values to a exiting bn

def messupbn(bn):
    bn1 = copybn(bn)
    nodelist = bn1.keys()
    for node in nodelist:
        for parstates in bn1[node]['cp']:
            for nodestate in bn1[node]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = random.random()
    bnnormalize(bn1)
    return bn1

# zerobn()
# assign zeros values to a exiting bn

def zerobn(bn):
    bn1 = copybn(bn)
    nodelist = bn1.keys()
    for node in nodelist:
        for parstates in bn1[node]['cp']:
            for nodestate in bn1[node]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = 0.0
    return bn1

# unitbn()
# assign unit values to a exiting bn

def unitbn(bn):
    bn1 = copybn(bn)
    nodelist = bn1.keys()
    for node in nodelist:
        for parstates in bn1[node]['cp']:
            for nodestate in bn1[node]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = 1.0
    return bn1

# timesbn()
# assign zeros values to a exiting bn

def timesbn(bn,factor):
    bn1 = copybn(bn)
    nodelist = bn1.keys()
    for node in nodelist:
        for parstates in bn1[node]['cp']:
            for nodestate in bn1[node]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = bn[node]['cp'][parstates][nodestate] * factor
    return bn1

# limitbn()
# limit the edge of the bn.

def limitbn(bn,limit):
    bn1 = copybn(bn)
    nodelist = bn1.keys()
    for node in nodelist:
        for parstates in bn1[node]['cp']:
            for nodestate in bn1[node]['cp'][parstates]:
                bn1[node]['cp'][parstates][nodestate] = max(limit,bn[node]['cp'][parstates][nodestate]) 
    bnnormalize(bn1)
    return bn1


# mleparam1()
# MLE estimate of the bn parameters given a complete dataset (slower)

def mleparam1(bn,dataset):
    bnn = copybn(bn)
    bnn = zerobn(bnn)
    countdata = datacompact(dataset)
    nodelist = bn.keys()

# assign the countdata to bn dict

    for data in countdata:
        dstate = data[0]
        for node in nodelist:
            parents = bnn[node]['pa'][:]
            for parstatestr in bnn[node]['cp']:
                match = True
                cpstate = dict(zip(parents,parstatestr.split('_')))
                for pnode in cpstate:
                    if match:
                        if cpstate[pnode] != dstate[pnode]:
                            match = False
                if match:
                    for nodestate in bnn[node]['cp'][parstatestr]:
                        if nodestate == dstate[node]:
                            bnn[node]['cp'][parstatestr][nodestate] += data[1]
    bnnormalize(bnn)
    return bnn

# mleparam()
# MLE estimate of the bn parameters given a complete dataset

def mleparam(bn,dataset,compact=False):
    bnn = copybn(bn)
    bnn = zerobn(bnn)
    if not compact:
        countdata = datacompact(dataset)
    else:
        countdata = dataset
    nodelist = bn.keys()

# assign the countdata to bn dict

    for data in countdata:
        dstate = data[0]
        for node in nodelist:
            parents = bnn[node]['pa'][:]
            parstatestr = '_'.join([dstate[pnode] for pnode in parents])
            bnn[node]['cp'][parstatestr][dstate[node]] += data[1]
    bnnormalize(bnn)
    return bnn

# mleparam2()
# MLE estimate of the bn parameters given a complete dataset

def mleparam2(bn,bn2,dataset):
    bnn = copybn(bn2)
    countdata = datacompact(dataset)
    nodelist = bn.keys()

# assign the countdata to bn dict

    for data in countdata:
        dstate = data[0]
        for node in nodelist:
            parents = bnn[node]['pa'][:]
            parstatestr = '_'.join([dstate[pnode] for pnode in parents])
            bnn[node]['cp'][parstatestr][dstate[node]] += data[1]
    bnnormalize(bnn)
    return bnn

# countbn()
# Conditional aggregations the bn parameters given a complete dataset

def countbn(bn,dataset):
    bnn = copybn(bn)
    bnn = zerobn(bnn)
    countdata = datacompact(dataset)
    nodelist = bn.keys()

# assign the countdata to bn dict

    for data in countdata:
        dstate = data[0]
        for node in nodelist:
            parents = bnn[node]['pa'][:]
            parstatestr = '_'.join([dstate[pnode] for pnode in parents])
            bnn[node]['cp'][parstatestr][dstate[node]] += data[1]
    return bnn

# goodturing()
# Performing Good-Turing estimate for all CPDs

def goodturing(countdict,maxzero=0.9):
# get the total of the count dictionary
    countlist = countdict.values()
    total = sum(countlist)
# get the frequency-of-frequency list
    fof = Counter(countlist)
    rlist = fof.keys()
    foflen = len(fof)
    if foflen >= 3 and 0 not in fof and total > 0:           
# use the Gale smooothing            
        rlist = fof.keys()
        rlist.sort()
        logzrlist = [numpy.log(fof[rlist[i]] * 2.0/float(rlist[i+1] - rlist[i-1])) for i in range(1,len(rlist)-1)]
        logzrlist.append(numpy.log(fof[rlist[0]] * 2.0/float(rlist[1] - 0)))
        logrlist = [numpy.log(rlist[i]) for i in range(1,len(rlist)-1)]
        logrlist.append(numpy.log(rlist[0]))
# build the linear model
        lm = numpy.polyfit(logrlist,logzrlist,1)
        a = lm[1]
        b = min(-1.0,lm[0])
        zeroest = numpy.exp(a) / float(total)
    else:
        b = -1.0
        zeroest = maxzero
        total = total / (1.0 - maxzero)

    smootheddict = {}
    for k in countdict:
        r = countdict[k]
        smootheddict[k] = r * numpy.power((1. + 1./float(r)),(b+1.)) / float(total)
    smootheddict['unobserved'] = zeroest
    return smootheddict

# goodturingbn()
# Performing Good-Turing estimate for all CPDs

def goodturingbn(bn):
    bnn = copybn(bn)
    bnn = zerobn(bnn)
    for node in bn:
        for parentstates in bn[node]['cp']:
            countdict = bn[node]['cp'][parentstates]
            gtcountdict = goodturing(countdict)
            for nodestate in bn[node]['cp'][parentstates]:
                bnn[node]['cp'][parentstates][nodestate] = gtcountdict[nodestate]
    bnnormalize(bnn)        
    return bnn

# parsecsv()
# parse the csv or any delimited data into dataset format

def parsecsv(delim,name):
    dataset = []
    datafile = file(name)
    nodelist = datafile.next().split(delim)
    nnode = len(nodelist)
    for line in datafile:
        ll = line.rstrip().split(delim)
        state = {}
        for inode in range(nnode):
            if ll[inode] != '?':
                state[nodelist[inode]] = ll[inode]
        dataset.append((state,1.0))
    return dataset

# populatebn()
# populate a BN dict with the data in a dataset

def populatebn(dataset,nodes,topo):
# populate a BayesNet from a dataset
# the network topology must be given 
    bn = {}
# build the parent lists from topo
    for node in nodes:
        bn[node] = {}
        bn[node]['cp'] = {}
        bn[node]['st'] = []
        bn[node]['pa'] = topo[node]
# read in the data
    for data in dataset:
        state = data[0]
        count = data[1]
        for inode in state:
            st = state[inode]
            if st not in bn[inode]['st']:
                bn[inode]['st'].append(st)
# populate the parent states and node state       
            pa = bn[inode]['pa']
            past = '_'.join([state[p] for p in pa])
            if past not in bn[inode]['cp']:
                bn[inode]['cp'][past] = {}
            if st not in bn[inode]['cp'][past]:
                bn[inode]['cp'][past][st] = 0
            bn[inode]['cp'][past][st] += count
    return bn

# parsehitlist()
# parse the dataset in hitlist style

def parsehitlist(bn,nstate,missing,name):
    f = file(name)
    dataset = []
    nodes = bn.keys()
    for l in f:
        state = dict.fromkeys(nodes,nstate[0])
        del state[missing]
        ll = l.split('\t')
        hits = ll[0].split('|')
        for rh in hits:
            if rh in state:
                state[rh] = nstate[1]
        dataset.append((state,float(ll[1])))
    return dataset

# gibbssample1()
# perform Gibbs sampling given a bn, an exponent dict

def gibbssample1(bn,likeexp):
    bn1 = copybn(bn)
    for node in bn1.keys():
        for parstates in bn1[node]['cp']:
            p1 = tuple(likeexp[node]['cp'][parstates].values())
            newparams = random.dirichlet(p1).tolist()
            skeys = bn[node]['cp'][parstates].keys()
            bn1[node]['cp'][parstates] = dict(zip(skeys,newparams))
    return bn1

# gibbssample2()
# perform Gibbs sampling given a bn, a prior and a likelihood exponent dict

def gibbssample2(bn,prior,likeexp):
    bn1 = copybn(bn)
    for node in bn1.keys():
        for parstates in bn1[node]['cp']:
            p0 = prior[node]['cp'][parstates].values()
            p1 = likeexp[node]['cp'][parstates].values()
            p2 = tuple([x+y for x,y in zip(p0,p1)])
            newparams = random.dirichlet(p2).tolist()
            skeys = bn[node]['cp'][parstates].keys()
            bn1[node]['cp'][parstates] = dict(zip(skeys,newparams))
    return bn1

# printbn()
# print out the BayesNet in the input format

def printbn(bn,name):

    fstate = file(name+'.state','w')
    ftopo = file(name+'.topo','w')
    fparam = file(name+'.param','w')

    nodelist = bn.keys()

    for node in nodelist:
        fstate.write(node+' '+' '.join(bn[node]['st'])+'\n')
    fstate.close()

    for node in nodelist:
        ftopo.write(node+' '+' '.join(bn[node]['pa'])+'\n')
    ftopo.close()

    for node in nodelist:
        for parstates in bn[node]['cp']:
            for nodestate in bn[node]['cp'][parstates]:
                fparam.write(node+' '+nodestate+' '+' '.join(parstates.split('_'))+' '+str(bn[node]['cp'][parstates][nodestate])+'\n')
    fparam.close()
    return

