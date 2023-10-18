# -*- coding: utf-8 -*-
"""
Tsunami Tehdidine Karşı İstanbul İli Marmara Denizi Kıyıları Tahliye Planlaması ve Yönetimi
Toy Problem
@author: Doruk & Ada
"""

#%%
#Import Libraries
from openpyxl import * 
import numpy as np
import math
import pandas as pd
import gurobipy as gp
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import re

#%%Assumptions

#Small Toy Network of Nodes and Arcs
#Vertical Shelter Costs
# tv[a,l] value in the objective function
# Risk set to 1 no paramter
# Pedestrian Capacity for Arcs (500)

#%%
#Read Data Files

#Arcs
Arcs = pd.read_excel("ToyData.xlsx", sheet_name = "Arcs") 
Arcs = Arcs.fillna(500)
Arcs_Len = len(Arcs.index)

#Select Arcs by Type (Vehicle or Pedestrian)
Vehicle_Arcs = []
for i in range(Arcs_Len):
    if Arcs.iloc[i]['Type'] == "v" or Arcs.iloc[i]['Type'] == "both":
        Vehicle_Arcs.append(Arcs.iloc[i][0])
 
Pedestrian_Arcs = []
for i in range(Arcs_Len):
    Pedestrian_Arcs.append(Arcs.iloc[i][0])    


Arc_Cap = Arcs.copy()
Arc_Cap = Arcs.drop(["Tail","Head","Type","Neighbor Zones","Length (km)","Speed Limit","t","risk"],axis=1)    
#Arc_Cap.set_index("Arc",inplace = True)
    
#Nodes
Nodes = pd.read_excel("ToyData.xlsx", sheet_name = "Nodes")   
Nodes_Len = len(Nodes.index)


#Classify Nodes
Vertical_Shelters = []
for i in range(Nodes_Len):
    if Nodes.iloc[i]['Vertical Shelter?'] == 1:
        Vertical_Shelters.append(Nodes.iloc[i][0])
        
Demand_Nodes = []
for i in range(Nodes_Len):
    if Nodes.iloc[i]['Origin?'] == 1:
       Demand_Nodes.append(Nodes.iloc[i][0])
       
Exit_Point = []
for i in range(Nodes_Len):
    if Nodes.iloc[i]['Exit Point?'] == 1:
       Exit_Point.append(Nodes.iloc[i][0])
       
'''    
#Define Incoming and Outgoing Arcs for Nodes
Arcs_Flow = [[]]
for i in range(Nodes_Len):
    for j in range(2):
        Arcs_Flow[i][2] == Nodes.iloc[i][7]
'''          
#print((Nodes.iloc[0]["Outgoing arc"]))

def convertList(stringElement):
    if type(stringElement) == str:
        
        return( re.findall("[A-Z][0-9][0-9]*", stringElement) )
    return([])
    


def changeArcsToList(Nodes):
    for index in (Nodes.index):
        Nodes.at[index,"Outgoing arc"] = convertList(Nodes.at[index,"Outgoing arc"])
        Nodes.at[index,"Incoming arc"] = convertList(Nodes.at[index,"Incoming arc"])

changeArcsToList(Nodes)
#print(Nodes.iloc[0]["Outgoing arc"])

#%%
#Define Parameter Values
V = Vertical_Shelters
O = Demand_Nodes
Origins =[]

for i in range(len(O)):
    if 'J' in O[i]:
        Origins.append(O[i])
        
Av = Vehicle_Arcs
Ap = Pedestrian_Arcs
Dv =  Nodes['V_Demand']
Nodes_List = Nodes['Nodes']
Dv = Dv.fillna(0)
Dp =  Nodes.copy()
Dp = Dp.drop(["Vertical Shelter?","Origin?","Exit Point?","V_Demand","Capacity","Outgoing arc","Incoming arc"],axis = 1)
Dp.set_index("Nodes",inplace = True)
Dp = Dp.fillna(0)


Qv =  Nodes.copy()
Qv = Qv.drop(["Vertical Shelter?","Origin?","Exit Point?","V_Demand","P_Demand","Outgoing arc","Incoming arc"],axis = 1)
Qv.set_index("Nodes",inplace = True)
Qv = Qv.fillna(0)


C = Arc_Cap
L = [1,2]
Vcost = 50
Budget = 100

t = Arcs["t"]

t =  Arcs.copy()
t  = Arcs.drop(["Tail","Head","Type","Capacity","Type","Neighbor Zones","Length (km)","Speed Limit","risk"],axis=1)
t.set_index("Arc",inplace = True)
t = t.fillna(0)

r = Arcs["risk"]

Flow = Nodes.copy()
Flow = Flow.drop(["Vertical Shelter?","Origin?","Exit Point?","V_Demand","P_Demand","Capacity"],axis=1)

#BPR Function Parameters
alpha = 0.15
beta = 4



#%%#Mathematical Model

model = gp.Model()


#Decision Variables

#fv[a,l] The amount of vehicle flow on arc a evacuating from origin o to shelter v
fv = {}
for a in Av:
    for l in L:
            fv[a,l] = model.addVar(vtype = gp.GRB.CONTINUOUS,name=f"fv{a}")


#fv2[a,o,v] The amount of vehicle flow on arc a evacuating from origin o to shelter v
fv2 = {}
for o in O:
    for v in V:
        for a in Ap:
            fv2[a,o,v] = model.addVar(vtype = gp.GRB.CONTINUOUS,name=f"fv2{a}_{o}_{v}")
        
#fp[o,v] The amount of pedestrian flow on arc a evacuating from origin o to shelter v
fp = {}
for o in O:
    for v in V:
        fp[o,v] = model.addVar(vtype = gp.GRB.CONTINUOUS,name=f"fp_{o}_{v}")
        
#x[o,v] Binary assignment variable of origin o to vertical shelter v
x = {}
for o in O:
    for v in V:
        x[o,v] = model.addVar(vtype = gp.GRB.BINARY,name=f"x_{o}_{v}")
        
#y[v] Binary assignment variable  vertical shelter v
y = {}
for v in V:
        y[v] = model.addVar(vtype = gp.GRB.BINARY,name=f"y_{v}")

#uv[a] Binary variable representing usage of arc a by vehicles from origin o to shelter V
uv = {}
for a in Av:
            uv[a] = model.addVar(vtype = gp.GRB.BINARY,name=f"uv_{a}")

#up[a] Binary variable representing usage of arc a by pedestrians from origin o to shelter V
up = {}
for a in Ap:
            up[a] = model.addVar(vtype = gp.GRB.BINARY,name=f"up_{a}")
#z[a,l] Speed Level Variable
z = {}
for a in Ap:
    for l in L:
           z[a,l] = model.addVar(vtype = gp.GRB.BINARY,name=f"z_{a}_{l}")
                   
            
#Constraints

#Constraint 1

for i in range(Nodes_Len):
    if 'J' not in (Flow.iloc[i][0]):
        if Flow.iloc[i][0] not in Exit_Point:
              # if (Flow.iloc[i][1][i2] != "A3" and  Flow.iloc[i][1][i2] != "A4") and (Flow.iloc[i][2][i3] != "A3" and  Flow.iloc[i][2][i3] != "A4"):
                    model.addConstr(((gp.quicksum(fv2[Flow.iloc[i][1][j],o,v] for j in range(len(Flow.iloc[i][1])) for o in O ))) ==
                                     gp.quicksum(((fv2[(Flow.iloc[i][2][k]),o,v]  for k in range(len(Flow.iloc[i][2])) for o in O ))))


for i in range(Nodes_Len):
    if 'J' in (Flow.iloc[i][0]):
            model.addConstr(((gp.quicksum(fv2[Flow.iloc[i][1][j],o,v] for j in range(len(Flow.iloc[i][1])) for o in O ))) ==
                                 ((Dv[i])))

Nodes_List2 = []
for i in range(Nodes_Len):
    for l in range(len(Nodes_List)):
        if Flow.iloc[i][0] in Exit_Point:
            if i == l:
                Nodes_List2. append(Nodes_List[l])
             

for i in Nodes_List2:
    temp = Nodes_List.index(i)
    print(temp)
              
model.addConstr(gp.quicksum(fv2[Flow.loc[i][2][j],o,v] for i in (Nodes_List2) for j in range(len(Flow.loc[i][2])) for o in O) ==
                       (sum(Dv)))

#Constraint 3
#for a in (Av):
#    for i in range(len(Av)):
#        if a.index(a) == i:
#            for l in L:
#                print(C.iloc[i][1])
#                model.addConstr(fv[a,l] <= C.iloc[i][1]*z[a,l])


#Constraint 4
for a in Av:
    for l in L:
        model.addConstr(gp.quicksum(fv2[a,o,v] for o in O) == fv[a,l])          

#Constraint 5
#for a in Av:
#     for i in range(len(Av)):
#        if a.index(a) == i:
#            model.addConstr(gp.quicksum(fv[a,l] for l in L) <= (C.iloc[i][1] * uv[a]))
for a in Ap:
    for i in range(len(Av)):
        if a.index(a) == i:
            for v in V:
                model.addConstr(gp.quicksum(fp[o,v] for o in O) <= C.iloc[i][1] * up[a])
        
#Constraint 6
for a in Av:
    model.addConstr(z[a,2] >= up[a])


#Constraint 7
for a in Ap:
    model.addConstr(gp.quicksum(z[a,l] for l in L) == 1)

#Constraint 8
for v in V:
    model.addConstr((gp.quicksum(Dp.loc[o] for o in O) * (gp.quicksum(x[o,v] for o in O)))  <= (Qv.loc[v][0] * y[v]))  
 
#Constraint 9
model.addConstr(gp.quicksum(Vcost * y[v] for v in V) <= Budget)


#Constraint 10
for a in Ap:
    for o in O:
        for i in range(len(Ap)):
            if a.index(a) == i:
                model.addConstr(fp[o,v] <= C.iloc[i][1] * up[a])
        
        
#Objective Function

#model.setObjective((gp.quicksum(t.loc[a] * fp[o,v] * 1 for a in Ap for o in O for v in V)) +
#                   (gp.quicksum(1 * t.loc[a] * fv2[a,o,v] for a in Av for o in O for v in V ))
#                   )


objfunction = gp.LinExpr()
for a in Ap:
    for o in O:
        for v in V:
            objfunction.addTerms(t.loc[a]* 1,fp[o,v])
            
for a in Av:
    for o in O:
        for v in V:
            objfunction.addTerms(t.loc[a] * 1, fv2[a,o,v])
            
            
model.setObjective(objfunction,gp.GRB.MINIMIZE)



model.optimize () #Solve the problem
model.getObjective()
model.computeIIS()
model.write("model.ilp")
model.write("model.lp")


for v in model.getVars():
    if v.x > 0:
        print('%s %g' % (v.varName, v.x))
'''        
print('Passenger Flow:')
for o in O:
    for v in V:
        print(fp[o,v].x)
        
print('Vehicle Flow:')
for o in O:
    for a in Av:
        for v in V:
            print(fv2[a,o,v].x)
'''

        