#!/usr/bin/env python
# coding: utf-8

# In[31]:


class structure (dict):

    # String representation of the structure
    def __repr__(self):
        return 'structure({})'.format(super().__repr__())
    
    # Get field value
    def __getattr__(self, field):
        if field not in dir(self):
            if field in self.keys():
                return self[field]
            else:
                return None
        else:
            return None
    
    # Set field value
    def __setattr__(self, field, value):
        if field not in dir(self):
            self[field] = value
        else:
            return super().__setattr__(field, value)
    
    # Get the list of structure fields
    def fields(self):
        return list(self.keys())

    # Deletes a field from structure
    def remove_field(self, field):
        if field in self.keys():
            del self[field]
    
    # Adds a new field to the structure
    def add_field(self, field, value = None):
        if field not in self.keys():
            self[field] = value

    # Creates a shallow copy of the structure
    def copy(self):
        import copy as cp
        self_copy = structure()
        for field in self.keys():
            if isinstance(self[field], structure):
                self_copy[field] = self[field].copy()
            else:
                self_copy[field] = cp.copy(self[field])
        
        return self_copy

    # Creates a deep copy of the strucre
    def deepcopy(self):
        import copy as cp
        self_copy = structure()
        for field in self.keys():
            if isinstance(self[field], structure):
                self_copy[field] = self[field].deepcopy()
            else:
                self_copy[field] = cp.deepcopy(self[field])
        
        return self_copy

    # Repeats (replicates) the structure to create an stratucre array (eg. for initialization)
    def repeat(self, n):
        return [self.deepcopy() for i in range(n)]


# In[32]:


from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import cross_val_score

num_features_to_select = 8

def fitness(solution):  
    idx = list(np.argsort(solution)[-num_features_to_select:])
    selected_features = data[:,idx]
    model = DecisionTreeClassifier()
    scores = cross_val_score(model, selected_features, labels, scoring='accuracy')
    score = np.mean(scores) 
    return score


# In[33]:


import pandas as pd
import numpy as np

dataset = pd.read_csv('pima-indians-diabetes.csv', header = None)
dataset.head(5)


# In[34]:


type(dataset)


# In[35]:


data = dataset.iloc[:,:-1]
data.head(5)


# In[36]:


labels = dataset.iloc[:,-1]
labels


# In[37]:


labels = np.asarray(labels)


# In[38]:


type(labels)


# In[39]:


from sklearn.preprocessing import normalize


# In[40]:


data = normalize(data)


# In[41]:


data


# In[42]:


import numpy as np

# Run Giza Pyramids Construction (GPC)
def run(problem, params):
    
    # Problem Definition
    objfunc = problem.objfunc          # Cost Function
    nvar = problem.nvar                # Number of Decision Variables
    varmin = problem.varmin            # Decision Variables Lower Bound
    varmax = problem.varmax            # Decision Variables Upper Bound

    # Params
    maxit = params.maxit               # Maximum Number of Iterations (Days of work)
    npop = params.npop                 # Number of workers
    G = params.G                       # Gravity
    Tetha = params.Tetha               # Angle of Ramp
    MuMin = params.MuMin               # Minimum Friction 
    MuMax = params.MuMax               # Maximum Friction
    pSS = params.pSS                   # Substitution Probability
    DisplayInfo = params.DisplayInfo

    # Empty Stones or Workers (Individual)
    empty_stones = structure()
    empty_stones.position = None
    empty_stones.cost = None

    # Best Solution
    best_worker = empty_stones.deepcopy()
    best_worker.cost = -np.inf

    # Best Costs History
    bestcosts = np.empty(maxit)

    # Initialization
    pop = empty_stones.repeat(npop)
    for i in range(0, npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = objfunc(pop[i].position)
        if pop[i].cost >= best_worker.cost:
            best_worker = pop[i].deepcopy()        # as Pharaoh's special agent

    # Construction Main Loop
    for it in range(0, maxit):
        for i in range(0, npop):
            
            V0 = np.random.rand(1)                                                             # Initial Velocity  
            temp_rand = np.random.rand(1)                                                      # Temp Random Number
            Mu= MuMin+(MuMax-MuMin)*temp_rand[0]                                               # Friction
            d = (V0[0]**2)/((2*G)*(np.sin(np.deg2rad(Tetha))+(Mu*np.cos(np.deg2rad(Tetha)))))  # Stone Destination  
            x = (V0[0]**2)/((2*G)*(np.sin(np.deg2rad(Tetha))))                                 # Worker Movement
            epsilon = np.random.uniform(-((varmax-varmin)/2),((varmax-varmin)/2), nvar)        # Epsilon
            new_position = apply_bounds((pop[i].position + d) * (x * epsilon), varmin, varmax) # Position of Stone and Worker

            # Substitution
            newsol = empty_stones.deepcopy()
            newsol.position = substitution(pop[i].position, new_position, pSS)
            newsol.cost = objfunc(newsol.position)

            if newsol.cost >= pop[i].cost:
                pop[i] = newsol
                if pop[i].cost >= best_worker.cost:
                    best_worker = pop[i].deepcopy()

        # Store Best Cost of Iteration
        bestcosts[it] = best_worker.cost

        # Show Iteration Info
        if DisplayInfo:
            print("Iteration {0}: Best fitness = {1}".format(it, best_worker.cost))
    
    # Return Results
    out = structure()
    out.best_worker = best_worker
    out.bestcost = best_worker.cost
    out.bestcosts = bestcosts
    out.pop = pop
    return out

# Apply Decision Variable Ranges
def apply_bounds(x, varmin, varmax):
    x = np.maximum(x, varmin)
    x = np.minimum(x, varmax)
    return x

# Substitution
def substitution(x, new_position, pSS):
    z = np.copy(x)
    nvar = x.size
    k = np.where(np.random.rand(nvar) <= pSS)
    k = np.append(k, np.random.randint(0, nvar))
    z[k] = new_position[k]
    return z
    


# In[ ]:



# External Libraries
import numpy as np
import matplotlib.pyplot as plt


# Problem Definition
problem = structure()
problem.objfunc = fitness  # See benchmarks module for other functions
problem.nvar = 8
problem.varmin = 0
problem.varmax = 1

# Parameters of Giza Pyramids Construction (GPC)
params = structure()
params.maxit = 200                   # Maximum Number of Iterations (Days of work)
params.npop = 30                     # Number of workers
params.G = 9.8                       # Gravity
params.Tetha = 14                    # Angle of Ramp
params.MuMin = 1                     # Minimum Friction 
params.MuMax = 10                    # Maximum Friction
params.pSS = 0.5                     # Substitution Probability
params.DisplayInfo = True

# Run GPC
out = run(problem, params)

# Print Final Result
print("Final Best Solution: {0}".format(out.best_worker))

# Plot of Best Costs History
plt.semilogy(out.bestcosts)
plt.xlim(0, params.maxit)
plt.xlabel("Iterations")
plt.ylabel("Best fitness")
plt.title("Giza Pyramids Construction")
plt.grid(True)
plt.show()


# In[ ]:


out.best_worker


# In[ ]:


np.argsort(out.best_worker.position)[-5:]


# In[ ]:




