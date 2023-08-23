import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Reshaped for Logistic function.
x = np.arange(46).reshape(-1,1)
y = np.array([1,1,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,
              0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1])

logr = linear_model.LogisticRegression()
logr.fit(x,y)

def sigmond(s):
    return 1.0 / (1.0 + np.exp(-s))
val = np.arange(-46, 46, 1)

def logit2prob(logr, x):
  log_odds = logr.coef_ * x + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)
print(logit2prob(logr, x))

plt.title("Does The Team That Picks Aatrox Have Higher Win")
plt.xlabel("Was Picked")
plt.ylabel("Won The Match")
plt.scatter(x, y)
plt.plot(val,sigmond(val))
plt.show()