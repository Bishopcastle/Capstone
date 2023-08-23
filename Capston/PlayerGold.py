import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
  
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)
  
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
    plt.xlim(0, 11)
    plt.ylim(0, 450)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
    
    #confidence interval
    ci = 1.91 * np.std(y)/np.sqrt(len(x))
    
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
    plt.plot(x, (y_pred+ci), linestyle = 'dotted')
    plt.plot(x, (y_pred-ci), linestyle = 'dotted')
    
    # calculating the r value
    r = np.corrcoef(x, y)[0,1]
    
    # place a text box in upper left in axes coords
    xl = len(x)
    plt.text(10, 480, 'Pop = %s'%(xl))
    plt.text(10, 500, 'r = %s'%(round(r,2)))
  
    
    # putting labels
    plt.title("Does Experience Give Better Average Gold?")
    plt.xlabel("Years Playing In Pro")
    plt.ylabel("Average Gold Per Minute")
    
    # function to show plot
    plt.grid()
    plt.show()
    
  
def main():
    # observations / data
    x = np.array([4,8,6,1,6,4,6,5,7,4,7,6,6,8,9,4,6,9,5,5,8,5,5,6,5,6,9,8,9,10,5,
                  6,9,10,3,2,3,3,4,5,4,5,6,5,5,6,7,7,7,5,4,8,7,4,8,5,9,2,8,4,6,6,6,7,
                  5,1,10,3,6,5,5,6,5,7,3,4,8,8,6,3,3,9])
    y = np.array([265,246,271,246,120,91,173,284,215,227,231,307,171,319,307,316,
                  237,266,225,259,250,199,323,149,341,314,129,288,88,200,210,355,191,234,
                  197,216,167,275,291,211,110,134,207,241,311,104,287,101,213,167,
                  114,106,106,116,282,255,215,229,199,205,168,222,307,276,272,285,
                  247,258,85,201,93,293,311,83,229,197,196,263,261,301,286,87])

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
  
    # plotting regression line
    plot_regression_line(x, y, b)
  
if __name__ == "__main__":
    main()