import numpy as np
import matplotlib.pyplot as plt
  
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
    plt.title("Does Experience Give Better Average Gold (Support)?")
    plt.xlabel("Years Playing In Pro")
    plt.ylabel("Average Gold Per Minute")
  
    # function to show plot
    plt.grid()
    plt.show()
  
def main():
    # observations / data
    x = np.array([6,4,9,9,4,5,6,7,4,8,7,4,6,5,7,9])
    y = np.array([120,91,129,88,110,134,104,101,114,106,106,116,85,93,83,87])
  
    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
  
    # plotting regression line
    plot_regression_line(x, y, b)
  
if __name__ == "__main__":
    main()
    
    