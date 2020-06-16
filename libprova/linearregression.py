def linreg(x,y):
    """
    Returns a and b coefficients from a single linear regression
    x: list of x values
    y: list of y values
    """
    n = len(x)
    sx = 0
    sy = 0
    sxx = 0
    sxy = 0
    for i in range(0,n):
        sx += x[i]
        sy += y[i]
        sxx += x[i]*x[i]
        sxy += x[i]*y[i]
    print('sx = ',sx)
    print('sy = ',sy)
    print('sxx = ',sxx)
    print('sxy = ',sxy)
    den = n*sxx-sx**2
    b = (sy*sxx-sx*sxy)/den
    a = (n*sxy-sx*sy)/den
    print('a = ',a)
    print('b = ',b)
    return [a,b]

def estimate(x,reg):
    """
    Gives a linear regression estimation from an x value
    x: float or list
    reg: list of a and b values, in this order
    """
    if isinstance(x,list):
        lista = []
        for i in x:
            lista.append(estimate(i,reg))
        return lista
    xhat = reg[0]*x+reg[1]
    print('xhat = ',xhat)
    return xhat

def RSS(y,yhat):
    """
    Calculates RSS value given y and yhat
    y and yhat: lists
    """
    n = len(y)
    sum = 0
    for i in range(0,n):
        factor = y[i]-yhat[i]
        print('------ factor ',i,' ------')
        print('y = ',y[i])
        print('yhat = ',yhat[i])
        print('y - yhat = ',factor)
        print('(y - yhat)**2 = ',factor**2)
        sum += factor**2
    print('RSS = ',sum)
    return sum