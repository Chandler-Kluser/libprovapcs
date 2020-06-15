def v(last_v,r,ac1,ac2,gamma):
    """
    Returns the new V vector (shape (2,)) from a VIP problem
    """
    a1 = r[0] + gamma*max(ac1[0]*last_v[0]+ac1[1]*last_v[1],ac1[2]*last_v[0]+ac1[3]*last_v[1])
    a2 = r[1] + gamma*max(ac2[0]*last_v[0]+ac2[1]*last_v[1],ac2[2]*last_v[0]+ac2[3]*last_v[1])
    return [a1,a2]
def iter(tol,initial_v,act1,act2):
    """
    Iterates until a tolerance from a VIP problem
    """
    diff=tol+1
    i=1
    old_v = initial_v
    while diff>tol:
        print('Iteration ',i)
        final_v = v(old_v,[3,-1],act1,act2)
        print('V = ',final_v)
        diff = abs(final_v[0]-old_v[0])+abs(final_v[1]-old_v[1])
        old_v=final_v
        i += 1