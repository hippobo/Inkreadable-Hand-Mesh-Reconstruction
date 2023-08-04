# Geometry functions
import math
import mathutils

def getXYLineCoeff(A, B):
    """
    getXYLineCoeff
    in: A, B (Point) 
    out: m, p (float) respectively slope and ordinate at origin of line
                      (AB): y=mx + p in plane (XY), not taking Z into account
                      0 and A.y if A and B are vertically aligned
    """
    # vertical line
    m = 0
    p = A.y 
    if (B.x - A.x != 0) :
        m = (B.y - A.y)/(B.x - A.x)
        p = B.y - (B.x * m)
    return m, p

def getXYBisectorCoeff(A, B):
    """
    getXYBisectorCoeff
    in: A, B (Point) 
    out: m, p (float) respectively slope and ordinate at origin 
                      of the perpendicular bisector to segment AB
                      inplane (XY), not taking Z into account
                      0 and A.y if A and B are vertically aligned
    """
    # vertical line
    mAB, pAB =  getXYLineCoeff(A,B)
    I = (A + B)/2
    m = 0
    p = I.y
    if (mAB != 0) :
        m = -1 / mAB
        p = I.y - (I.x * m)
    return m, p


def rotateXYPoint(C, P, alpha):
    """
    rotateXYPoint
    in: C (Point) center of rotation
        P (Point) point to rotate
        alpha (float) angle of rotation (radians)
    out: rotated point in (XY) plane (Tuple)
    """
    dx = P.x - C.x
    dy = P.y - C.y
    return (C.x + dx * math.cos(alpha) - dy * math.sin(alpha), C.y + dx * math.sin(alpha) + dy * math.cos(alpha),0)



def solvePoly2(a, b, c):
    """
    solvePoly2
    in: a, b, c (float) 
    out: return solutions of ax2 + bx + c = 0 or 0,0 if no solution  
    """        
    delta = b**2 - (4 * a * c)
    try :        
        delta = math.sqrt(b**2 - (4 * a * c))
        x0 = ((-1.0 * b) + delta)/(2*a)
        x1 = ((-1.0 * b) - delta)/(2*a)       
        return x0, x1
    except ValueError :
        print ("no solution to ax2 + bx + c = 0")
        return 0, 0

def getXYCircleCenter(A, B, alpha, leftside):
    """
    getXYCircleCenter
    in : A, B (mathutils.Vector)
         alpha (float) in radians
    out : C (Point) returns the center of circle passing through A and B
          with angle alpha between CA and CB,
          in (XY) plane, don't take account of z, C.z set to 0
    """
    M = (A + B)/2
    ls = (B - M).length
    lc = ls /( math.tan(alpha/2))
    m, p = getXYBisectorCoeff(A, B)
    a = 1 + m**2
    b = -2*(M.x + m*(M.y - p))
    c = M.x**2 + (M.y - p)**2 - lc**2
    x0, x1 = solvePoly2(a, b, c)
    C0 =(x0,m * x0 + p,0) 
    C1 =(x1,m * x1 + p,0)
    if (leftside and C1[0] > C0[0]) or (not(leftside) and C1[0] < C0[0]) :
        return C0
    else:
        return C1

    
