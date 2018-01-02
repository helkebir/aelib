import math
import numpy as np
from scipy import optimize

# Gravitational constant (m3/kg.s2)
G = 6.67408e-11

# ASTROPHYSICAL DATA OF CELESTIAL BODIES
# Sources: International Astronomical Union Resolution B3; NASA Planetary Fact Sheet
# Masses (kg)
dict_M = {
    'Sun'     : 1.98855e30,
    'Mercury' : 0.33011e24,
    'Venus'   : 4.8675e24,
    'Earth'   : 5.9722e24,
    'Moon'    : 0.07346e24,
    'Mars'    : 0.64171e24,
    'Jupiter' : 1898.19e24,
    'Saturn'  : 568.34e24,
    'Uranus'  : 86.813e24,
    'Neptune' : 102.413e24,
    'Pluto'   : 0.01303e24
}

# Volumetric mean radii (m)
dict_R = {
    'Sun'     : 6.957e8,
    'Mercury' : 2439.7e3,
    'Venus'   : 6051.8e3,
    'Earth'   : 6.3781e6,
    'Moon'    : 1737.4e3,
    'Mars'    : 3389.5e3,
    'Jupiter' : 7.1492e7,
    'Saturn'  : 58232e3,
    'Uranus'  : 25362e3,
    'Neptune' : 24622e3,
    'Pluto'   : 1187e3
}

# Gravitational acceleration
def g(body='Earth', h=0.0, m=1.0, R=None):
    try:
        M = dict_M[body]
    except:
        raise SyntaxError, 'Invalid celestial body'
    try:
        h = float(h)
    except TypeError:
        raise TypeError, 'Invalid altitude'
    if R != None:
        try:
            R = float(R)
        except TypeError:
            raise TypeError, 'Invalid radius'
        else:
            if R <= 0:
                raise ValueError, 'Radius must be greater than 0'
    else:
        R = dict_R[body]
    try:
        m = float(m)
    except TypeError:
        raise TypeError, 'Invalid mass'
    else:
        m = float(m)
        if m <= 0:
            raise ValueError, 'Mass must be greater than 0'
    return G*M*m/((R + h)**2)

# Mean anomaly solve function [M/n - t_tp = 0]
def M_solve(E, *data):
    a, e, mu, t_tp = data
    if E < 0:
        return 99999
    else:
        return np.sqrt((a**3)/mu)*(E - e*np.sin(E)) - t_tp

# Kepler's first law of planetary motion
def kepler(body='Earth', Q=None, q=None, e=None, a=None, t_tp=None, theta=None, degrees=False, interactive=False):
    try:
        M_b = dict_M[body]
        R = dict_R[body]
    except:
        raise SyntaxError, 'Invalid celestial body'
    if a != None and e != None:
        pass
    try:
        e = float(e)
    except TypeError:
        if e == None:
            try:
                Q = float(Q)
            except:
                if Q == None:
                    try:
                        a = float(a)
                    except TypeError:
                        raise TypeError, 'Invalid semi-major axis'
                    else:
                        if a <= 0:
                            raise ValueError, 'Semi-major axis must be greater than 0'
                        else:
                            try:
                                q = float(q)
                            except TypeError:
                                raise TypeError, 'Invalid periapsis'
                            else:
                                if q <= 0:
                                    raise ValueError, 'Periapsis must be greater than 0'
                                else:
                                    Q = 2*a - q
                                    e = (Q - q)/(Q + q)
                else:
                    raise ValueError, 'Invalid apoapsis'
            else:
                try:
                    q = float(q)
                except:
                    if q == None:
                        try:
                            a = float(a)
                        except TypeError:
                            raise TypeError, 'Invalid semi-major axis'
                        else:
                            if a <= 0:
                                raise ValueError, 'Semi-major axis must be greater than 0'
                            else:
                                try:
                                    Q = float(Q)
                                except TypeError:
                                    raise TypeError, 'Invalid apoapsis'
                                else:
                                    if Q <= 0:
                                        raise ValueError, 'Apoapsis must be greater than 0'
                                    else:
                                        q = 2*a - Q
                                        e = (Q - q)/(Q + q)
                    else:
                        raise ValueError, 'Invalid periapsis'
                else:
                    if q < 0 and Q < 0:
                        raise ValueError, 'Apoapsis and periapsis must be greater than 0'
                    else:
                        a = (Q + q)/2
                        e = (Q - q)/(Q + q)
    else:
        if -1 < e < 1:
            try:
                Q = float(Q)
            except:
                if Q == None:
                    try:
                        q = float(q)
                    except TypeError:
                        if q == None:
                            pass
                        else:
                            raise TypeError, 'Invalid periapsis'
                    else:
                        if q <= 0:
                            raise ValueError, 'Periapsis must be greater than 0'
                        else:
                            Q = q*(-1 - e)/(e - 1)
                            a = (Q + q)/2
                else:
                    raise ValueError, 'Invalid apoapsis'
            else:
                try:
                    q = float(q)
                except:
                    if q == None:
                        try:
                            Q = float(Q)
                        except TypeError:
                            raise TypeError, 'Invalid apoapsis'
                        else:
                            if Q <= 0:
                                raise ValueError, 'Apoapsis must be greater than 0'
                            else:
                                q = Q*(1 - e)/(e + 1)
                                a = (Q + q)/2
                    else:
                        raise ValueError, 'Invalid periapsis'
                else:
                    if q < 0 and Q < 0:
                        raise ValueError, 'Apoapsis and periapsis must be greater than 0'
                    else:
                        if a != (Q + q)/2:
                            raise ValueError, 'Conflicting semi-major axis and apsides'
                        elif e != (Q - q)/(Q + q):
                            raise ValueError, 'Conflicting eccentricity and apsides'
                        else:
                            a = (Q + q)/2
                            e = (Q - q)/(Q + q)
        else:
            raise ValueError, 'Invalid eccentricity (greater or equal |1|)'
    try:
        e = float(e)
    except TypeError:
        raise TypeError, 'Invalid eccentricity'
    else:
        if -1 < e < 1:
            pass
        else:
            raise ValueError, 'Invalid eccentricity (greater or equal |1|)'
    try:
        a = float(a)
    except TypeError:
        raise TypeError, 'Invalid semi-major axis'
    else:
        if a < 0:
            raise ValueError, 'Semi-major axis must be greater than 0'
    if theta != None:
        try:
            float(theta)
        except:
            try:
                theta = str(theta)
            except:
                raise TypeError, 'Invalid angle'
            else:
                if theta.find('pi') > 0:
                    try:
                        float(theta.split('pi')[0])
                    except:
                        raise SyntaxError, 'Invalid angle'
                    else:
                        theta = float(theta.split('pi')[0]) * np.pi
                else:
                    if theta.split('pi')[0] == '':
                        theta = np.pi
                    else:
                        raise SyntaxError, 'Invalid angle'
        else:
            theta = float(theta)
        if degrees == False:
            theta = theta
        elif degrees == True:
            theta = np.radians(theta)
        else:
            raise SyntaxError, 'Invalid degrees bool'
    T = (2*np.pi*np.sqrt((a**3)/(G*M_b)))
    try:
        t_tp = float(t_tp)
    except TypeError:
        if t_tp != None:
            raise TypeError, 'Invalid time from periapsis'
    else:
        if theta != None:
            raise ValueError, 'Only true anomaly or theta may be supplied'
        else:
            if t_tp >= T:
                t_tp = ((t_tp / T) - np.floor(t_tp / T))*T
            data = (a,e,(G*M_b),t_tp)
            E = optimize.fsolve(M_solve, 1, args=data)
            E = E[0]
            theta = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    p = a*(1-e**2)
    b = a*np.sqrt(1-np.absolute(e)**2)
    out = {
        'body' : body,
        'a'    : a,
        'b'    : b,
        'e'    : e,
        'p'    : p,
        'T'    : T
        }
    if e < 0:
        Q = a*(1-e)
        q = a*(1+e)
        V_p = np.sqrt((G*M_b/a)*((1+e)/(1-e)))
        V_a = np.sqrt((G*M_b/a)*((1+e)/(1-e)))
    elif e >= 0:
        Q = a*(1+e)
        q = a*(1-e)
        V_p = np.sqrt((G*M_b/a)*((1+e)/(1-e)))  
        V_a = np.sqrt((G*M_b/a)*((1-e)/(1+e)))
    h_a = Q - R
    h_p = q - R
    out['h_a'] = h_a
    out['h_p'] = h_p
    out['Q'] = Q
    out['q'] = q
    out['V_p'] = V_p
    out['V_a'] = V_a
    if theta != None:
        r = ((a*(1 - e**2)) / (1 + e * np.cos(theta)))
        E = 2*np.arctan(np.sqrt((1-np.absolute(e))/(1+np.absolute(e)))*np.tan(theta/2))
        M = E - e*np.sin(E)
        n = np.sqrt((G*M_b)/(a**3))
        t_tp = M/n
        tp_t = T - t_tp
        V_esc = np.sqrt((2*G*M_b)/r)
        out['theta'] = theta
        out['r'] = r
        out['E'] = E
        out['M'] = M
        out['n'] = n
        out['t_tp'] = t_tp
        out['tp_t'] = tp_t
        out['V_esc'] = V_esc
        V = np.sqrt((G*M_b)*((2/r)-(1/a)))
        out['V'] = V
    if interactive==True:
        print'+------------------------+'
        print'|GENERAL ORBIT PROPERTIES|'
        print'+------------------------+\n'
        print'CELESTIAL BODY (body): {}'.format(body.upper())
        print'BODY RADIUS       (R): {:.9e} m'.format(R)
        print'BODY MASS       (M_b): {:.9e} kg'.format(M_b)
        print'ECCENTRICITY      (e): {:.9e}'.format(e)
        print'SEMI-MAJOR AXIS   (a): {:.9e} m'.format(a)  
        print'SEMI-MINOR AXIS   (b): {:.9e} m'.format(b)  
        print'SEMI-LATUS RECTUM (p): {:.9e} m'.format(p)
        print'PERIOD            (T): {:.9e} s'.format(T)  
        print'APOAPSIS         (Ap):'        
        print'|=> RADIUS        (Q): {:.9e} m'.format(Q)
        print'|=> ALTITUDE    (h_a): {:.9e} m'.format(h_a)
        print'+=> VELOCITY    (V_a): {:.9e} m/s'.format(V_a)
        print'PERIAPSIS        (Pe):'        
        print'|=> RADIUS        (q): {:.9e} m'.format(q)
        print'|=> ALTITUDE    (h_p): {:.9e} m'.format(h_p)
        print'+=> VELOCITY    (V_p): {:.9e} m/s'.format(V_p)
        if theta != None:
            print'\n\n+------------------------------+'
            print'|INSTANTANEOUS ORBIT PROPERTIES|'
            print'+------------------------------+\n'
            print'TRUE ANOMALY  (theta): {:.9e} rad'.format(theta)
            print'ECCENTRIC ANOMALY (E): {:.9e} rad'.format(E)
            print'MEAN ANOMALY      (M): {:.9e} rad'.format(M)
            print'MEAN MOTION       (n): {:.9e} rad/s'.format(n)
            print'TIME FROM Pe   (t_tp): {:.9e} s'.format(t_tp)
            print'TIME TO Pe     (tp_t): {:.9e} s'.format(tp_t)
            print'INST. RADIUS      (r): {:.9e} m'.format(r)
            print'INST. VELOCITY    (V): {:.9e} m/s'.format(V)
            print'ESC. VELOCITY (V_esc): {:.9e} m/s'.format(V_esc)
    elif interactive==False:
        return out
    else:
        raise SyntaxError, 'Invalid interactive bool'
# Hohmann transfer
def hohmann(body='Earth', r_CO1=None, r_CO2=None, i_CO1=None, i_CO2=None, degrees=True, interactive=False): 
    try:
        M_b = dict_M[body]
        R = dict_R[body]
    except:
        raise SyntaxError, 'Invalid celestial body'
    try:
        i_CO1 = float(i_CO1)
    except TypeError:
        if i1 != None:
            raise TypeError, 'Invalid inclination angle (i_CO1)'
    else:
        try:
            i_CO2 = float(i_CO2)
        except TypeError:
            if i_CO2 != None:
                raise TypeError, 'Invalid inclination angle (i_CO2)'
            else:
                raise ValueError, 'Missing final inclination angle (i_CO2)'
        else:
            if degrees == False:
                i_CO1 = np.rad2deg(i_CO1)
                i_CO2 = np.rad2deg(i_CO2)
            elif degrees == True:
                pass
            else:
                raise TypeError, 'Invalid degrees bool'
            if 0 <= i_CO1 <= 180:
                pass
            else:
                raise ValueError, 'Inclination must be 0 <= i_CO1 <= (180 deg or pi rad)'
            if 0 <= i_CO2 <= 180:
                pass
            else:
                raise ValueError, 'Inclination must be 0 <= i_CO2 <= (180 deg or pi rad)'
    try:
        r_CO1 = float(r_CO1)
    except TypeError:
        raise TypeError, 'Invalid radius (r_CO1)'
    else:
        try:
            r_CO2 = float(r_CO2)
        except TypeError:
            raise TypeError, 'Invalid radius (r_CO2)'
        else:
            if r_CO1 > r_CO2:
                raise ValueError, 'Final radius must be greater than initial radius'
    h_CO1 = r_CO1 - R
    h_CO2 = r_CO2 - R
    mu = G*M_b
    a_TO = (r_CO1 + r_CO2)/2
    e_TO = (r_CO2 - r_CO1)/(r_CO2 + r_CO1)
    b_TO = a_TO*np.sqrt(1 - e_TO**2)
    p_TO = a_TO*(1 - e_TO**2)
    Q_TO = r_CO2
    q_TO = r_CO1
    h_a_TO = h_CO2
    h_p_TO = h_CO1
    T_CO1 = 2*np.pi*np.sqrt((r_CO1**3)/mu)
    T_CO2 = 2*np.pi*np.sqrt((r_CO2**3)/mu)
    T_TO = 2*np.pi*np.sqrt((a_TO**3)/mu)
    V_CO1 = np.sqrt(mu/r_CO1)
    V_p_TO = np.sqrt(mu*((2/r_CO1)-(1/a_TO)))
    V_a_TO = np.sqrt(mu*((2/r_CO2)-(1/a_TO)))
    V_CO2 = np.sqrt(mu/r_CO2)
    DeltaV_1 = np.absolute(V_p_TO - V_CO1)
    DeltaV_2 = np.absolute(V_CO2 - V_a_TO)
    out = {}
    out['body'] = body
    out['R'] = R
    out['r_CO1'] = r_CO1 
    out['r_CO2'] = r_CO2
    out['h_CO1'] = h_CO1
    out['h_CO2'] = h_CO2
    out['V_CO1'] = V_CO1
    out['V_CO2'] = V_CO2
    out['T_CO1'] = T_CO1
    out['T_CO2'] = T_CO2
    out['h_a_TO'] = h_a_TO
    out['h_p_TO'] = h_p_TO
    out['Q_TO'] = Q_TO
    out['q_TO'] = q_TO
    out['V_a_TO'] = V_a_TO
    out['V_p_TO'] = V_p_TO
    out['DeltaV_1'] = DeltaV_1
    out['DeltaV_2'] = DeltaV_2
    if i_CO1 != None:
        DeltaV_3 = np.sqrt(np.absolute((2*V_CO2**2)*(1 - np.cos(np.radians(i_CO1 - i_CO2)))))
        out['DeltaV_3'] = DeltaV_3
        out['i_CO1'] = i_CO1
        out['i_CO2'] = i_CO2
        DeltaV_tot = DeltaV_1 + DeltaV_2 + DeltaV_3
    else:
        DeltaV_3 = None
        DeltaV_tot = DeltaV_1 + DeltaV_2
    out['DeltaV_tot'] = DeltaV_tot
    if interactive == True:
        print'+-------------------+'
        print'|MANEUVER PROPERTIES|'
        print'+-------------------+\n'
        print'CELESTIAL BODY     (body): {}'.format(body.upper())
        print'BODY RADIUS           (R): {:.9e} m'.format(R)
        print'INITIAL CIRC. ORBIT (CO1):'
        print'|=> RADIUS   (r_CO1): {:.9e} m'.format(r_CO1)
        print'|=> ALTITUDE (h_CO1): {:.9e} m'.format(h_CO1)
        print'|=> VELOCITY (V_CO1): {:.9e} m/s'.format(V_CO1)
        print'+=> PERIOD   (T_CO1): {:.9e} s'.format(T_CO1)
        print'\n=========================='
        print'TRANSFER BURN        (TB):'
        print'+=> DELTA V (DeltaV_1): {:.9e} m/s'.format(DeltaV_1)
        print'==========================\n'
        print'TRANSFER ORBIT       (TO):'
        print'|=> ECCENTRICITY      (e_TO): {:.9e}'.format(e_TO)
        print'|=> SEMI-MAJOR AXIS   (a_TO): {:.9e} m'.format(a_TO)
        print'|=> SEMI-MINOR AXIS   (b_TO): {:.9e} m'.format(b_TO)
        print'|=> SEMI-LATUS RECTUM (p_TO): {:.9e} m'.format(p_TO)
        print'|=> PERIOD            (T_TO): {:.9e} s'.format(T_TO)
        print'|=> APOAPSIS (Q_TO):'
        print'|   |=> RADIUS     (Q_TO): {:.9e} m'.format(Q_TO)
        print'|   |=> ALTITUDE (h_a_TO): {:.9e} m'.format(h_a_TO)
        print'|   +=> VELOCITY (V_a_TO): {:.9e} m/s'.format(V_a_TO)
        print'+=> PERIAPSIS (q_TO):'
        print'    |=> RADIUS     (q_TO): {:.9e} m'.format(q_TO)
        print'    |=> ALTITUDE (h_p_TO): {:.9e} m'.format(h_p_TO)
        print'    +=> VELOCITY (V_p_TO): {:.9e} m/s'.format(V_p_TO)
        print'\n=========================='
        print'CIRCULARIZATION BURN (CB):'
        print'+=> DELTA V (DeltaV_2): {:.9e} m/s'.format(DeltaV_2)
        print'==========================\n'
        print'FINAL CIRC. ORBIT   (CO2):'
        print'|=> RADIUS   (r_CO2): {:.9e} m'.format(r_CO2)
        print'|=> ALTITUDE (h_CO2): {:.9e} m'.format(h_CO2)
        print'|=> VELOCITY (V_CO2): {:.9e} m/s'.format(V_CO2)
        print'+=> PERIOD   (T_CO2): {:.9e} s'.format(T_CO2)
        if DeltaV_3 != None:
            print'\n=========================='
            print'INCLINATION CHANGE   (IC):'
            print'|=> INITIAL INCLINATION (i_CO1): {:.9e} deg'.format(i_CO1)
            print'|=> FINAL INCLINATION   (i_CO2): {:.9e} deg'.format(i_CO2)
            print'+=> DELTA V          (DeltaV_3): {:.9e} m/s'.format(DeltaV_3)
            print'==========================\n'
        print'+-----------------------+'
        print'|OVERALL MANEUVER PROP\'S|'
        print'+-----------------------+\n'
        print'TOT. DELTA V (DeltaV_tot): {:.9e} m/s'.format(DeltaV_tot)
    else:
        return out
