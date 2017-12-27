# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from matplotlib import rc
rc('mathtext', default='regular')

#RAW AERODYNAMIC DATA

alfa = np.linspace(-4.0,11.0,num=76)
CL = np.array([-0.045012,-0.021045,0.004286,0.03136,0.058865,0.08379,0.106369,0.129043,0.153038,0.178472,0.20125,0.221113,0.242899,0.264935,0.28566,0.303924,0.320166,0.349729,0.381481,0.405462,0.427334,0.447064,0.466523,0.486592,0.505741,0.525211,0.544567,0.563632,0.582763,0.602193,0.621595,0.640664,0.65992,0.67887,0.697738,0.716764,0.736224,0.755249,0.774052,0.792928,0.812238,0.83124,0.849798,0.868473,0.88777,0.906576,0.92498,0.943429,0.962478,0.981082,0.999402,1.017451,1.036057,1.054223,1.072074,1.089735,1.107360,1.124718,1.142030,1.159264,1.175409,1.192119,1.208218,1.223509,1.239098,1.253754,1.268017,1.281829,1.294423,1.306207,1.316859,1.326179,1.333502,1.338089,1.340471,1.320323])
Cd = np.array([0.020963,0.020433,0.019843,0.019289,0.018823,0.018463,0.018147,0.017838,0.017535,0.01726,0.017039,0.016861,0.016659,0.016404,0.016128,0.015803,0.015288,0.015091,0.015122,0.01535,0.015562,0.015801,0.016103,0.016411,0.016765,0.017127,0.017518,0.017931,0.018365,0.018831,0.019322,0.019816,0.020344,0.020893,0.021476,0.022068,0.022696,0.023337,0.02398,0.024642,0.025349,0.026063,0.026766,0.027493,0.028275,0.029047,0.029816,0.030591,0.031425,0.032259,0.033085,0.0339,0.034761,0.035614,0.036474,0.037311,0.038163,0.038998,0.039841,0.040706,0.041519,0.042376,0.043221,0.044039,0.044899,0.04573,0.046571,0.047414,0.048243,0.049099,0.049984,0.050878,0.051797,0.052759,0.053819,0.058003])
CD = np.array([])
CLmin = 0.004286
CLmax = 1.340471

#ISA

grad = -0.0065 #K/m
R = 287 #J/kg.K
T0 = 288.15 #K

def frho(h):
    T = T0 + grad*h
    return 1.225*(T/T0)**(-g/(grad*R) - 1)

def fp(h):
    T = T0 + grad*h
    return 101325*(T/T0)**(-g/(grad*R))

h_mars = np.linspace(0,100000,num=101)
rho_mars = np.array([0.015029863,0.013793119,0.012658359,0.011617156,0.010661782,0.009785149,0.008980753,0.00824263,0.007636285,0.007046484,0.006502845,0.006001719,0.005539749,0.005113845,0.004721162,0.004359082,0.004025194,0.003717281,0.003433298,0.003171365,0.002929752,0.002706862,0.002501228,0.002311498,0.002136427,0.001974868,0.001825764,0.001688144,0.001561111,0.00144384,0.001335569,0.001235599,0.001143285,0.00105803,0.000979289,0.000906554,0.000839362,0.000777282,0.000719921,0.000666913,0.000617922,0.000572639,0.000530779,0.000492077,0.000456292,0.0004232,0.000392594,0.000364283,0.000338093,0.000313862,0.000291439,0.000270688,0.000251481,0.0002337,0.000217239,0.000201995,0.000187879,0.000174803,0.000162691,0.000151468,0.00014107,0.000131433,0.0001225,0.000114219,0.000106542,9.94229E-05,9.28207E-05,8.66971E-05,8.10167E-05,7.57467E-05,7.0857E-05,6.63197E-05,6.21091E-05,5.82013E-05,5.45744E-05,5.12081E-05,4.80837E-05,4.51839E-05,4.24927E-05,3.99954E-05,3.76786E-05,3.55296E-05,3.3537E-05,3.16903E-05,2.99798E-05,2.83967E-05,2.69328E-05,2.55808E-05,2.43342E-05,2.31871E-05,2.21342E-05,2.1171E-05,2.02937E-05,1.94993E-05,1.87855E-05,1.8151E-05,1.75955E-05,1.71201E-05,1.67273E-05,1.64219E-05,1.62114E-05])

r_L = P.polyfit(h_mars,rho_mars,2,full=False)
r0 = float(r_L[0])
r1 = float(r_L[1])
r2 = float(r_L[2])

def frho_mars(h):
    return r0 + r1*h + r2*h**2

def fp_mars(h):
    return 669*np.exp(-0.00009*h)

def fa(h, mars):
    if mars == True:
        return np.sqrt(1.3*((fp_mars(h))/(frho_mars(h))))
    else:
        return np.sqrt(1.4*((fp(h))/(frho(h))))

def fM(V, h, mars):
    return V / fa(h,mars)

#WING PROPERTIES

b = 3
S = 0.45
c_r = 0.18
mac = 0.15
A = 20
l = 1.5
e = 1

#AIRCRAFT PROPERTIES

Pel = 800 #Wh
Pel = Pel*(60*60) #W
n_motor = 0.87
n_prop = 0.79
Pbr = Pel * n_motor
Pa = Pbr * n_prop
m = 12 #kg
g = 9.80665
g_mars = 0.375 #g_earth
W = m*g
rho = 1.225
opt = 'cl/cd'
CLopt = 0

def init():
    global Pa, S, CD, A, W, CD0, k1, k2
    i_batt = float(raw_input('BATTERY MASS (4) [kg]: ') or 4.0) / 4.0
    i_payl = float(raw_input('PAYLOAD MASS (4.5) [kg]: ') or 4.5) / 4.5
    m = 3.5 + i_payl*4.5 + i_batt*4.0
    W = m*g
    Pel = 800*i_batt*(60*60)
    Pbr = Pel * n_motor
    Pa = Pbr * n_prop
    i_span = float(raw_input('SPAN (3) [m]: ') or 3.0) / 3.0
    b = 3*i_span
    S = b*mac
    A = b**2 / S
    CD = np.array([])
    for x in range(0,len(CL)):
        CD = np.append(CD, (Cd[x] + (CL[x])**2/(np.pi * A * e)))
    print'MASS:',m,' [kg]'
    print'WING AREA:',S,' [m^2]'
    print'ASPECT RATIO:',A
    CD = np.array([])
    for x in range(0,len(CL)):
        CD = np.append(CD, (Cd[x] + (CL[x])**2/(np.pi * A * e)))
    k_L = P.polyfit(CL,CD,2,full=False)
    CD0 = float(k_L[0])
    k1 = float(k_L[1])
    k2 = float(k_L[2])
    

for x in range(0,len(CL)):
    CD = np.append(CD, (Cd[x] + (CL[x])**2/(np.pi * A * e)))

k_L = P.polyfit(CL,CD,2,full=False)
CD0 = float(k_L[0])
k1 = float(k_L[1])
k2 = float(k_L[2])
CL_L = np.arange(-0.045,1.35,0.005)
CLpos_L = np.arange(0.01,1.36,0.005)

a_L = P.polyfit(alfa,CL,1,full=False)
CL0 = float(a_L[0])
a = float(a_L[1])

#AERODYNAMIC FUNCTIONS

def fCD(x):
    return CD0 + k1*x + k2*x**2

def fCL(x):
    return CL0 + a*x

def fALFA(x):
    return (x - CL0) / a

def optf(CL):
    if opt == 'cl/cd':
        return CL / fCD(CL)
    elif opt == 'cl/cd2':
        return CL / fCD(CL)**2
    elif opt == 'cl3/cd2':
        return CL**3 / fCD(CL)**2

def invf(x):
    if x == 0:
        return float("inf")
    else:
        return 1.0 / optf(x)

def fV(CL,mars):
    if mars == True:
        return np.sqrt(((W*g_mars)/S)*(2.0/rho)*(1.0/CL))
    else:
        return np.sqrt((W/S)*(2.0/rho)*(1.0/CL))

def fV_rho(CL,r,mars):
    if mars == True:
        return np.sqrt(((W*g_mars)/S)*(2.0/r)*(1.0/CL))
    else:
        return np.sqrt((W/S)*(2.0/r)*(1.0/CL))

def fPr(h, CL, mars):
    if mars == True:
        return fCD(CL)*0.5*frho_mars(h)*S*(((W*g_mars)/S)*(2/frho_mars(h))*(1/CL))**(1.5)
    else:
        return fCD(CL)*0.5*frho(h)*S*((W/S)*(2/frho(h))*(1/CL))**(1.5)

def fhmax(CL, mars):
    if mars == True:
        REAL_Pa = int(np.round((400.0*n_motor*n_prop),0))
        h = 0
        while h < 11000:
            Pr = np.round(fPr(h, CL, True),0).astype(np.int64)
            print Pr
            if Pr == REAL_Pa:
                return h
                break
            else:
                h += 1
        return 0
    else:
        REAL_Pa = int(np.round((400.0*n_motor*n_prop),0))
        h = 0
        while h < 11000:
            Pr = np.round(fPr(h, CL, False),0).astype(np.int64)
            print Pr
            if Pr == REAL_Pa:
                return h
                break
            else:
                h += 1
        return 0

def fVmax(h, mars):
    x = 0.01
    if mars == True:
        while x < 1.36:
            REAL_Pa = int(np.round((400.0*n_motor*n_prop),0))
            Pr = np.round(fPr(h, x, True),0).astype(np.int64)
            if Pr == REAL_Pa:
                return fV_rho(x,frho(h),True)
                break
            else:
                x += 0.00001
        return 0
    else:
        while x < 1.36:
            REAL_Pa = int(np.round((400.0*n_motor*n_prop),0))
            Pr = np.round(fPr(h, x, False),0).astype(np.int64)
            if Pr == REAL_Pa:
                return fV_rho(x,frho(h),False)
                break
            else:
                x += 0.00001
        return 0
lE_Vmax = np.array([38.492384,38.61024387,38.7291222,38.84835939,38.96863055,39.08861033,39.20964047,39.33106284,39.45355166,39.57578433,39.69910049,39.822179,39.94702254,40.07164663,40.19672505,40.32226714,40.44894506,40.57610484,40.70375608,40.83124688,40.95991061,41.0890952,41.21881063,41.34972693,41.48053399,41.61190247,41.74384277,41.87636539,42.00948092,42.14320008,42.27753368,42.41249264,42.54743227,42.68367562,42.81992305,42.95749575,43.09509641,43.23339127,43.37239201,43.51145856,43.65125588,43.79179598,43.93309103,44.07450386,44.21669751,44.3596845,44.50282996,44.64679543,44.79094766,44.93594711,45.08116222,45.22725232,45.37423075,45.5208261,45.66898154,45.81678581,45.96553768,46.11461185,46.26402491,46.41443115,46.56520825,46.71637329,46.86794353,47.01993643,47.17236964,47.32589302,47.47925951,47.63312029,47.78749374,47.94239839,48.09722601,48.25325045,48.409238,48.56521125,48.72181557,48.8790706,49.03637605,49.19375553,49.35185058,49.51006519,49.66842376,49.82695093,49.98628398,50.14522199,50.30440412,50.46385604,50.62299703,50.78246288,50.94228003,51.10187314,51.26127385,51.42111295,51.58022366,51.73983448,51.89878605,52.05770884,52.21604659,52.37442699,52.53229823,52.68970079,52.84609342,53.00268381,53.15835156,53.31314522,53.46711397,53.62087882,53.77277617,53.92400405,54.07404965,54.22297214,54.3702723])
#lM_Vmax = np.array([38.492384,38.61024387,38.7291222,38.84835939,38.96863055,39.08861033,39.20964047,39.33106284,39.45355166,39.57578433,39.69910049,39.822179,39.94702254,40.07164663,40.19672505,40.32226714,40.44894506,40.57610484,40.70375608,40.83124688,40.95991061,41.0890952,41.21881063,41.34972693,41.48053399,41.61190247,41.74384277,41.87636539,42.00948092,42.14320008,42.27753368,42.41249264,42.54743227,42.68367562,42.81992305,42.95749575,43.09509641,43.23339127,43.37239201,43.51145856,43.65125588,43.79179598,43.93309103,44.07450386,44.21669751,44.3596845,44.50282996,44.64679543,44.79094766,44.93594711,45.08116222,45.22725232,45.37423075,45.5208261,45.66898154,45.81678581,45.96553768,46.11461185,46.26402491,46.41443115,46.56520825,46.71637329,46.86794353,47.01993643,47.17236964,47.32589302,47.47925951,47.63312029,47.78749374,47.94239839,48.09722601,48.25325045,48.409238,48.56521125,48.72181557,48.8790706,49.03637605,49.19375553,49.35185058,49.51006519,49.66842376,49.82695093,49.98628398,50.14522199,50.30440412,50.46385604,50.62299703,50.78246288,50.94228003,51.10187314,51.26127385,51.42111295,51.58022366,51.73983448,51.89878605,52.05770884,52.21604659,52.37442699,52.53229823,52.68970079,52.84609342,53.00268381,53.15835156,53.31314522,53.46711397,53.62087882,53.77277617,53.92400405,54.07404965,54.22297214,54.3702723,0.])
"""
lE_Vmax = np.array([])
lM_Vmax = np.array([])
h_L = np.linspace(0,11000,100)
x = 0
while x < 11001:
    global lE_Vmax
    global lM_Vmax
    lE_Vmax = np.append(lE_Vmax,fVmax(x, False))
    lM_Vmax = np.append(lE_Vmax,fVmax(x, True))
    print x,'m',(float(x) / 11000.0)*100.0, '%'
    x += 100
"""
print'EARTH'
print len(lE_Vmax)
print'\n'
print lE_Vmax
print'======='
print'H'
h_L = np.arange(0,11001,100)
print len(h_L)
print'\n'

def fD(CL,mars):
    if mars == True:
        return fCD(CL)*0.5*frho_mars(0)*fV(CL,mars)**2*S
    else:
        return fCD(CL)*0.5*rho*fV(CL,False)**2*S

V_L = np.array(fV(CLpos_L,False))
D_L = np.array(fD(CLpos_L,False))
"""
CLposV500_L = np.array([])
i = 0
while fV(CLpos_L[i]) < 501:
    global CLposV500_L
    CLposV500_L = np.append(CLposV500_L, CLpos_L[i])
    i += 1
print CLposV500_L
V500_L = np.array(fV(CLposV500_L))
print V500_L
D500_L = np.array(fD(CLposV500_L))
print D500_L
"""
Vext_L = np.array([V_L[0], V_L[-1]])

def PROPCALC():
    print'\n'
    inp_COMM = (raw_input('COMM >> ')).lower().split()
    if len(inp_COMM) > 0:
        if inp_COMM[0] == '?':
            print'ALFA    *ALFA* => CL, CD'
            print'CL      *CL*   => CD, ALFA'
            print'INIT'
            print'OPT     *RANGE (^EARTH^ / ^MARS^) (^ALT^)* / *ENDURANCE (^EARTH^ / ^MARS^) (^ALT^)* / *VELOCITY (^EARTH^ / ^MARS^)*'
            print'PLR     *CL* / *CD*'
            print'PLT     *CL/CD* / *ALFA/CL* / *ALFA/CD* / *V/P* / *V/F*'
        elif inp_COMM[0] == 'init':
            init()
        elif inp_COMM[0] == 'alfa':
            if len(inp_COMM) > 1:
                inp_alfa = float(inp_COMM[1])
                if -2.1 < inp_alfa < 9.6:
                    print'CL:',fCL(inp_alfa)
                    print'CD:',fCD(fCL(inp_alfa))
                else:
                    print'ERR.: ALFA OUT OF LINEAR FLIGHT ENVELOPE'
            else:
                print'ERR.: PROVIDE ALFA VALUE'
        elif inp_COMM[0] == 'cl':
            if len(inp_COMM) > 1:
                inp_CL = float(inp_COMM[1])
                if -0.045 < inp_CL < 1.35:
                    print'CD:',fCD(inp_CL)
                    print'///ALFA ACCURATE BETWEEN -2°~9.5°///'
                    print'ALFA:',fALFA(inp_CL)
                else:
                    print'ERR.: CL OUT OF FLIGHT ENVELOPE'
            else:
                print'ERR.: PROVIDE CL VALUE'
        elif inp_COMM[0] == 'opt':
            if len(inp_COMM) > 1:
                global opt
                global CLopt
                if inp_COMM[1] == 'range':
                    fig, ax1 = plt.subplots()
                    opt = 'cl/cd'
                    mars = False
                    h = 0
                    CLopt = fmin(invf,0)[0]
                    print'\n'
                    print'========='
                    print'MAX RANGE'
                    print'========='
                    print'(CL / CD)max'
                    print'CLopt:',CLopt
                    print'CDopt:',fCD(CLopt)
                    print'\n**ENVIRONMENTAL DETAILS**'
                    if len(inp_COMM) > 2 and inp_COMM[2] == 'mars':
                        mars = True
                        if len(inp_COMM) > 3 and isinstance( float(inp_COMM[3]), ( float, long ) ):
                            h = float(inp_COMM[3])
                            if h == 0:
                                plt.suptitle('MAX. RANGE PERFORMANCE ON MARS (SL CONDITIONS)', ha='center')
                            else:
                                plt.suptitle('MAX. RANGE PERFORMANCE ON MARS (h = {}m)'.format(h), ha='center')
                            rho = frho_mars(h)
                            print'//MARS//'
                            print'h: ',h,' [m]'
                            print'rho: ',rho,' [kg/m^3]'
                        else:
                            plt.suptitle('MAX. RANGE PERFORMANCE ON MARS (SL CONDITIONS)', ha='center')
                            rho = frho_mars(0)
                            print'//MARS//'
                            print'h: SL'
                            print'rho: ',rho,' [kg/m^3]'
                    else:
                        mars = False
                        if len(inp_COMM) > 3 and isinstance( float(inp_COMM[3]), ( float, long ) ) and float(inp_COMM[3]) <= 11000:
                            h = float(inp_COMM[3])
                            if h == 0:
                                plt.suptitle('MAX. RANGE PERFORMANCE ON EARTH (SL CONDITIONS)', ha='center')
                            else:
                                plt.suptitle('MAX. RANGE PERFORMANCE ON EARTH (h = {}m)'.format(h), ha='center')
                            rho = frho(h)
                            print'//EARTH//'
                            print'h: ',h,' [m]'
                            print'rho: ',rho,' [kg/m^3]'
                        else:
                            plt.suptitle('MAX. RANGE PERFORMANCE ON EARTH (SL CONDITIONS)', ha='center')
                            rho = frho(0)
                            print'//EARTH//'
                            print'h: SL'
                            print'rho: ',rho,' [kg/m^3]'
                    print'a: ',fa(h,mars),' [m/s]'
                    print'\n'
                    Vopt = fV_rho(CLopt,rho,mars)
                    print'Vopt:',Vopt,' [m/s]'
                    Mopt = fM(Vopt,h,mars)
                    print'Mopt:',Mopt
                    Pr = (np.sqrt((W**3/S)*(2.0/rho)*(fCD(CLopt)**2/CLopt**3)))
                    print'Pr:',Pr,' [W]'
                    Pa_iter = Pa
                    Pa_iter_L = np.array([Pa])
                    t = 0
                    t_L = np.array([0])
                    s = 0
                    s_L = np.array([0])
                    while Pa_iter >= 0:
                        Pa_iter = Pa_iter - Pr
                        Pa_iter_L = np.append(Pa_iter_L, Pa_iter)
                        t += 1 #sec
                        t_L = np.append(t_L, t)
                        s += fV(CLopt,mars) #meters
                        s_L = np.append(s_L, s)
                    Pa_iter = Pa_iter + Pr
                    t += Pa_iter / Pr
                    s += fV(CLopt,mars) * (Pa_iter / Pr)
                    print'AIRTIME:',t,' [s]'
                    print'MAX RANGE:',s,' [m]'
                    plt.title('Opt. Velocity: {} m/s ; Opt. Mach: {} ; Range: {} m ; Endurance: {} s'.format(Vopt, Mopt, s, t), fontsize=9, ha='center')
                    ax1.plot(t_L, Pa_iter_L, 'b', label='Power Available')
                    ax1.set_xlabel('t [s]')
                    # Make the y-axis label, ticks and tick labels match the line color.
                    ax1.set_ylabel('Pa [W]', color='b')
                    ax1.tick_params('y', colors='b')
        
                    ax2 = ax1.twinx()
                    ax2.plot(t_L, s_L, 'r', label='Distance')
                    ax2.set_ylabel('s [m]', color='r')
                    ax2.tick_params('y', colors='r')
                    ax2.legend(loc=1, shadow=True)
                    ax1.legend(loc=2, shadow=True)
                    fig.tight_layout()
                    plt.show()
                elif inp_COMM[1] == 'cl/cd2':
                    opt = 'cl/cd2'
                    CLopt = fmin(invf,0)[0]
                    print'CLopt:',CLopt
                    print'Vopt:',fV(CLopt,mars),' [m/s]'
                elif inp_COMM[1] == 'endurance':
                    fig, ax1 = plt.subplots()
                    opt = 'cl3/cd2'
                    mars = False
                    h = 0
                    CLopt = fmin(invf,0)[0]
                    print'\n'
                    print'============='
                    print'MAX ENDURANCE'
                    print'============='
                    print'(CL^3 / CD^2)max'
                    print'CLopt:',CLopt
                    print'CDopt:',fCD(CLopt)
                    print'\n**ENVIRONMENTAL DETAILS**'
                    if len(inp_COMM) > 2 and inp_COMM[2] == 'mars':
                        mars = True
                        if len(inp_COMM) > 3 and isinstance( float(inp_COMM[3]), ( float, long ) ):
                            h = float(inp_COMM[3])
                            if h == 0:
                                plt.suptitle('MAX. ENDURANCE PERFORMANCE ON MARS (SL CONDITIONS)', ha='center')
                            else:
                                plt.suptitle('MAX. ENDURANCE PERFORMANCE ON MARS (h = {}m)'.format(h), ha='center')
                            rho = frho_mars(h)
                            print'//MARS//'
                            print'h: ',h,' [m]'
                            print'rho: ',rho,' [kg/m^3]'
                        else:
                            plt.suptitle('MAX. ENDURANCE PERFORMANCE ON MARS (SL CONDITIONS)', ha='center')
                            rho = frho_mars(0)
                            print'//MARS//'
                            print'h: SL'
                            print'rho: ',rho,' [kg/m^3]'
                    else:
                        mars = False
                        if len(inp_COMM) > 3 and isinstance( float(inp_COMM[3]), ( float, long ) ) and float(inp_COMM[3]) <= 11000:
                            h = float(inp_COMM[3])
                            if h == 0:
                                plt.suptitle('MAX. ENDURANCE PERFORMANCE ON EARTH (SL CONDITIONS)', ha='center')
                            else:
                                plt.suptitle('MAX. ENDURANCE PERFORMANCE ON EARTH (h = {}m)'.format(h), ha='center')
                            rho = frho(h)
                            print'//EARTH//'
                            print'h: ',h,' [m]'
                            print'rho: ',rho,' [kg/m^3]'
                        else:
                            plt.suptitle('MAX. ENDURANCE PERFORMANCE ON EARTH (SL CONDITIONS)', ha='center')
                            rho = frho(0)
                            print'//MARS//'
                            print'h: SL'
                            print'rho: ',rho,' [kg/m^3]'
                    print'a: ',fa(h,mars),' [m/s]'
                    print'\n'
                    Vopt = fV_rho(CLopt,rho,mars)
                    print'Vopt:',Vopt,' [m/s]'
                    Mopt = fM(Vopt,h,mars)
                    print'Mopt:',Mopt
                    Pr = (np.sqrt((W**3/S)*(2.0/rho)*(fCD(CLopt)**2/CLopt**3)))
                    print'Pr:',Pr,' [W]'
                    Pa_iter = Pa
                    Pa_iter_L = np.array([Pa])
                    t = 0
                    t_L = np.array([0])
                    s = 0
                    s_L = np.array([0])
                    while Pa_iter >= 0:
                        Pa_iter = Pa_iter - Pr
                        Pa_iter_L = np.append(Pa_iter_L, Pa_iter)
                        t += 1 #sec
                        t_L = np.append(t_L, t)
                        s += fV(CLopt,mars) #meters
                        s_L = np.append(s_L, s)
                    Pa_iter = Pa_iter + Pr
                    t += Pa_iter / Pr
                    s += fV(CLopt,mars) * (Pa_iter / Pr)
                    print'MAX AIRTIME:',t,' [s]'
                    print'RANGE:',s,' [m]'
                    plt.title('Opt. Velocity: {} m/s ; Opt. Mach: {} ; Range: {} m ; Endurance: {} s'.format(Vopt, Mopt, s, t), fontsize=9, ha='center')
                    ax1.plot(t_L, Pa_iter_L, 'b', label='Power Available')
                    ax1.set_xlabel('t [s]')
                    # Make the y-axis label, ticks and tick labels match the line color.
                    ax1.set_ylabel('Pa [W]', color='b')
                    ax1.tick_params('y', colors='b')
        
                    ax2 = ax1.twinx()
                    ax2.plot(t_L, s_L, 'r', label='Distance')
                    ax2.set_ylabel('s [m]', color='r')
                    ax2.tick_params('y', colors='r')
                    ax2.legend(loc=1, shadow=True)
                    ax1.legend(loc=2, shadow=True)
                    fig.tight_layout()
                    plt.show()
                elif inp_COMM[1] == 'velocity':
                    if len(inp_COMM) > 2:
                        if inp_COMM[2] == 'earth':
                            mars = False
                            h_L = np.arange(0,11001,100)
                            plt.plot(h_L, fV_rho(CLmax, frho(h_L), mars), 'g', label='Min Velocity')
                            plt.plot(h_L, lE_Vmax, 'm', label='Max Velocity')
                            #plt.plot(h_L, fV_rho(CLmin, frho(h_L), mars), 'm', label='Max Velocity')
                            opt = 'cl/cd'
                            CLopt_R = fmin(invf,0)[0]
                            #plt.plot(h_L, fV_rho(CLopt_R, frho(h_L), mars), 'b', label='Max Range')
                            opt = 'cl3/cd2'
                            CLopt_E = fmin(invf,0)[0]
                            #plt.plot(h_L, fV_rho(CLopt_E, frho(h_L), mars), 'r', label='Max Endurance')
                            plt.xlabel('h [m]')
                            plt.ylabel('V [m/s]')
                            legend = plt.legend(loc='lower right', shadow=True)
                            plt.title('VELOCITY VS ALTITUDE ON EARTH', ha='center')
                            plt.grid(True)
                            plt.show()
                        elif inp_COMM[2] == 'mars':
                            mars = True
                            h_L = np.linspace(0,11000,110)
                            plt.plot(h_L, fV_rho(CLmax, frho_mars(h_L), mars), 'g', label='Min Velocity')
                            plt.plot(h_L, lM_Vmax, 'm', label='Max Velocity')
                            #plt.plot(h_L, fV_rho(CLmin, frho_mars(h_L), mars), 'm', label='Max Velocity')
                            opt = 'cl/cd'
                            CLopt_R = fmin(invf,0)[0]
                            #plt.plot(h_L, fV_rho(CLopt_R, frho_mars(h_L), mars), 'b', label='Max Range')
                            opt = 'cl3/cd2'
                            CLopt_E = fmin(invf,0)[0]
                            #plt.plot(h_L, fV_rho(CLopt_E, frho_mars(h_L), mars), 'r', label='Max Endurance')
                            plt.xlabel('h [m]')
                            plt.ylabel('V [m/s]')
                            legend = plt.legend(loc='lower right', shadow=True)
                            plt.title('VELOCITY VS ALTITUDE ON MARS', ha='center')
                            plt.grid(True)
                            plt.show()
                        else:
                            print'ERR.: MARS OR EARTH CONDITIONS'
                    else:
                        print'ERR.: MARS OR EARTH CONDITIONS'
                else:
                    print'ERR.: INVALID ARGUMENT'
            else:
                print'ERR.: PROVIDE ARGUMENT'
        elif inp_COMM[0] == 'plr':
            if len(inp_COMM) > 1:
                if inp_COMM[1] == 'cl':
                    print'CL = {} + {}*α'.format(CL0, a)
                elif inp_COMM[1] == 'cd':
                    print'CD = {} + {}*CL + {}*CL^2'.format(CD0, k1, k2)
                else:
                    print'ERR.: INVALID ARGUMENT'
            else:
                print'ERR.: PROVIDE ARGUMENT'
        elif inp_COMM[0] == 'plt':
            if len(inp_COMM) > 1:
                if inp_COMM[1] == 'cl/cd':
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    #ax.annotate('{} + {}*CL + {}*CL^2'.format(CD0, k1, k2), xy=(0.8, 0.08), xytext=(.8, 0.08))
                    plt.plot(CL, CD, 'k+', label='Raw Data')
                    plt.plot(CL_L, (CD0 + k1*CL_L + k2*CL_L**2), 'b', label='Interpolation')
                    plt.suptitle('CL/CD POLAR T1-1', ha='center')
                    plt.title('CD = {} + {}*CL + {}*CL^2'.format(CD0, k1, k2), fontsize=8, ha='center')
                    plt.xlabel('CL')
                    plt.ylabel('CD')
                    legend = plt.legend(loc='lower right', shadow=True)
                    plt.grid(True)
                    plt.show()
                elif inp_COMM[1] == 'alfa/cl':
                    plt.plot(alfa, CL, 'k+', label='Raw Data')
                    plt.plot(alfa, (CL0 + a*alfa), 'b', label='Interpolation')
                    plt.suptitle('α/CL POLAR T1-1', ha='center')
                    plt.title('CL = {} + {}*α'.format(CL0, a), fontsize=8, ha='center')
                    plt.xlabel('α [°]')
                    plt.ylabel('CL')
                    legend = plt.legend(loc='lower right', shadow=True)
                    plt.grid(True)
                    plt.show()
                elif inp_COMM[1] == 'alfa/cd':
                    plt.plot(alfa, CD, 'k+', label='Raw Data')
                    plt.suptitle('α/CD POLAR T1-1', ha='center')
                    plt.xlabel('α [°]')
                    plt.ylabel('CD')
                    legend = plt.legend(loc='lower right', shadow=True)
                    plt.grid(True)
                    plt.show()
                elif inp_COMM[1] == 'v/p':
                    plt.plot(V_L, D_L*V_L/(10**3), 'b', label='Power Required')
                    #plt.plot(Vext_L, Pa*(Vext_L / Vext_L)/(10**3), 'r', label='Power Available')
                    
                    plt.suptitle('V/P POLAR T1-1', ha='center')
                    plt.xlabel('V [m/s]')
                    plt.ylabel('P [kW]')
                    legend = plt.legend(loc='lower right', shadow=True)
                    plt.grid(True)
                    plt.show()
                elif inp_COMM[1] == 'v/f':
                    plt.plot(V_L, D_L, 'b', label='Drag')
                    #plt.plot(V_L, Pa/V_L, 'r', label='Thrust')
                    
                    plt.suptitle('V/F POLAR T1-1', ha='center')
                    plt.xlabel('V [m/s]')
                    plt.ylabel('F [N]')
                    legend = plt.legend(loc='lower right', shadow=True)
                    plt.grid(True)
                    plt.show()
                else:
                    print'ERR.: INVALID ARGUMENT'
            else:
                print'ERR.: PROVIDE ARGUMENT'
        else:
            print'ERR.: PROVIDE COMMAND (? FOR HELP)'
    else:
        print'ERR.: PROVIDE COMMAND (? FOR HELP)'
    PROPCALC()

PROPCALC()
