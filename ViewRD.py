from numpy import absolute, append, arctan, array, asarray
from numpy import diag, dot, exp
from numpy import float64, iscomplexobj, linspace, linalg, log
from numpy import pi, shape, sin, tan, vstack, zeros
from numpy import apply_along_axis
from numpy.linalg import eig, inv, norm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TkAgg') # Set Tk backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

from warnings import simplefilter
# Prevents divide by zero warnings from printing to user.
simplefilter("ignore") 

# Import tkinter for python 2 or 3; depedning on user system.
from sys import version_info
if version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

# Setting Some Inital Values
Field = 600 # B0 in Mhz.
lmf = lmf0 = 150.870 # 13C larmor freq in 600 Mhz
pB0, pC0 = .01, 0 # pB and pC populations as probabilities
dwB0, dwC0 = 3, 0 # Chemical shifts in ppmchr
kexAB0, kexAC0, kexBC0 = 1000, 0, 0 # Exchange rates
R1a0 = R1b0 = R1c0 = 2.5 # R1 values
R2a0 = R2b0 = R2c0 = 20 # R2 values

# Functions for the calculation of the Bloch equations.
def ExpDecay(x,a,b):
    return a*exp(-b*x)

def isnumber(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def MatrixBM3(k12,k21,k13,k31,k23,k32,delta1,delta2,delta3,
              w1, R1, R2, R1b, R1c, R2b, R2c):

    # Exchange Matrix (exchange rates)
    #  See Trott & Palmer JMR 2004, n-site chemical exchange
    K = array([[-k12 -k13, k21, k31, 0., 0., 0., 0., 0., 0.],
               [k12, -k21 - k23, k32, 0., 0., 0., 0., 0., 0.],
               [k13, k23, -k31 - k32, 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., -k12 -k13, k21, k31, 0., 0., 0.],
               [0., 0., 0., k12, -k21 - k23, k32, 0., 0., 0.],
               [0., 0., 0., k13, k23, -k31 - k32, 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., -k12 -k13, k21, k31],
               [0., 0., 0., 0., 0., 0., k12, -k21 - k23, k32],
               [0., 0., 0., 0., 0., 0., k13, k23, -k31 - k32]], float64)
    
    # Delta matrix (offset and population)
    Der = array([[0., 0., 0., -delta1, 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., -delta2, 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., -delta3, 0., 0., 0.],
                 [delta1, 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., delta2, 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., delta3, 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.]], float64)

    # Spinlock power matrix (SLP/w1)
    #  Matrix is converted to rad/s here
    W = array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0., -w1, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., -w1, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., -w1],
               [0., 0., 0., w1, 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., w1, 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., w1, 0., 0., 0.]], float64)

    # Intrinsic rate constant matrix (R1 and R2)
    R = array([[-R2, 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., -R2b, 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., -R2c, 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., -R2, 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., -R2b, 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., -R2c, 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0., -R1, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., -R1b, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., -R1c]], float64) 

    return(K + Der + W + R)

def matrix_exponential(A, w1, wrf, t, EigVal=False):
    W, V = eig(A)
    W_orig = asarray(list(W))
    if iscomplexobj(W):
        W = W.real
    eA = dot(dot(V, diag(exp(W))), inv(V))
    if EigVal == False:
        return eA.real
    else:
        return eA.real, W

def SimMagVecs(dt,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf):
    # Sim mag at time increment
    M = dot((matrix_exponential(Ms*dt, w1, wrf, dt)), M0)
    Mxa = M[0]
    Mxb = M[1]
    Mxc = M[2]
    Mya = M[3]
    Myb = M[4]
    Myc = M[5]
    Mza = M[6]
    Mzb = M[7]
    Mzc = M[8]
    # Project effective mag along indv effective lOmegas
    PeffA = dot(vstack((Mxa,Mya,Mza)).T, lOmegaA)[0]
    PeffB = dot(vstack((Mxb,Myb,Mzb)).T, lOmegaB)[0]
    PeffC = dot(vstack((Mxc,Myc,Mzc)).T, lOmegaC)[0]
    # Project mag along average effective
    Peff = PeffA + PeffB + PeffC
    return Peff

def AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC):
    if pB > pC:
        exchReg = kexAB / absolute(dwB)
    else:
        exchReg = kexAC / absolute(dwC) 
    
    if exchReg <= 1.:

        uOmega1 = 0.0
        uOmega2 = dwB
        uOmega3 = dwC
        uOmegaAvg = uOmega1

        delta1 = (uOmega1 - wrf) # rad/s
        delta2 = (uOmega2 - wrf) # rad/s
        delta3 = (uOmega3 - wrf) # rad/s
        deltaAvg = (uOmega1 - wrf) # rad/s, avg delta is GS - carrier
        thetaAvg = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*2pi))
        theta1 = theta2 = theta3 = thetaAvg    

        ## GS,ES1,ES2 along average state

        tempState = asarray([w1,0.,deltaAvg], float64)
        # Normalize vector
        lOmegaA = tempState / linalg.norm(tempState)
        lOmegaB = lOmegaC = lOmegaA

        # If exchange regime is non-slow (>0.5), the align along average
    else:
            
        uOmega1 = -(pB*dwB + pC*dwC) / ((pA + pB + pC)) # (rad/sec)
        uOmega2 = uOmega1 + dwB # (rad/sec)
        uOmega3 = uOmega1 + dwC # (rad/sec)
        uOmegaAvg = pA*uOmega1 + pB*uOmega2 + pC*uOmega3 #Average Resonance Offset (rad/sec)

        delta1 = (uOmega1 - wrf) # rad/s
        delta2 = (uOmega2 - wrf) # rad/s
        delta3 = (uOmega3 - wrf) # rad/s
        deltaAvg = (uOmegaAvg - wrf) # rad/s

        theta1 = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*2pi))
        theta2 = theta3 = thetaAvg = theta1

        tempState = asarray([w1,0.,deltaAvg], float64)
            
        lOmegaA = tempState / linalg.norm(tempState)
        lOmegaB = lOmegaA
        lOmegaC = lOmegaA

    return (lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,
            delta1, delta2, delta3, deltaAvg,
            theta1, theta2, theta3, thetaAvg)

def CalcR2eff(wrf, w1, lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, time, Flag = "OffRes"):
    # Function is applyed to an array of offset values in the data function, to calcuate an R2eff value. 
    if Flag == 'OnRes': # Hacky way to switch the passed input variable from data functions ApplyAlongAxis call for the OnRes profile.
        w1 = wrf
        wrf = 0
    pA = 1 - (pB + pC)
    dwB = dwB * lmf * 2 * pi * -1
    dwC = dwC * lmf * 2 * pi * -1
    k12 = kexAB * pB / (pB + pA)
    k21 = kexAB * pA / (pB + pA)
    k13 = kexAC * pC / (pC + pA)
    k31 = kexAC * pA / (pC + pA)
    if kexBC != 0.:
        k23 = kexBC * pC / (pB + pC)
        k32 = kexBC * pB / (pB + pC)
    else:
        k23 = 0.
        k32 = 0.

    lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,\
        delta1, delta2, delta3, deltaAvg, theta1, theta2, theta3, thetaAvg = \
                            AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC)
    Ma = pA*lOmegaA 
    Mb = pB*lOmegaB 
    Mc = pC*lOmegaC 
    # Magnetization matrix
    Ms = MatrixBM3(k12,k21,k13,k31,k23,k32,delta1,delta2,delta3,
               w1, R1a, R2a, R1b, R1c, R2b, R2c)
    # Initial magnetization
    M0 = array([Ma[0],Mb[0],Mc[0],Ma[1],Mb[1],Mc[1],Ma[2],Mb[2],Mc[2]], float64)
    magVecs = asarray([SimMagVecs(x,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf) for x in time])
    
    # Fit decay for R1rho
    popt, pcov = curve_fit(ExpDecay, time, magVecs, (1., R1a))
    R1p = popt[1]
    R1p_err = 0.0
    preExp = popt[0]

    R2eff = (R1p/sin(thetaAvg)**2.) - (R1a/(tan(thetaAvg)**2.))
    return R2eff                   

def data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, offset, w1, Flag = 'OffRes'):
    # Takes in exchange parameters and calcuates R2eff values. 
    time = linspace(0, 0.2, 3) #Points are sacrificed for speed; may cause issues; should be okay since err = 0 
    if Flag == 'OffRes':   
        w1 = w1 * 2 * pi
        offset2pi = vstack(array(offset * 2 * pi))
        return apply_along_axis(CalcR2eff, 1, offset2pi, w1, lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, time, Flag = 'OffRes')
    elif Flag == 'OnRes':
        offset2pi = 0
        w1 = vstack(array(w1 * 2 * pi))
        return apply_along_axis(CalcR2eff, 1, w1, offset2pi, lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, time, Flag = 'OnRes')
# Function for setting shared plot sliders
def initializeSliders():
    # Initialize sliders
    axcolor = 'lightgrey'
    ax_pB = plt.axes([0.45, 0.15, 0.2, 0.015], facecolor=axcolor)
    ax_pC = plt.axes([0.72, 0.15, 0.2, 0.015], facecolor=axcolor)
    ax_dwB = plt.axes([0.45, 0.13, 0.2, 0.015], facecolor=axcolor)
    ax_dwC = plt.axes([0.72, 0.13, 0.2, 0.015], facecolor=axcolor)
    ax_R2a = plt.axes([0.45, 0.11, 0.11, 0.015], facecolor = axcolor)
    ax_R2b = plt.axes([0.625, 0.11, 0.11, 0.015], facecolor = axcolor)
    ax_R2c = plt.axes([0.80, 0.11, 0.11, 0.015], facecolor = axcolor)
    ax_kexAB = plt.axes([0.45, 0.07, 0.4, 0.015], facecolor=axcolor)
    ax_kexAC = plt.axes([0.45, 0.05, 0.4, 0.015], facecolor=axcolor)
    ax_kexBC = plt.axes([0.45, 0.03, 0.4, 0.015], facecolor=axcolor)

    # Set slider ID and values
    slider_pB = Slider(ax_pB, 'p$_B$', 0, .2, valfmt = '%.3f', valinit = pB0)
    slider_pC = Slider(ax_pC, 'p$_C$', 0, .2, valfmt = '%.3f', valinit = pC0)
    slider_dwB = Slider(ax_dwB, '$\Delta$$\omega$$_B$', -10, 10, valinit = dwB0)
    slider_dwC = Slider(ax_dwC, '$\Delta$$\omega$$_C$', -10, 10, valinit = dwC0)
    slider_R2a = Slider(ax_R2a, 'R$_{2a}$', 0, 50, valinit = R2a0)
    slider_R2b = Slider(ax_R2b, 'R$_{2b}$', 0, 50, valinit = R2a0)
    slider_R2c = Slider(ax_R2c, 'R$_{2c}$', 0, 50, valinit = R2a0)
    slider_kexAB = Slider(ax_kexAB, 'k$_{ex}$AB (s$^{-1}$)', 0, 15000, valinit = kexAB0)
    slider_kexAC = Slider(ax_kexAC, 'k$_{ex}$AC (s$^{-1}$)', 0, 15000, valinit = kexAC0)
    slider_kexBC = Slider(ax_kexBC, 'k$_{ex}$BC (s$^{-1}$)', 0, 15000, valinit = kexBC0)
    
    return slider_pB, slider_pC, slider_dwB,\
        slider_dwC, slider_R2a, slider_R2b, slider_R2c,\
        slider_kexAB, slider_kexAC, slider_kexBC

def get_lmf(label):
    if Field == 600:
        lmfdict = {'Carbon':150.870, 'Nitrogen':60.821}
    if Field == 700:
        lmfdict = {'Carbon':176.015, 'Nitrogen':70.957}
    if Field == 800:
        lmfdict = {'Carbon':201.160, 'Nitrogen':81.094}
    if Field == 1100:
        lmfdict = {'Carbon':276.595, 'Nitrogen':111.505}
    global lmf
    lmf = lmfdict[label]

def init_custom(event):
    def custom_update():
        if isnumber(pB_u.get()) == True and float(pB_u.get()) < 1 and float(pB_u.get()) >= 0:
            slider_pB.set_val(float(pB_u.get()))
        if isnumber(pC_u.get()) == True and float(pC_u.get()) < 1 and float(pC_u.get()) >= 0:
            slider_pC.set_val(float(pC_u.get()))
        if isnumber(dwB_u.get()) == True and float(dwB_u.get()) < 80 and float(dwB_u.get()) > -80:
            slider_dwB.set_val(float(dwB_u.get()))
        if isnumber(dwC_u.get()) == True and float(dwC_u.get()) < 80 and float(dwC_u.get()) > -80:
            slider_dwC.set_val(float(dwC_u.get()))
        if isnumber(R2b_u.get()) == True and float(R2b_u.get()) >= 0:
            slider_R2b.set_val(float(R2b_u.get()))
        if isnumber(R2c_u.get()) == True and float(R2c_u.get()) >= 0:
            slider_R2c.set_val(float(R2c_u.get()))
        if isnumber(kexAB_u.get()) == True and float(kexAB_u.get()) >= 0:
            slider_kexAB.set_val(float(kexAB_u.get()))
        if isnumber(kexAC_u.get()) == True and float(kexAC_u.get()) >= 0:
            slider_kexAC.set_val(float(kexAC_u.get()))
        if isnumber(kexBC_u.get()) == True and float(kexBC_u.get()) >= 0:    
            slider_kexBC.set_val(float(kexBC_u.get()))
        if isnumber(R2a_u.get()) == True and float(R2a_u.get()) >= 0:    
            R2a = (float(R2a_u.get()))

    global Variables
    Variables = Tk.Tk()
    Variables.wm_title('User Variable Entry')
    Tk.Label(Variables, text="pB").grid(row=0, column = 0)
    Tk.Label(Variables, text="pC").grid(row=0, column = 2)
    Tk.Label(Variables, text="dwB").grid(row=1, column = 0)
    Tk.Label(Variables, text="dwC").grid(row=1, column = 2)
    Tk.Label(Variables, text="R2a").grid(row=4, column = 0)
    Tk.Label(Variables, text="R2b").grid(row=2, column = 0)
    Tk.Label(Variables, text="R2c").grid(row=2, column = 2)

    Tk.Label(Variables, text="kexAB").grid(row=3, column = 0)
    Tk.Label(Variables, text="kexAC").grid(row=3, column = 2)
    Tk.Label(Variables, text="kexBC").grid(row=4, column = 0)

    pB_u = Tk.Entry(Variables)
    pC_u = Tk.Entry(Variables)
    dwB_u = Tk.Entry(Variables)
    dwC_u = Tk.Entry(Variables)
    R2a_u = Tk.Entry(Variables)
    R2b_u = Tk.Entry(Variables)
    R2c_u = Tk.Entry(Variables)
    kexAB_u = Tk.Entry(Variables)
    kexAC_u = Tk.Entry(Variables)
    kexBC_u = Tk.Entry(Variables)

    pB_u.grid(row = 0, column = 1)
    pC_u.grid(row = 0, column = 3)
    dwB_u.grid(row = 1, column = 1)
    dwC_u.grid(row = 1, column = 3)
    R2b_u.grid(row = 2, column = 1)
    R2c_u.grid(row = 2, column = 3)
    kexAB_u.grid(row = 3, column = 1)
    kexAC_u.grid(row = 3, column = 3)
    kexBC_u.grid(row = 4, column = 1)
    R2a_u.grid(row = 4, column = 3)

    Tk.Button(Variables, text='Update', command=custom_update).grid(row=5, column=1, pady=4)

def redraw(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c):
    if slps.get_status()[0] == True:
        l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, offset1, 100))
    if slps.get_status()[1] == True:
        l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, offset2, 500))
    if slps.get_status()[2] == True: 
        l3.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, offset3, 1000))
    if slps.get_status()[3] == True:
        l4.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, offset4, 2000))
    lo1.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, 0, slp, Flag = 'OnRes'))
    fig.canvas.draw_idle()

#fig, ax = plt.subplots(figsize = (14, 7))
fig = plt.figure(figsize = (12, 7))
ax = fig.add_subplot(121)
plt.subplots_adjust(bottom=0.25)
ax.set_xlabel(r'$\Omega\,2\pi^{-1}\,{(Hz)}$', fontsize=16)
ax.set_ylabel(r'$R_{2}+R_{ex}\,(s^{-1})$', fontsize=16)
ax.set_title('Off-Res RD Plot', fontsize = 20)
plt.axis([-3000, 3000, 10, 60])
axcolor = 'lightgrey'
    
# Initial plotted data
offset1 = linspace(-350, 350, 24)
offset2 = linspace(-1750, 1750, 24)
offset3 = linspace(-3500, 3500, 24)
offset4 = linspace(-7000, 7000, 24)
l, = plt.plot(offset1, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset1, 100), lw = 0, marker = 'o', color = 'C0')
l2, = plt.plot(offset2, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset2, 500), lw = 0, marker = 'o', color = 'C2')
l3, = plt.plot(offset3, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset3, 1000), lw = 0, marker = 'o', color = 'C1')
l4, = plt.plot(offset4, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset4, 2000), lw = 0, marker = 'o', color = 'C3')

ax2 = fig.add_subplot(122)
#plt.subplots_adjust(left=0.25, bottom=0.25)
ax2.set_xlabel(r'$\omega$$_1$(Hz)', fontsize=16)
ax2.set_ylabel(r'$R_{2}+R_{ex}\,(s^{-1})$', fontsize=16)
ax2.set_title('On-Res RD Plot ($\Omega$=0 Hz)', fontsize = 20)
plt.axis([0, 3000, 10, 60])

slp = linspace(50, 3000, 25)
lo1, = ax2.plot(slp, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, 0, slp, Flag = 'OnRes'), lw = 0, marker = 'o', color = 'C0')

    ## Update the y-data values when sliders are changed
    # Set Sliders
slider_pB, slider_pC, slider_dwB, \
    slider_dwC, slider_R2a, slider_R2b, slider_R2c, \
    slider_kexAB, slider_kexAC, slider_kexBC = initializeSliders()
    
    # Function to update y-data when slider changed
def update(val):
    pB = slider_pB.val 
    pC = slider_pC.val
    dwB = slider_dwB.val 
    dwC = slider_dwC.val 
    R2a = slider_R2a.val
    R2b = slider_R2b.val
    R2c = slider_R2c.val
    kexAB = slider_kexAB.val 
    kexAC = slider_kexAC.val 
    kexBC = slider_kexBC.val
    redraw(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a, R2b, R2c) 

# Update profile when a slider is changed
slider_pB.on_changed(update)
slider_pC.on_changed(update)
slider_dwB.on_changed(update)
slider_dwC.on_changed(update)
slider_R2a.on_changed(update)
slider_R2b.on_changed(update)
slider_R2c.on_changed(update)
slider_kexAB.on_changed(update)
slider_kexAC.on_changed(update)
slider_kexBC.on_changed(update)

## CheckButtons for turning SLP on/off
# Make the Check Button axes
lines = [l, l2, l3, l4]
vis_ax = plt.axes([0.01, 0.01, 0.1, 0.15])
labels = (['100 Hz', '500 Hz', '1000 Hz', '2000 Hz'])
visibility = [line.get_visible() for line in lines]
slps = CheckButtons(vis_ax, labels, visibility)
# Function when check buttons changed
def show_slps(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    pB = slider_pB.val 
    pC = slider_pC.val
    dwB = slider_dwB.val 
    dwC = slider_dwC.val 
    R2b = slider_R2b.val
    R2c = slider_R2c.val
    kexAB = slider_kexAB.val 
    kexAC = slider_kexAC.val 
    kexBC = slider_kexBC.val
    redraw(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a, R2b, R2c, offset)

# Call function on click
slps.on_clicked(show_slps)

## RadioButtons to switch atom type
# Make the RadioButton axes
type_ax = plt.axes([0.11, 0.11, 0.1, 0.08])
atomtypeButton = RadioButtons(type_ax, ('Carbon', 'Nitrogen'))
# Function
def changelmf(label):
    get_lmf(label)
    pB = slider_pB.val 
    pC = slider_pC.val
    dwB = slider_dwB.val 
    dwC = slider_dwC.val 
    R2b = slider_R2b.val
    R2c = slider_R2c.val
    kexAB = slider_kexAB.val 
    kexAC = slider_kexAC.val 
    kexBC = slider_kexBC.val
    redraw(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a, R2b, R2c, offset)

# Execute on click
atomtypeButton.on_clicked(changelmf)

## RadioButtons to switch B0
# Makes axes
Fieldax = plt.axes([0.11, 0.01, 0.1, 0.1])
FieldRadio = RadioButtons(Fieldax, ('600 MHz' ,'700 MHz', '800MHz', '1.1GHz'))
# Function
def changeField(label):
    FieldDict = {'600 MHz':600, '700 MHz':700, '800MHz':800, '1.1GHz':1100}
    global Field
    Field = FieldDict[label]
    # Once the field is changes, update the lmf
    changelmf(atomtypeButton.value_selected)
# Execute on click
FieldRadio.on_clicked(changeField)

## Slider to adjust x-axis
# Make
ax_axis = plt.axes([0.45, 0.01, 0.4, 0.015], facecolor=axcolor)
slider_axis = Slider(ax_axis, 'x-axis lim', 15, 1000, valinit = 40)
# Function
def update_axis(val):
    ax.axis([-3000, 3000, 10, val])
    ax2.axis([0, 3000, 10, val])
# Call on changed
slider_axis.on_changed(update_axis)

## Reset Button
# Make
resetax = plt.axes([0.01, 0.16, 0.1, 0.03])
button = Button(resetax, 'Reset')
# Function
def reset(event):
    slider_pB.reset()
    slider_pC.reset()
    slider_dwB.reset()
    slider_dwC.reset()
    slider_R2a.reset()
    slider_R2b.reset()
    slider_R2c.reset()
    slider_kexAB.reset()
    slider_kexAC.reset()
    slider_kexBC.reset()
    slider_axis.reset()
    redraw(lmf, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0)

# Call
button.on_clicked(reset)

## Button to start custom entry GUI
customax = plt.axes([0.22, 0.06, 0.15, 0.055])
GuiButton = Button(customax, 'Define Custom\nSlider Values')
# Call
GuiButton.on_clicked(init_custom)

# All set now show it
plt.show()
