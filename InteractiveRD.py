from numpy import absolute, append, arctan, array, asarray
from numpy import diag, dot, exp
from numpy import float64, iscomplexobj, linspace, linalg, log
from numpy import pi, shape, sin, tan, vstack, zeros
from numpy import apply_along_axis
from numpy.linalg import eig, inv, norm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

# Setting Some Inital Values
Field = 600
lmf = lmf0 = 150.870
pB0, pC0 = .01, 0
dwB0, dwC0 = 3, 0
kexAB0, kexAC0, kexBC0 = 1000, 0, 0
R1a0 = R1b0 = R1c0 = 2.5
R2a0 = R2b0 = R2c0 = 20

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

        thetaAvg = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*))
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
                                                           # Given as uOmega-bar = sum(i=1, N)[ p_i + uOmega_i)
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

def CalcR2eff(wrf, w1, lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, time):
    # Function is applyed to an array of offset values in the data function, to calcuate an R2eff value. 
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


def data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, offset, w1):
    # Takes in exchange parameters and calcuates R2eff values. 
    time = linspace(0, 0.2, 3) #Points are sacrificed for speed; may cause issues; should be okay since err = 0 
    offset2pi = array(offset * 2 * pi)
    offset2pi = vstack(offset2pi)
    w1 = w1 * 2 * pi
    return apply_along_axis(CalcR2eff, 1, offset2pi, w1, lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, time)

# Function for setting shared plot sliders
def initializeSliders():
    # Initialize sliders
    axcolor = 'lightgrey'
    ax_pB = plt.axes([0.18, 0.15, 0.3, 0.015], facecolor=axcolor)
    ax_pC = plt.axes([0.62, 0.15, 0.3, 0.015], facecolor=axcolor)
    ax_dwB = plt.axes([0.18, 0.13, 0.3, 0.015], facecolor=axcolor)
    ax_dwC = plt.axes([0.62, 0.13, 0.3, 0.015], facecolor=axcolor)
    ax_R2b = plt.axes([0.18, 0.11, 0.3, 0.015], facecolor = axcolor)
    ax_R2c = plt.axes([0.62, 0.11, 0.3, 0.015], facecolor = axcolor)
    ax_kexAB = plt.axes([0.25, 0.07, 0.65, 0.015], facecolor=axcolor)
    ax_kexAC = plt.axes([0.25, 0.05, 0.65, 0.015], facecolor=axcolor)
    ax_kexBC = plt.axes([0.25, 0.03, 0.65, 0.015], facecolor=axcolor)

    # Set slider ID and values
    slider_pB = Slider(ax_pB, 'p$_B$', 0, .2, valfmt = '%.3f', valinit = pB0)
    slider_pC = Slider(ax_pC, 'p$_C$', 0, .2, valfmt = '%.3f', valinit = pC0)
    slider_dwB = Slider(ax_dwB, '$\Delta$$\omega$$_B$', -10, 10, valinit = dwB0)
    slider_dwC = Slider(ax_dwC, '$\Delta$$\omega$$_C$', -10, 10, valinit = dwC0)
    slider_R2b = Slider(ax_R2b, 'R$_{2b}$', 0, 50, valinit = R2a0)
    slider_R2c = Slider(ax_R2c, 'R$_{2c}$', 0, 50, valinit = R2a0)
    slider_kexAB = Slider(ax_kexAB, 'k$_{ex}$AB (s$^{-1}$)', 0, 15000, valinit = kexAB0)
    slider_kexAC = Slider(ax_kexAC, 'k$_{ex}$AC (s$^{-1}$)', 0, 15000, valinit = kexAC0)
    slider_kexBC = Slider(ax_kexBC, 'k$_{ex}$BC (s$^{-1}$)', 0, 15000, valinit = kexBC0)
    
    return slider_pB, slider_pC, slider_dwB,\
        slider_dwC, slider_R2b, slider_R2c,\
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

def plot1():    
    fig, ax = plt.subplots(figsize = (7, 7))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.xlabel(r'$\Omega\,2\pi^{-1}\,{(Hz)}$', fontsize=16)
    plt.ylabel(r'$R_{2}+R_{ex}\,(s^{-1})$', fontsize=16)
    plt.title('Interactive RD Plot', fontsize = 20)
    plt.axis([-3000, 3000, 10, 60])
    axcolor = 'lightgrey'
    
    # Initial plotted data
    offset = linspace(-3000, 3000, 50)
    l, = plt.plot(offset, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500), lw = 0, marker = 'o', color = 'C0')
    l2, = plt.plot(offset, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000), lw = 0, marker = 'o', color = 'C3')

    ## Update the y-data values when sliders are changed
    # Set Sliders
    slider_pB, slider_pC, slider_dwB, \
        slider_dwC, slider_R2b, slider_R2c, \
        slider_kexAB, slider_kexAC, slider_kexBC = initializeSliders()
    
    # Function to update y-data when slider changed
    def update(val):
        pB = slider_pB.val 
        pC = slider_pC.val
        dwB = slider_dwB.val 
        dwC = slider_dwC.val 
        R2b = slider_R2b.val
        R2c = slider_R2c.val
        kexAB = slider_kexAB.val 
        kexAC = slider_kexAC.val 
        kexBC = slider_kexBC.val 
        if slps.get_status()[0] == True: # These check status of check boxes to see if visable and should be plotted
            l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c, offset, 500))
        if slps.get_status()[1] == True:
            l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c, offset, 1000))
        fig.canvas.draw_idle()
    # Update profile when a slider is changed
    slider_pB.on_changed(update)
    slider_pC.on_changed(update)
    slider_dwB.on_changed(update)
    slider_dwC.on_changed(update)
    slider_R2b.on_changed(update)
    slider_R2c.on_changed(update)
    slider_kexAB.on_changed(update)
    slider_kexAC.on_changed(update)
    slider_kexBC.on_changed(update)

    ## CheckButtons for turning SLP on/off
    # Make the Check Button axes
    lines = [l, l2]
    vis_ax = plt.axes([0.01, 0.2, 0.15, 0.1])
    labels = (['500 Hz', '1000 Hz'])
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
        if slps.get_status()[0] == True:
            l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c, offset, 500))
        if slps.get_status()[1] == True:
            l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c, offset, 1000))
        plt.draw()
    # Call function on click
    slps.on_clicked(show_slps)

    ## RadioButtons to switch atom type
    # Make the RadioButton axes
    type_ax = plt.axes([0.01, 0.3, 0.15, 0.1])
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
        if slps.get_status()[0] == True:
            l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c, offset, 500))
        if slps.get_status()[1] == True:
            l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c, offset, 1000))
        plt.draw()
    # Execute on click
    atomtypeButton.on_clicked(changelmf)

    ## RadioButtons to switch B0
    # Makes axes
    Fieldax = plt.axes([0.01, 0.4, 0.15, 0.1])
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
    ax_axis = plt.axes([0.25, 0.01, 0.65, 0.015], facecolor=axcolor)
    slider_axis = Slider(ax_axis, 'x-axis lim', 15, 300, valinit = 40)
    # Function
    def update_axis(val):
        ax.axis([-3000, 3000, 10, val])
    # Call on changed
    slider_axis.on_changed(update_axis)

    ## Reset Button
    # Make
    resetax = plt.axes([0.01, 0.16, 0.07, 0.03])
    button = Button(resetax, 'Reset')
    # Function
    def reset(event):
        slider_pB.reset()
        slider_pC.reset()
        slider_dwB.reset()
        slider_dwC.reset()
        slider_R2b.reset()
        slider_R2c.reset()
        slider_kexAB.reset()
        slider_kexAC.reset()
        slider_kexBC.reset()
        slider_axis.reset()
        l.set_ydata(data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500))
        l2.set_ydata(data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000))
        plt.draw()
    # Call
    button.on_clicked(reset)
    
    ## Button to start custom entry GUI
    customax = plt.axes([0.01, 0.90, 0.2, 0.03])
    GuiButton = Button(customax, 'Define Values')
    # Fucntion
    def init_custom(event):
        def custom_update():
            if isnumber(pB_u.get()) == True:
                if float(pB_u.get()) < 1 and float(pB_u.get()) >= 0:
                    slider_pB.set_val(float(pB_u.get()))
            if isnumber(pC_u.get()) == True:
                if float(pC_u.get()) < 1 and float(pC_u.get()) >= 0:
                    slider_pC.set_val(float(pC_u.get()))
            if isnumber(dwB_u.get()) == True:
                if float(dwB_u.get()) < 80 and float(dwB_u.get()) > -80:
                    slider_dwB.set_val(float(dwB_u.get()))
            if isnumber(dwC_u.get()) == True:
                if float(dwC_u.get()) < 80 and float(dwC_u.get()) > -80:
                    slider_dwC.set_val(float(dwC_u.get()))
            if isnumber(R2b_u.get()) == True:
                if float(R2b_u.get()) >= 0:
                    slider_R2b.set_val(float(R2b_u.get()))
            if isnumber(R2c_u.get()) == True:
                if float(R2c_u.get()) >= 0:
                    slider_R2c.set_val(float(R2c_u.get()))
            if isnumber(kexAB_u.get()) == True:
                if float(kexAB_u.get()) >= 0:
                    slider_kexAB.set_val(float(kexAB_u.get()))
            if isnumber(kexAC_u.get()) == True:
                if float(kexAC_u.get()) >= 0:
                    slider_kexAC.set_val(float(kexAC_u.get()))
            if isnumber(kexBC_u.get()) == True:
                if float(kexBC_u.get()) >= 0:    
                    slider_kexBC.set_val(float(kexBC_u.get()))

        global Variables
        Variables = Tk.Tk()
        Variables.wm_title('User Variable Entry')
        Tk.Label(Variables, text="pB").grid(row=0, column = 0)
        Tk.Label(Variables, text="pC").grid(row=0, column = 2)
        Tk.Label(Variables, text="dwB").grid(row=1, column = 0)
        Tk.Label(Variables, text="dwC").grid(row=1, column = 2)
        Tk.Label(Variables, text="R2b").grid(row=2, column = 0)
        Tk.Label(Variables, text="R2c").grid(row=2, column = 2)

        Tk.Label(Variables, text="kexAB").grid(row=3, column = 0)
        Tk.Label(Variables, text="kexAC").grid(row=3, column = 2)
        Tk.Label(Variables, text="kexBC").grid(row=4, column = 0)

        pB_u = Tk.Entry(Variables)
        pC_u = Tk.Entry(Variables)
        dwB_u = Tk.Entry(Variables)
        dwC_u = Tk.Entry(Variables)
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

        Tk.Button(Variables, text='Update', command=custom_update).grid(row=5, column=1, pady=4)
    
    #Call on button click
    GuiButton.on_clicked(init_custom)
    

    ## Button to swtich plot type
    switchax = plt.axes([0.01, 0.95, 0.2, 0.03])
    switchButton = Button(switchax, 'Change SLP Range')
    # Fucntion
    def switchPlots(event):
        plt.close() # Close current plot
        try:
            Variables.destroy()
        except:
            pass
        global lmf
        lmf = lmf0 # reset the lmf
        plot2() # open plot2
        # close custom window if open
    # Call
    switchButton.on_clicked(switchPlots)

    # All set now show it
    plt.show()

def plot2():
    fig, ax = plt.subplots(figsize = (7, 7))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.xlabel(r'$\Omega\,2\pi^{-1}\,{(Hz)}$', fontsize=16)
    plt.ylabel(r'$R_{2}+R_{ex}\,(s^{-1})$', fontsize=16)
    plt.title('Interactive RD Plot', fontsize = 20)
    plt.axis([-6000, 6000, 10, 60])
    axcolor = 'lightgrey'
    
    # Initial plotted data
    offset1 = linspace(-450, 450, 24) # 150 Hz x 3
    offset2 = linspace(-1500, 1500, 24) # 500 Hz x 3
    offset3 = linspace(-3000, 3000, 24) # 1000 Hz x 3
    offset4 = linspace(-6000, 6000, 24) # 2000 Hz x 3
    l1, = plt.plot(offset1, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset1, 150), lw = 0, marker = 'o', color = 'C0', label = '150 Hz')
    l2, = plt.plot(offset2, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset2, 500), lw = 0, marker = 'o', color = 'C1', label = '500 Hz')
    l3, = plt.plot(offset3, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset3, 1000), lw = 0, marker = 'o', color = 'C2', label = '1000 Hz')
    l4, = plt.plot(offset4, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset4, 2000), lw = 0, marker = 'o', color = 'C3', label = '2000 Hz')
    plt.legend()
    
    def update_yValues(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0):
        # When updating, call this function rather then all four individual
        l1.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset1, 150))
        l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset2, 500))
        l3.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset3, 1000))
        l4.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset4, 2000))
    
    ## Update the y-data values when sliders are changed
    slider_pB, slider_pC, slider_dwB, \
        slider_dwC, slider_R2b, slider_R2c, \
        slider_kexAB, slider_kexAC, slider_kexBC = initializeSliders()
    def update(val):
        pB = slider_pB.val 
        pC = slider_pC.val
        dwB = slider_dwB.val 
        dwC = slider_dwC.val 
        R2b = slider_R2b.val
        R2c = slider_R2c.val
        kexAB = slider_kexAB.val 
        kexAC = slider_kexAC.val 
        kexBC = slider_kexBC.val 
        update_yValues(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c)
        fig.canvas.draw_idle()
    # Update profile when a slider is changed
    slider_pB.on_changed(update)
    slider_pC.on_changed(update)
    slider_dwB.on_changed(update)
    slider_dwC.on_changed(update)
    slider_R2b.on_changed(update)
    slider_R2c.on_changed(update)
    slider_kexAB.on_changed(update)
    slider_kexAC.on_changed(update)
    slider_kexBC.on_changed(update)


    ## RadioButtons to switch atom type
    type_ax = plt.axes([0.01, 0.3, 0.15, 0.1])
    atomtypeButton = RadioButtons(type_ax, ('Carbon', 'Nitrogen'))
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
        update_yValues(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b, R2c)
        plt.draw()
    atomtypeButton.on_clicked(changelmf)

    ## RadioButtons to switch B0
    Fieldax = plt.axes([0.01, 0.4, 0.15, 0.1])
    FieldRadio = RadioButtons(Fieldax, ('600 MHz' ,'700 MHz', '800MHz', '1.1GHz'))
    def changeField(label):
        FieldDict = {'600 MHz':600, '700 MHz':700, '800MHz':800, '1.1GHz':1100}
        global Field
        Field = FieldDict[label]
        changelmf(atomtypeButton.value_selected)
    FieldRadio.on_clicked(changeField)

    ## Slider to adjust x-axis
    ax_axis = plt.axes([0.25, 0.01, 0.65, 0.015], facecolor=axcolor)
    slider_axis = Slider(ax_axis, 'x-axis lim', 15, 300, valinit = 40)
    def update_axis(val):
        ax.axis([-6000, 6000, 10, val])
    slider_axis.on_changed(update_axis)

    ## Reset Button
    resetax = plt.axes([0.01, 0.16, 0.07, 0.03])
    button = Button(resetax, 'Reset')
    def reset(event):
        slider_pB.reset()
        slider_pC.reset()
        slider_dwB.reset()
        slider_dwC.reset()
        slider_R2b.reset()
        slider_R2c.reset()
        slider_kexAB.reset()
        slider_kexAC.reset()
        slider_kexBC.reset()
        slider_axis.reset()
        update_yValues(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0)
        plt.draw()
    button.on_clicked(reset)


        ## Button to start custom entry GUI
    customax = plt.axes([0.01, 0.90, 0.2, 0.03])
    GuiButton = Button(customax, 'Define Values')
    # Fucntion
    def init_custom(event):
        def custom_update():
            if isnumber(pB_u.get()) == True:
                if float(pB_u.get()) < 1 and float(pB_u.get()) >= 0:
                    slider_pB.set_val(float(pB_u.get()))
            if isnumber(pC_u.get()) == True:
                if float(pC_u.get()) < 1 and float(pC_u.get()) >= 0:
                    slider_pC.set_val(float(pC_u.get()))
            if isnumber(dwB_u.get()) == True:
                if float(dwB_u.get()) < 80 and float(dwB_u.get()) > -80:
                    slider_dwB.set_val(float(dwB_u.get()))
            if isnumber(dwC_u.get()) == True:
                if float(dwC_u.get()) < 80 and float(dwC_u.get()) > -80:
                    slider_dwC.set_val(float(dwC_u.get()))
            if isnumber(R2b_u.get()) == True:
                if float(R2b_u.get()) >= 0:
                    slider_R2b.set_val(float(R2b_u.get()))
            if isnumber(R2c_u.get()) == True:
                if float(R2c_u.get()) >= 0:
                    slider_R2c.set_val(float(R2c_u.get()))
            if isnumber(kexAB_u.get()) == True:
                if float(kexAB_u.get()) >= 0:
                    slider_kexAB.set_val(float(kexAB_u.get()))
            if isnumber(kexAC_u.get()) == True:
                if float(kexAC_u.get()) >= 0:
                    slider_kexAC.set_val(float(kexAC_u.get()))
            if isnumber(kexBC_u.get()) == True:
                if float(kexBC_u.get()) >= 0:    
                    slider_kexBC.set_val(float(kexBC_u.get()))

        global Variables
        Variables = Tk.Tk()
        Variables.wm_title('User Variable Entry')
        Tk.Label(Variables, text="pB").grid(row=0, column = 0)
        Tk.Label(Variables, text="pC").grid(row=0, column = 2)
        Tk.Label(Variables, text="dwB").grid(row=1, column = 0)
        Tk.Label(Variables, text="dwC").grid(row=1, column = 2)
        Tk.Label(Variables, text="R2b").grid(row=2, column = 0)
        Tk.Label(Variables, text="R2c").grid(row=2, column = 2)

        Tk.Label(Variables, text="kexAB").grid(row=3, column = 0)
        Tk.Label(Variables, text="kexAC").grid(row=3, column = 2)
        Tk.Label(Variables, text="kexBC").grid(row=4, column = 0)

        pB_u = Tk.Entry(Variables)
        pC_u = Tk.Entry(Variables)
        dwB_u = Tk.Entry(Variables)
        dwC_u = Tk.Entry(Variables)
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

        Tk.Button(Variables, text='Update', command=custom_update).grid(row=5, column=1, pady=4)
    
    #Call on button click
    GuiButton.on_clicked(init_custom)

    ## Button to swtich plot type
    switchax = plt.axes([0.01, 0.95, 0.2, 0.03])
    switchButton = Button(switchax, 'Change SLP Range')
    def switchPlots(event):
        plt.close() # Close current plot
        try:
            Variables.destroy()
        except:
            pass
        global lmf
        lmf = lmf0 # reset the lmf
        plot1()
    switchButton.on_clicked(switchPlots)
    # All set now show it
    plt.show()

plot1() 