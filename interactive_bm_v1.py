from numpy import absolute, append, arctan, array, asarray
from numpy import diag, dot, exp
from numpy import float64, iscomplexobj, linspace, linalg, log
from numpy import pi, shape, sin, tan, vstack, zeros
from numpy import apply_along_axis
from numpy.linalg import eig, inv, norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

fig, ax = plt.subplots(figsize = (7, 7))
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.xlabel(r'$\Omega\,2\pi^{-1}\,{(Hz)}$', fontsize=16)
plt.ylabel(r'$R_{2}+R_{ex}\,(s^{-1})$', fontsize=16)
plt.title('Interactive RD Plot', fontsize = 20)
plt.axis([-3000, 3000, 10, 60])

offset = linspace(-3000, 3000, 50)
lmf = lmf0 = 150.784627
pB0, pC0 = .01, 0
dwB0, dwC0 = 3, 0
kexAB0, kexAC0, kexBC0 = 1000, 0, 0
R1a0 = R1b0 = R1c0 = 2.5
R2a0 = R2b0 = R2c0 = 20

def ExpDecay(x,a,b):
    return a*exp(-b*x)

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

        thetaAvg = arctan(w1/deltaAvg) # == arccot(deltaAvg/(w1*))
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
    #Points are sacrificed for speed; may cause issues; should be okay since err = 0 
    time = linspace(0, 0.2, 3)
    offset2pi = array(offset * 2 * pi)
    offset2pi = vstack(offset2pi)
    w1 = w1 * 2 * pi 
    return apply_along_axis(CalcR2eff, 1, offset2pi, w1, lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a, R1b, R1c, R2a, R2b, R2c, time)

    
# Initial plotted data
l, = plt.plot(offset, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500), lw = 0, marker = 'o', color = 'C0')
l2, = plt.plot(offset, data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000), lw = 0, marker = 'o', color = 'C3')


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
slider_pB = Slider(ax_pB, 'p$_B$', .001, .2, valinit = pB0)
slider_pC = Slider(ax_pC, 'p$_C$', .001, .2, valinit = pC0)
slider_dwB = Slider(ax_dwB, '$\Delta$$\omega$$_B$', -10, 10, valinit = dwB0)
slider_dwC = Slider(ax_dwC, '$\Delta$$\omega$$_C$', -10, 10, valinit = dwC0)
slider_R2b = Slider(ax_R2b, 'R$_{2b}$', 0, 50, valinit = R2a0)
slider_R2c = Slider(ax_R2c, 'R$_{2c}$', 0, 50, valinit = R2a0)
slider_kexAB = Slider(ax_kexAB, 'k$_{ex}$AB (s$^{-1}$)', 0, 50000, valinit = kexAB0)
slider_kexAC = Slider(ax_kexAC, 'k$_{ex}$AC (s$^{-1}$)', 0, 50000, valinit = kexAC0)
slider_kexBC = Slider(ax_kexBC, 'k$_{ex}$BC (s$^{-1}$)', 0, 50000, valinit = kexBC0)

# Function to update the y-data values when sliders are changed
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

    if slps.get_status()[0] == True:
        l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500))
    if slps.get_status()[1] == True:
        l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000))

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

# CheckButtons for SLP
lines = [l, l2]
rax = plt.axes([0.01, 0.2, 0.15, 0.15])
labels = (['500 Hz', '1000 Hz'])
visibility = [line.get_visible() for line in lines]
slps = CheckButtons(rax, labels, visibility)

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
        l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500))
    if slps.get_status()[1] == True:
        l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000))
    plt.draw()
slps.on_clicked(show_slps)

# RadioButtons to switch atom type
rax = plt.axes([0.01, 0.35, 0.15, 0.1])
radio = RadioButtons(rax, ('Carbon', 'Nitrogen'))

def changelmf(label):
    lmfdict = {'Carbon':150.784627, 'Nitrogen':60.76302}
    global lmf
    lmf = lmfdict[label]
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
        l.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500))
    if slps.get_status()[1] == True:
        l2.set_ydata(data(lmf, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000))
    plt.draw()
radio.on_clicked(changelmf)

# Slider to adjust x-axis
def update_axis(val):
    ax.axis([-3000, 3000, 10, val])

ax_axis = plt.axes([0.25, 0.01, 0.65, 0.015], facecolor=axcolor)
slider_axis = Slider(ax_axis, 'x-axis lim', 15, 300, valinit = 40)
slider_axis.on_changed(update_axis)

# Reset Button
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
    l.set_ydata(data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 500))
    l2.set_ydata(data(lmf0, pB0, pC0, dwB0, dwC0, kexAB0, kexAC0, kexBC0, R1a0, R1b0, R1c0, R2a0, R2b0, R2c0, offset, 1000))
    plt.draw()

button.on_clicked(reset)

# All set now show it
plt.show()