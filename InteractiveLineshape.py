from numpy import arange, argmax, array, diag, dot, exp, linalg, linspace, pi, zeros
from nmrglue import proc_base
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

fig, ax = plt.subplots(figsize = (7, 7))
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.ylabel('Intensity', fontsize = 16)
plt.xlabel(r'$\omega\,(ppm)$', fontsize=16)
plt.ylim(0, 100)

pB0 = .1
wA0, wB0 = 0, 3
kexAB0 = 2800
R1a0 = R1b0 = 2
R2a0, R2b0 = 22.5, 22.5
lmf = 150.784627  # 1H for 700MHz spectrometer
aq = 1.0
N = 1024 * 4
T = linspace(0., aq, N)
dt = aq / N
xf = arange(-1./(2*dt), 1./(2*dt), 1./dt/N)/lmf

def isnumber(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def setplot(lmf, pB, wA, wB, kexAB, R1a0, R1b0, R2a, R2b):
    new_y = data(lmf, pB, wA, wB, kexAB, R1a0, R1b0, R2a, R2b)
    l.set_ydata(new_y)
    xmax = xf[argmax(new_y)]
    ymax = max(new_y)
    text = 'Main peak @ %s ppm | Peak Int: %s' % (format(xmax, '.4f'), format(ymax, '.2f'))
    ax.set_title(text)

def set_x(lmf):
	return arange(-1./(2*dt), 1./(2*dt), 1./dt/N)/lmf

def data(lmf, pB, wA, wB, kexAB, R1a, R1b, R2a, R2b):
	pA = 1 - pB
	M0 = array([1-pB, pB])
	k12 = kexAB * pB / (pB + pA)
	k21 = kexAB * pA / (pB + pA)
	wA = wA * lmf * 2 * pi
	wB = wB * lmf * 2 * pi
	dw = wB - wA
	dR2 = R2b - R2a

	R = zeros((2, 2), dtype=complex)
	R[0, 0] = -R2a + 1j*wA - k12
	R[0, 1] = k21
	R[1, 0] = k12
	R[1, 1] = -R2b + 1j*wB - k21

	v,G = linalg.eig(R)
	G_1 = linalg.inv(G)

	
	fid = zeros(T.shape, dtype=complex)
	for i,t in enumerate(T):
		A = dot(dot(G,diag(exp(v*t))), G_1)
		fid[i] = sum(dot(A, M0))
	return proc_base.fft(fid).real

# Inital Plotted Data
y0 = data(lmf, pB0, wA0, wB0, kexAB0, R1a0, R1b0, R2a0, R2b0)
l, = plt.plot(xf, y0)
xmax = xf[argmax(y0)]
text = 'Main peak @ %s ppm' % format(xmax, '.4f')
ax.set_title(text)

# Initallize Sliders
axcolor = 'lightgrey'
ax_pB = plt.axes([0.16, 0.15, 0.3, 0.015], facecolor=axcolor)
ax_wA = plt.axes([0.16, 0.13, 0.3, 0.015], facecolor=axcolor)
ax_wB = plt.axes([0.16, 0.11, 0.3, 0.015], facecolor=axcolor)
ax_R2a = plt.axes([0.60, 0.13, 0.3, 0.015], facecolor = axcolor)
ax_R2b = plt.axes([0.60, 0.11, 0.3, 0.015], facecolor = axcolor)
ax_kexAB = plt.axes([0.60, 0.15, 0.3, 0.015], facecolor=axcolor)

# Set slider ID and values
slider_pB = Slider(ax_pB, 'p$_B$', .00, .5, valfmt = '%.3f', valinit = pB0)
slider_wA = Slider(ax_wA, '$\omega$$_A$', -10, 10, valinit = wA0)
slider_wB = Slider(ax_wB, '$\omega$$_B$', -10, 10, valinit = wB0)
slider_R2a = Slider(ax_R2a, 'R$_{2a}$', 0, 50, valinit = R2a0)
slider_R2b = Slider(ax_R2b, 'R$_{2b}$', 0, 50, valinit = R2b0)
slider_kexAB = Slider(ax_kexAB, 'k$_{ex}$AB', 0, 50000, valinit = kexAB0)

def update(val):
    pB = slider_pB.val 
    wA = slider_wA.val 
    wB = slider_wB.val 
    R2a = slider_R2a.val
    R2b = slider_R2b.val
    kexAB = slider_kexAB.val 
    setplot(lmf, pB, wA, wB, kexAB, R1a0, R1b0, R2a, R2b)
    fig.canvas.draw_idle()

slider_pB.on_changed(update)
slider_wA.on_changed(update)
slider_wB.on_changed(update)
slider_R2a.on_changed(update)
slider_R2b.on_changed(update)
slider_kexAB.on_changed(update)

rax = plt.axes([0.01, 0.35, 0.15, 0.1])
radio = RadioButtons(rax, ('Carbon', 'Nitrogen', 'Proton'))

def init_custom(event):
    def custom_update():
        if isnumber(pB_u.get()) == True and float(pB_u.get()) < 1 and float(pB_u.get()) >= 0:
            slider_pB.set_val(float(pB_u.get()))
        if isnumber(wA_u.get()) == True and float(wA_u.get()) < 80 and float(wA_u.get()) > -80:
            slider_wA.set_val(float(wA_u.get()))
        if isnumber(wB_u.get()) == True and float(wB_u.get()) < 80 and float(wB_u.get()) > -80:
            slider_wB.set_val(float(wB_u.get()))
        if isnumber(R2a_u.get()) == True and float(R2a_u.get()) >= 0:
            slider_R2a.set_val(float(R2a_u.get()))
        if isnumber(R2b_u.get()) == True and float(R2b_u.get()) >= 0:
            slider_R2b.set_val(float(R2b_u.get()))
        if isnumber(kexAB_u.get()) == True and float(kexAB_u.get()) >= 0:
            slider_kexAB.set_val(float(kexAB_u.get()))

    global Variables
    Variables = Tk.Tk()
    Variables.wm_title('User Variable Entry')
    Tk.Label(Variables, text="pB").grid(row=0, column = 0)
    Tk.Label(Variables, text="wA").grid(row=1, column = 0)
    Tk.Label(Variables, text="wB").grid(row=1, column = 2)
    Tk.Label(Variables, text="R2a").grid(row=2, column = 0)
    Tk.Label(Variables, text="R2b").grid(row=2, column = 2)
    Tk.Label(Variables, text="kexAB").grid(row=3, column = 0)

    pB_u = Tk.Entry(Variables)
    wA_u = Tk.Entry(Variables)
    wB_u = Tk.Entry(Variables)
    R2a_u = Tk.Entry(Variables)
    R2b_u = Tk.Entry(Variables)
    kexAB_u = Tk.Entry(Variables)

    pB_u.grid(row = 0, column = 1)
    wA_u.grid(row = 1, column = 1)
    wB_u.grid(row = 1, column = 3)
    R2a_u.grid(row = 2, column = 1)
    R2b_u.grid(row = 2, column = 3)
    kexAB_u.grid(row = 3, column = 1)

    Tk.Button(Variables, text='Update', command=custom_update).grid(row=5, column=1, pady=4)

customax = plt.axes([0.01, 0.9, 0.17, 0.055])
GuiButton = Button(customax, 'User Defined\nSlider Values')
#Call on button click
GuiButton.on_clicked(init_custom)

# RadioButtons for lmf
def changelmf(label):
    lmfdict = {'Carbon':150.784627, 'Nitrogen':60.76302, 'Proton':600}
    global lmf
    lmf = lmfdict[label] 
    pB = slider_pB.val 
    wA = slider_wA.val 
    wB = slider_wB.val 
    R2a = slider_R2a.val
    R2b = slider_R2b.val
    kexAB = slider_kexAB.val 
    l.set_xdata(set_x(lmf))
    setplot(lmf, pB, wA, wB, kexAB, R1a0, R1b0, R2a, R2b)
    ax.axis([min(set_x(lmf)), max(set_x(lmf)), None, None])
    plt.draw()

radio.on_clicked(changelmf)


plt.show()
