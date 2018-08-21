# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 01:35:49 2018

An analytic secondary source model of edge diffraction impulse responses
"""

# to do list:
#   performance:
#   - "apex point" optimization (up to 2x speed improvement)
#   - 
#   ditching the sympy.geometry stuff for something more reliable

import numpy
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sympy import Point3D, Line3D, Ray3D, Segment3D
from speaker import Speaker, read_ts_param

def raddeg(x):
    return x*180.0/numpy.pi
def ddelta(taxis, d, sig, method):
    t = d / c_sound
    A = 1.0 / d
    if method == 'gauss':
        return A*((2*numpy.pi*sig*sig)**(-0.5))*numpy.exp(-((taxis-t)**2.0)/(2*sig**2.0))
    else:
        print('Error')

def angle(e1, e2):
    tol = 0.001    
    alpha = e1.angle_between(e2).evalf()
    if abs(alpha-0.5*numpy.pi) < tol:
        arrangement = 'perpendicular'
    elif abs(alpha-numpy.pi) < tol: # 180 degree
        arrangement = 'paralell'
        e2 = Line3D(e2.p2, e2.p1)
    elif abs(alpha) < tol:          #   0 degree
        arrangement = 'paralell'        
    else:
        arrangement = 'general'
    return arrangement, e1, e2
        
def vecnorm(v):
    # v: sympy geometry segment
    p1, p2 = v.p1.evalf(), v.p2.evalf()
    V = numpy.array(p2.x-p1.x, p2.y-p1.y, p2.z-p1.z)
    norm = 1.0 / sum(V**2.0)**0.5
    return V*norm

""" ===========================================================================
    Calculate the impulse response contribution of an edge
    ===========================================================================
"""
def delta_h(c, m, l, teta_R, teta_S, teta_w, rS,rR, zR, z, dz):
    # m, l = z / numpy.sin(alpha), (z-zR) / numpy.sin(gamma) 
    # tau = (m + l) / c
    v = numpy.pi/teta_w
    s_pp = numpy.sin(v*(numpy.pi+teta_R+teta_S))
    s_pm = numpy.sin(v*(numpy.pi+teta_R-teta_S))
    s_mp = numpy.sin(v*(numpy.pi-teta_R+teta_S))
    s_mm = numpy.sin(v*(numpy.pi-teta_R-teta_S))
    c_pp = numpy.cos(v*(numpy.pi+teta_R+teta_S))
    c_pm = numpy.cos(v*(numpy.pi+teta_R-teta_S))
    c_mp = numpy.cos(v*(numpy.pi-teta_R+teta_S))
    c_mm = numpy.cos(v*(numpy.pi-teta_R-teta_S))
    # !!! needs check: why is _y < 1 in some cases?
    # _y = (m*l + z*(z-zR)) / (rS*rR) # due some numerical errors ->
    # print(min(_y), max(_y))
    _y = numpy.maximum(((m*l + z*(z-zR)) / (rS*rR)), 1.0)
    _A = (_y**2.0 - 1.0)**0.5 + _y
    cosh_veta = 0.5*(_A**v + _A**(-v))
    b_pp = s_pp / (cosh_veta - c_pp)
    b_pm = s_pm / (cosh_veta - c_pm)
    b_mp = s_mp / (cosh_veta - c_mp)
    b_mm = s_mm / (cosh_veta - c_mm)
    beta = b_pp + b_pm + b_mp + b_mm
    if False:
        print(s_pp,s_pm,s_mp,s_mm)
        print(c_pp,c_pm,c_mp,c_mm)
        print(min(cosh_veta),max(cosh_veta))
        print(min(beta),max(beta))
    return -v/(4.0*numpy.pi) * beta / (m*l) * dz

""" ===========================================================================
    Calculate 2nd order impulse response contribution of a pair of edges
    ===========================================================================
"""
def delta_h12(c_sound, l1, l2, l3, teta_R, teta_S, teta_w1, teta_w2, rS, rR, r12, zrec, zax1, zax2, dz):
    # m, l = z / numpy.sin(alpha), (z-zR) / numpy.sin(gamma) 
    # tau = (m + l) / c
    v1 = numpy.pi/teta_w1
    v2 = numpy.pi/teta_w2

    # !!! needs check: why is _y < 1 in some cases?
    # _y = (m*l + z*(z-zR)) / (rS*rR) # due some numerical errors ->

    l1m = l1[:, numpy.newaxis]
    l3m = l3[numpy.newaxis, :]
    zm1 = zax1[:, numpy.newaxis]
    zm2 = zax2[numpy.newaxis, :]    
    # >>> beta(S -> edge1 -> edge2) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
    s_pp = numpy.sin(v1*(numpy.pi+teta_R+teta_S))
    s_pm = numpy.sin(v1*(numpy.pi+teta_R-teta_S))
    s_mp = numpy.sin(v1*(numpy.pi-teta_R+teta_S))
    s_mm = numpy.sin(v1*(numpy.pi-teta_R-teta_S))
    c_pp = numpy.cos(v1*(numpy.pi+teta_R+teta_S))
    c_pm = numpy.cos(v1*(numpy.pi+teta_R-teta_S))
    c_mp = numpy.cos(v1*(numpy.pi-teta_R+teta_S))
    c_mm = numpy.cos(v1*(numpy.pi-teta_R-teta_S))    
    _y = numpy.maximum(((l1m*l2 + zm1*(zm1-zm2)) / (rS*r12)), 1.0)
    _A = (_y**2.0 - 1.0)**0.5 + _y    
    cosh_veta = 0.5*(_A**v1 + _A**(-v1))
    b_pp = s_pp / (cosh_veta - c_pp)
    b_pm = s_pm / (cosh_veta - c_pm)
    b_mp = s_mp / (cosh_veta - c_mp)
    b_mm = s_mm / (cosh_veta - c_mm)
    beta1 = b_pp + b_pm + b_mp + b_mm
    # >>> beta(edge1 -> edge2 -> R) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    s_pp = numpy.sin(v2*(numpy.pi+teta_R+teta_S))
    s_pm = numpy.sin(v2*(numpy.pi+teta_R-teta_S))
    s_mp = numpy.sin(v2*(numpy.pi-teta_R+teta_S))
    s_mm = numpy.sin(v2*(numpy.pi-teta_R-teta_S))
    c_pp = numpy.cos(v2*(numpy.pi+teta_R+teta_S))
    c_pm = numpy.cos(v2*(numpy.pi+teta_R-teta_S))
    c_mp = numpy.cos(v2*(numpy.pi-teta_R+teta_S))
    c_mm = numpy.cos(v2*(numpy.pi-teta_R-teta_S))    
    _y = numpy.maximum(((l2*l3m + (zm2-zm1)*(zm2-zrec)) / (rR*r12)), 1.0)
    _A = (_y**2.0 - 1.0)**0.5 + _y    
    cosh_veta = 0.5*(_A**v2 + _A**(-v2))
    b_pp = s_pp / (cosh_veta - c_pp)
    b_pm = s_pm / (cosh_veta - c_pm)
    b_mp = s_mp / (cosh_veta - c_mp)
    b_mm = s_mm / (cosh_veta - c_mm)
    beta2 = b_pp + b_pm + b_mp + b_mm    

    const = v1*v2/(32.0*numpy.pi*numpy.pi) * dz
    return const  * beta1*beta2 / (l1*l2*l3) 


""" ===========================================================================
    Determine the parameters of and edge: 
    teta_w, teta_S, teta_R, 
    ===========================================================================
"""
def edge_param(edge, v1, v2, source, receiver):
    # source, receiver :: Point3D
    # edge :: Line3D(Point3D(), Point3D())    
    # v1, v2: vectors describing the edge
    proj_src_edge = edge.projection(source)
    proj_rec_edge = edge.projection(receiver)
    
    ray_edge_src = Line3D(proj_src_edge, source)   #  Proj(S, edge) ---> S
    ray_edge_rec = Line3D(proj_rec_edge, receiver) #  Proj(R, edge) ---> R    
    
    x = float(v1.angle_between(v2))    
    teta_w = max(2.0*numpy.pi - x, x)
    
    teta_S = float(ray_edge_src.angle_between(v1))
    teta_R = float(ray_edge_rec.angle_between(v1))
    
    # print(raddeg(teta_S), raddeg(teta_R))
    return teta_w, teta_R, teta_S


""" ===========================================================================
    Calculate m,l for each segment of an edge
    ===========================================================================
"""
def edge_quant(edge, source, receiver, res):
    # the apex point optimaziation is not implemented
    # source, receiver :: Point3D
    # edge :: Line3D(Point3D(), Point3D())
    edgelength = float(edge.p1.distance(edge.p2))
    zaxis = numpy.linspace(res*0.5, edgelength-res*0.5, int(edgelength/(res)))
    
    rS, rR = float(edge.distance(source)), float(edge.distance(receiver))
    _p1, _p2 = edge.p1, edge.p2
    proj_src_edge = edge.projection(source)
    proj_rec_edge = edge.projection(receiver)
    
    # >>> determine relative z position of the source: >>>>>>>>>>>>>>>>>>>>>>>>
    alpha = edge.angle_between(Line3D(_p1, proj_src_edge)).evalf()     
    if abs(alpha) < 0.01:              #   0 degree
        z_shift = float(proj_src_edge.distance(_p1))
    elif abs(alpha-numpy.pi) < 0.01:   # 180 degree
        z_shift = -float(proj_src_edge.distance(_p1))
    else:
        raise ValueError('z_shift: alpha = %.3f  Should be 0 or pi.' % alpha)
    zaxis += -z_shift
    
    # >>> z(receiver): >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #
    abs_zR = float(proj_src_edge.distance(proj_rec_edge))
    if abs_zR < 0.001:
        zR = 0.0
    else:
        alpha = edge.angle_between(Line3D(proj_src_edge, proj_rec_edge)).evalf()        
        if abs(alpha) < 0.01:              #   0 degree
            zR =  abs_zR
        elif abs(alpha-numpy.pi) < 0.01:   # 180 degree
            zR = -abs_zR
        else:
            raise ValueError('zR: alpha = %.3f  Should be 0 or pi.' % alpha)

    m, l = (zaxis**2.0 + rS**2.0)**0.5, (zaxis**2.0 + rR**2.0)**0.5
    return m, l, zaxis, rS, rR, zR

""" ===========================================================================
    Calculate l1, l2, l3
    ===========================================================================
"""
def edge_quant2(edge1, edge2, source, receiver, res):
    # source, receiver :: Point3D
    # edge :: Line3D(Point3D(), Point3D())
    arrangement, edge1, edge2 = angle(edge1, edge2)
    rS, rR = float(edge1.distance(source)), float(edge2.distance(receiver))
    
    edge1length = float(edge1.p1.distance(edge1.p2))
    edge2length = float(edge2.p1.distance(edge2.p2))    
    
    if arrangement == 'paralell': 
        # z axis: edge1 p1 -> p2
        # z(source) = 0
        P1 = edge1.projection(source)
        P2 = edge2.projection(source)
        P3 = edge2.projection(receiver)        
        r12 = float(edge1.distance(edge2.p1))
        Dz1 = float(P1.distance(edge1.p1)) 
        Dz2 = float(P2.distance(edge2.p1))
        Dz3 = float(P3.distance(P2))
        al1 = edge1.angle_between(Line3D(edge1.p1, P1)).evalf()
        al2 = edge2.angle_between(Line3D(edge2.p1, P2)).evalf()
        al3 = edge2.angle_between(Line3D(P1, P2)).evalf()                           

        if Dz1 < 0.001: # --- zaxis1 shift -----------------------------------
            z_shift1 = 0.0
        else:
            if al1 < 0.01:                   #   0 degree
                z_shift1 = Dz1
            elif abs(al1-numpy.pi) < 0.01:   # 180 degree
                z_shift1 = -Dz1
            else:
                raise ValueError('z_shift1: alpha = %.3f  Should be 0 or pi.' % al1)
        if Dz2 < 0.001: # --- zaxis2 shift -----------------------------------
            z_shift2 = 0.0
        else:
            if al2 < 0.01:                   #   0 degree
                z_shift2 = Dz2
            elif abs(al2-numpy.pi) < 0.01:   # 180 degree
                z_shift2 = -Dz2
            else:
                raise ValueError('z_shift2: alpha = %.3f  Should be 0 or pi.' % al1)
        if Dz3 < 0.001: # --- z(receiver) _-----------------------------------
            zrec = 0.0
        else:
            if al3 < 0.01:                   #   0 degree
                zrec = Dz3
            elif abs(al3-numpy.pi) < 0.01:   # 180 degree
                zrec = -Dz3
            else:
                raise ValueError('z_shift3: alpha = %.3f  Should be 0 or pi.' % al1)

        zaxis1 = numpy.linspace(res*0.5-z_shift1, edge1length-res*0.5-z_shift1, int(edge1length/(res)))
        zaxis2 = numpy.linspace(res*0.5-z_shift2, edge2length-res*0.5-z_shift2, int(edge2length/(res)))
        
        # >>> l1(n x 1), l2(n x m), l3(1 x m) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        l1 = (zaxis1**2.0 + rS**2.0)**0.5
        l2 = ((zaxis1[:, numpy.newaxis]-zaxis2[numpy.newaxis, :])**2.0 + r12**2.0)**0.5
        l3 = ((zaxis2-zrec)**2.0 + rS**2.0)**0.5        
    elif arrangement == 'perpendicular':
        pass

    else:
        raise ValueError('Only paralell and 90degree configurations are supported.')
    return l1, l2, l3, zaxis1, zaxis2, rS, rR, r12, zrec


""" ===========================================================================
    Determine the array of point sources
    ===========================================================================
"""
def sourcegrid(v, DS, src_res, vs1, vs2):
    o = numpy.array(v)
    # --- rectangular pattern ----------------------------------
    tmp = numpy.arange(-DS*0.5-src_res, DS*0.5+src_res, src_res)    
    ngrid, shft = len(tmp), min(abs(tmp-0.0))
    tmp = tmp+shft
    v1x, v1y, v1z = tmp*vs1[0], tmp*vs1[1], tmp*vs1[2]
    v2x, v2y, v2z = tmp*vs2[0], tmp*vs2[1], tmp*vs2[2]
    sources = []
    for i in range(ngrid):
        for j in range(ngrid):
            v = numpy.array([v1x[i]+v2x[j], v1y[i]+v2y[j], v1z[i]+v2z[j]])
            r = sum(v**2.0)**0.5
            if r < DS*0.5:
                sources += [Point3D(v+o, evaluate=geomeval)]
    # print(sources)            
    return sources

""" ===========================================================================
    Draw the geometry projected to the (x,y) plane
    ===========================================================================
"""
def draw_xy(sources, edges, receiver, quiveropts, DS):
    x0,y0,x1,y1, xs, ys = [], [], [], [], [],[]
    for e in edges:
        x0 += [float(e[0].p1.x)]
        y0 += [float(e[0].p1.y)]
        x1 += [float(e[0].p2.x)]
        y1 += [float(e[0].p2.y)]
    x0, y0 = numpy.array(x0), numpy.array(y0)
    x1, y1 = numpy.array(x1), numpy.array(y1)
    for src in sources:
        xs += [float(src.x)]
        ys += [float(src.y)]
    #rec = plt.Circle((xs, ys), DS, color='b', fill=False)       
    xr,yr = [float(receiver.x)],float(receiver.y)
    with PdfPages('geometry.pdf') as pdf:
        fig, ax = plt.subplots()
        plt.quiver(x0,y0,x1-x0,y1-y0,  **quiveropts)
        plt.plot(xs,ys,'o', markersize=2)
        plt.plot(xr,yr,'o', markersize=10)        
        ax.set_aspect('equal', 'datalim')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.show()
        pdf.savefig(fig)
        plt.close()
    
def inf_baffle_resp(Re, Le, Res, Lces, Cmes, f_axis):
    # print("Re=%f Le=%fmH Res=%f Lces=%f Cmes=%f " % (Re, Le, Res, Lces, Cmes))
    # U(w) = 1V 
    w_ax = f_axis*2.0*numpy.pi
    Zs = Re + 1j*w_ax*Le
    Zp = 1.0 /(1.0/Res + 1j*w_ax*Cmes + 1.0/(1j*w_ax*Lces))
    I = Zp / (Zp+Zs) * (1j*w_ax*Cmes)
    return I

def closedbox_reps(Re, Le, Res, Lces, Cmes, Vbox, Vas, f_axis):
    alpha = Vas / Vbox
    # print("Re=%f Le=%fmH Res=%f Lces=%f Cmes=%f " % (Re, Le, Res, Lces, Cmes))
    # U(w) = 1V 
    Lceb = Lces / alpha
    w_ax = f_axis*2.0*numpy.pi
    Zs = Re + 1j*w_ax*Le
    Zp = 1.0 /(1.0/Res + 1j*w_ax*Cmes + 1.0/(1j*w_ax*Lces) + 1.0/(1j*w_ax*Lceb))
    I = Zp / (Zp+Zs) * (1j*w_ax*Cmes)
    return I

def edge_v1(edge, source, receiver, dz, f_s, c_sound, Np):
    e,v1,v2 = edge[0], edge[1], edge[2]
    teta_w, teta_R, teta_S = edge_param(e, v1, v2, source, receiver)
    m, l, zax, rS, rR, zR = edge_quant(e, source, receiver, dz)
    bin_index =  list(map(int, map(numpy.around, f_s*(m+l) / c_sound)))

    dh = delta_h(c_sound, m, l, teta_R, teta_S, teta_w, rS, rR, zR, zax, dz)
    # 'temparr[]' would not be necessary if h_t is directly accessed from this func
    temparr = numpy.zeros(Np, dtype=float)
    numpy.add.at(temparr, bin_index, dh)
    return temparr


def edge_v1v2(edge1, edge2, source, receiver, dz, f_s, c_sound, Np):
    # >>> edge1, edge2 parameters:
    e1,v11,v12 = edge1[0], edge1[1], edge1[2]
    e2,v21,v22 = edge2[0], edge2[1], edge2[2]    
    teta_w1, teta_R, _temp = edge_param(e1, v11, v12, source, receiver)
    teta_w2, _temp, teta_S = edge_param(e2, v21, v22, source, receiver)
    
    l1, l2, l3, zax1, zax2, rS, rR, r12, zrec = edge_quant2(e1, e2, source, receiver, dz)
    _n, _m = len(l1), len(l2)
    nm = _n * _m
    
    temp = numpy.reshape(f_s*(l1[:, numpy.newaxis] + l2 + l3[numpy.newaxis, :]) / c_sound, nm)
    bin_index =  list(map(int, map(round, temp)))
    
    dh = numpy.reshape(delta_h12(c_sound, l1, l2, l3, teta_R, teta_S, teta_w1, teta_w2, rS, rR, r12, zrec, zax1, zax2, dz), nm)
    
    # 'temparr[]' would not be necessary if h_t is directly accessed from this func
    temparr = numpy.zeros(Np, dtype=float)
    numpy.add.at(temparr, bin_index, dh)
    return temparr

""" 
    ---------------------------------------------------------------------------
    ===========================================================================
    >> >  >   >    >     >            M A I N            <     <    <   <  < <<
    ===========================================================================    
    ---------------------------------------------------------------------------
"""
# units: cm, ms, Hz
#
#

# === sympy geometry ======================================
#geomeval = False
geomeval = True


# === numerical settings, sampling, etc. ==================
speedofsound = 343.0
c_sound = speedofsound*100.0 # cm/ms
f_s = 192000.0
sigma = 1.0 / (2*f_s)
dz = c_sound / f_s
# >>> measurement time:
measure_interval = 0.02
# resolution of the source (in cm):
src_res = 4.0

# >>> possible resolutions:
temp = [0.025, 0.05, 0.1,0.2, 0.25, 0.5, 1.0]
for x1,x2 in zip(temp[:-1], temp[1:]):
    # print x1,x2,dz
    if x1 < dz <= x2:
        dz = x1
        break
# >>> impulse response (time):        
n_tsamples = int(2**(math.ceil(numpy.log2(f_s * measure_interval))))
t_axis = numpy.linspace(0.0, (n_tsamples-1)/f_s, n_tsamples)
f_axis = numpy.fft.rfftfreq(n_tsamples, 1.0/f_s)
h_t = numpy.zeros(n_tsamples, dtype=float)
h2_t = numpy.zeros(n_tsamples, dtype=float)
print('  '+'='*60+'\n'+20*' '+' Diffraction\n'+'  '+'='*60+'\n')
print("   > Sample frequency: %.1f Hz" % f_s)
print("   > Time samples: %d " % n_tsamples)
print("   > dt: %f ms    df: %f Hz" % ((t_axis[1]-t_axis[0])*1000.0,f_axis[1]-f_axis[0]))
print("   > Edge resolution: %f cm" % (dz))

# === source and receiver =================================
x_s, y_s, z_s = 5.0, 10.0, 0.1
x_r, y_r, z_r = 0.0, 0.0, 250.0  
# receiver diameter (set zero for point source)
# DS = 2.54*6.5
DS = 0.0

# receiver plane (defined by two _unit_ vectors):
vs1, vs2 = [1.0,0.0,0.0], [0.0,1.0,0.0]
if DS < 1.0:
    rec_model = 'pointsource'    
    source, receiver = Point3D(x_s, y_s, z_s, evaluate=geomeval), Point3D(x_r, y_r, z_r, evaluate=geomeval)
    Nsource = 1    
else:
    rec_model = 'pointarray'
    receiver = Point3D(x_r, y_r, z_r,evaluate=geomeval)
    sources = sourcegrid([x_s, y_s, z_s], DS, src_res, numpy.array(vs1), numpy.array(vs2))
    Nsource = len(sources)

print('\n   > Number of pointsources: %d' % Nsource)


# === rectangle model =====================================
h = 60.0
w = 45.0
th = 2.0

R1 = Point3D(-w*0.5,-h*0.5, 0.0, evaluate=geomeval)
R2 = Point3D(+w*0.5,-h*0.5, 0.0, evaluate=geomeval)
R3 = Point3D(+w*0.5,+h*0.5, 0.0, evaluate=geomeval)
R4 = Point3D(-w*0.5,+h*0.5, 0.0, evaluate=geomeval)
R10 = Point3D(-w*0.5,-h*0.5, -th, evaluate=geomeval)
R20 = Point3D(+w*0.5,-h*0.5, -th, evaluate=geomeval)
R30 = Point3D(+w*0.5,+h*0.5, -th, evaluate=geomeval)
R40 = Point3D(-w*0.5,+h*0.5, -th, evaluate=geomeval)
# edges [edge, v1, v2]:
e1 = [Line3D(R1,R2), Ray3D(R2, R3), Ray3D(R1,R10)]
e2 = [Line3D(R2,R3), Ray3D(R3, R4), Ray3D(R2,R20)]
e3 = [Line3D(R3,R4), Ray3D(R4, R1), Ray3D(R3,R30)]
e4 = [Line3D(R4,R1), Ray3D(R1, R2), Ray3D(R4,R40)]
edges = [e1,e2,e3,e4]

# === plot the geometry of the setup =======================

quiveropts = dict(headlength=0, scale=1, scale_units='xy', headwidth=1, headaxislength=0)
if rec_model == 'pointsource':
    draw_xy([source], edges, receiver, quiveropts, DS)
elif rec_model == 'pointarray':
    draw_xy(sources, edges, receiver, quiveropts, DS)

    
# === infinite open baffle response ========================

# reading Thiele-Small parameters:
sp1 = read_ts_param('eminence15a.txt')
# Frequency response in frequency:
f_ax_log = numpy.logspace(numpy.log10(20.0), numpy.log10(20000.0), num=100)
speaker_resp_w = 10.0*numpy.log10(abs(inf_baffle_resp(sp1.Re, sp1.Le, sp1.Res, sp1.Lces, sp1.Cmes, f_ax_log)))
# Impulse response:
f_ax_lin = numpy.linspace(0.0, 40950.0, num=4096); f_ax_lin[0] = 0.001
t_ax = numpy.linspace(10.0, 40950.0+10.0, num=(4096-1)*2) # df = f_s/n
speaker_temp = numpy.nan_to_num(inf_baffle_resp(sp1.Re, sp1.Le, sp1.Res, sp1.Lces, sp1.Cmes, f_ax_lin))
speaker_resp_t = abs(numpy.fft.irfft(speaker_temp))

# === test speaker =========================================
# tmp = closedbox_reps(6.1, 0.11*0.001, 8.9, 7.5*0.001, 0.306*0.001, 1.5, 1.0, f_ax)
# I = 20.0*numpy.log10(abs(tmp))

# === first order diffraction ==============================
if rec_model == 'pointsource':
    tempx = ddelta(t_axis, float(source.distance(receiver)), sigma, 'gauss')
    h_t += tempx/f_s    
    for edge in edges:
        h_t += edge_v1(edge, source, receiver, dz, f_s, c_sound, n_tsamples)
elif rec_model == 'pointarray':
    for i, source in enumerate(sources):
        if i%10==0:
            print(i)
        tempx = ddelta(t_axis, float(source.distance(receiver)), sigma, 'gauss')
        h_t += tempx/f_s 
        for edge in edges:
            h_t += edge_v1(edge, source, receiver, dz, f_s, c_sound, n_tsamples)


# test second order diffraction =================================
# > > > paralell
sources = Point3D(5.0, 8.0, 0.0, evaluate=True)
edge1, edge2 = edges[0], edges[2]
# h2_t += edge_v1v2(edge1, edge2, source, receiver, dz, f_s, c_sound, n_tsamples)
'''            

for i, edge1 in enumerate(edges):
    for j, edge2 in enumerate(edges):
        if i == j:
            continue
        # first edge
        e,v1,v2 = edge1[0], edge1[1], edge1[2]
        teta_1w, teta_1R, teta_1S = edge_param(e,v1,v2, source, receiver)
        
        # second edge
        e,v1,v2 = edge2[0], edge2[1], edge2[2]
        teta_2w, teta_2R, teta_2S = edge_param(e,v1,v2, source, receiver)        
        
        bin_index =  list(map(int, map(round, f_s*(spath) / c_sound)))    
        dh = delta_h(c_sound, alpha, gamma, teta_R, teta_S, teta_w*3.0, rS, rR, zR, zax, dz)
        numpy.add.at(h_t, bin_index, dh)
'''

# === Fourier transform ====================================
h_f = abs(numpy.fft.rfft(h_t))


""" ===========================================================================
          Plotting 
    ===========================================================================
"""


plt.subplot(2,1,1)
plt.xlim((7, 10)) 
plt.plot(t_axis*1000.0, h_t, '-')
plt.title('Impulse response')
plt.xlabel('time (ms)')
plt.ylabel('amplitude')

plt.subplot(2,1,2)
plt.xlim((20, 20000))
plt.ylim((-15, 0))
plt.xscale('log')
plt.plot(f_axis, 10.0*numpy.log10(h_f/max(h_f)), '-')
plt.title(' ')
plt.xlabel('time (Hz)')
plt.ylabel('amplitude')
plt.show()

plt.subplot(2,1,1)
plt.plot(f_ax_log, speaker_resp_w, '-')
plt.xscale('log')
plt.grid(True)

plt.subplot(2,1,2)
plt.xlim((0, 500))
plt.plot(speaker_resp_t, '-')
plt.title(' ')
plt.show()

