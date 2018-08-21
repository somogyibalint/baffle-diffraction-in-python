# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 01:11:03 2018

@author: balint
"""
import numpy

class Speaker:
  def __init__(self, model, Fs, Re, Le, Qms, Qes, Qts, Vas, Vd, Cms, BL, Mms, EBP, 
               Xmax, Sd, Xlim, Cmes, Res, Lces):
    self.model = model
    self.Fs, self.Re, self.Le, self.Qms, self.Qes = Fs, Re, Le, Qms, Qes
    self.Qts, self.Vas, self.Vd, self.Cms, self.BL  = Qts, Vas, Vd, Cms, BL
    self.Mms, self.EBP, self.Xmax, self.Sd, =  Mms, EBP, Xmax, Sd
    self.Xlim, self.Cmes, self.Res, self.Lces = Xlim, Cmes, Res, Lces

def read_ts_param(filename):
    f = open('speakerdata/'+filename, 'r'); INPUT = f.readlines(); f.close()
    Fs, Re, Le, Qms, Qes, Qts, Vas, Vd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    Cms, BL, Mms, EBP, Xmax, Sd, Xlim = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    model = ''
    for line in INPUT:
        line = line.split()
        if line:
            if line[0] == 'model':
                model = line[1]
            elif line[0] == 'Fs': # Hz
                Fs = float(line[1])
            elif line[0] == 'Re': # ohm
                Re = float(line[1])
            elif line[0] == 'Le': # uH, mH, H
                if line[2] == 'uH':
                    Le = float(line[1])*0.000001
                elif line[2] == 'mH':
                    Le = float(line[1])*0.001
                elif line[2] == 'H':
                    Le = float(line[1])
                else:
                    raise ValueError('Incorrect unit for Le')
            elif line[0] == 'Qms':
                Qms = float(line[1])
            elif line[0] == 'Qes':
                Qes = float(line[1])
            elif line[0] == 'Qts':
                Qts = float(line[1])
            elif line[0] == 'Vas': # l, liters, cu.ft.
                if line[2] in ['l', 'liters', 'liter']:
                    Vas = float(line[1])
                elif line[2] in ['cu.ft.', 'cu.ft', 'cubic feet', 'cubic feets']:
                    Vas = float(line[1])*28.3168
                else:
                    raise ValueError('Incorrect unit for Vas') 
            elif line[0] == 'Vd':
                if line[2] in ['cc', 'cc.']:
                    Vd = float(line[1])
                elif line[2] in ['l', 'liters', 'liter']:
                    Vd = float(line[1])*1000.0
                else:
                    raise ValueError('Incorrect unit for Vd')
            elif line[0] == 'Cms':
                if line[2] in ['mm/N']:
                    Cms = float(line[1])
                elif line[2] in ['cm/N']:
                    Cms = float(line[1])*10.0
                else:
                    raise ValueError('Incorrect unit for Cms')
            elif line[0] == 'BL':
                BL = float(line[1])
            elif line[0] == 'Mms': # grams
                if line[2] in ['g', 'gram', 'grams']:
                    Mms = float(line[1])
                else:
                    raise ValueError('Incorrect unit for Mms')                    
            elif line[0] == 'EBP':
                EBP = float(line[1])
            elif line[0] == 'Xmax': # [mm]
                Xmax = float(line[1])
            elif line[0] == 'Sd':
                if line[2] in ['cm2']:                
                    Sd = float(line[1])
                else:
                    raise ValueError('Incorrect unit for Cms')                    
            elif line[0] == 'Xlim': # [mm]
                Xlim = float(line[1])
    # derived parameters:
    Cmes = Qes / (2.0*numpy.pi*Fs*Re)
    Res = Qms / (2.0*numpy.pi*Fs*Cmes)
    Lces = 1.0 / (Fs**2.0 * Cmes*4.0*(numpy.pi)**2.0)
    return Speaker(model, Fs, Re, Le, Qms, Qes, Qts, Vas, Vd, Cms, BL, Mms, EBP, 
                   Xmax, Sd, Xlim, Cmes, Res, Lces)