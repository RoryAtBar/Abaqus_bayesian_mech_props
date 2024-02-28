#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:38:59 2024

@author: w10944rb
"""

import numpy as np
import pandas as pd
import subprocess
import os
import time
from sys import argv

index = int(argv[1])

results_df = pd.read_pickle('Quad_plasticity.pkl')

def model_wrapper(strain, params, SR, T):
    #This function serves as a wrapper for the parameters, so that the flow 
    #stress model can be easily copied into other scripts where variable
    #parameters are distinguished
    C0, C1, C2, C3, C4, C5 = params
    flow_stress = Flow_stress_model(strain,
                      C0,
                      C1,
                      C2,
                      C3,
                      C4,
                      C5,
                      SR,
                      T)
    return flow_stress

def Flow_stress_model(strain,
                      C0,
                      C1,
                      C2,
                      C3,
                      C4,
                      C5,
                      SR,
                      T):
    log_SR = np.log10(SR)
    stress = C0 + C1*T + C2*log_SR + C3*T*log_SR + C4*(T**2) + C5*(log_SR**2)
    flow_stress = np.ones(len(strain)) * stress
    return flow_stress



def Abaqus_plastic_table_quadratic(params,
                         strain=np.linspace(0,2,201), 
                         plastic_rate=[0.001,0.01,0.1,1.0,10.0],
                         Temperature = np.linspace(600,1100,11)):
    Out = []
    for SR in plastic_rate:
        if SR == np.min(plastic_rate):
            Out.append("*Plastic, rate=0.")
        else:
            Out.append(f"*Plastic, rate={SR}")
        for Temp_C in Temperature:
            fs = model_wrapper(strain, params, SR, Temp_C)
            flow_curve = np.ones((len(strain),3))
            flow_curve[:,0] = fs
            flow_curve[:,1] = strain
            flow_curve[:,2] *= Temp_C
            for row in flow_curve:
                Out.append(f"{row[0]:.2f}, {row[1]:.2f}, {row[2]}")
    return Out



subprocess.run(['pwd'])
#Version 2 uses a modified version of read_Force_PEEQ_NT11_barrelling based on a macro
#Version 3 imports a different odb for each value of platen conductance
def call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,count,SR, Temp_C):
    #This function calls the 'generate_input_file' function to create a model with randomised parameters
    #It's purpose is is call abaqus with the compression test model, then call abaqus cae to interpret the
    #output data base. The sub process generates a text file with force values vs time step called 'force_output.txt'
    #which is then read and returned as the output of this function
    #The list_of_material_coefficients is a numpy array of randomised multiples of material data, original_inp_file is a 
    #string locating the inp file, and output directory is where the .odb file is to be placed
    #print('abaqus function called')
    output_filename='Doesitwork'
    input_file = generate_input_file_quadratic_plasticity(list_of_material_coefficients, original_inp_file, SR, Temp_C)
    Run_Abaqus = subprocess.run(['abq2022','job=sub_script_check', 'input='+original_inp_file, 'interactive'])
    read_odb_into_text_file = subprocess.run(['abq2022','cae', 'noGUI=read_Force_PEEQ_NT11_barrelling_forcemac.py'])
    subprocess.run(['ls','-l'])
    file_count = str(count)
    try:
        with open('Force_sample_set1.rpt','r') as f:
            force_vals1=f.read().split('\n')[:-1]
        f.close()
        with open('Force_sample_set2.rpt','r') as f:
            force_vals2=f.read().split('\n')[:-1]
        f.close()
    #compression_force = np.zeros(len(force_vals))
        with open('PEEQ_output.rpt','r') as f:
            PEEQ_vals=f.read().split('\n')[:-1]
        f.close()
        with open('outer_sample_xcoords.rpt','r') as f:
            barrelling_profile=f.read().split('\n')[:-1]
        f.close()
        with open('NT11.rpt','r') as f:
            NT11=f.read().split('\n')[:-1]
        f.close()    
        #for i, force in enumerate(force_vals):
        #    compression_force[i] = float(force)
        #print('abaqus function completed')
        results_df.at[count,'Force Results1'] = force_vals1
        results_df.at[count,'Force Results2'] = force_vals2
        results_df.at[count,'Barrelling Profile'] = barrelling_profile
        results_df.at[count,'PEEQ Results'] = PEEQ_vals
        results_df.at[count,'Temperature profile'] = NT11
        subprocess.run(['mv','PEEQ_output.rpt',f'PEEQ_output{file_count}.rpt'])
        subprocess.run(['mv','outer_sample_xcoords.rpt',f'outer_sample_xcoords{file_count}.rpt'])
        subprocess.run(['mv','NT11.rpt',f'NT11_{file_count}.rpt'])
        subprocess.run(['mv','Force_sample_set1.rpt',f'Force_sample_set1{file_count}.rpt'])
        subprocess.run(['mv','Force_sample_set2.rpt',f'Force_sample_set2{file_count}.rpt'])
    except FileNotFoundError:
        results_df.at[count,'Force Results1'] = float('NaN')
        results_df.at[count,'Force Results2'] = float('NaN')
        results_df.at[count,'Barrelling Profile'] = float('NaN')
        results_df.at[count,'PEEQ Results'] = float('NaN')
    if np.random.rand() > 0.95:
        subprocess.run(['mv','sub_script_check.odb',f'{file_count}.odb'])
        subprocess.run(['cp','AFRC_plasticity.inp',f'{file_count}.inp'])
    subprocess.run(['rm','sub_script_check*'])
    #return compression_force

def modify_friction(inp_text, coefficient_of_friction):
    new_text = inp_text
    replacement_line = ' '+str(coefficient_of_friction)+','
    for n,line in enumerate(inp_text):
        if line == '*Surface Interaction, name=FRICTION':
            new_text[n+3] = replacement_line
    return new_text

def modify_platen_conductance(inp_text, platen_conductance):
    new_text = inp_text
    replacement_line = str(platen_conductance[0])+',    0.'
    next_line = (len(replacement_line)- len('0., 0.001'))*' '
    for n,line in enumerate(inp_text):
        if line == '*Surface Interaction, name=SAMPLE_PLATEN_CONDUCTANCE':
            new_text[n+3] = replacement_line
            new_text[n+4] = next_line + '0., 0.001'
    curworkdir = os.getcwd()        
    for n,line in enumerate(new_text):
        if line == "** Name: Predefined Field-2   Type: Temperature":
            new_text[n+1] = f"*Initial Conditions, type=TEMPERATURE, file={curworkdir}/{platen_conductance[1]}, step=1, inc=0, interpolate"
    return new_text

def modify_power(inp_text, p):
    new_text = inp_text
    for n,line in enumerate(inp_text):

        if line == '*Amplitude, name=POWER':
            new_text[n+1] = f'             0., {p},             0.5, {p},             2.5, {p},              3., {p}'
            new_text[n+2] = f'             5., {p},             5.5, {p},             7.5, {p},              8., {p}'
            new_text[n+3] = f'            10., {p},            10.5, {p},            12.5, {p},             13., {p}'
            new_text[n+4] = f'            15., {p},            15.5, {p},            17.5, {p},             18., {p}'
            new_text[n+5] = f'            20., {p},            20.5, {p},            22.5, {p},             23., {p}'
            new_text[n+6] = f'            25., {p},            25.5, {p},            27.5, {p},             28., {p}'
            new_text[n+7] = f'            30., {p},            30.5, {p},            32.5, {p},             33., {p}'
            new_text[n+8] = f'            35., {p},            35.5, {p},            37.5, {p},             38., {p}'
            new_text[n+9] = f'            40., {p},            40.5, {p},            42.5, {p},             43., {p}'
            new_text[n+10] = f'            45., {p},            45.5, {p},            47.5, {p},             48., {p}'
            new_text[n+11] = f'            50., {p}'
    return new_text            

#Main function for generating inp files. This function organises the above functions.
#It takes the location of the inp file, and a seperate file with a plasticity data lookup table in the 
#same format as the inp file, reads them and feeds them through all of the above functions in order.
def generate_input_file_friction_conductance_power(parameters, inp_file):
    inp_data = open(inp_file).read().split('\n')
    inp_data = modify_friction(inp_data, parameters[0])
    inp_data = modify_platen_conductance(inp_data, parameters[1])
    inp_data = modify_power(inp_data, parameters[2])
    #print(new_plasticity_data==plasticity_data_table)
    #print(list_of_material_coefficients)
    new_inp = ''
    for line in inp_data:
        new_inp += line + '\n'
    with open(inp_file,'w') as f:
        f.write(new_inp)
    f.close()
    return new_inp

def model_sensitivity_lib_format(Friction_Coefficient,Sample_Platen_Thermal_Conductivity):
    #This function is to recycle existing code into the sensitivity library
    parameters = [Friction_Coefficient,Sample_Platen_Thermal_Conductivity]
    inp_file = 'AFRC_plasticity.inp'
    output_directory = ''
    comp_force = call_abaqus_with_new_params(parameters, inp_file, output_directory)
    return comp_force


def Extract_plastic_data(inp_file):
    #The compression test inp file is opened and read before being passed to this function
    #It finds the lines where the plasticity data is kept within the original inp file and returns the original
    #lookup table of plasticity data (which is not used in this iteration of script), but more importantly
    #finds the index numbers of the lines where the plasticity data starts and ends
    Titanium_sample = False
    for n, line in enumerate(inp_file):
        if line == '**  Titanium Sample':
            Titanium_sample = True
        if line == '*Plastic, rate=0.' and Titanium_sample:
            Plastic_start = n
        if line == '*Specific Heat' and Titanium_sample and Plastic_start > 0:
            Plastic_end = n
    return inp_file[Plastic_start:Plastic_end], Plastic_start, Plastic_end

    

def feed_modified_table_into_inp(converted_new_plastic_data, old_inp_file, plastic_start, plastic_end, inp_filename):
    #Takes the randomised material data and inserts it into the compression model. Converted plastic data is the plasticity
    #data after it has been fed back through convert_table_back_to_inp. old_inp_file is the original input file including the old plasticity data
    #plastic start and end are the indices of the lines where the plasticity data in the inp file needs to be replaced
    #with the randomised data. Get this from Extract_plastic_data
    new_inp = old_inp_file[:plastic_start]+converted_new_plastic_data+old_inp_file[plastic_end:]
#    for line in old_inp_file[:plastic_start]:
#        new_inp += line + '\n'
#    for line in converted_new_plastic_data:
#        new_inp += line +'\n'
#    for line in old_inp_file[plastic_end:]:
#        new_inp += line + '\n'
    new_file = open(inp_filename+'.inp','w')
    for line in new_inp:
        new_file.write(f"{line}\n")
    new_file.close()
    return new_inp

def modify_array_constitutive_flow_stress(list_of_coeffs, strains, plastic_data_tabled):
    keys = plastic_data_tabled.keys()
    new_plastic_data = plastic_data_tabled
    for i, s in enumerate(strains):
        for key in keys:
            for i in range(len(plastic_data_tabled[key])):
                if float(key[11:]) == 0:
                    log_SR = 0# log of strain rate
                else:
                    log_SR = np.log(float(key[11:]))
                S = plastic_data_tabled[key][i][1] #strain
                T = plastic_data_tabled[key][i][2] + 273#Temperature
                alph, n, Qu, logA = list_of_coeffs
                stress = np.arcsinh(np.exp(((log_SR/n)+(Qu/(n*8.314*T))-(logA/n))))/alph
                new_plastic_data[key][i][0] = stress
    return new_plastic_data
            

#Main function for generating inp files. This function organises the above functions.
#It takes the location of the inp file, and a seperate file with a plasticity data lookup table in the 
#same format as the inp file, reads them and feeds them through all of the above functions in order.
def generate_input_file_quadratic_plasticity(parameters, inp_file, SR, Temp_C,):
    inp_data = open(inp_file).read().split('\n')
    plastic_data, plastic_start, plastic_end = Extract_plastic_data(inp_data)
    new_plasticity_data = Abaqus_plastic_table_quadratic(parameters)
    new_inp=feed_modified_table_into_inp(new_plasticity_data, inp_data, plastic_start, plastic_end, 'AFRC_plasticity')
    return new_inp

def convert_table_back_to_inp(plastic_data_table):
    #converts the dict of flow stress numpy arrays under each strain rate back into a string so that it may 
    #be inserted into the model inp file
    keys = plastic_data_table.keys()
    num_cells = 0
    for key in keys:
        num_cells += len(plastic_data_table[key])
    text_inp = [None] * (num_cells + len(keys))
    i = 0
    for key in keys:
        text_inp[i] = '*Plastic, rate=' + key[11:]
        i += 1
        for x,y,z in plastic_data_table[key]:
            text_inp[i] = ' '+str(x) + ', ' + str(y) + ', ' +str(z)
            i += 1
    return text_inp

list_of_material_coefficients = [results_df["C0"][index],
                                 results_df["C1"][index],
                                 results_df["C2"][index],
                                 results_df["C3"][index],
                                 results_df["C4"][index],
                                 results_df["C5"][index]]
original_inp_file = 'Quad_plasticity.inp'
output_directory = ''
friction = results_df["Friction"][index]
conductance = [results_df["Conductance"][index], results_df["Conductance input file"][index]]
power = results_df["Power"][index]
generate_input_file_friction_conductance_power([friction,conductance,power], original_inp_file)

call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,index)