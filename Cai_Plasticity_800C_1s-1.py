import numpy as np
import pandas as pd
import subprocess
import os
import time
from sys import argv
from scipy.stats import qmc

print(argv)

alpha_min = float(argv[1])
alpha_max = float(argv[2])
n_param_min = float(argv[3])
n_param_max = float(argv[4])
Q_min = float(argv[5])
Q_max = float(argv[6])
ln_A_min = float(argv[7])
ln_A_max = float(argv[8])
runs = int(argv[9])
friction = float(argv[10])
conductance = [float(argv[11]), argv[12]]
power = float(argv[13])

subprocess.run(['pwd'])
#Version 2 uses a modified version of read_Force_PEEQ_NT11_barrelling based on a macro
#Version 3 imports a different odb for each value of platen conductance
def call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,count):
    #This function calls the 'generate_input_file' function to create a model with randomised parameters
    #It's purpose is is call abaqus with the compression test model, then call abaqus cae to interpret the
    #output data base. The sub process generates a text file with force values vs time step called 'force_output.txt'
    #which is then read and returned as the output of this function
    #The list_of_material_coefficients is a numpy array of randomised multiples of material data, original_inp_file is a 
    #string locating the inp file, and output directory is where the .odb file is to be placed
    #print('abaqus function called')
    output_filename='Doesitwork'
    input_file = generate_input_file_quadratic_surf(list_of_material_coefficients, original_inp_file, Titanium_plasticity_data)
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
    if np.random.rand() > 0.9:
        subprocess.run(['mv','sub_script_check.odb',f'{file_count}.odb'])
        subprocess.run(['cp','Quad_plasticity.inp',f'{file_count}.inp'])
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
    inp_file = 'Quad_plasticity.inp'
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

def convert_table_to_numbers(table):
    #The plasticity data after being split into individual strain rates is passed to this function to 
    #be converted into a numpy array
    new_table = np.zeros((len(table),3))
    for n, line in enumerate(table):
        split_line = line.split(',')
        new_table[n,0] = float(split_line[0])
        new_table[n,1] = float(split_line[1])
        new_table[n,2] = float(split_line[2])
    return new_table
    
def separate_plastic_table_by_strain_rate(plastic):
    #After the plasticity data is read and split into lines, it is broken down into a dict with the strain
    #rate as the key, converted to a numpy array of [flow stress, temperature, strain] under each strain rate
    single_strain_rate_data = {}
    strain_rate = None
    first_line_given_strain_rate = 0
    for n,line in enumerate(plastic):   
        if (strain_rate == None) & (line[:15] == '*Plastic, rate='):
            strain_rate = line[15:]
            first_line_given_strain_rate = n+1
        elif line[:15] == '*Plastic, rate=':
            new_strain_rate = line[15:]
            last_line_given_strain_rate = n-1
            single_strain_rate_data['strain rate'+strain_rate]=convert_table_to_numbers(plastic[first_line_given_strain_rate:last_line_given_strain_rate])
            strain_rate = new_strain_rate
            first_line_given_strain_rate = n+1
        elif n+2 > len(plastic):
            last_line_given_strain_rate = n+1
            single_strain_rate_data['strain rate'+strain_rate]=convert_table_to_numbers(plastic[first_line_given_strain_rate:last_line_given_strain_rate])
    return single_strain_rate_data

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

def modify_array_quad_surf(list_of_coeffs, strains, plastic_data_tabled):
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
def generate_input_file_quadratic_surf(parameters, inp_file, material_data_txt):
    inp_data = open(inp_file).read().split('\n')
    input_plastic_data = open(material_data_txt).read().split('\n')
    plastic_data, plastic_start, plastic_end = Extract_plastic_data(inp_data)
    plasticity_data_table = separate_plastic_table_by_strain_rate(input_plastic_data)
    strains = list(set(plasticity_data_table['strain rate0.'][:,1]))
    new_plasticity_data = modify_array_quad_surf(parameters, strains, plasticity_data_table)
    #print(new_plasticity_data==plasticity_data_table)
    #print(list_of_material_coefficients)
    new_plasticity_data_in_inp_format = convert_table_back_to_inp(new_plasticity_data)
    new_inp=feed_modified_table_into_inp(new_plasticity_data_in_inp_format, inp_data, plastic_start, plastic_end, 'Quad_plasticity')
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

Titanium_plasticity_data = 'Patryk_mat_data.txt'
hypercube_obj = qmc.LatinHypercube(d=4)
samples = hypercube_obj.random(runs)

lower_bounds = [alpha_min,n_param_min,Q_min,ln_A_min]
upper_bounds = [alpha_max,n_param_max,Q_max,ln_A_max]

scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

starting_time = time.time()


#results = {'power input': np.zeros(no_samples) 'coefficient of friction':np.zeros(no_samples), 'platen sample interface conductance':np.zeros(no_samples),'Force Results':np.zeros(no_samples)}
results = {'Alpha':np.zeros(runs), 
           'n':np.zeros(runs), 
           'Q':np.zeros(runs), 
           'ln_A':np.zeros(runs), 
           'Force Results1':np.zeros(runs), 
           'Force Results2':np.zeros(runs),
           'PEEQ Results':np.zeros(runs), 
           'Barrelling Profile':np.zeros(runs),
           'Temperature profile':np.zeros(runs)}
results_df = pd.DataFrame(results)
results_df['Force Results1'] = results_df['Force Results1'].astype(object)
results_df['Force Results2'] = results_df['Force Results2'].astype(object)
results_df['Barrelling Profile'] = results_df['Barrelling Profile'].astype(object)
results_df['PEEQ Results'] = results_df['PEEQ Results'].astype(object)
results_df['Temperature profile'] = results_df['Temperature profile'].astype(object)


subprocess.run(['rm','sub_script_check*'])


output_directory = ''
original_inp_file = 'Quad_plasticity.inp'
generate_input_file_friction_conductance_power([friction,conductance,power], original_inp_file)

for n, list_of_material_coefficients in enumerate(scaled_samples):
    results_df.loc[n,'Alpha'] = list_of_material_coefficients[0]
    results_df.loc[n,'n'] = list_of_material_coefficients[1]
    results_df.loc[n,'Q'] = list_of_material_coefficients[2]
    results_df.loc[n,'ln_A'] = list_of_material_coefficients[3]
    call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,n)
    


results_df.to_pickle('Quad_plasticity.pkl')
