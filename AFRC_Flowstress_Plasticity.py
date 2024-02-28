import numpy as np
import pandas as pd
import subprocess
import os
import time
from sys import argv
from scipy.stats import qmc

print(argv)

Temp_C = float(argv[1])
SR = float(argv[2])
βtransus = float(argv[3])
initial_stress_min = float(argv[4])
initial_stress_max = float(argv[5])
Q_min = float(argv[6])
Q_max = float(argv[7])
m_α_min = float(argv[8])
m_α_max = float(argv[9])
m_β_min = float(argv[10])
m_β_max = float(argv[11])
n_α_min = float(argv[12])
n_α_max = float(argv[13])
n_β_min = float(argv[14])
n_β_max =float(argv[15])
Q_β_min = float(argv[16])
Q_β_max = float(argv[17])
Q_α_min = float(argv[18])
Q_α_max = float(argv[19])
B_min = float(argv[20])
B_max = float(argv[21])
a_min = float(argv[22])
a_max = float(argv[23])
b_min = float(argv[24])
b_max = float(argv[25])
runs = int(argv[26])
friction = float(argv[27])
conductance = [float(argv[28]), argv[29]]
power = float(argv[30])

alloy = {"Al" : 6.31,
         "V" : 4.1,
         "Fe" : 0.16,
         "O" : 0.18,
         "C" : 0.031,
         "N" : 0.006}

def model_wrapper(strain, params, alloy, Temp_C, SR, βtransus):
    #This function serves as a wrapper for the parameters, so that the flow 
    #stress model can be easily copied into other scripts where variable
    #parameters are distinguished
    initial_stress,Q,m_α,m_β,n_α,n_β,Q_β,Q_α,B,a,b = params
    flow_stress = Flow_stress_model(strain,
                      alloy,
                      Temp_C,
                      SR,
                      βtransus,
                      initial_stress,
                      Q,
                      m_α,
                      m_β,
                      n_α,
                      n_β,
                      Q_β,
                      Q_α,
                      B,
                      a,
                      b)
    return flow_stress

def Flow_stress_model(strain,
                      alloy,
                      Temp_C,
                      SR,
                      βtransus,
                      initial_stress,
                      Q,
                      m_α,
                      m_β,
                      n_α,
                      n_β,
                      Q_β,
                      Q_α,
                      B,
                      a,
                      b):
    R = 8.3145
    Temp_K = Temp_C + 273
    Z = SR * np.exp(Q/(R*Temp_K)) #Zener Holloman
    m_Avg = np.mean([0.22,0.24])
    V_eqv = alloy["V"] + alloy["Al"] * 0.27
    K_α = 10 ** ((0.37 * alloy["Al"]) - 3.375)
    K_β = 10 ** ((0.000718*(V_eqv**3)) - (0.0316*(V_eqv**2))
                                     + (0.555*V_eqv) - 1.483)
    α_fraction = (1-(0.925 * np.exp(-0.0085*(βtransus-Temp_C))
                 + 0.075))*(Temp_C < βtransus)
    σ_46_α = K_α * (np.exp(Q_α/(R*Temp_K))) * SR
    σ_42_β = K_β * (np.exp(Q_β/(R*Temp_K))) * SR
    σ_α = np.exp(np.log(σ_46_α)/n_α)
    σ_β = np.exp(np.log(σ_42_β)/n_β)
    ratio_kα_kβ = (σ_α/(SR ** m_α))/(σ_β/(SR ** m_β))
    ratio_kβ_kα = 1/ratio_kα_kβ
    ρ = np.linspace(0.11,1,90)
    kLsc_kL1 =(1/6)*((3 - 2 * ρ) + (5 * (1 - α_fraction) * (ρ - 1)) 
                + ((((3 - (2*ρ)+(5*(1-α_fraction) * (ρ - 1)))) ** 2)
                + (24 * ρ))**0.5)
    k_k1 = ((kLsc_kL1 ** ((m_Avg + 1)/2))
        * ((α_fraction +(1 + α_fraction)
        * (ρ**((m_Avg + 1)/(m_Avg - 1)))
        *(ratio_kβ_kα ** (2/(1 - m_Avg))))
        ** ((1 - m_Avg)/2)))
    k_kα = np.min(k_k1)
    kα = σ_α / (SR ** m_α)
    kβ = σ_β / (SR ** m_β)
    ϵ̇1_ϵ̇ov = np.exp((1 / m_Avg) * np.log(((k_kα-(((1-α_fraction)**(1-m_Avg))
                * ratio_kβ_kα))/(α_fraction * (1-(((1-α_fraction)**(1-m_Avg))
                                                 *ratio_kβ_kα))))))
    ϵ̇2_ϵ̇ov = (1 - (α_fraction * ϵ̇1_ϵ̇ov)) / (1 - α_fraction)
    σov = k_kα * kα * (SR ** m_Avg)
    Sat_stress = σov
    Peak_stress = Sat_stress
    peak_strain = 4e-9 * np.log10(Z) ** 5.0931
    steady_state_stress = np.max([5, 210.78 + (-49.527 * np.log10(Z))+ 1.79 * (np.log10(Z)**2)])
    sig1 = np.sqrt((Peak_stress ** 2) 
        + ((initial_stress ** 2) 
            - (Peak_stress ** 2)) 
        * np.exp(-2 * B * strain))
    X_vals = [(1 - np.exp(-a*((s-peak_strain) ** b))) for s in strain if s > peak_strain]
    X = np.zeros(len(strain))
    with open("output.txt", "a") as f:
        f.write(f"with initial_stress={initial_stress}, Q={Q}, m_α={m_α}, m_β={m_β}, n_α={n_α}, n_β={n_β}, Q_β={Q_β}, B={B}, a={a}, b={b}\n")
        f.write(f"X_vals: {X_vals}")
    f.close()
    if X_vals: #Carries out the next operation if the list is not empty
        X[-len(X_vals):] = X_vals
    flow_stress = (sig1-X*(sig1-steady_state_stress))
    with open("output.txt", "a") as f:
        f.write(f"Flow stress outputs: {flow_stress}\n")
    f.close()
    return flow_stress



def Abaqus_plastic_table_AFRC(params, 
                         alloy, 
                         Temp_C, 
                         βtransus, 
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
            fs = model_wrapper(strain, params, alloy, Temp_C, SR, βtransus)
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
def call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,count,alloy,Temp_C,SR,βtransus):
    #This function calls the 'generate_input_file' function to create a model with randomised parameters
    #It's purpose is is call abaqus with the compression test model, then call abaqus cae to interpret the
    #output data base. The sub process generates a text file with force values vs time step called 'force_output.txt'
    #which is then read and returned as the output of this function
    #The list_of_material_coefficients is a numpy array of randomised multiples of material data, original_inp_file is a 
    #string locating the inp file, and output directory is where the .odb file is to be placed
    #print('abaqus function called')
    output_filename='Doesitwork'
    input_file = generate_input_file_AFRC_plasticity(list_of_material_coefficients, original_inp_file, alloy, Temp_C, SR, βtransus)
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
def generate_input_file_AFRC_plasticity(parameters, inp_file, alloy, Temp_C, SR, βtransus):
    inp_data = open(inp_file).read().split('\n')
    plastic_data, plastic_start, plastic_end = Extract_plastic_data(inp_data)
    new_plasticity_data = Abaqus_plastic_table_AFRC(parameters, 
                             alloy, 
                             Temp_C, 
                             βtransus)
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

hypercube_obj = qmc.LatinHypercube(d=11)
samples = hypercube_obj.random(runs)

lower_bounds = [initial_stress_min, 
                Q_min,
                m_α_min,
                m_β_min,
                n_α_min,
                n_β_min,
                Q_β_min,
                Q_α_min,
                B_min,
                a_min,
                b_min]

upper_bounds = [initial_stress_max, 
                Q_max,
                m_α_max,
                m_β_max,
                n_α_max,
                n_β_max,
                Q_β_max,
                Q_α_max,
                B_max,
                a_max,
                b_max]
difference = np.array(upper_bounds)-np.array(lower_bounds)
with open("output.txt","w") as f:
    f.write(f"input args: {argv}\n")
    f.write(f"Lower bounds: {lower_bounds}\n")
    f.write(f"Upper bounds: {upper_bounds}\n")
    f.write(f"difference: {difference}\n")
f.close()
scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

starting_time = time.time()


#results = {'power input': np.zeros(no_samples) 'coefficient of friction':np.zeros(no_samples), 'platen sample interface conductance':np.zeros(no_samples),'Force Results':np.zeros(no_samples)}
results = {'initial stress':np.zeros(runs), 
           'Q':np.zeros(runs), 
           'm_α':np.zeros(runs), 
           'm_β':np.zeros(runs), 
           'n_α':np.zeros(runs), 
           'n_β':np.zeros(runs),
           'Q_β':np.zeros(runs),
           'Q_α':np.zeros(runs), 
           'B':np.zeros(runs),
           'a':np.zeros(runs),
           'b':np.zeros(runs),
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
original_inp_file = 'AFRC_plasticity.inp'
generate_input_file_friction_conductance_power([friction,conductance,power], original_inp_file)

for n, list_of_material_coefficients in enumerate(scaled_samples):
    results_df.loc[n,'initial stress'] = list_of_material_coefficients[0]
    results_df.loc[n,'Q'] = list_of_material_coefficients[1]
    results_df.loc[n,'m_α'] = list_of_material_coefficients[2]
    results_df.loc[n,'m_β'] = list_of_material_coefficients[3]
    results_df.loc[n,'n_α'] = list_of_material_coefficients[4]
    results_df.loc[n,'n_β'] = list_of_material_coefficients[5]
    results_df.loc[n,'Q_β'] = list_of_material_coefficients[6]
    results_df.loc[n,'Q_α'] = list_of_material_coefficients[7]
    results_df.loc[n,'B'] = list_of_material_coefficients[8]
    results_df.loc[n,'a'] = list_of_material_coefficients[9]
    results_df.loc[n,'b'] = list_of_material_coefficients[10]
    call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory, n, alloy, Temp_C , SR, βtransus)
    results_df.to_pickle('AFRC_plasticity.pkl')
    


results_df.to_pickle('AFRC_plasticity.pkl')
