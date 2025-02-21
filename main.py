# Create a spectral library
import numpy as np
import pandas as pd
from pepfrag import MassType, ModSite, Peptide
from pepfrag import IonType
from pyteomics import parser
from keras.models import model_from_json
from sklearn import linear_model


def mz_from_mass(mass, charge):

    # Calculate the mass to charge
    mz = (mass + (1.007276466879 * charge)) / charge

    return mz


def load_model(model_json, model_h5):

    # Open the json file with the model
    json_file = open(model_json, 'r')
    # Read in the json file
    l_loaded_model_json = json_file.read()
    # CLose the json file
    json_file.close()
    # Load the model structure from the json file
    l_loaded_model = model_from_json(l_loaded_model_json)
    # Load weights into new model
    l_loaded_model.load_weights(model_h5)
    # Print the model has been loaded
    print("Loaded model from disk")

    return l_loaded_model


def skip_peptide(peptide):
    # Define the modifications that will not be compatible
    skip_mods = ['-17.0265', '42.0106', '119.0041', '-18.0106', '304.2071']

    # Determine if we will want to skip this peptide due to incompatible modification
    skip = False
    for mod in skip_mods:
        if mod in peptide:
            skip = True

    return skip


def create_peptide(peptide, charge=3, tmt=False):
    # Clean the peptide
    clean_pep = peptide.replace('M[15.9949]', 'B')
    clean_pep = clean_pep.replace('M(ox)', 'B')
    clean_pep = clean_pep.replace('C[57.0215]', 'U')
    clean_pep = clean_pep.replace('R[10.0082]', 'O')
    clean_pep = clean_pep.replace('K[8.0142]', 'Z')

    if '.' in clean_pep:
        clean_pep = clean_pep.split('.')[1]

    mod_list = []
    residue = 1
    rebuild_pep = ""

    if tmt:
        mod_list.append(ModSite(304.2071, "nterm", "TMTpro"))

    for char in clean_pep:
        if char == 'K' and tmt:
            mod_list.append(ModSite(304.2071, residue, "TMTpro"))
            rebuild_pep += 'K'
        elif char == 'C':
            mod_list.append(ModSite(57.02146, residue, "CAM"))
            rebuild_pep += 'C'
        elif char == 'B':
            mod_list.append(ModSite(15.994915, residue, "Ox"))
            rebuild_pep += 'M'
        elif char == 'U':
            mod_list.append(ModSite(57.02146, residue, "CAM"))
            rebuild_pep += 'C'
        elif char == 'O':
            mod_list.append(ModSite(10.0082, residue, "HeavyR"))
            rebuild_pep += 'R'
        elif char == 'Z':
            mod_list.append(ModSite(312.2213, residue, "HeavyK"))
            rebuild_pep += 'K'
        else:
            rebuild_pep += char

        residue += 1

    peptide_obj = Peptide(
        rebuild_pep,
        charge,
        mod_list,
        mass_type=MassType.mono
    )

    frag_dict = {}
    for frag in peptide_obj.fragment(ion_types={IonType.b: [], IonType.y: []}):
        frag_dict[frag[1]] = frag[0]

    return peptide_obj, frag_dict


def get_experimental_int_dict(frag_dict, spectrum, tol=20.0, tol_type='PPM', normalize=False):
    # The dictionary to save the extracted intensities
    frag_int_dict = {}


    # Save the spectrum values for more clear access
    spectrum_mz = spectrum['mz']
    spectrum_int = spectrum['int']

    # Save the base peak information
    base_peak_mz = -1
    base_peak_int = -1

    # Save the total TIC
    spectrum_tic = sum(spectrum['int'])
    matched_tic = 0
    tallest_frag_int = -1

    # Iterate though the fragments and look for the mz in the
    for frag, mz in frag_dict.items():
        # Set the rang for the fragment ion to match
        low_mz = mz - (mz * (tol * 0.000001))
        high_mz = mz + (mz * (tol * 0.000001))

        # If the tolerance is in Da then change it here
        if tol_type == 'DA':
            low_mz = mz - tol
            high_mz = mz + tol

        # Next we want to extract the peak that matches this fragment
        tallest_mz = -1
        tallest_int = -1

        # This will get the index of the value near the low_mz
        peak_index = np.searchsorted(spectrum_mz, low_mz)

        # Do a quick check to make sure that the index is within the bound
        if 0 <= peak_index < len(spectrum_mz):

            # If it is then grab the values at the index
            current_mz = spectrum_mz[peak_index]
            current_int = spectrum_int[peak_index]

            # Keep checking values until the mz is higher than the upper limit
            while current_mz <= high_mz:
                if current_int > tallest_int:
                    tallest_mz = current_mz
                    tallest_int = current_int

                    # Save the base peak for final normalization
                    if tallest_int > base_peak_int:
                        base_peak_mz = tallest_mz
                        base_peak_int = tallest_int

                # Increment the counter
                peak_index += 1

                # Check the bounds again just to make sure before looping again
                if peak_index < 0 or peak_index >= len(spectrum_mz):
                    break

        # If we didn't find a peak just add a 0
        if tallest_mz == -1:
            frag_int_dict[frag] = 0
        # If we did find a peak then add the tallest intensity
        else:
            frag_int_dict[frag] = tallest_int
            matched_tic += tallest_int

            if tallest_int > tallest_frag_int:
                tallest_frag_int = tallest_int

    # Calculate the fragment percent of the tic
    matched_percent_tic = matched_tic / spectrum_tic

    # If we want to normalize then divide every peak by the base peak
    return_frag_int_dict = {}
    if normalize and not tallest_frag_int == -1:
        for frag, peak_int in frag_int_dict.items():
            return_frag_int_dict[frag] = peak_int / tallest_frag_int
    else:
        return_frag_int_dict = frag_int_dict

    # Return the frag intensity dictionary
    return return_frag_int_dict, matched_percent_tic


def get_frag_ion_array():
    # Set up the dimensions that will make up the array
    ions = ['y', 'b']
    charges = ['+', '2+', '3+']
    ion_nums = list(np.arange(1, 30))

    # Arrays that we will be filling out
    ion_key_array = []

    # Do all the position = 1 ions first
    for ion_num in ion_nums:
        # Do all the y1 ions before b1 ions
        for ion in ions:
            # Group the charge states for y1 ions first
            for charge in charges:
                # These ions keys should match up to the fragment dictionary we made before
                ion_key = str(ion) + str(ion_num) + '[' + str(charge) + ']'
                ion_key_array.append(ion_key)

    return ion_key_array


def get_experimental_mz_array(frag_dict):
    # Arrays that we will be filling out
    ion_key_array = get_frag_ion_array()
    ret_mz_array = []

    # Now that we have the ions ordered correctly we can fill in the experimental array
    for ion_key in ion_key_array:
        # Check to see if the ion_key is in the frag_int_dict
        if ion_key in frag_dict:
            ret_mz_array.append(frag_dict[ion_key])
        # If it is not present then it shouldn't be possible, and we can return a -1
        else:
            ret_mz_array.append(-1)

    return ret_mz_array


def switch_aa(aa):
    # This switches an amino acid with the integer that represents the amino acid in the ML model
    switcher = {
        'A': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'K': 9,
        'L': 10,
        'M': 11,
        'N': 12,
        'P': 13,
        'Q': 14,
        'R': 15,
        'S': 16,
        'T': 17,
        'V': 18,
        'W': 19,
        'Y': 20,
        'B': 21,
        'X': 22
    }

    return switcher[aa]


def get_peptide_array(peptide_sequence, tmt=False):
    # Skip the peptide if we can't handle the mods
    if not skip_peptide(peptide_sequence):

        # Clean the peptide, converting modifications into character representation that can be switched to integer
        clean_pep = peptide_sequence.replace('M[15.9949]', 'B')
        clean_pep = clean_pep.replace('M(ox)', 'B')
        clean_pep = clean_pep.replace('C[57.0215]', 'C')
        clean_pep = clean_pep.replace('R[10.0082]', 'R')
        clean_pep = clean_pep.replace('K[8.0142]', 'K')

        # If we are doing TMT then K becomes X
        if tmt:
            clean_pep = clean_pep.replace('K', 'X')

        # If there are flanking residues then remove those
        if '.' in clean_pep:
            clean_pep = clean_pep.split('.')[1]

        # For each amino acid switch with the integer value
        peptide_array = []
        for aa in clean_pep:
            peptide_array.append(switch_aa(aa))

        # Make the array into the standard 30 aa length
        for i in range(len(peptide_array), 30):
            peptide_array.append(0)

        return peptide_array


def get_charge_array(pep_charge):
    # One hot encoding of charge
    pep_charge_array = []

    # Make an array of length 6 with zeros
    for i in range(0, 6):
        pep_charge_array.append(False)

    # Set the value associated with this charge to 1
    pep_charge_array[pep_charge - 1] = True

    # Return the charge array
    return pep_charge_array


def peptides_to_spectral_library(peptide_data, loaded_model):
    # List that will hold the inputs to do a batch prediction
    expanded_row_list = []
    peptide_mz_list = []
    peptide_charge_list = []
    peptide_array_list = []
    charge_array_list = []
    col_energy_array_list = []
    ion_key_array_list = []
    mz_array_list = []

    # List of charges that we will predict
    charges = [1, 2, 3]

    # Same for all peptides
    frag_ion_array = get_frag_ion_array()

    # Iterate through the pin data
    for index, row in peptide_data.iterrows():
        # Get the peptide sequence and the charge from the pin data
        peptide = row['peptide']

        for charge in charges:
            # Here we will create a peptide object from the sequence and also calculate the fragments
            peptide_obj, frag_dict = create_peptide(peptide, charge=charge)

            # Get any arrays that will always be the same per peptide
            mz_array = get_experimental_mz_array(frag_dict)
            peptide_array = get_peptide_array(peptide)

            # Save the expanded row
            expanded_row_list.append(row)

            # Save the precursor mz and charge
            peptide_mz_list.append(mz_from_mass(peptide_obj.mass, charge))
            peptide_charge_list.append(charge)

            # Save the ion key array
            ion_key_array_list.append(frag_ion_array)

            # Save the m/z array for the ions
            mz_array_list.append(mz_array)

            # Save the peptide array - the int representation of the aa
            peptide_array_list.append(peptide_array)

            # Save the one-hot vector for the charge
            charge_array = get_charge_array(charge)
            charge_array_list.append(charge_array)

            # Calculate the ce to use for prediction
            ce = .10 + (.05 * charge)

            # Save the collision energy
            col_energy_array_list.append([ce])

    # Prepare the arrays needed to perform prediction
    peptide_array = np.array(peptide_array_list)
    charge_array = np.array(charge_array_list)
    col_energy_array = np.array(col_energy_array_list)

    # Perform the prediction of the spectrum
    ml_input = [peptide_array, charge_array, col_energy_array]
    prediction = loaded_model.predict(ml_input)

    # Create the lists that will make up the library
    lib_prec_mz = []
    lib_prod_mz = []
    lib_frag_annotation = []
    lib_protein_id = []
    lib_gene_name = []
    lib_peptide_seq = []
    lib_peptide_seq_mod = []
    lib_prec_charge = []
    lib_prod_ion_int = []
    lib_norm_rt = []
    lib_prec_ion_mob = []
    lib_frag_series = []
    lib_frag_charge = []
    lib_frag_number = []
    lib_frag_loss = []
    lib_average_exp_rt = []

    # Each entry in the mz_array_list is a unique peptide sequence and charge state
    i = 0
    for l_mz_array in mz_array_list:
        # Get the associated arrays that were saved above
        l_ion_key_array = ion_key_array_list[i]
        l_prediction_int_array = prediction[i]

        l_prec_mz = peptide_mz_list[i]
        l_prec_charge = peptide_charge_list[i]
        l_protein_id = expanded_row_list[i]['gene_id']
        l_gene_name = expanded_row_list[i]['gene_symbol']
        l_peptide_seq = expanded_row_list[i]['peptide'].replace('(ox)', '')
        l_peptide_seq_mod = expanded_row_list[i]['peptide'].replace('(ox)', '(UniMod:35)').replace('C', 'C(UniMod:4)')
        l_peptide_ion_mob = ''
        l_peptide_rt = expanded_row_list[i]['normalized_retention_time']
        l_peptide_expt_rt = ''

        l_prod_mz_list = []
        l_frag_annotation_list = []
        l_frag_series_list = []
        l_frag_charge_list = []
        l_frag_number_list = []
        l_frag_loss_list = []
        l_prod_ion_int_list = []

        # Add to the arrays but don't add impossible ions
        j = 0
        for mz in l_mz_array:
            if not mz == -1:
                # See if the predicted intensity is greater than 0
                predicted_int = l_prediction_int_array[j]

                if predicted_int > 0: # and int(frag_charge) <= int(prec_charge)
                    # Parse out the fragment ion information from this format y10[+];y10[2+];y10[3+]
                    frag_str = l_ion_key_array[j]
                    frag_charge_str = frag_str.split('[')[1].split(']')[0]
                    frag_charge = frag_charge_str.replace('3+', '3').replace('2+', '2').replace('+','1')  # TODO: Clean up this code to convert to charge state
                    frag_series = frag_str[0]
                    frag_number = frag_str.replace(frag_series, '').replace('[' + frag_charge_str + ']', '')

                    l_prod_mz_list.append(mz)
                    l_frag_annotation_list.append(str(frag_series) + str(frag_number) + '^' + str(frag_charge))
                    l_frag_series_list.append(frag_series)
                    l_frag_charge_list.append(frag_charge)
                    l_frag_number_list.append(frag_number)
                    l_frag_loss_list.append('') # TODO: Update when we have fragment ions with neutral loss
                    l_prod_ion_int_list.append(predicted_int)

            j += 1

        l_norm_prod_ion_int_list = [10000*(x / max(l_prod_ion_int_list)) for x in l_prod_ion_int_list]

        k = 0
        for prod_mz in l_prod_mz_list:
            # First get the general information from the input file
            lib_protein_id.append(l_protein_id)
            lib_gene_name.append(l_gene_name)
            lib_peptide_seq.append(l_peptide_seq)
            lib_peptide_seq_mod.append(l_peptide_seq_mod)

            # Grab the precursor mz
            lib_prec_mz.append(l_prec_mz)
            lib_prec_charge.append(l_prec_charge)

            # Other peptide specific parameters
            lib_norm_rt.append(l_peptide_rt)
            lib_prec_ion_mob.append(l_peptide_ion_mob)
            lib_average_exp_rt.append(l_peptide_expt_rt)

            # Grab fragment specific
            lib_prod_mz.append(l_prod_mz_list[k])
            lib_frag_annotation.append(l_frag_annotation_list[k])
            lib_frag_series.append(l_frag_series_list[k])
            lib_frag_charge.append(l_frag_charge_list[k])
            lib_frag_number.append(l_frag_number_list[k])
            lib_frag_loss.append(l_frag_loss_list[k])
            lib_prod_ion_int.append(l_norm_prod_ion_int_list[k])

            # Increment the counter on the frag ion list
            k += 1

        # Increment the counter on the unique peptide list
        i += 1

    spec_library_data = pd.DataFrame({
        'PrecursorMz': lib_prec_mz,
        'ProductMz': lib_prod_mz,
        'Annotation': lib_frag_annotation,
        'ProteinId': lib_protein_id,
        'GeneName': lib_gene_name,
        'PeptideSequence': lib_peptide_seq,
        'ModifiedPeptideSequence': lib_peptide_seq_mod,
        'PrecursorCharge': lib_prec_charge,
        'LibraryIntensity': lib_prod_ion_int,
        'NormalizedRetentionTime': lib_norm_rt,
        'PrecursorIonMobility': lib_prec_ion_mob,
        'FragmentType': lib_frag_series,
        'FragmentCharge': lib_frag_charge,
        'FragmentSeriesNumber': lib_frag_number,
        'FragmentLossType': lib_frag_loss,
        'AverageExperimentalRetentionTime': lib_average_exp_rt,
    })

    return spec_library_data


def peptides_to_spectral_library_irt(peptide_data, loaded_irt_model):
    # Peptide array list to predict iRT values
    peptide_array_list = []

    # Iterate through the pin data
    for index, row in peptide_data.iterrows():
        # Get the peptide sequence and the charge from the pin data
        peptide = row['peptide']

        # Create the peptide array input for the ML model
        peptide_array = get_peptide_array(peptide)

        # Add the peptide array to the list of arrays to rescore
        peptide_array_list.append(peptide_array)

    # Do the prediction of the iRT values
    irt_prediction_input = np.array(peptide_array_list)
    irt_prediction = loaded_irt_model.predict(irt_prediction_input)

    # Change the prediction results into a list that we can compare to the experimental RTs
    irt_list = []
    for irt_value_array in irt_prediction:
        # Grab the value from the array
        irt_list.append(float(irt_value_array)*100/2)

    # Add the raw irt values to the pin data
    peptide_data.insert(len(peptide_data.columns), "normalized_retention_time", np.array(irt_list))

    return peptide_data


def peptides_to_spectral_library_ionmob(peptide_data, im_predictions):
    # Create a dataframe that will be used for the merge, leaving out the mz for this
    data_to_merge = pd.read_csv(im_predictions, sep='\t')

    # Calculate the 1/K0 given reference data
    data_to_merge = calculate_ion_mobility(data_to_merge)

    # Do the merge with the input data
    return_peptide_data = pd.merge(peptide_data, data_to_merge, how='left',
                                   left_on=['ModifiedPeptideSequence', 'PrecursorCharge'],
                                   right_on=['ModifiedPeptideSequence', 'PrecursorCharge'])

    return_peptide_data['PrecursorIonMobility'] = return_peptide_data['IonMobility']

    return_peptide_data = return_peptide_data.drop('IonMobility', axis=1)

    # Return the data
    return return_peptide_data


def calculate_ion_mobility(predicted_ccs_all):
        # Use a previous file of IM values to do the coversion
        reference_data_file = "./inputfiles/HLA_peptides_ionmobility_predictedCCS.csv"
        reference_data_all = pd.read_csv(reference_data_file)

        # Make the CCS predictions for both the reference and the predicted peptides
        charges = [1, 2, 3]
        return_data = pd.DataFrame(columns=predicted_ccs_all.columns)
        for charge in charges:
            reference_data = reference_data_all.loc[reference_data_all['PrecursorCharge'] == charge].reset_index(drop=True)
            predicted_data = predicted_ccs_all.loc[predicted_ccs_all['PrecursorCharge'] == charge].reset_index(drop=True)

            # Get the dataa in right format for regression
            l_x = np.array(reference_data['ccs_predicted']).reshape(-1, 1)
            l_y = np.array(reference_data['PrecursorIonMobility']).reshape(-1, 1)

            # Fit line using all data
            model = linear_model.LinearRegression()
            model.fit(l_x, l_y)

            # Robustly fit linear model with RANSAC algorithm
            model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
            model_ransac.fit(l_x, l_y)

            calculated_ion_mobility = []
            for index, row in predicted_data.iterrows():
                predicted_ccs = row['PredictedCCS']
                ion_mobility = float(model_ransac.predict([[predicted_ccs]]))
                calculated_ion_mobility.append(ion_mobility)

            predicted_data['IonMobility'] = calculated_ion_mobility

            return_data = pd.concat([return_data, predicted_data])

        return return_data


def add_variant_peptides(peptide_data):
    # Make a temporary data frame that we will iterate through
    temp_data = peptide_data.copy()

    # Iterate through the data frame and make additional peptide variants based on oxidation of methionine
    for index, row in temp_data.iterrows():
        # Get the peptide sequence
        peptide_sequence = row['peptide']

        # Replace the oxM with M(ox) which is a bit easier to handle
        isoforms = parser.isoforms(peptide_sequence, variable_mods={'ox': ['M']})

        # For each of the isoforms make a new row within the initial data frame
        for isoform in isoforms:
            if 'ox' in isoform:
                add_sequence = isoform.replace("oxM", "M(ox)")
                row['peptide'] = add_sequence
                peptide_data = pd.concat([peptide_data, pd.DataFrame([row])], ignore_index=True)

    # Return the data
    return peptide_data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Input files needed to run the code
    # Files needed to load the ML model
    frag_model_json = "./models/model_MS_hcd.json"
    frag_model_h5 = "./models/model_MS_hcd.h5"

    # This will load the model and get it ready for predictions
    loaded_frag_model = load_model(frag_model_json, frag_model_h5)

    # Files need to load the iRT model
    irt_model_json = "./models/model_IRT.json"
    irt_model_h5 = "./models/model_IRT.h5"

    # Load the iRT model
    loaded_iRT_model = load_model(irt_model_json, irt_model_h5)

    # File that contains peptides that are predicted binders from HLApollo
    pep_file = "./inputfiles/A375_Apollo_predictedbinders_waitlist.tsv"

    # File that contains CCS predictions from IonMob
    ionmob_predictions = "./inputfiles/A375_Apollo_predictedbinders_waitlist_ionmob.tsv"

    #######
    # Code to create a spectral library based on prediction of spectra and RT
    # Read in the file
    print("Loading Peptide File")
    peptide_input = pd.read_csv(pep_file, sep='\t', usecols=range(27), header=0)

    # The peptides contain odd characters, just get rid of them for now
    peptide_input['peptide'] = peptide_input['peptide'].str.replace('$', '')
    peptide_input['peptide'] = peptide_input['peptide'].str.replace('X', '')

    # Add variants for PTMs - right now it is just Oxidized M
    print("Creating Variant Peptides")
    print("Peptides before variants added: " + str(len(peptide_input)))
    peptide_input_w_variants = add_variant_peptides(peptide_input)
    print("Peptides after variants added: " + str(len(peptide_input_w_variants)))

    # Add predicted irt with an in-house built Prosit model
    print("Predicting iRT")
    peptide_input_with_irt = peptides_to_spectral_library_irt(peptide_input_w_variants, loaded_iRT_model)
    print("iRT Prediction Finished")

    # Add in the predicted spectrum using an in-house version of the Prosit model
    print("Predicting Spectra")
    spectral_library = peptides_to_spectral_library(peptide_input_with_irt, loaded_frag_model)
    print("Spectra Prediction Finished")

    # Add in ion mobility using ionMob
    print("Predicting Ion Mobility")
    spectral_library_with_ionmob = peptides_to_spectral_library_ionmob(spectral_library, ionmob_predictions)
    print("Ion Mobility Prediction Finished")

    # Print out the spectral library
    print("Printing Results")
    spectral_library_with_ionmob.to_csv(pep_file.replace('.tsv', '_speclib.tsv'), sep='\t', index=False)
    #######

    # Completed
    print("Done")

