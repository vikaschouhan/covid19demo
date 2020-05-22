from modules.utils import *

def get_form_schema():
    form_schema = {
            'patient_age' : {
                'weights' : {
                    '65+': 3,
                    '15-64': 2,
                    'Less than 14': 1,
                },
                'type': 'single',
            },
            'patient_gender': {
                'weights': {
                    'Male': 2,
                    'Female': 1,
                },
                'type': 'single',
            },
            'dermatologic_change': {
                'weights': {
                    'Yes': 3,
                    'Yes but resolved by it self': 2,
                    'No': 1,
                },
                'type': 'single',
            },
            'patient_contact': {
                'weights': {
                    'Close contact with confirmed case of Covid 19 infection': 3,
                    'Close contact with probable case of Covid 19 infection': 2,
                    'Presence in a healthcare facility': 2,
                    'Unknown': 1,
                    'None': 1,
                },
                'type': 'single',
            },
            'patient_symptoms': {
                'weights': {
                    'Shortness of breath': 4,
                    'Dry cough': 3,
                    'Sore throat': 3,
                    'Headache/Muscle ache': 3,
                    'Fever/Chills': 3,
                    'Runny nose': 2,
                    'Joint pain': 2,
                    'Chest pain': 2,
                    'Abdominal pain': 2,
                    'Diarrhea, vomiting or nausea': 2,
                    'Irritability/confusion': 2,
                    'Loss of smell': 2,
                    'Malaise/Fatigue': 2,
                    'Decrease in sense of taste': 2,
                    'None (Asymptomatic)': 1,
                },
                'type': 'multiple',
            },
            'patient_smoking_status': {
                'weights': {
                    'Current smoker': 2,
                    'Former smoker': 1,
                    'Never smoked': 1,
                },
                'type': 'single',
            },
            'pre_existing_medical_conditions': {
                'weights': {
                    'Lung disease (asthma, COPD, pulmonary fibrosis etc.)': 2,
                    'Diabetes': 2,
                    'Morbid obesity (BMI 40+)': 2,
                    'Hypertension': 2,
                    'Heart disease (coronary artery disease, congestive heart failure)': 2,
                    'Cancer': 2,
                    'Kidney disease': 1,
                    'Organ transplant recipient': 1,
                    'Immunodeficiency': 1,
                    'Inflammatory bowel disease': 1,
                    'Liver disease': 1,
                    'Other': 1
                },
                'type': 'multiple',
            }
        }

    return form_schema
# enddef

def _str(x):
    return str(x)
# enddef

def validate_form(form):
    schema = get_form_schema_details()
    weights = get_form_weights()

    if not isdict(form):
        return False, 'form should be a dictionary.'
    # endif
    for key_t in form:
        # Convert to string
        key_t = _str(key_t)

        if key_t not in schema:
            return False, 'Unknown key {}. Supported {}'.format(key_t, schema.keys())
        # endif

        if schema[key_t] == 'single':
            if not isinstance(form[key_t], str):
                return False, 'key {} should have a single value in string format.'.format(key_t)
            # endif
            if form[key_t] not in weights[key_t]:
                return False, 'value {} for key {} not found. Supported {}'.format(form[key_t], key_t, list(weights[key_t].keys()))
            # endif
        else:
            if not islist(form[key_t]):
                return False, 'key {} should have a list of values in string format.'.format(key_t)
            # endif
            new_keys = set(form[key_t]) - set(weights[key_t].keys())
            if len(new_keys) != 0:
                return False, 'keys {} not found. Supported {}'.format(new_keys, list(weights[key_t].keys()))
            # endif
        # endif
    # endfor

    return True, None
# enddef

@static('f', None)
def get_form_format():
    if get_form_format.f is None:
        _f_t = {}
        w_t = get_form_schema()
        for key_t in w_t:
            _f_t[key_t] = list(w_t[key_t]['weights'].keys())
        # endfor
        get_form_format.f = _f_t
    # endif

    return get_form_format.f
# enddef

@static('w', None)
def get_form_weights():
    if get_form_weights.w is None:
        _w_t = {}
        w_t = get_form_schema()
        for key_t in w_t:
            _w_t[key_t] = w_t[key_t]['weights']
        # endfor
        get_form_weights.w = _w_t
        return _w_t
    # endif
    
    return get_form_weights.w
# enddef

@static('z', None)
def get_form_schema_details():
    if get_form_schema_details.z is None:
        _z_t = {}
        z_t = get_form_schema()
        for key_t in z_t:
            _z_t[key_t] = z_t[key_t]['type']
        # endfor
        get_form_schema_details.z = _z_t
        return _z_t
    # endif

    return get_form_schema_details.z
# enddef

@static("nw", None)
def get_normalized_form_weights():
    if get_normalized_form_weights.nw is not None:
        return get_normalized_form_weights.nw
    # endif

    _fweights = get_form_weights()
    for key_t in _fweights:
        _fweights[key_t] = {k:v/sum(_fweights[key_t].values()) for k,v in _fweights[key_t].items()}
    # endfor

    get_normalized_form_weights.nw = _fweights
    return _fweights
# enddef

# max_score is 4 since maximum score is 4 in the json schema
# For lack of time, we are hardcoding it as such. It should be automatically
# derived from the schema
def calculate_form_weights(forms, max_score=3):
    norm_weights = get_form_weights()
    sub_keys_weights_dict = {}
    for key_t in forms:
        # Convert to string
        key_t = _str(key_t)

        if islist(forms[key_t]):
            for sub_key_t in forms[key_t]:
                # Convert to string
                sub_key_t = _str(sub_key_t)

                sub_keys_weights_dict[sub_key_t] = norm_weights[key_t][sub_key_t]
            # endfor
        else:
            sub_key_t = _str(forms[key_t])
            sub_keys_weights_dict[sub_key_t] = norm_weights[key_t][sub_key_t]
        # endif
    # endfor

    return sum(sub_keys_weights_dict.values())/(len(sub_keys_weights_dict) * max_score)
# enddef

def predict_form(forms):
    return calculate_form_weights(forms)
# enddef
