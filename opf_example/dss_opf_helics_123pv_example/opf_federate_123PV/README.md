This is the OPF Federate directory 

Files Structure:
- flex_load_info.json: information on flexible load values
- opf_federate_123PV.py: main opf code which calls scripts for running the optimal power flow. Current central optimizaiton is implemented as a proof-of-concept. 
- opf_grid_utility_script.py: utility script to unpack the helics values to be usable by OPF and packaging the results to be sent to feeder federate
- opf_input_mapping.json: subscribed messages from feeder federate
- opf_static_inputs.json: configuration files 
- /base_vals: safe-fail variables if OPF fails